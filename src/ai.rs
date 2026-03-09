use std::{
    collections::{HashMap, HashSet},
    error::Error as StdError,
    thread,
    time::{Duration, Instant},
};

use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{config::AiConfig, error::AppError, logging, mask::mask_sensitive};

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ToolUsePolicy {
    Auto,
    RequireAtLeastOne,
}

#[derive(Debug, Clone)]
pub struct ChatToolResponse {
    pub content: String,
    pub thinking: Option<String>,
    pub metrics: ChatMetrics,
    pub stop_reason: Option<ChatStopReason>,
    pub tool_rounds_used: usize,
    pub total_tool_calls: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatStopReason {
    ToolCallLimitExceeded,
    RepeatedSameToolCall,
    RepeatedToolTimeout,
    TooManyToolTimeouts,
    MaxToolRoundsReached,
}

impl ChatStopReason {
    pub fn code(self) -> &'static str {
        match self {
            ChatStopReason::ToolCallLimitExceeded => "tool_call_limit_exceeded",
            ChatStopReason::RepeatedSameToolCall => "repeated_same_tool_call",
            ChatStopReason::RepeatedToolTimeout => "repeated_tool_timeout",
            ChatStopReason::TooManyToolTimeouts => "too_many_tool_timeouts",
            ChatStopReason::MaxToolRoundsReached => "max_tool_rounds_reached",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatRoundEvent {
    pub round: usize,
    pub content: String,
    pub thinking: Option<String>,
    pub has_tool_calls: bool,
    pub tool_call_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ChatMetrics {
    pub api_rounds: usize,
    pub api_duration_ms: u128,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub estimated_cost_usd: f64,
}

#[derive(Debug, Clone)]
pub struct ExternalToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct AiClient {
    client: Client,
    base_url: String,
    token: String,
    model: String,
    max_retries: u32,
    backoff_millis: u64,
    input_price_per_million: f64,
    output_price_per_million: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolChoiceMode {
    Policy,
    AutoOnly,
    Disabled,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ApiMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Clone)]
struct ApiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct ApiTool {
    r#type: String,
    function: ApiFunctionDefinition,
}

#[derive(Debug, Serialize, Clone)]
struct ApiFunctionDefinition {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ApiToolCall {
    id: String,
    r#type: String,
    function: ApiToolFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ApiToolFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Debug, Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: Option<serde_json::Value>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    #[serde(default)]
    tool_calls: Vec<ApiToolCall>,
}

impl AiClient {
    pub fn new(cfg: &AiConfig) -> Result<Self, AppError> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(8))
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|err| AppError::Ai(format!("failed to build AI http client: {err}")))?;
        Ok(Self {
            client,
            base_url: build_chat_url(&cfg.base_url),
            token: cfg.token.clone(),
            model: cfg.model.clone(),
            max_retries: cfg.retry.max_retries,
            backoff_millis: cfg.retry.backoff_millis,
            input_price_per_million: cfg.input_price_per_million,
            output_price_per_million: cfg.output_price_per_million,
        })
    }

    pub fn validate_connectivity(&self) -> Result<(), AppError> {
        let system_prompt = "You are a connectivity checker. Reply with one word: OK.";
        let user_prompt = "Respond with OK only.";
        let response = self.chat(&[], system_prompt, user_prompt)?;
        if response.trim().is_empty() {
            return Err(AppError::Ai(
                "AI validation returned empty response".to_string(),
            ));
        }
        Ok(())
    }

    pub fn chat(
        &self,
        history: &[ChatMessage],
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, AppError> {
        let messages = build_base_messages(history, system_prompt, user_prompt);
        let call = self.send_chat_completion(&ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.2,
            tools: None,
            tool_choice: None,
        })?;
        let assistant = call.assistant;

        let content = assistant_content_text(&assistant);
        if content.trim().is_empty() {
            return Err(AppError::Ai("AI returned empty content".to_string()));
        }
        Ok(content)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chat_with_shell_tool<F>(
        &self,
        history: &[ChatMessage],
        system_prompt: &str,
        user_prompt: &str,
        policy: ToolUsePolicy,
        max_tool_rounds: usize,
        max_total_tool_calls: usize,
        extra_tools: &[ExternalToolDefinition],
        mut execute_tool: F,
        mut on_round_event: impl FnMut(ChatRoundEvent),
    ) -> Result<ChatToolResponse, AppError>
    where
        F: FnMut(&ToolCallRequest) -> String,
    {
        const MAX_REPEAT_SAME_TOOL: usize = 3;
        const MAX_TIMEOUT_TOOL_CALLS_TOTAL: usize = 2;
        const MAX_TIMEOUT_SAME_TOOL_CALL: usize = 1;
        const MAX_FORCE_CONTINUE_HINT_ROUNDS: usize = 2;

        let mut messages = build_base_messages(history, system_prompt, user_prompt);
        let mut tools = vec![shell_tool_definition()];
        tools.extend(extra_tools.iter().map(external_tool_definition));
        let mut forced_retry_used = false;
        let mut total_tool_calls: usize = 0;
        let mut tool_result_cache: HashMap<String, String> = HashMap::new();
        let mut same_tool_counter: HashMap<String, usize> = HashMap::new();
        let mut timeout_tool_counter: HashMap<String, usize> = HashMap::new();
        let mut timeout_total: usize = 0;
        let mut thinking_chunks: Vec<String> = Vec::new();
        let mut metrics = ChatMetrics::default();
        let mut tool_rounds_used: usize = 0;
        let mut forced_continue_hint_rounds: usize = 0;
        let mut tool_choice_mode = if model_prefers_omit_tool_choice(&self.model) {
            logging::info(
                "model prefers omitting tool_choice for compatibility; starting with tool_choice omitted",
            );
            ToolChoiceMode::Disabled
        } else {
            ToolChoiceMode::Policy
        };

        for _ in 0..max_tool_rounds {
            let tool_choice = match tool_choice_mode {
                ToolChoiceMode::Disabled => None,
                ToolChoiceMode::AutoOnly => Some(json!("auto")),
                ToolChoiceMode::Policy => match policy {
                    ToolUsePolicy::Auto => Some(json!("auto")),
                    ToolUsePolicy::RequireAtLeastOne => {
                        if forced_retry_used {
                            Some(json!("auto"))
                        } else {
                            Some(json!("required"))
                        }
                    }
                },
            };
            let call = match self.send_chat_completion(&ChatCompletionRequest {
                model: self.model.clone(),
                messages: messages.clone(),
                temperature: 0.2,
                tools: Some(tools.clone()),
                tool_choice,
            }) {
                Ok(call) => call,
                Err(err) => {
                    if is_tool_choice_unsupported_error(&err) {
                        match tool_choice_mode {
                            ToolChoiceMode::Policy => {
                                logging::warn(
                                    "model does not support requested tool_choice; fallback to tool_choice=auto",
                                );
                                tool_choice_mode = ToolChoiceMode::AutoOnly;
                                continue;
                            }
                            ToolChoiceMode::AutoOnly => {
                                logging::warn(
                                    "model does not support tool_choice=auto; fallback to tool_choice omitted",
                                );
                                tool_choice_mode = ToolChoiceMode::Disabled;
                                continue;
                            }
                            ToolChoiceMode::Disabled => {}
                        }
                    }
                    return Err(err);
                }
            };
            metrics.api_rounds += 1;
            tool_rounds_used += 1;
            metrics.api_duration_ms += call.elapsed_ms;
            metrics.prompt_tokens += call.usage.prompt_tokens;
            metrics.completion_tokens += call.usage.completion_tokens;
            metrics.total_tokens += call.usage.total_tokens;
            let assistant = call.assistant;
            let round_thinking = assistant_reasoning_text(&assistant);
            append_unique_thinking_chunk(&mut thinking_chunks, round_thinking.clone());
            let assistant_content = sanitize_assistant_content(&assistant_content_text(&assistant));
            let mut tool_calls = assistant.tool_calls.clone();
            if tool_calls.is_empty() {
                tool_calls =
                    parse_dsml_tool_calls(&assistant_content_text(&assistant), metrics.api_rounds)
                        .into_iter()
                        .map(|item| ApiToolCall {
                            id: item.id,
                            r#type: "function".to_string(),
                            function: ApiToolFunction {
                                name: item.name,
                                arguments: item.arguments,
                            },
                        })
                        .collect();
            }
            if !assistant_content.trim().is_empty()
                || !round_thinking.trim().is_empty()
                || !tool_calls.is_empty()
            {
                on_round_event(ChatRoundEvent {
                    round: metrics.api_rounds,
                    content: assistant_content.clone(),
                    thinking: if round_thinking.trim().is_empty() {
                        None
                    } else {
                        Some(round_thinking)
                    },
                    has_tool_calls: !tool_calls.is_empty(),
                    tool_call_count: tool_calls.len(),
                });
            }
            if !tool_calls.is_empty() {
                messages.push(ApiMessage {
                    role: "assistant".to_string(),
                    content: optional_content(assistant_content),
                    reasoning_content: Some(build_reasoning_content_for_tool_round(
                        assistant.reasoning_content.as_deref(),
                        assistant.reasoning.as_ref(),
                    )),
                    tool_calls: Some(tool_calls.clone()),
                    tool_call_id: None,
                });

                let mut finalize_reason: Option<ChatStopReason> = None;
                for tool_call in tool_calls {
                    total_tool_calls += 1;
                    let request = ToolCallRequest {
                        id: tool_call.id,
                        name: tool_call.function.name,
                        arguments: tool_call.function.arguments,
                    };
                    let signature = normalize_tool_signature(&request.name, &request.arguments);
                    let tool_result = if let Some(reason) = finalize_reason {
                        build_guard_tool_result(reason.code())
                    } else if total_tool_calls > max_total_tool_calls {
                        finalize_reason = Some(ChatStopReason::ToolCallLimitExceeded);
                        build_guard_tool_result("tool_call_limit_exceeded")
                    } else {
                        let count_entry = same_tool_counter.entry(signature.clone()).or_insert(0);
                        *count_entry += 1;
                        if *count_entry > MAX_REPEAT_SAME_TOOL {
                            finalize_reason = Some(ChatStopReason::RepeatedSameToolCall);
                            build_guard_tool_result("repeated_same_tool_call")
                        } else if let Some(cached) = tool_result_cache.get(&signature) {
                            if tool_result_timed_out(cached) {
                                timeout_total += 1;
                                let timeout_count =
                                    timeout_tool_counter.entry(signature.clone()).or_insert(0);
                                *timeout_count += 1;
                                if *timeout_count > MAX_TIMEOUT_SAME_TOOL_CALL {
                                    finalize_reason = Some(ChatStopReason::RepeatedToolTimeout);
                                } else if timeout_total > MAX_TIMEOUT_TOOL_CALLS_TOTAL {
                                    finalize_reason = Some(ChatStopReason::TooManyToolTimeouts);
                                }
                            }
                            cached.clone()
                        } else {
                            let result = execute_tool(&request);
                            if tool_result_timed_out(&result) {
                                timeout_total += 1;
                                let timeout_count =
                                    timeout_tool_counter.entry(signature.clone()).or_insert(0);
                                *timeout_count += 1;
                                if *timeout_count > MAX_TIMEOUT_SAME_TOOL_CALL {
                                    finalize_reason = Some(ChatStopReason::RepeatedToolTimeout);
                                } else if timeout_total > MAX_TIMEOUT_TOOL_CALLS_TOTAL {
                                    finalize_reason = Some(ChatStopReason::TooManyToolTimeouts);
                                }
                            }
                            tool_result_cache.insert(signature, result.clone());
                            result
                        }
                    };
                    messages.push(ApiMessage {
                        role: "tool".to_string(),
                        content: Some(tool_result),
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: Some(request.id),
                    });
                }
                if finalize_reason.is_some() {
                    logging::warn(&format!(
                        "tool-calling guard triggered, reason={}",
                        finalize_reason.map(|item| item.code()).unwrap_or("unknown")
                    ));
                    return self.finalize_without_tools(
                        messages,
                        thinking_chunks,
                        metrics,
                        finalize_reason,
                        tool_rounds_used,
                        total_tool_calls,
                    );
                }
                continue;
            }

            if matches!(policy, ToolUsePolicy::RequireAtLeastOne) && !forced_retry_used {
                forced_retry_used = true;
                messages.push(ApiMessage {
                    role: "system".to_string(),
                    content: Some("You are running locally with direct tool access. You MUST call run_shell_command at least once before giving a final answer for this request.".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
                continue;
            }

            if !assistant_content.trim().is_empty()
                && forced_continue_hint_rounds < MAX_FORCE_CONTINUE_HINT_ROUNDS
                && should_force_continue_with_tools(&assistant_content)
            {
                forced_continue_hint_rounds += 1;
                logging::warn(
                    "assistant asked user to run/provide command output; forcing another tool-calling round",
                );
                messages.push(ApiMessage {
                    role: "assistant".to_string(),
                    content: optional_content(assistant_content.clone()),
                    reasoning_content: Some(build_reasoning_content_for_tool_round(
                        assistant.reasoning_content.as_deref(),
                        assistant.reasoning.as_ref(),
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                });
                messages.push(ApiMessage {
                    role: "system".to_string(),
                    content: Some("You have direct local tool access in this CLI. Continue investigating by calling tools yourself. Do NOT ask the user to execute commands or provide command output unless execution is explicitly blocked by policy/safety constraints.".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
                continue;
            }

            if !assistant_content.trim().is_empty() {
                return Ok(ChatToolResponse {
                    content: assistant_content,
                    thinking: merge_thinking_chunks(thinking_chunks),
                    metrics: with_cost(
                        metrics,
                        self.input_price_per_million,
                        self.output_price_per_million,
                    ),
                    stop_reason: None,
                    tool_rounds_used,
                    total_tool_calls,
                });
            }

            return Err(AppError::Ai("AI returned empty content".to_string()));
        }

        self.finalize_without_tools(
            messages,
            thinking_chunks,
            metrics,
            Some(ChatStopReason::MaxToolRoundsReached),
            tool_rounds_used,
            total_tool_calls,
        )
    }

    fn send_chat_completion(
        &self,
        body: &ChatCompletionRequest,
    ) -> Result<ApiChatCallResult, AppError> {
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            logging::info(&format!("AI request started, attempt={attempt}"));
            let started = Instant::now();
            let resp = self
                .client
                .post(&self.base_url)
                .bearer_auth(&self.token)
                .json(body)
                .send();

            match resp {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let body = resp
                            .text()
                            .unwrap_or_else(|_| "<unreadable body>".to_string());
                        let safe_body = mask_sensitive(&body);
                        let err_msg = format!("AI HTTP status={status}, body={safe_body}");
                        logging::warn(&err_msg);
                        if attempt <= self.max_retries {
                            thread::sleep(Duration::from_millis(self.backoff_millis));
                            continue;
                        }
                        return Err(AppError::Ai(err_msg));
                    }

                    let parsed: ChatCompletionResponse = resp.json().map_err(|err| {
                        AppError::Ai(format!("failed to parse AI response: {err}"))
                    })?;
                    if let Some(choice) = parsed.choices.into_iter().next() {
                        logging::info("AI request finished successfully");
                        return Ok(ApiChatCallResult {
                            assistant: choice.message,
                            usage: parsed.usage,
                            elapsed_ms: started.elapsed().as_millis(),
                        });
                    }

                    let err_msg = "AI returned empty choices".to_string();
                    logging::warn(&err_msg);
                    if attempt <= self.max_retries {
                        thread::sleep(Duration::from_millis(self.backoff_millis));
                        continue;
                    }
                    return Err(AppError::Ai(err_msg));
                }
                Err(err) => {
                    let err_msg = format!("AI request failed: {}", format_reqwest_error(&err));
                    logging::warn(&err_msg);
                    if attempt <= self.max_retries {
                        thread::sleep(Duration::from_millis(self.backoff_millis));
                        continue;
                    }
                    return Err(AppError::Ai(err_msg));
                }
            }
        }
    }

    fn finalize_without_tools(
        &self,
        mut messages: Vec<ApiMessage>,
        mut thinking_chunks: Vec<String>,
        mut metrics: ChatMetrics,
        stop_reason: Option<ChatStopReason>,
        tool_rounds_used: usize,
        total_tool_calls: usize,
    ) -> Result<ChatToolResponse, AppError> {
        messages.push(ApiMessage {
            role: "system".to_string(),
            content: Some("Now provide a final answer based on available tool outputs. Do not call any more tools.".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        });
        let call = self.send_chat_completion(&ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.2,
            tools: None,
            tool_choice: None,
        })?;
        metrics.api_rounds += 1;
        metrics.api_duration_ms += call.elapsed_ms;
        metrics.prompt_tokens += call.usage.prompt_tokens;
        metrics.completion_tokens += call.usage.completion_tokens;
        metrics.total_tokens += call.usage.total_tokens;
        let assistant = call.assistant;
        append_unique_thinking_chunk(&mut thinking_chunks, assistant_reasoning_text(&assistant));
        let content = sanitize_assistant_content(&assistant_content_text(&assistant));
        if content.trim().is_empty() {
            return Err(AppError::Ai(
                "AI returned empty content after tool-calling finalization".to_string(),
            ));
        }
        Ok(ChatToolResponse {
            content,
            thinking: merge_thinking_chunks(thinking_chunks),
            metrics: with_cost(
                metrics,
                self.input_price_per_million,
                self.output_price_per_million,
            ),
            stop_reason,
            tool_rounds_used,
            total_tool_calls,
        })
    }
}

fn format_reqwest_error(err: &reqwest::Error) -> String {
    let category = if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connect"
    } else if err.is_decode() {
        "decode"
    } else if err.is_status() {
        "status"
    } else if err.is_request() {
        "request"
    } else {
        "unknown"
    };

    let mut causes = Vec::<String>::new();
    let mut source = err.source();
    while let Some(item) = source {
        let text = item.to_string();
        if !text.trim().is_empty() {
            causes.push(text);
        }
        if causes.len() >= 4 {
            break;
        }
        source = item.source();
    }

    let hint = if err.is_timeout() {
        "hint=检查网络连通性/代理与防火墙策略，或适当增大重试与超时"
    } else if err.is_connect() {
        "hint=检查 ai.base-url、DNS 解析、证书链和出口网络策略"
    } else {
        "hint=检查 AI 服务可用性与配置项是否正确"
    };

    if causes.is_empty() {
        format!("kind={category}, error={err}, {hint}")
    } else {
        format!(
            "kind={category}, error={err}, causes={}, {hint}",
            causes.join(" | ")
        )
    }
}

static DSML_FUNCTION_CALLS_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<[|｜]dsml[|｜]function_calls>.*?</[|｜]dsml[|｜]function_calls>"#)
        .expect("valid dsml function call block regex")
});
static DSML_INVOKE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<[|｜]dsml[|｜]invoke\s+name=\"([^\"]+)\"[^>]*>(.*?)</[|｜]dsml[|｜]invoke>"#,
    )
    .expect("valid dsml invoke regex")
});
static DSML_PARAMETER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<[|｜]dsml[|｜]parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</[|｜]dsml[|｜]parameter>"#,
    )
    .expect("valid dsml parameter regex")
});

#[derive(Debug)]
struct ApiChatCallResult {
    assistant: AssistantMessage,
    usage: Usage,
    elapsed_ms: u128,
}

fn assistant_content_text(message: &AssistantMessage) -> String {
    let Some(content) = message.content.as_ref() else {
        return String::new();
    };
    extract_text_from_value(content).trim().to_string()
}

fn extract_text_from_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.to_string(),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(extract_text_like_from_item)
            .filter(|item| !item.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(|v| v.as_str()) {
                return text.to_string();
            }
            if let Some(text) = map.get("content").and_then(|v| v.as_str()) {
                return text.to_string();
            }
            if let Some(content) = map.get("content") {
                return extract_text_from_value(content);
            }
            String::new()
        }
        _ => String::new(),
    }
}

fn extract_text_like_from_item(item: &serde_json::Value) -> Option<String> {
    if let Some(text) = item.as_str() {
        return Some(text.to_string());
    }
    if let Some(obj) = item.as_object() {
        if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
            return Some(text.to_string());
        }
        if let Some(text) = obj.get("output_text").and_then(|v| v.as_str()) {
            return Some(text.to_string());
        }
        if let Some(text) = obj.get("content").and_then(|v| v.as_str()) {
            return Some(text.to_string());
        }
        if let Some(content) = obj.get("content") {
            let nested = extract_text_from_value(content);
            if !nested.trim().is_empty() {
                return Some(nested);
            }
        }
    }
    None
}

fn assistant_reasoning_text(message: &AssistantMessage) -> String {
    let mut chunks = Vec::new();
    if let Some(text) = message.reasoning_content.as_deref() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
    }
    if let Some(reasoning) = message.reasoning.as_ref() {
        collect_reasoning_from_value(reasoning, &mut chunks);
    }
    if let Some(content) = message.content.as_ref() {
        collect_reasoning_from_value(content, &mut chunks);
    }
    chunks.join("\n")
}

fn collect_reasoning_from_value(value: &serde_json::Value, output: &mut Vec<String>) {
    match value {
        serde_json::Value::String(_) | serde_json::Value::Null | serde_json::Value::Bool(_) => {}
        serde_json::Value::Number(_) => {}
        serde_json::Value::Array(items) => {
            for item in items {
                collect_reasoning_from_value(item, output);
            }
        }
        serde_json::Value::Object(map) => {
            let is_reasoning_block = map
                .get("type")
                .and_then(|v| v.as_str())
                .map(|t| t.to_ascii_lowercase().contains("reasoning"))
                .unwrap_or(false);
            if is_reasoning_block {
                for key in [
                    "text",
                    "summary",
                    "content",
                    "reasoning",
                    "reasoning_content",
                ] {
                    if let Some(value) = map.get(key) {
                        let text = extract_text_from_value(value);
                        if !text.trim().is_empty() {
                            output.push(text.trim().to_string());
                        }
                    }
                }
            }
            for key in ["reasoning", "reasoning_content", "summary"] {
                if let Some(value) = map.get(key) {
                    let text = extract_text_from_value(value);
                    if !text.trim().is_empty() {
                        output.push(text.trim().to_string());
                    }
                }
            }
            for value in map.values() {
                collect_reasoning_from_value(value, output);
            }
        }
    }
}

fn append_unique_thinking_chunk(chunks: &mut Vec<String>, chunk: String) {
    let cleaned_chunk = sanitize_assistant_content(&chunk);
    let normalized = cleaned_chunk.trim();
    if normalized.is_empty() {
        return;
    }
    if chunks.iter().any(|existing| existing.trim() == normalized) {
        return;
    }
    chunks.push(normalized.to_string());
}

fn merge_thinking_chunks(chunks: Vec<String>) -> Option<String> {
    let mut seen = HashSet::new();
    let merged = chunks
        .into_iter()
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .filter(|item| seen.insert(item.clone()))
        .collect::<Vec<_>>()
        .join("\n\n");
    if merged.is_empty() {
        return None;
    }
    Some(merged)
}

fn build_guard_tool_result(reason: &str) -> String {
    json!({
        "ok": false,
        "skipped": true,
        "reason": reason
    })
    .to_string()
}

fn normalize_tool_signature(name: &str, arguments: &str) -> String {
    if name == "run_shell_command"
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments)
    {
        let command = value
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim();
        let mode = value
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("read")
            .trim()
            .to_ascii_lowercase();
        let normalized_mode = if mode == "write" { "write" } else { "read" };
        return format!("{name}::{normalized_mode}::{command}");
    }
    format!("{name}::{}", arguments.trim())
}

fn tool_result_timed_out(payload: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()
        .and_then(|value| value.get("timed_out").and_then(|item| item.as_bool()))
        .unwrap_or(false)
}

fn build_base_messages(
    history: &[ChatMessage],
    system_prompt: &str,
    user_prompt: &str,
) -> Vec<ApiMessage> {
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(ApiMessage {
        role: "system".to_string(),
        content: Some(system_prompt.to_string()),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    });
    for item in history {
        let normalized_role = match item.role.as_str() {
            "assistant" => "assistant",
            "system" => "system",
            _ => "user",
        };
        messages.push(ApiMessage {
            role: normalized_role.to_string(),
            content: Some(item.content.clone()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }
    messages.push(ApiMessage {
        role: "user".to_string(),
        content: Some(user_prompt.to_string()),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    });
    messages
}

fn build_reasoning_content_for_tool_round(
    direct_reasoning: Option<&str>,
    structured_reasoning: Option<&serde_json::Value>,
) -> String {
    if let Some(text) = direct_reasoning {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Some(value) = structured_reasoning {
        let extracted = extract_text_from_value(value).trim().to_string();
        if !extracted.is_empty() {
            return extracted;
        }
    }
    String::new()
}

fn shell_tool_definition() -> ApiTool {
    ApiTool {
        r#type: "function".to_string(),
        function: ApiFunctionDefinition {
            name: "run_shell_command".to_string(),
            description: "Execute local shell command on current machine. Prefer read-only commands. Use mode=write only when mutation is necessary.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Short label of this command"},
                    "command": {"type": "string", "description": "Command string to execute"},
                    "mode": {"type": "string", "enum": ["read", "write"], "description": "read for non-mutating, write for mutating command"}
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        },
    }
}

fn external_tool_definition(tool: &ExternalToolDefinition) -> ApiTool {
    ApiTool {
        r#type: "function".to_string(),
        function: ApiFunctionDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        },
    }
}

fn with_cost(
    mut metrics: ChatMetrics,
    input_price_per_million: f64,
    output_price_per_million: f64,
) -> ChatMetrics {
    let input_cost = (metrics.prompt_tokens as f64 / 1_000_000.0) * input_price_per_million;
    let output_cost = (metrics.completion_tokens as f64 / 1_000_000.0) * output_price_per_million;
    metrics.estimated_cost_usd = input_cost + output_cost;
    metrics
}

fn optional_content(content: String) -> Option<String> {
    if content.trim().is_empty() {
        return None;
    }
    Some(content)
}

fn sanitize_assistant_content(raw: &str) -> String {
    let cleaned = DSML_FUNCTION_CALLS_BLOCK_RE.replace_all(raw, "");
    cleaned.trim().to_string()
}

fn parse_dsml_tool_calls(raw: &str, round: usize) -> Vec<ToolCallRequest> {
    if !raw.to_ascii_lowercase().contains("dsml") {
        return Vec::new();
    }
    let mut output = Vec::new();
    for (idx, invoke_caps) in DSML_INVOKE_RE.captures_iter(raw).enumerate() {
        let function_name_raw = invoke_caps
            .get(1)
            .map(|m| m.as_str().trim())
            .unwrap_or_default();
        let invoke_body = invoke_caps.get(2).map(|m| m.as_str()).unwrap_or_default();
        let mut params = serde_json::Map::new();
        for param_caps in DSML_PARAMETER_RE.captures_iter(invoke_body) {
            let key = param_caps
                .get(1)
                .map(|m| m.as_str().trim())
                .unwrap_or_default();
            let value = param_caps
                .get(2)
                .map(|m| decode_dsml_text(m.as_str()))
                .unwrap_or_default();
            if !key.is_empty() {
                params.insert(key.to_string(), serde_json::Value::String(value));
            }
        }
        let normalized_name = normalize_dsml_tool_name(function_name_raw);
        if normalized_name == "run_shell_command" && !params.contains_key("command") {
            continue;
        }
        output.push(ToolCallRequest {
            id: format!("dsml_round_{}_{}", round, idx + 1),
            name: normalized_name,
            arguments: serde_json::Value::Object(params).to_string(),
        });
    }
    output
}

fn normalize_dsml_tool_name(name: &str) -> String {
    let normalized = name.trim().to_ascii_lowercase().replace('_', "");
    if normalized == "runshellcommand" || normalized == "run_shell_command" {
        return "run_shell_command".to_string();
    }
    name.trim().to_string()
}

fn decode_dsml_text(raw: &str) -> String {
    raw.replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .trim()
        .to_string()
}

fn is_tool_choice_unsupported_error(err: &AppError) -> bool {
    let AppError::Ai(detail) = err else {
        return false;
    };
    let lowered = detail.to_ascii_lowercase();
    lowered.contains("does not support this tool_choice")
        || lowered.contains("unsupported tool_choice")
}

fn model_prefers_omit_tool_choice(model: &str) -> bool {
    let normalized = model.trim().to_ascii_lowercase();
    normalized == "deepseek-reasoner"
}

fn should_force_continue_with_tools(content: &str) -> bool {
    let lowered = content.to_ascii_lowercase();
    let patterns = [
        "请执行",
        "请在终端执行",
        "请手动执行",
        "把结果发给我",
        "把输出发给我",
        "请提供输出",
        "请提供结果",
        "我无法自动执行",
        "无法自动执行命令",
        "please run",
        "run this command",
        "paste the output",
        "provide the output",
        "i cannot execute",
        "can't execute commands",
        "execute it yourself",
    ];
    patterns.iter().any(|p| lowered.contains(p))
}

fn build_chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        return trimmed.to_string();
    }
    format!("{trimmed}/chat/completions")
}
