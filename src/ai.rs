use std::{
    collections::{HashMap, HashSet},
    thread,
    time::{Duration, Instant},
};

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

    pub fn chat_with_shell_tool<F>(
        &self,
        history: &[ChatMessage],
        system_prompt: &str,
        user_prompt: &str,
        policy: ToolUsePolicy,
        extra_tools: &[ExternalToolDefinition],
        mut execute_tool: F,
    ) -> Result<ChatToolResponse, AppError>
    where
        F: FnMut(&ToolCallRequest) -> String,
    {
        const MAX_TOOL_ROUNDS: usize = 8;
        const MAX_TOTAL_TOOL_CALLS: usize = 20;
        const MAX_REPEAT_SAME_TOOL: usize = 3;

        let mut messages = build_base_messages(history, system_prompt, user_prompt);
        let mut tools = vec![shell_tool_definition()];
        tools.extend(extra_tools.iter().map(external_tool_definition));
        let mut forced_retry_used = false;
        let mut total_tool_calls: usize = 0;
        let mut tool_result_cache: HashMap<String, String> = HashMap::new();
        let mut same_tool_counter: HashMap<String, usize> = HashMap::new();
        let mut thinking_chunks: Vec<String> = Vec::new();
        let mut metrics = ChatMetrics::default();

        for _ in 0..MAX_TOOL_ROUNDS {
            let tool_choice = match policy {
                ToolUsePolicy::Auto => Some(json!("auto")),
                ToolUsePolicy::RequireAtLeastOne => {
                    if forced_retry_used {
                        Some(json!("auto"))
                    } else {
                        Some(json!("required"))
                    }
                }
            };
            let call = self.send_chat_completion(&ChatCompletionRequest {
                model: self.model.clone(),
                messages: messages.clone(),
                temperature: 0.2,
                tools: Some(tools.clone()),
                tool_choice,
            })?;
            metrics.api_rounds += 1;
            metrics.api_duration_ms += call.elapsed_ms;
            metrics.prompt_tokens += call.usage.prompt_tokens;
            metrics.completion_tokens += call.usage.completion_tokens;
            metrics.total_tokens += call.usage.total_tokens;
            let assistant = call.assistant;

            append_unique_thinking_chunk(
                &mut thinking_chunks,
                assistant_reasoning_text(&assistant),
            );
            let assistant_content = assistant_content_text(&assistant);
            let tool_calls = assistant.tool_calls;
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

                let mut finalize_reason: Option<&str> = None;
                for tool_call in tool_calls {
                    total_tool_calls += 1;
                    let request = ToolCallRequest {
                        id: tool_call.id,
                        name: tool_call.function.name,
                        arguments: tool_call.function.arguments,
                    };
                    let signature = normalize_tool_signature(&request.name, &request.arguments);
                    let tool_result = if let Some(reason) = finalize_reason {
                        build_guard_tool_result(reason)
                    } else if total_tool_calls > MAX_TOTAL_TOOL_CALLS {
                        finalize_reason = Some("tool_call_limit_exceeded");
                        build_guard_tool_result("tool_call_limit_exceeded")
                    } else {
                        let count_entry = same_tool_counter.entry(signature.clone()).or_insert(0);
                        *count_entry += 1;
                        if *count_entry > MAX_REPEAT_SAME_TOOL {
                            finalize_reason = Some("repeated_same_tool_call");
                            build_guard_tool_result("repeated_same_tool_call")
                        } else if let Some(cached) = tool_result_cache.get(&signature) {
                            cached.clone()
                        } else {
                            let result = execute_tool(&request);
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
                        finalize_reason.unwrap_or("unknown")
                    ));
                    return self.finalize_without_tools(messages, thinking_chunks, metrics);
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

            if !assistant_content.trim().is_empty() {
                return Ok(ChatToolResponse {
                    content: assistant_content,
                    thinking: merge_thinking_chunks(thinking_chunks),
                    metrics: with_cost(
                        metrics,
                        self.input_price_per_million,
                        self.output_price_per_million,
                    ),
                });
            }

            return Err(AppError::Ai("AI returned empty content".to_string()));
        }

        self.finalize_without_tools(messages, thinking_chunks, metrics)
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
                    let err_msg = format!("AI request failed: {err}");
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
        let content = assistant_content_text(&assistant);
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
        })
    }
}

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
    let normalized = chunk.trim();
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

fn build_chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        return trimmed.to_string();
    }
    format!("{trimmed}/chat/completions")
}
