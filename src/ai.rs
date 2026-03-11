use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    error::Error as StdError,
    fs,
    io::{BufRead, BufReader, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    sync::RwLock,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::RETRY_AFTER;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    config::AiConfig, error::AppError, i18n, logging, mask::mask_sensitive, render,
    shell::take_interactive_input_refresh_hint, tls::ensure_rustls_crypto_provider,
};

const MODEL_PRICE_CACHE_TTL_SECS: u64 = 7 * 24 * 60 * 60;
const MODEL_PRICE_CACHE_VERSION: u32 = 1;
const MAX_AUTO_RATE_LIMIT_WAIT_SECS: u64 = 15;
const AI_HTTP_CONNECT_TIMEOUT_SECS: u64 = 8;
const AI_HTTP_REQUEST_TIMEOUT_SECS: u64 = 60;
const AI_HTTP_POOL_IDLE_TIMEOUT_SECS: u64 = 30;
const AI_HTTP_TCP_KEEPALIVE_SECS: u64 = 30;

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
    pub archived_content: String,
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
    RateLimited,
}

impl ChatStopReason {
    pub fn code(self) -> &'static str {
        match self {
            ChatStopReason::ToolCallLimitExceeded => "tool_call_limit_exceeded",
            ChatStopReason::RepeatedSameToolCall => "repeated_same_tool_call",
            ChatStopReason::RepeatedToolTimeout => "repeated_tool_timeout",
            ChatStopReason::TooManyToolTimeouts => "too_many_tool_timeouts",
            ChatStopReason::MaxToolRoundsReached => "max_tool_rounds_reached",
            ChatStopReason::RateLimited => "rate_limited",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatRoundEvent {
    pub round: usize,
    pub content: String,
    pub thinking: Option<String>,
    pub streamed_content: bool,
    pub streamed_thinking: bool,
    pub has_tool_calls: bool,
    pub tool_call_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatStreamEventKind {
    Content,
    Thinking,
}

#[derive(Debug, Clone)]
pub struct ChatStreamEvent {
    pub kind: ChatStreamEventKind,
    pub text: String,
}

#[derive(Debug, Clone, Default)]
pub struct ChatMetrics {
    pub api_rounds: usize,
    pub api_duration_ms: u128,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub estimated_cost_usd: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ExternalToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug)]
pub struct AiClient {
    client: RwLock<Client>,
    base_url: String,
    token: String,
    model: String,
    colorful: bool,
    model_price_cache_path: PathBuf,
    debug: bool,
    max_retries: u32,
    backoff_millis: u64,
    input_price_per_million: f64,
    output_price_per_million: f64,
    runtime_model_prices: RwLock<Option<(f64, f64)>>,
}

impl Clone for AiClient {
    fn clone(&self) -> Self {
        let client = self
            .client
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_else(|_| {
                ensure_rustls_crypto_provider();
                Client::builder().build().expect("fallback AI client clone")
            });
        let cached_prices = self
            .runtime_model_prices
            .read()
            .ok()
            .and_then(|guard| *guard);
        Self {
            client: RwLock::new(client),
            base_url: self.base_url.clone(),
            token: self.token.clone(),
            model: self.model.clone(),
            colorful: self.colorful,
            model_price_cache_path: self.model_price_cache_path.clone(),
            debug: self.debug,
            max_retries: self.max_retries,
            backoff_millis: self.backoff_millis,
            input_price_per_million: self.input_price_per_million,
            output_price_per_million: self.output_price_per_million,
            runtime_model_prices: RwLock::new(cached_prices),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPriceSource {
    Configured,
    LocalCache,
    RuntimeProbe,
    Builtin,
    Unavailable,
}

#[derive(Debug, Clone)]
pub struct ModelPriceCheckResult {
    pub source: ModelPriceSource,
    pub prices: Option<(f64, f64)>,
    pub probe_skipped: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolChoiceMode {
    Policy,
    AutoOnly,
    Disabled,
}

#[derive(Debug, Serialize, Clone)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ApiMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize, Clone)]
struct ApiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "option_string_is_blank")]
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

#[derive(Debug, Deserialize)]
struct ChatCompletionStreamResponse {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default, deserialize_with = "deserialize_usage_or_default")]
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

fn deserialize_usage_or_default<'de, D>(deserializer: D) -> Result<Usage, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<Usage>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: AssistantDeltaMessage,
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

#[derive(Debug, Deserialize, Default)]
struct AssistantDeltaMessage {
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    #[serde(default)]
    tool_calls: Vec<StreamToolCallDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct StreamToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    r#type: Option<String>,
    #[serde(default)]
    function: Option<StreamToolFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct StreamToolFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct PersistedModelPriceCatalog {
    #[serde(default = "default_model_price_cache_version")]
    version: u32,
    #[serde(default)]
    models: BTreeMap<String, PersistedModelPriceEntry>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PersistedModelPriceEntry {
    input_price_per_million: f64,
    output_price_per_million: f64,
    checked_at_epoch_secs: u64,
}

impl AiClient {
    fn build_http_client() -> Result<Client, AppError> {
        ensure_rustls_crypto_provider();
        Client::builder()
            .connect_timeout(Duration::from_secs(AI_HTTP_CONNECT_TIMEOUT_SECS))
            .timeout(Duration::from_secs(AI_HTTP_REQUEST_TIMEOUT_SECS))
            .pool_idle_timeout(Duration::from_secs(AI_HTTP_POOL_IDLE_TIMEOUT_SECS))
            .tcp_keepalive(Duration::from_secs(AI_HTTP_TCP_KEEPALIVE_SECS))
            .build()
            .map_err(|err| AppError::Ai(format!("failed to build AI http client: {err}")))
    }

    pub fn new(
        cfg: &AiConfig,
        model_price_cache_path: PathBuf,
        colorful: bool,
    ) -> Result<Self, AppError> {
        let client = Self::build_http_client()?;
        Ok(Self {
            client: RwLock::new(client),
            base_url: build_chat_url(&cfg.base_url),
            token: cfg.token.clone(),
            model: cfg.model.clone(),
            colorful,
            model_price_cache_path,
            debug: cfg.debug,
            max_retries: cfg.retry.max_retries,
            backoff_millis: cfg.retry.backoff_millis,
            input_price_per_million: cfg.input_price_per_million,
            output_price_per_million: cfg.output_price_per_million,
            runtime_model_prices: RwLock::new(None),
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

    pub fn prepare_model_pricing(&self, skip_probe: bool) -> ModelPriceCheckResult {
        if let Some(prices) =
            configured_model_prices(self.input_price_per_million, self.output_price_per_million)
        {
            return ModelPriceCheckResult {
                source: ModelPriceSource::Configured,
                prices: Some(prices),
                probe_skipped: true,
            };
        }
        if skip_probe {
            let prices = builtin_model_prices(&self.model);
            return ModelPriceCheckResult {
                source: if prices.is_some() {
                    ModelPriceSource::Builtin
                } else {
                    ModelPriceSource::Unavailable
                },
                prices,
                probe_skipped: true,
            };
        }
        if let Some(prices) = self.cached_runtime_model_prices() {
            return ModelPriceCheckResult {
                source: ModelPriceSource::RuntimeProbe,
                prices: Some(prices),
                probe_skipped: false,
            };
        }
        if let Some(prices) = self.cached_persisted_model_prices() {
            self.set_runtime_model_prices(prices);
            return ModelPriceCheckResult {
                source: ModelPriceSource::LocalCache,
                prices: Some(prices),
                probe_skipped: false,
            };
        }
        match self.probe_model_price_catalog() {
            Ok(probed_prices) => {
                if !probed_prices.is_empty()
                    && let Err(err) = self.persist_model_price_catalog(&probed_prices)
                {
                    self.debug_emit(
                        AiDebugLevel::Warn,
                        &format!(
                            "failed to persist model price catalog: {}",
                            mask_sensitive(&err.to_string())
                        ),
                    );
                }
                if let Some(prices) = probed_prices.get(&normalize_priced_model_name(&self.model)) {
                    let prices = *prices;
                    self.set_runtime_model_prices(prices);
                    return ModelPriceCheckResult {
                        source: ModelPriceSource::RuntimeProbe,
                        prices: Some(prices),
                        probe_skipped: false,
                    };
                }
                let prices = builtin_model_prices(&self.model);
                ModelPriceCheckResult {
                    source: if prices.is_some() {
                        ModelPriceSource::Builtin
                    } else {
                        ModelPriceSource::Unavailable
                    },
                    prices,
                    probe_skipped: false,
                }
            }
            Err(err) => {
                self.debug_emit(
                    AiDebugLevel::Warn,
                    &format!(
                        "model pricing probe failed: {}",
                        mask_sensitive(&err.to_string())
                    ),
                );
                let prices = builtin_model_prices(&self.model);
                ModelPriceCheckResult {
                    source: if prices.is_some() {
                        ModelPriceSource::Builtin
                    } else {
                        ModelPriceSource::Unavailable
                    },
                    prices,
                    probe_skipped: false,
                }
            }
        }
    }

    pub fn chat(
        &self,
        history: &[ChatMessage],
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, AppError> {
        let messages = build_base_messages(history, system_prompt, user_prompt);
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.2,
            tools: None,
            tool_choice: None,
            stream: None,
        };
        let call = self.send_chat_completion(&request)?;
        let assistant = call.assistant;

        let content = assistant_content_text(&assistant);
        if content.trim().is_empty() {
            return Err(AppError::Ai("AI returned empty content".to_string()));
        }
        Ok(content)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chat_with_shell_tool<F, G>(
        &self,
        history: &[ChatMessage],
        system_prompt: &str,
        user_prompt: &str,
        policy: ToolUsePolicy,
        max_tool_rounds: usize,
        max_total_tool_calls: usize,
        stream_output: bool,
        extra_tools: &[ExternalToolDefinition],
        mut execute_tool: F,
        mut on_round_event: impl FnMut(ChatRoundEvent),
        mut on_stream_event: G,
    ) -> Result<ChatToolResponse, AppError>
    where
        F: FnMut(&ToolCallRequest) -> String,
        G: FnMut(ChatStreamEvent),
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
        let mut visible_assistant_chunks: Vec<String> = Vec::new();
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
            let request = ChatCompletionRequest {
                model: self.model.clone(),
                messages: messages.clone(),
                temperature: 0.2,
                tools: Some(tools.clone()),
                tool_choice,
                stream: None,
            };
            let call = match self.send_chat_completion_with_optional_streaming(
                &request,
                stream_output,
                &mut on_stream_event,
            ) {
                Ok(call) => call,
                Err(err) => {
                    if is_rate_limited_error(&err) {
                        logging::warn(
                            "AI rate limited during tool-calling round; returning recoverable local fallback",
                        );
                        return Ok(self.build_recoverable_chat_response(
                            FinalizeWithoutToolsState {
                                thinking_chunks,
                                visible_assistant_chunks,
                                metrics,
                                tool_rounds_used,
                                total_tool_calls,
                            },
                            Some(ChatStopReason::RateLimited),
                        ));
                    }
                    if is_transient_ai_error(&err)
                        && (total_tool_calls > 0
                            || !thinking_chunks.is_empty()
                            || !visible_assistant_chunks.is_empty())
                    {
                        logging::warn(
                            "transient AI failure during tool-calling round; returning recoverable local fallback",
                        );
                        return Ok(self.build_transient_ai_failure_chat_response(
                            FinalizeWithoutToolsState {
                                thinking_chunks,
                                visible_assistant_chunks,
                                metrics,
                                tool_rounds_used,
                                total_tool_calls,
                            },
                            &err,
                        ));
                    }
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
            append_unique_visible_chunk(&mut visible_assistant_chunks, assistant_content.clone());
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
                    streamed_content: call.streamed_content,
                    streamed_thinking: call.streamed_thinking,
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
                        } else {
                            let cacheable_request = is_cacheable_tool_request(&request);
                            if cacheable_request {
                                if let Some(cached) = tool_result_cache.get(&signature) {
                                    if tool_result_timed_out(cached) {
                                        timeout_total += 1;
                                        let timeout_count = timeout_tool_counter
                                            .entry(signature.clone())
                                            .or_insert(0);
                                        *timeout_count += 1;
                                        if *timeout_count > MAX_TIMEOUT_SAME_TOOL_CALL {
                                            finalize_reason =
                                                Some(ChatStopReason::RepeatedToolTimeout);
                                        } else if timeout_total > MAX_TIMEOUT_TOOL_CALLS_TOTAL {
                                            finalize_reason =
                                                Some(ChatStopReason::TooManyToolTimeouts);
                                        }
                                    }
                                    cached.clone()
                                } else {
                                    let result = execute_tool(&request);
                                    if tool_result_timed_out(&result) {
                                        timeout_total += 1;
                                        let timeout_count = timeout_tool_counter
                                            .entry(signature.clone())
                                            .or_insert(0);
                                        *timeout_count += 1;
                                        if *timeout_count > MAX_TIMEOUT_SAME_TOOL_CALL {
                                            finalize_reason =
                                                Some(ChatStopReason::RepeatedToolTimeout);
                                        } else if timeout_total > MAX_TIMEOUT_TOOL_CALLS_TOTAL {
                                            finalize_reason =
                                                Some(ChatStopReason::TooManyToolTimeouts);
                                        }
                                    }
                                    tool_result_cache.insert(signature, result.clone());
                                    result
                                }
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
                                result
                            }
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
                        FinalizeWithoutToolsState {
                            thinking_chunks,
                            visible_assistant_chunks,
                            metrics,
                            tool_rounds_used,
                            total_tool_calls,
                        },
                        finalize_reason,
                        stream_output,
                        &mut on_stream_event,
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

            if matches!(policy, ToolUsePolicy::RequireAtLeastOne)
                && total_tool_calls == 0
                && forced_continue_hint_rounds < MAX_FORCE_CONTINUE_HINT_ROUNDS
            {
                forced_continue_hint_rounds += 1;
                logging::warn(
                    "policy requires at least one tool call; assistant returned no tool calls, forcing another round",
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
                    content: Some("Policy requires at least one local tool call in this round. You must call a tool now instead of giving a final text-only response.".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
                continue;
            }

            if !assistant_content.trim().is_empty() {
                return Ok(ChatToolResponse {
                    archived_content: merge_visible_chunks(
                        &visible_assistant_chunks,
                        Some(assistant_content.as_str()),
                    ),
                    content: assistant_content,
                    thinking: merge_thinking_chunks(thinking_chunks),
                    metrics: self.attach_cost(metrics),
                    stop_reason: None,
                    tool_rounds_used,
                    total_tool_calls,
                });
            }

            if total_tool_calls > 0
                || !thinking_chunks.is_empty()
                || !visible_assistant_chunks.is_empty()
            {
                logging::warn(
                    "assistant returned empty final content after tool-calling round; using recoverable fallback response",
                );
                return Ok(self.build_recoverable_chat_response(
                    FinalizeWithoutToolsState {
                        thinking_chunks,
                        visible_assistant_chunks,
                        metrics,
                        tool_rounds_used,
                        total_tool_calls,
                    },
                    None,
                ));
            }

            return Err(AppError::Ai("AI returned empty content".to_string()));
        }

        self.finalize_without_tools(
            messages,
            FinalizeWithoutToolsState {
                thinking_chunks,
                visible_assistant_chunks,
                metrics,
                tool_rounds_used,
                total_tool_calls,
            },
            Some(ChatStopReason::MaxToolRoundsReached),
            stream_output,
            &mut on_stream_event,
        )
    }

    fn send_chat_completion(
        &self,
        body: &ChatCompletionRequest,
    ) -> Result<ApiChatCallResult, AppError> {
        let mut effective_body = normalize_request_for_provider(body, &self.base_url);
        let mut stripped_reasoning_retry_used = false;
        let mut retry_with_fresh_client = false;
        let mut refreshed_for_idle_hint = false;
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            logging::info(&format!("AI request started, attempt={attempt}"));
            self.debug_emit(
                AiDebugLevel::Info,
                &format!(
                    "AI request start: attempt={attempt}, url={}, model={}, stream=false, messages={}",
                    self.base_url,
                    effective_body.model,
                    effective_body.messages.len(),
                ),
            );
            self.debug_emit(
                AiDebugLevel::Debug,
                &format!("AI request body: {}", serialize_debug_json(&effective_body)),
            );
            if !refreshed_for_idle_hint && take_interactive_input_refresh_hint() {
                self.reconnect_http_client()?;
                maybe_print_ai_reconnect_notice(
                    i18n::chat_ai_reconnecting_after_idle(),
                    self.colorful,
                );
                refreshed_for_idle_hint = true;
            }
            let started = Instant::now();
            let client = if retry_with_fresh_client {
                self.debug_emit(
                    AiDebugLevel::Warn,
                    "AI request retry is using a fresh HTTP client to avoid stale idle connections",
                );
                self.reconnect_http_client()?;
                self.current_http_client()?
            } else {
                self.current_http_client()?
            };
            let resp = client
                .post(&self.base_url)
                .bearer_auth(&self.token)
                .json(&effective_body)
                .send();

            match resp {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let retry_after = resp
                            .headers()
                            .get(RETRY_AFTER)
                            .and_then(|value| value.to_str().ok())
                            .map(|value| value.to_string());
                        let body = resp
                            .text()
                            .unwrap_or_else(|_| "<unreadable body>".to_string());
                        let safe_body = mask_sensitive(&body);
                        let err_msg = format!("AI HTTP status={status}, body={safe_body}");
                        self.debug_emit(
                            AiDebugLevel::Error,
                            &format!("AI response error body: {safe_body}"),
                        );
                        if !stripped_reasoning_retry_used
                            && should_retry_without_reasoning_content(status, &body)
                            && request_contains_reasoning_content(&effective_body)
                        {
                            logging::warn(
                                "AI rejected chat message; retrying once without reasoning_content for compatibility",
                            );
                            self.debug_emit(
                                AiDebugLevel::Warn,
                                &format!(
                                    "AI compatibility fallback triggered: retrying without reasoning_content, status={status}, body={safe_body}"
                                ),
                            );
                            strip_reasoning_content_from_request(&mut effective_body);
                            stripped_reasoning_retry_used = true;
                            continue;
                        }
                        logging::warn(&err_msg);
                        if attempt <= self.max_retries {
                            if let Some(delay) =
                                parse_rate_limit_retry_delay(status, retry_after.as_deref())
                            {
                                thread::sleep(delay);
                                continue;
                            }
                            if should_retry_status(status) {
                                thread::sleep(Duration::from_millis(self.backoff_millis));
                                continue;
                            }
                        }
                        return Err(AppError::Ai(err_msg));
                    }

                    let response_text = resp.text().map_err(|err| {
                        AppError::Ai(format!("failed to read AI response body: {err}"))
                    })?;
                    self.debug_emit(
                        AiDebugLevel::Debug,
                        &format!("AI response body: {}", mask_sensitive(&response_text)),
                    );
                    let parsed = parse_chat_completion_response_text(&response_text)?;
                    logging::info("AI request finished successfully");
                    self.debug_emit(
                        AiDebugLevel::Info,
                        &format!(
                            "AI request finished: elapsed_ms={}, prompt_tokens={}, completion_tokens={}, total_tokens={}",
                            started.elapsed().as_millis(),
                            parsed.usage.prompt_tokens,
                            parsed.usage.completion_tokens,
                            parsed.usage.total_tokens,
                        ),
                    );
                    return Ok(ApiChatCallResult {
                        assistant: parsed.message,
                        usage: parsed.usage,
                        elapsed_ms: started.elapsed().as_millis(),
                        streamed_content: false,
                        streamed_thinking: false,
                    });
                }
                Err(err) => {
                    let err_msg = format!("AI request failed: {}", format_reqwest_error(&err));
                    logging::warn(&err_msg);
                    if attempt <= self.max_retries {
                        retry_with_fresh_client =
                            should_refresh_http_client_after_reqwest_error(&err);
                        if retry_with_fresh_client {
                            maybe_print_ai_reconnect_notice(
                                &i18n::chat_ai_reconnecting(attempt, self.max_retries),
                                self.colorful,
                            );
                        }
                        thread::sleep(Duration::from_millis(self.backoff_millis));
                        continue;
                    }
                    return Err(AppError::Ai(err_msg));
                }
            }
        }
    }

    fn send_chat_completion_with_optional_streaming<G>(
        &self,
        body: &ChatCompletionRequest,
        stream_output: bool,
        on_stream_event: &mut G,
    ) -> Result<ApiChatCallResult, AppError>
    where
        G: FnMut(ChatStreamEvent),
    {
        if !stream_output {
            return self.send_chat_completion(body);
        }

        let mut effective_body = normalize_request_for_provider(body, &self.base_url);
        let mut stripped_reasoning_retry_used = false;
        let mut retry_with_fresh_client = false;
        let mut refreshed_for_idle_hint = false;
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            logging::info(&format!("AI streaming request started, attempt={attempt}"));
            self.debug_emit(
                AiDebugLevel::Info,
                &format!(
                    "AI streaming request start: attempt={attempt}, url={}, model={}, stream=true, messages={}",
                    self.base_url,
                    effective_body.model,
                    effective_body.messages.len(),
                ),
            );
            let started = Instant::now();
            let mut stream_body = effective_body.clone();
            stream_body.stream = Some(true);
            self.debug_emit(
                AiDebugLevel::Debug,
                &format!(
                    "AI streaming request body: {}",
                    serialize_debug_json(&stream_body)
                ),
            );
            if !refreshed_for_idle_hint && take_interactive_input_refresh_hint() {
                self.reconnect_http_client()?;
                maybe_print_ai_reconnect_notice(
                    i18n::chat_ai_reconnecting_after_idle(),
                    self.colorful,
                );
                refreshed_for_idle_hint = true;
            }
            let client = if retry_with_fresh_client {
                self.debug_emit(
                    AiDebugLevel::Warn,
                    "AI streaming retry is using a fresh HTTP client to avoid stale idle connections",
                );
                self.reconnect_http_client()?;
                self.current_http_client()?
            } else {
                self.current_http_client()?
            };
            let resp = client
                .post(&self.base_url)
                .bearer_auth(&self.token)
                .json(&stream_body)
                .send();

            match resp {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let retry_after = resp
                            .headers()
                            .get(RETRY_AFTER)
                            .and_then(|value| value.to_str().ok())
                            .map(|value| value.to_string());
                        let response_body = resp
                            .text()
                            .unwrap_or_else(|_| "<unreadable body>".to_string());
                        let safe_body = mask_sensitive(&response_body);
                        self.debug_emit(
                            AiDebugLevel::Error,
                            &format!("AI streaming response error body: {safe_body}"),
                        );
                        if !stripped_reasoning_retry_used
                            && should_retry_without_reasoning_content(status, &response_body)
                            && request_contains_reasoning_content(&effective_body)
                        {
                            logging::warn(
                                "AI streaming request rejected chat message; retrying once without reasoning_content for compatibility",
                            );
                            self.debug_emit(
                                AiDebugLevel::Warn,
                                &format!(
                                    "AI streaming compatibility fallback triggered: retrying without reasoning_content, status={status}, body={safe_body}"
                                ),
                            );
                            strip_reasoning_content_from_request(&mut effective_body);
                            stripped_reasoning_retry_used = true;
                            continue;
                        }
                        if should_fallback_to_non_streaming(status, &response_body) {
                            logging::warn(&format!(
                                "AI streaming unsupported, fallback to non-streaming, status={status}, body={safe_body}"
                            ));
                            self.debug_emit(
                                AiDebugLevel::Warn,
                                &format!(
                                    "AI streaming fallback to non-streaming triggered: status={status}, body={safe_body}"
                                ),
                            );
                            return self.send_chat_completion(body);
                        }
                        let err_msg = format!("AI HTTP status={status}, body={safe_body}");
                        logging::warn(&err_msg);
                        if attempt <= self.max_retries {
                            if let Some(delay) =
                                parse_rate_limit_retry_delay(status, retry_after.as_deref())
                            {
                                thread::sleep(delay);
                                continue;
                            }
                            if should_retry_status(status) {
                                thread::sleep(Duration::from_millis(self.backoff_millis));
                                continue;
                            }
                        }
                        return Err(AppError::Ai(err_msg));
                    }

                    match parse_streaming_chat_response(resp, on_stream_event, self.debug) {
                        Ok(parsed) => {
                            logging::info("AI streaming request finished successfully");
                            self.debug_emit(
                                AiDebugLevel::Info,
                                &format!(
                                    "AI streaming request finished: elapsed_ms={}, streamed_content={}, streamed_thinking={}, prompt_tokens={}, completion_tokens={}, total_tokens={}",
                                    started.elapsed().as_millis(),
                                    parsed.streamed_content,
                                    parsed.streamed_thinking,
                                    parsed.usage.prompt_tokens,
                                    parsed.usage.completion_tokens,
                                    parsed.usage.total_tokens,
                                ),
                            );
                            return Ok(ApiChatCallResult {
                                assistant: parsed.assistant,
                                usage: parsed.usage,
                                elapsed_ms: started.elapsed().as_millis(),
                                streamed_content: parsed.streamed_content,
                                streamed_thinking: parsed.streamed_thinking,
                            });
                        }
                        Err(StreamParseResult::FallbackJson(body_text)) => {
                            logging::warn(
                                "AI streaming response was not SSE; fallback to non-streaming JSON parsing",
                            );
                            self.debug_emit(
                                AiDebugLevel::Warn,
                                &format!(
                                    "AI streaming response fallback to JSON body: {}",
                                    mask_sensitive(&body_text)
                                ),
                            );
                            let parsed = parse_chat_completion_response_text(&body_text)?;
                            return Ok(ApiChatCallResult {
                                assistant: parsed.message,
                                usage: parsed.usage,
                                elapsed_ms: started.elapsed().as_millis(),
                                streamed_content: false,
                                streamed_thinking: false,
                            });
                        }
                        Err(StreamParseResult::Retryable(err_msg)) => {
                            self.debug_emit(AiDebugLevel::Warn, &err_msg);
                            logging::warn(&err_msg);
                            if attempt <= self.max_retries {
                                retry_with_fresh_client = true;
                                maybe_print_ai_reconnect_notice(
                                    &i18n::chat_ai_reconnecting(attempt, self.max_retries),
                                    self.colorful,
                                );
                                thread::sleep(Duration::from_millis(self.backoff_millis));
                                continue;
                            }
                            return Err(AppError::Ai(err_msg));
                        }
                        Err(StreamParseResult::Fatal(err_msg)) => {
                            self.debug_emit(AiDebugLevel::Error, &err_msg);
                            logging::warn(&err_msg);
                            return Err(AppError::Ai(err_msg));
                        }
                    }
                }
                Err(err) => {
                    let err_msg = format!("AI request failed: {}", format_reqwest_error(&err));
                    self.debug_emit(AiDebugLevel::Error, &err_msg);
                    logging::warn(&err_msg);
                    if attempt <= self.max_retries {
                        retry_with_fresh_client =
                            should_refresh_http_client_after_reqwest_error(&err);
                        if retry_with_fresh_client {
                            maybe_print_ai_reconnect_notice(
                                &i18n::chat_ai_reconnecting(attempt, self.max_retries),
                                self.colorful,
                            );
                        }
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
        mut state: FinalizeWithoutToolsState,
        stop_reason: Option<ChatStopReason>,
        stream_output: bool,
        on_stream_event: &mut impl FnMut(ChatStreamEvent),
    ) -> Result<ChatToolResponse, AppError> {
        messages.push(ApiMessage {
            role: "system".to_string(),
            content: Some("Now provide a final answer based on available tool outputs. Do not call any more tools.".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        });
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.2,
            tools: None,
            tool_choice: None,
            stream: None,
        };
        let call = match self.send_chat_completion_with_optional_streaming(
            &request,
            stream_output,
            on_stream_event,
        ) {
            Ok(call) => call,
            Err(err) => {
                if is_rate_limited_error(&err) {
                    logging::warn(
                        "AI rate limited during finalization; returning recoverable local fallback",
                    );
                    return Ok(self.build_recoverable_chat_response(
                        state,
                        Some(ChatStopReason::RateLimited),
                    ));
                }
                if is_transient_ai_error(&err) {
                    logging::warn(
                        "transient AI failure during finalization; returning recoverable local fallback",
                    );
                    return Ok(self.build_transient_ai_failure_chat_response(state, &err));
                }
                return Err(err);
            }
        };
        state.metrics.api_rounds += 1;
        state.metrics.api_duration_ms += call.elapsed_ms;
        state.metrics.prompt_tokens += call.usage.prompt_tokens;
        state.metrics.completion_tokens += call.usage.completion_tokens;
        state.metrics.total_tokens += call.usage.total_tokens;
        let assistant = call.assistant;
        append_unique_thinking_chunk(
            &mut state.thinking_chunks,
            assistant_reasoning_text(&assistant),
        );
        let raw_content = assistant_content_text(&assistant);
        let content = sanitize_assistant_content(&raw_content);
        let archived_content =
            merge_visible_chunks(&state.visible_assistant_chunks, Some(&content));
        if content.trim().is_empty() {
            logging::warn(
                "AI returned empty content after tool-calling finalization; using local fallback response",
            );
            return Ok(self.build_recoverable_chat_response(state, stop_reason));
        }
        Ok(ChatToolResponse {
            archived_content,
            content,
            thinking: merge_thinking_chunks(state.thinking_chunks),
            metrics: self.attach_cost(state.metrics),
            stop_reason,
            tool_rounds_used: state.tool_rounds_used,
            total_tool_calls: state.total_tool_calls,
        })
    }

    fn build_recoverable_chat_response(
        &self,
        state: FinalizeWithoutToolsState,
        stop_reason: Option<ChatStopReason>,
    ) -> ChatToolResponse {
        let archived_content = merge_visible_chunks(&state.visible_assistant_chunks, None);
        let content = if archived_content.trim().is_empty() {
            build_finalization_fallback(stop_reason, state.tool_rounds_used, state.total_tool_calls)
        } else {
            archived_content.clone()
        };
        ChatToolResponse {
            archived_content: if archived_content.trim().is_empty() {
                content.clone()
            } else {
                archived_content
            },
            content,
            thinking: merge_thinking_chunks(state.thinking_chunks),
            metrics: self.attach_cost(state.metrics),
            stop_reason,
            tool_rounds_used: state.tool_rounds_used,
            total_tool_calls: state.total_tool_calls,
        }
    }

    fn build_transient_ai_failure_chat_response(
        &self,
        state: FinalizeWithoutToolsState,
        err: &AppError,
    ) -> ChatToolResponse {
        let archived_content = merge_visible_chunks(&state.visible_assistant_chunks, None);
        let fallback = build_transient_ai_failure_fallback(err);
        let content = if archived_content.trim().is_empty() {
            fallback.clone()
        } else {
            format!("{archived_content}\n\n{fallback}")
        };
        ChatToolResponse {
            archived_content: content.clone(),
            content,
            thinking: merge_thinking_chunks(state.thinking_chunks),
            metrics: self.attach_cost(state.metrics),
            stop_reason: None,
            tool_rounds_used: state.tool_rounds_used,
            total_tool_calls: state.total_tool_calls,
        }
    }

    fn debug_emit(&self, level: AiDebugLevel, message: &str) {
        emit_ai_debug(self.debug, level, message);
    }

    fn current_http_client(&self) -> Result<Client, AppError> {
        self.client
            .read()
            .map(|guard| guard.clone())
            .map_err(|_| AppError::Ai("failed to read AI http client".to_string()))
    }

    pub fn reconnect_http_client(&self) -> Result<(), AppError> {
        let new_client = Self::build_http_client()?;
        let mut guard = self
            .client
            .write()
            .map_err(|_| AppError::Ai("failed to refresh AI http client".to_string()))?;
        *guard = new_client;
        self.debug_emit(
            AiDebugLevel::Info,
            "AI http client refreshed to recover from idle/disconnected transport state",
        );
        Ok(())
    }

    fn attach_cost(&self, metrics: ChatMetrics) -> ChatMetrics {
        with_cost(
            metrics,
            &self.model,
            self.input_price_per_million,
            self.output_price_per_million,
            self.cached_runtime_model_prices(),
        )
    }

    fn cached_runtime_model_prices(&self) -> Option<(f64, f64)> {
        self.runtime_model_prices
            .read()
            .ok()
            .and_then(|guard| *guard)
    }

    fn set_runtime_model_prices(&self, prices: (f64, f64)) {
        if let Ok(mut guard) = self.runtime_model_prices.write() {
            *guard = Some(prices);
        }
    }

    fn cached_persisted_model_prices(&self) -> Option<(f64, f64)> {
        let normalized_model = normalize_priced_model_name(&self.model);
        let catalog = match self.load_model_price_catalog() {
            Ok(Some(catalog)) => catalog,
            Ok(None) => return None,
            Err(err) => {
                self.debug_emit(
                    AiDebugLevel::Warn,
                    &format!(
                        "failed to load model price cache: {}",
                        mask_sensitive(&err.to_string())
                    ),
                );
                return None;
            }
        };
        let prices = fresh_model_price_catalog(&catalog, now_epoch_secs())
            .get(&normalized_model)
            .copied();
        if let Some((input, output)) = prices {
            self.debug_emit(
                AiDebugLevel::Info,
                &format!(
                    "using persisted model pricing cache for model={} input={} output={}",
                    normalized_model, input, output
                ),
            );
        }
        prices
    }

    fn load_model_price_catalog(&self) -> Result<Option<PersistedModelPriceCatalog>, AppError> {
        let raw = match fs::read_to_string(&self.model_price_cache_path) {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(AppError::Runtime(format!(
                    "failed to read model price cache {}: {err}",
                    self.model_price_cache_path.display()
                )));
            }
        };
        let parsed = serde_json::from_str::<PersistedModelPriceCatalog>(&raw).map_err(|err| {
            AppError::Runtime(format!(
                "failed to parse model price cache {}: {err}",
                self.model_price_cache_path.display()
            ))
        })?;
        Ok(Some(parsed))
    }

    fn persist_model_price_catalog(
        &self,
        incoming_prices: &BTreeMap<String, (f64, f64)>,
    ) -> Result<(), AppError> {
        if incoming_prices.is_empty() {
            return Ok(());
        }
        let mut catalog = match self.load_model_price_catalog() {
            Ok(Some(existing)) => existing,
            Ok(None) => PersistedModelPriceCatalog::default(),
            Err(err) => {
                self.debug_emit(
                    AiDebugLevel::Warn,
                    &format!(
                        "failed to load existing model price cache before persist: {}",
                        mask_sensitive(&err.to_string())
                    ),
                );
                PersistedModelPriceCatalog::default()
            }
        };
        let now = now_epoch_secs();
        catalog.version = MODEL_PRICE_CACHE_VERSION;
        catalog.models = retained_fresh_model_price_entries(catalog.models, now);
        for (model, (input, output)) in incoming_prices {
            if !is_valid_model_price(*input, *output) {
                continue;
            }
            catalog.models.insert(
                normalize_priced_model_name(model),
                PersistedModelPriceEntry {
                    input_price_per_million: *input,
                    output_price_per_million: *output,
                    checked_at_epoch_secs: now,
                },
            );
        }
        let raw = serde_json::to_string_pretty(&catalog).map_err(|err| {
            AppError::Runtime(format!("failed to serialize model price cache: {err}"))
        })?;
        write_string_atomically(&self.model_price_cache_path, &raw)?;
        self.debug_emit(
            AiDebugLevel::Info,
            &format!(
                "persisted {} model price entries to {}",
                catalog.models.len(),
                self.model_price_cache_path.display()
            ),
        );
        Ok(())
    }

    fn probe_model_price_catalog(&self) -> Result<BTreeMap<String, (f64, f64)>, AppError> {
        let candidate_models = price_probe_candidate_models(&self.model);
        self.debug_emit(
            AiDebugLevel::Info,
            &format!(
                "probing model pricing catalog for model={} candidates={}",
                self.model,
                candidate_models.join(",")
            ),
        );
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ApiMessage {
                    role: "system".to_string(),
                    content: Some("You are a pricing probe. Return exactly one strict minified JSON object and nothing else.".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                ApiMessage {
                    role: "user".to_string(),
                    content: Some(format!(
                        "Return exactly one minified JSON object whose top-level keys are model ids and whose values are objects in the form {{\"input_price_per_million\":number,\"output_price_per_million\":number}}. Only include models you are confident about from this candidate list: {}. Always include `{}` if you are confident. If a model is uncertain, omit it entirely. Do not include markdown, code fences, explanations, comments, trailing commas, or extra keys.",
                        serde_json::to_string(&candidate_models).unwrap_or_else(|_| "[]".to_string()),
                        self.model
                    )),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            temperature: 0.0,
            tools: None,
            tool_choice: None,
            stream: None,
        };
        let call = self.send_chat_completion(&request)?;
        let response_text = assistant_content_text(&call.assistant);
        self.debug_emit(
            AiDebugLevel::Debug,
            &format!("model pricing probe raw response: {response_text}"),
        );
        Ok(parse_model_price_catalog_response(
            &response_text,
            &normalize_priced_model_name(&self.model),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
enum AiDebugLevel {
    Info,
    Debug,
    Warn,
    Error,
}

impl AiDebugLevel {
    fn as_str(self) -> &'static str {
        match self {
            AiDebugLevel::Info => "info",
            AiDebugLevel::Debug => "debug",
            AiDebugLevel::Warn => "warn",
            AiDebugLevel::Error => "error",
        }
    }

    fn localized_tag(self) -> &'static str {
        match self {
            AiDebugLevel::Info => i18n::chat_tag_debug_info(),
            AiDebugLevel::Debug => i18n::chat_tag_debug_debug(),
            AiDebugLevel::Warn => i18n::chat_tag_debug_warn(),
            AiDebugLevel::Error => i18n::chat_tag_debug_error(),
        }
    }
}

fn emit_ai_debug(enabled: bool, level: AiDebugLevel, message: &str) {
    if !enabled {
        return;
    }
    let sanitized = mask_sensitive(message);
    println!("{} {}", level.localized_tag(), sanitized);
    let logging_line = format!("ai-debug[{}] {}", level.as_str(), sanitized);
    match level {
        AiDebugLevel::Info | AiDebugLevel::Debug => logging::info(&logging_line),
        AiDebugLevel::Warn => logging::warn(&logging_line),
        AiDebugLevel::Error => logging::error(&logging_line),
    }
}

fn serialize_debug_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unserializable json>".to_string())
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
    streamed_content: bool,
    streamed_thinking: bool,
}

struct ParsedChatResponse {
    message: AssistantMessage,
    usage: Usage,
}

struct ParsedStreamingChatResponse {
    assistant: AssistantMessage,
    usage: Usage,
    streamed_content: bool,
    streamed_thinking: bool,
}

enum StreamParseResult {
    FallbackJson(String),
    Retryable(String),
    Fatal(String),
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    r#type: String,
    name: String,
    arguments: String,
}

struct FinalizeWithoutToolsState {
    thinking_chunks: Vec<String>,
    visible_assistant_chunks: Vec<String>,
    metrics: ChatMetrics,
    tool_rounds_used: usize,
    total_tool_calls: usize,
}

fn parse_chat_completion_response_text(body: &str) -> Result<ParsedChatResponse, AppError> {
    let parsed: ChatCompletionResponse = serde_json::from_str(body)
        .map_err(|err| AppError::Ai(format!("failed to parse AI response: {err}")))?;
    let Some(choice) = parsed.choices.into_iter().next() else {
        return Err(AppError::Ai("AI returned empty choices".to_string()));
    };
    Ok(ParsedChatResponse {
        message: choice.message,
        usage: parsed.usage,
    })
}

fn parse_streaming_chat_response<G>(
    response: reqwest::blocking::Response,
    on_stream_event: &mut G,
    debug_enabled: bool,
) -> Result<ParsedStreamingChatResponse, StreamParseResult>
where
    G: FnMut(ChatStreamEvent),
{
    let mut reader = BufReader::new(response);
    let mut line = String::new();
    let mut buffered_body = String::new();
    let mut saw_sse = false;
    let mut usage = Usage::default();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::<PartialToolCall>::new();
    let mut streamed_content = false;
    let mut streamed_thinking = false;

    loop {
        line.clear();
        let read = reader.read_line(&mut line).map_err(|err| {
            if streamed_content || streamed_thinking {
                StreamParseResult::Fatal(format!("failed to read AI streaming response: {err}"))
            } else {
                StreamParseResult::Retryable(format!("failed to read AI streaming response: {err}"))
            }
        })?;
        if read == 0 {
            break;
        }
        let trimmed_line = line.trim_end_matches(['\r', '\n']);
        let meaningful = trimmed_line.trim();
        if meaningful.is_empty() || meaningful.starts_with(':') {
            continue;
        }
        if !meaningful.starts_with("data:") {
            if !saw_sse {
                buffered_body.push_str(trimmed_line);
                buffered_body.push('\n');
                reader.read_to_string(&mut buffered_body).map_err(|err| {
                    StreamParseResult::Retryable(format!("failed to read AI fallback body: {err}"))
                })?;
                return Err(StreamParseResult::FallbackJson(buffered_body));
            }
            continue;
        }

        saw_sse = true;
        let payload = meaningful.trim_start_matches("data:").trim();
        if payload == "[DONE]" {
            break;
        }
        emit_ai_debug(
            debug_enabled,
            AiDebugLevel::Debug,
            &format!("AI streaming chunk: {payload}"),
        );
        let parsed: ChatCompletionStreamResponse =
            serde_json::from_str(payload).map_err(|err| {
                emit_ai_debug(
                    debug_enabled,
                    AiDebugLevel::Error,
                    &format!("AI streaming chunk parse error: {err}; payload={payload}"),
                );
                if streamed_content || streamed_thinking {
                    StreamParseResult::Fatal(format!("failed to parse AI streaming chunk: {err}"))
                } else {
                    StreamParseResult::Retryable(format!(
                        "failed to parse AI streaming chunk: {err}"
                    ))
                }
            })?;
        usage = merge_usage(usage, parsed.usage);
        for choice in parsed.choices {
            let thinking_delta = assistant_delta_reasoning_text(&choice.delta);
            let added_thinking = merge_text_delta(&mut reasoning, &thinking_delta);
            if !added_thinking.is_empty() {
                streamed_thinking = true;
                on_stream_event(ChatStreamEvent {
                    kind: ChatStreamEventKind::Thinking,
                    text: added_thinking,
                });
            }

            let content_delta = assistant_delta_content_text(&choice.delta);
            let added_content = merge_text_delta(&mut content, &content_delta);
            if !added_content.is_empty() {
                streamed_content = true;
                on_stream_event(ChatStreamEvent {
                    kind: ChatStreamEventKind::Content,
                    text: added_content,
                });
            }

            for tool_call_delta in choice.delta.tool_calls {
                merge_tool_call_delta(&mut tool_calls, &tool_call_delta);
            }
        }
    }

    if !saw_sse {
        return Err(StreamParseResult::Retryable(
            "AI streaming response was empty".to_string(),
        ));
    }

    Ok(ParsedStreamingChatResponse {
        assistant: AssistantMessage {
            content: optional_json_text(content),
            reasoning_content: optional_text(reasoning),
            reasoning: None,
            tool_calls: finalize_tool_calls(tool_calls),
        },
        usage,
        streamed_content,
        streamed_thinking,
    })
}

fn assistant_delta_content_text(message: &AssistantDeltaMessage) -> String {
    let Some(content) = message.content.as_ref() else {
        return String::new();
    };
    extract_text_delta_from_value(content)
}

fn assistant_delta_reasoning_text(message: &AssistantDeltaMessage) -> String {
    let assistant = AssistantMessage {
        content: message.content.clone(),
        reasoning_content: message.reasoning_content.clone(),
        reasoning: message.reasoning.clone(),
        tool_calls: Vec::new(),
    };
    assistant_reasoning_text(&assistant)
}

fn merge_text_delta(target: &mut String, incoming: &str) -> String {
    if incoming.is_empty() {
        return String::new();
    }
    if target.is_empty() {
        target.push_str(incoming);
        return incoming.to_string();
    }
    if incoming.starts_with(target.as_str()) {
        let suffix = &incoming[target.len()..];
        if !suffix.is_empty() {
            target.push_str(suffix);
        }
        return suffix.to_string();
    }
    if target.ends_with(incoming) {
        return String::new();
    }
    if target.contains(incoming) {
        return String::new();
    }
    target.push_str(incoming);
    incoming.to_string()
}

fn merge_tool_call_delta(tool_calls: &mut Vec<PartialToolCall>, delta: &StreamToolCallDelta) {
    let index = delta.index.unwrap_or(tool_calls.len());
    while tool_calls.len() <= index {
        tool_calls.push(PartialToolCall::default());
    }
    let item = &mut tool_calls[index];
    if let Some(id) = delta.id.as_deref()
        && !id.is_empty()
    {
        item.id = id.to_string();
    }
    if let Some(tool_type) = delta.r#type.as_deref()
        && !tool_type.is_empty()
    {
        item.r#type = tool_type.to_string();
    }
    if let Some(function) = delta.function.as_ref() {
        if let Some(name) = function.name.as_deref() {
            let _ = merge_text_delta(&mut item.name, name);
        }
        if let Some(arguments) = function.arguments.as_deref() {
            let _ = merge_text_delta(&mut item.arguments, arguments);
        }
    }
}

fn finalize_tool_calls(tool_calls: Vec<PartialToolCall>) -> Vec<ApiToolCall> {
    tool_calls
        .into_iter()
        .filter_map(|item| {
            if item.id.trim().is_empty()
                || item.name.trim().is_empty()
                || item.arguments.trim().is_empty()
            {
                return None;
            }
            Some(ApiToolCall {
                id: item.id,
                r#type: if item.r#type.trim().is_empty() {
                    "function".to_string()
                } else {
                    item.r#type
                },
                function: ApiToolFunction {
                    name: item.name,
                    arguments: item.arguments,
                },
            })
        })
        .collect()
}

fn optional_json_text(value: String) -> Option<serde_json::Value> {
    optional_text(value).map(serde_json::Value::String)
}

fn option_string_is_blank(value: &Option<String>) -> bool {
    value.as_ref().is_none_or(|item| item.trim().is_empty())
}

fn optional_text(value: String) -> Option<String> {
    if value.trim().is_empty() {
        return None;
    }
    Some(value)
}

fn merge_usage(current: Usage, next: Usage) -> Usage {
    Usage {
        prompt_tokens: current.prompt_tokens.max(next.prompt_tokens),
        completion_tokens: current.completion_tokens.max(next.completion_tokens),
        total_tokens: current.total_tokens.max(next.total_tokens),
    }
}

fn should_fallback_to_non_streaming(status: StatusCode, body: &str) -> bool {
    if !status.is_client_error() {
        return false;
    }
    let lowered = body.to_ascii_lowercase();
    lowered.contains("stream")
        && (lowered.contains("unsupported")
            || lowered.contains("not support")
            || lowered.contains("not_supported")
            || lowered.contains("unknown field")
            || lowered.contains("invalid parameter")
            || lowered.contains("unrecognized"))
}

fn assistant_content_text(message: &AssistantMessage) -> String {
    let Some(content) = message.content.as_ref() else {
        return String::new();
    };
    trim_blank_line_edges(&extract_text_from_value(content))
}

fn extract_text_from_value(value: &serde_json::Value) -> String {
    let mut chunks = Vec::new();
    collect_visible_text_from_value(value, &mut chunks);
    merge_visible_chunks(&chunks, None)
}

fn extract_text_delta_from_value(value: &serde_json::Value) -> String {
    let mut chunks = Vec::new();
    collect_visible_text_delta_from_value(value, &mut chunks);
    chunks.concat()
}

fn collect_visible_text_from_value(value: &serde_json::Value, output: &mut Vec<String>) {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
        serde_json::Value::String(text) => append_unique_visible_chunk(output, text.to_string()),
        serde_json::Value::Array(items) => {
            for item in items {
                collect_visible_text_from_value(item, output);
            }
        }
        serde_json::Value::Object(map) => {
            for key in [
                "text",
                "output_text",
                "content",
                "message",
                "value",
                "input_text",
                "output",
                "summary",
                "refusal",
            ] {
                if let Some(value) = map.get(key) {
                    collect_visible_text_from_value(value, output);
                }
            }
        }
    }
}

fn collect_visible_text_delta_from_value(value: &serde_json::Value, output: &mut Vec<String>) {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
        serde_json::Value::String(text) => append_visible_delta_chunk(output, text.to_string()),
        serde_json::Value::Array(items) => {
            for item in items {
                collect_visible_text_delta_from_value(item, output);
            }
        }
        serde_json::Value::Object(map) => {
            for key in [
                "text",
                "output_text",
                "content",
                "message",
                "value",
                "input_text",
                "output",
                "summary",
                "refusal",
            ] {
                if let Some(value) = map.get(key) {
                    collect_visible_text_delta_from_value(value, output);
                }
            }
        }
    }
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

fn append_unique_visible_chunk(chunks: &mut Vec<String>, chunk: String) {
    let cleaned_chunk = sanitize_assistant_content(&chunk);
    let display = normalize_visible_text_for_display(&cleaned_chunk);
    let compare = normalize_visible_text_for_compare(&display);
    if compare.is_empty() {
        return;
    }
    if chunks
        .iter()
        .any(|existing| normalize_visible_text_for_compare(existing) == compare)
    {
        return;
    }
    chunks.push(display);
}

fn append_visible_delta_chunk(chunks: &mut Vec<String>, chunk: String) {
    let cleaned_chunk = sanitize_assistant_content(&chunk);
    if cleaned_chunk.is_empty() {
        return;
    }
    chunks.push(normalize_visible_text_for_display(&cleaned_chunk));
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

fn merge_visible_chunks(chunks: &[String], extra: Option<&str>) -> String {
    let mut seen = HashSet::new();
    let mut merged = Vec::new();
    for item in chunks {
        let compare = normalize_visible_text_for_compare(item);
        if !compare.is_empty() && seen.insert(compare) {
            merged.push(trim_blank_line_edges(item));
        }
    }
    if let Some(extra_item) = extra {
        let compare = normalize_visible_text_for_compare(extra_item);
        if !compare.is_empty() && seen.insert(compare) {
            merged.push(trim_blank_line_edges(extra_item));
        }
    }
    merged.join("\n\n")
}

fn normalize_visible_text_for_display(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn normalize_visible_text_for_compare(text: &str) -> String {
    trim_blank_line_edges(&normalize_visible_text_for_display(text))
}

fn trim_blank_line_edges(text: &str) -> String {
    let normalized = normalize_visible_text_for_display(text);
    let lines = normalized.split('\n').collect::<Vec<_>>();
    let Some(start) = lines.iter().position(|line| !line.trim().is_empty()) else {
        return String::new();
    };
    let end = lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .unwrap_or(start);
    lines[start..=end].join("\n")
}

fn build_guard_tool_result(reason: &str) -> String {
    json!({
        "ok": false,
        "skipped": true,
        "reason": reason
    })
    .to_string()
}

fn is_cacheable_tool_request(request: &ToolCallRequest) -> bool {
    if request.name != "run_shell_command" {
        return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&request.arguments) else {
        return false;
    };
    let mode = value
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    mode == "read"
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
    model: &str,
    input_price_per_million: f64,
    output_price_per_million: f64,
    runtime_model_prices: Option<(f64, f64)>,
) -> ChatMetrics {
    let Some((resolved_input_price, resolved_output_price)) = resolve_effective_model_prices(
        model,
        input_price_per_million,
        output_price_per_million,
        runtime_model_prices,
    ) else {
        metrics.estimated_cost_usd = None;
        return metrics;
    };
    let input_cost = (metrics.prompt_tokens as f64 / 1_000_000.0) * resolved_input_price;
    let output_cost = (metrics.completion_tokens as f64 / 1_000_000.0) * resolved_output_price;
    metrics.estimated_cost_usd = Some(input_cost + output_cost);
    metrics
}

fn resolve_effective_model_prices(
    model: &str,
    configured_input_price: f64,
    configured_output_price: f64,
    runtime_model_prices: Option<(f64, f64)>,
) -> Option<(f64, f64)> {
    configured_model_prices(configured_input_price, configured_output_price)
        .or(runtime_model_prices.filter(|(input, output)| *input > 0.0 && *output > 0.0))
        .or_else(|| builtin_model_prices(model))
}

fn configured_model_prices(
    configured_input_price: f64,
    configured_output_price: f64,
) -> Option<(f64, f64)> {
    if is_valid_model_price(configured_input_price, configured_output_price) {
        return Some((configured_input_price, configured_output_price));
    }
    None
}

fn builtin_model_prices(model: &str) -> Option<(f64, f64)> {
    let normalized = normalize_priced_model_name(model);
    if normalized.contains("deepseek-chat") || normalized.contains("deepseek-v3") {
        return Some((0.27, 1.10));
    }
    if normalized.contains("deepseek-reasoner") || normalized.contains("deepseek-r1") {
        return Some((0.55, 2.19));
    }
    match normalized.as_str() {
        "gpt-5.2" => return Some((1.75, 14.0)),
        "gpt-5.2-pro" => return Some((21.0, 168.0)),
        "gpt-5"
        | "gpt-5.1"
        | "gpt-5-chat-latest"
        | "gpt-5.1-chat-latest"
        | "gpt-5-codex"
        | "gpt-5.1-codex" => return Some((1.25, 10.0)),
        "gpt-5-pro" => return Some((15.0, 120.0)),
        "gpt-5-mini" => return Some((0.25, 2.0)),
        "gpt-5-nano" => return Some((0.05, 0.4)),
        "gpt-4.1" => return Some((2.0, 8.0)),
        "gpt-4.1-mini" => return Some((0.4, 1.6)),
        "gpt-4.1-nano" => return Some((0.1, 0.4)),
        "gpt-4o" => return Some((2.5, 10.0)),
        "gpt-4o-mini" => return Some((0.15, 0.6)),
        _ => {}
    }
    None
}

fn is_valid_model_price(input: f64, output: f64) -> bool {
    input.is_finite() && output.is_finite() && input > 0.0 && output > 0.0
}

fn normalize_priced_model_name(model: &str) -> String {
    let trimmed = model.trim().to_ascii_lowercase();
    let last_segment = trimmed
        .rsplit(['/', ':'])
        .next()
        .unwrap_or(trimmed.as_str())
        .trim();
    last_segment
        .replace(['_', ' '], "-")
        .trim_matches('-')
        .to_string()
}

fn known_priced_models() -> &'static [&'static str] {
    &[
        "deepseek-chat",
        "deepseek-v3",
        "deepseek-reasoner",
        "deepseek-r1",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5",
        "gpt-5.1",
        "gpt-5-chat-latest",
        "gpt-5.1-chat-latest",
        "gpt-5-codex",
        "gpt-5.1-codex",
        "gpt-5-pro",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
    ]
}

fn price_probe_candidate_models(model: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    let raw_model = model.trim();
    if !raw_model.is_empty() && seen.insert(raw_model.to_string()) {
        candidates.push(raw_model.to_string());
    }
    let normalized = normalize_priced_model_name(model);
    if !normalized.is_empty() && seen.insert(normalized.clone()) {
        candidates.push(normalized.clone());
    }
    let family_prefix = if normalized.starts_with("gpt-5") {
        Some("gpt-5")
    } else if normalized.starts_with("gpt-4") {
        Some("gpt-4")
    } else if normalized.starts_with("deepseek") {
        Some("deepseek")
    } else {
        None
    };
    let Some(family_prefix) = family_prefix else {
        return candidates;
    };
    for candidate in known_priced_models() {
        if candidate.starts_with(family_prefix) && seen.insert((*candidate).to_string()) {
            candidates.push((*candidate).to_string());
        }
    }
    candidates
}

#[derive(Debug, Deserialize)]
struct ModelPriceProbePayload {
    input_price_per_million: Option<f64>,
    output_price_per_million: Option<f64>,
}

fn parse_model_price_probe_response(raw: &str) -> Option<(f64, f64)> {
    let candidate = strip_code_fence(raw.trim());
    let json_body = extract_first_json_object(candidate)?;
    let parsed: ModelPriceProbePayload = serde_json::from_str(json_body).ok()?;
    let input = parsed.input_price_per_million?;
    let output = parsed.output_price_per_million?;
    if !is_valid_model_price(input, output) {
        return None;
    }
    Some((input, output))
}

fn parse_model_price_catalog_response(
    raw: &str,
    current_model: &str,
) -> BTreeMap<String, (f64, f64)> {
    let candidate = strip_code_fence(raw.trim());
    let json_body = match extract_first_json_object(candidate) {
        Some(json_body) => json_body,
        None => return BTreeMap::new(),
    };
    let parsed = match serde_json::from_str::<serde_json::Value>(json_body) {
        Ok(parsed) => parsed,
        Err(_) => return BTreeMap::new(),
    };
    let Some(object) = parsed.as_object() else {
        return BTreeMap::new();
    };
    if object.contains_key("input_price_per_million")
        || object.contains_key("output_price_per_million")
    {
        return parse_model_price_probe_response(json_body)
            .map(|prices| {
                let mut single = BTreeMap::new();
                single.insert(normalize_priced_model_name(current_model), prices);
                single
            })
            .unwrap_or_default();
    }
    let mut prices = BTreeMap::new();
    for (model, value) in object {
        let Some(entry) = parse_model_price_value(value) else {
            continue;
        };
        prices.insert(normalize_priced_model_name(model), entry);
    }
    prices
}

fn parse_model_price_value(value: &serde_json::Value) -> Option<(f64, f64)> {
    let parsed: ModelPriceProbePayload = serde_json::from_value(value.clone()).ok()?;
    let input = parsed.input_price_per_million?;
    let output = parsed.output_price_per_million?;
    if !is_valid_model_price(input, output) {
        return None;
    }
    Some((input, output))
}

fn fresh_model_price_catalog(
    catalog: &PersistedModelPriceCatalog,
    now_epoch_secs: u64,
) -> BTreeMap<String, (f64, f64)> {
    catalog
        .models
        .iter()
        .filter_map(|(model, entry)| {
            if !is_fresh_model_price_entry(entry, now_epoch_secs) {
                return None;
            }
            Some((
                normalize_priced_model_name(model),
                (
                    entry.input_price_per_million,
                    entry.output_price_per_million,
                ),
            ))
        })
        .collect()
}

fn retained_fresh_model_price_entries(
    models: BTreeMap<String, PersistedModelPriceEntry>,
    now_epoch_secs: u64,
) -> BTreeMap<String, PersistedModelPriceEntry> {
    models
        .into_iter()
        .filter(|(_, entry)| is_fresh_model_price_entry(entry, now_epoch_secs))
        .collect()
}

fn is_fresh_model_price_entry(entry: &PersistedModelPriceEntry, now_epoch_secs: u64) -> bool {
    is_valid_model_price(
        entry.input_price_per_million,
        entry.output_price_per_million,
    ) && now_epoch_secs.saturating_sub(entry.checked_at_epoch_secs) <= MODEL_PRICE_CACHE_TTL_SECS
}

fn default_model_price_cache_version() -> u32 {
    MODEL_PRICE_CACHE_VERSION
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed;
    }
    let without_prefix = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```JSON")
        .trim_start_matches("```");
    without_prefix.trim().trim_end_matches("```").trim()
}

fn extract_first_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(&text[start..=end])
}

fn write_string_atomically(path: &Path, content: &str) -> Result<(), AppError> {
    let parent = path.parent().ok_or_else(|| {
        AppError::Runtime(format!(
            "failed to resolve parent directory for {}",
            path.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|err| {
        AppError::Runtime(format!(
            "failed to create parent directory {}: {err}",
            parent.display()
        ))
    })?;
    let file_name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("model-price-cache");
    let temp_path = parent.join(format!(".{}.{}.tmp", file_name, uuid::Uuid::new_v4()));
    fs::write(&temp_path, content).map_err(|err| {
        AppError::Runtime(format!(
            "failed to write temporary file {}: {err}",
            temp_path.display()
        ))
    })?;
    if let Err(err) = fs::rename(&temp_path, path) {
        if path.exists() {
            let _ = fs::remove_file(path);
            fs::rename(&temp_path, path).map_err(|rename_err| {
                AppError::Runtime(format!(
                    "failed to replace file {} after rename error {}: {}",
                    path.display(),
                    err,
                    rename_err
                ))
            })?;
        } else {
            let _ = fs::remove_file(&temp_path);
            return Err(AppError::Runtime(format!(
                "failed to move temporary file into place {}: {err}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn optional_content(content: String) -> Option<String> {
    if content.trim().is_empty() {
        return None;
    }
    Some(content)
}

fn sanitize_assistant_content(raw: &str) -> String {
    let cleaned = DSML_FUNCTION_CALLS_BLOCK_RE.replace_all(raw, "");
    cleaned.replace("\r\n", "\n").replace('\r', "\n")
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

fn is_rate_limited_error(err: &AppError) -> bool {
    let AppError::Ai(detail) = err else {
        return false;
    };
    let lowered = detail.to_ascii_lowercase();
    lowered.contains("http status=429")
        || lowered.contains("too many requests")
        || lowered.contains("\"type\":\"rate_limited\"")
        || lowered.contains("rate limited")
        || lowered.contains("request limited rpm reached")
}

pub(crate) fn is_transient_ai_error(err: &AppError) -> bool {
    let AppError::Ai(detail) = err else {
        return false;
    };
    let lowered = detail.to_ascii_lowercase();
    lowered.contains("kind=timeout")
        || lowered.contains("kind=connect")
        || lowered.contains("operation timed out")
        || lowered.contains("connection timed out")
        || lowered.contains("timed out")
        || lowered.contains("temporarily unavailable")
        || lowered.contains("service unavailable")
        || lowered.contains("bad gateway")
        || lowered.contains("gateway timeout")
        || lowered.contains("http status=502")
        || lowered.contains("http status=503")
        || lowered.contains("http status=504")
}

fn should_retry_status(status: StatusCode) -> bool {
    status.is_server_error() || matches!(status, StatusCode::REQUEST_TIMEOUT)
}

fn parse_rate_limit_retry_delay(status: StatusCode, retry_after: Option<&str>) -> Option<Duration> {
    if status != StatusCode::TOO_MANY_REQUESTS {
        return None;
    }
    let seconds = retry_after
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<u64>().ok())?;
    if seconds == 0 || seconds > MAX_AUTO_RATE_LIMIT_WAIT_SECS {
        return None;
    }
    Some(Duration::from_secs(seconds))
}

fn should_refresh_http_client_after_reqwest_error(err: &reqwest::Error) -> bool {
    if err.is_timeout() || err.is_connect() || err.is_request() {
        return true;
    }
    let lowered = err.to_string().to_ascii_lowercase();
    lowered.contains("connection reset")
        || lowered.contains("broken pipe")
        || lowered.contains("connection closed")
        || lowered.contains("unexpected eof")
        || lowered.contains("channel closed")
        || lowered.contains("tls handshake eof")
}

fn maybe_print_ai_reconnect_notice(message: &str, colorful: bool) {
    if !std::io::stdout().is_terminal() {
        return;
    }
    // Spinner uses carriage-return in-place refresh; clear that line before printing reconnect hint.
    print!("\r{: <220}\r", "");
    println!(
        "{}",
        render::render_chat_reconnect_notice(message, colorful)
    );
    let _ = std::io::stdout().flush();
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

fn build_finalization_fallback(
    stop_reason: Option<ChatStopReason>,
    tool_rounds_used: usize,
    total_tool_calls: usize,
) -> String {
    let reason_text = stop_reason
        .map(|item| item.code().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "本轮工具调用已结束（reason={reason_text}, rounds={tool_rounds_used}, tool_calls={total_tool_calls}）。AI 在收尾阶段未返回可展示文本。请继续提问（可缩小排查范围或指定目标日志/目录），我会基于当前已收集证据继续分析。"
    )
}

fn build_transient_ai_failure_fallback(err: &AppError) -> String {
    format!(
        "本轮 AI 请求暂时失败（{}）。会话不会退出；如需继续，请直接重试上一条指令或稍后再试。",
        mask_sensitive(&err.to_string())
    )
}

fn build_chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        return trimmed.to_string();
    }
    format!("{trimmed}/chat/completions")
}

fn normalize_request_for_provider(
    body: &ChatCompletionRequest,
    base_url: &str,
) -> ChatCompletionRequest {
    let mut normalized = body.clone();
    if provider_requires_reasoning_content_omission(base_url) {
        normalize_stepfun_chat_request(&mut normalized);
    }
    normalized
}

fn normalize_stepfun_chat_request(body: &mut ChatCompletionRequest) {
    strip_reasoning_content_from_request(body);
    ensure_assistant_content_is_non_null(body);
    normalize_tool_call_ids_in_request(body);
}

fn provider_requires_reasoning_content_omission(base_url: &str) -> bool {
    let normalized = base_url.trim().to_ascii_lowercase();
    normalized.contains("stepfun.com")
}

fn ensure_assistant_content_is_non_null(body: &mut ChatCompletionRequest) {
    for message in &mut body.messages {
        if message.role == "assistant" && message.content.is_none() {
            message.content = Some(String::new());
        }
    }
}

fn request_contains_reasoning_content(body: &ChatCompletionRequest) -> bool {
    body.messages.iter().any(|message| {
        message
            .reasoning_content
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    })
}

fn strip_reasoning_content_from_request(body: &mut ChatCompletionRequest) {
    for message in &mut body.messages {
        message.reasoning_content = None;
    }
}

fn normalize_tool_call_ids_in_request(body: &mut ChatCompletionRequest) {
    let mut seen_ids = HashSet::new();
    let mut pending_ids = HashMap::<String, VecDeque<String>>::new();
    let mut generated_counter: usize = 0;

    for message in &mut body.messages {
        if message.role == "assistant" {
            let Some(tool_calls) = message.tool_calls.as_mut() else {
                continue;
            };
            for tool_call in tool_calls {
                let original_id = tool_call.id.trim().to_string();
                let normalized_id = if original_id.is_empty() || seen_ids.contains(&original_id) {
                    generate_unique_tool_call_id(
                        &original_id,
                        &mut seen_ids,
                        &mut generated_counter,
                    )
                } else {
                    seen_ids.insert(original_id.clone());
                    original_id.clone()
                };
                tool_call.id = normalized_id.clone();
                pending_ids
                    .entry(original_id)
                    .or_default()
                    .push_back(normalized_id);
            }
            continue;
        }

        if message.role != "tool" {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_mut() else {
            continue;
        };
        let original_id = tool_call_id.trim().to_string();
        if original_id.is_empty() {
            continue;
        }
        if let Some(mapped_id) = pending_ids
            .get_mut(&original_id)
            .and_then(|queue| queue.pop_front())
        {
            *tool_call_id = mapped_id;
        }
    }
}

fn generate_unique_tool_call_id(
    original_id: &str,
    seen_ids: &mut HashSet<String>,
    generated_counter: &mut usize,
) -> String {
    let base = sanitize_tool_call_id_base(original_id);
    loop {
        *generated_counter += 1;
        let candidate = format!("{base}_ctx_{}", generated_counter);
        if seen_ids.insert(candidate.clone()) {
            return candidate;
        }
    }
}

fn sanitize_tool_call_id_base(original_id: &str) -> String {
    let sanitized = original_id
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if sanitized.is_empty() {
        return "tool_call".to_string();
    }
    sanitized
}

fn should_retry_without_reasoning_content(status: StatusCode, body: &str) -> bool {
    if status != StatusCode::BAD_REQUEST {
        return false;
    }
    let lowered = body.to_ascii_lowercase();
    lowered.contains("unrecognized chat message")
        || lowered.contains("reasoning_content")
        || lowered.contains("reasoning format")
}

#[cfg(test)]
mod tests {
    use reqwest::StatusCode;
    use reqwest::blocking::Client;
    use serde_json::json;
    use std::{env, fs, sync::RwLock, time::Duration};

    use super::{
        AiClient, ApiMessage, ApiToolCall, ApiToolFunction, ChatCompletionRequest,
        ChatCompletionStreamResponse, ChatMetrics, ChatStopReason, FinalizeWithoutToolsState,
        ModelPriceSource, PersistedModelPriceCatalog, PersistedModelPriceEntry,
        builtin_model_prices, extract_text_delta_from_value, extract_text_from_value,
        fresh_model_price_catalog, is_rate_limited_error, merge_text_delta, merge_visible_chunks,
        normalize_stepfun_chat_request, parse_model_price_catalog_response,
        parse_model_price_probe_response, parse_rate_limit_retry_delay,
        price_probe_candidate_models, provider_requires_reasoning_content_omission,
        request_contains_reasoning_content, resolve_effective_model_prices,
        should_fallback_to_non_streaming, should_refresh_http_client_after_reqwest_error,
        should_retry_without_reasoning_content, strip_reasoning_content_from_request, with_cost,
    };
    use crate::{error::AppError, tls::ensure_rustls_crypto_provider};

    fn test_ai_client() -> AiClient {
        ensure_rustls_crypto_provider();
        AiClient {
            client: RwLock::new(Client::builder().build().expect("test client")),
            base_url: "https://example.com/chat/completions".to_string(),
            token: "token".to_string(),
            model: "unknown-model".to_string(),
            colorful: true,
            model_price_cache_path: env::temp_dir()
                .join(format!("machineclaw-test-{}.json", uuid::Uuid::new_v4())),
            debug: false,
            max_retries: 0,
            backoff_millis: 0,
            input_price_per_million: 0.0,
            output_price_per_million: 0.0,
            runtime_model_prices: RwLock::new(None),
        }
    }

    #[test]
    fn extracts_nested_visible_text_blocks() {
        let payload = json!([
            {"type":"text","text":{"value":"first line"}},
            {"type":"output_text","output_text":"second line"},
            {"type":"wrapper","content":[{"text":"third line"}]},
            {"type":"tool_call","arguments":"{\"ignored\":true}"}
        ]);
        assert_eq!(
            extract_text_from_value(&payload),
            "first line\n\nsecond line\n\nthird line"
        );
    }

    #[test]
    fn merges_visible_chunks_without_duplicates() {
        let chunks = vec!["alpha".to_string(), "beta".to_string(), "alpha".to_string()];
        assert_eq!(merge_visible_chunks(&chunks, Some("beta")), "alpha\n\nbeta");
    }

    #[test]
    fn preserves_multiline_code_block_text_from_visible_value() {
        let payload = json!([
            {"type":"text","text":{"value":"```rust\nfn main() {\n    println!(\"hi\");\n}\n```"}}
        ]);
        assert_eq!(
            extract_text_from_value(&payload),
            "```rust\nfn main() {\n    println!(\"hi\");\n}\n```"
        );
    }

    #[test]
    fn preserves_stream_delta_newlines_and_indentation() {
        let payload = json!([
            {"type":"text","text":{"value":"\nfn main() {\n    println!(\"hi\");\n}"}}
        ]);
        assert_eq!(
            extract_text_delta_from_value(&payload),
            "\nfn main() {\n    println!(\"hi\");\n}"
        );
    }

    #[test]
    fn resolves_builtin_model_prices_when_config_missing() {
        assert_eq!(
            resolve_effective_model_prices("deepseek-chat", 0.0, 0.0, None),
            Some((0.27, 1.10))
        );
        assert_eq!(
            resolve_effective_model_prices("gpt-5.2", 0.0, 0.0, None),
            Some((1.75, 14.0))
        );
        assert_eq!(
            resolve_effective_model_prices("openai/gpt-5.2", 0.0, 0.0, None),
            Some((1.75, 14.0))
        );
        assert_eq!(
            resolve_effective_model_prices("closeai:gpt-5-mini", 0.0, 0.0, None),
            Some((0.25, 2.0))
        );
        assert_eq!(builtin_model_prices("unknown-model"), None);
    }

    #[test]
    fn configured_model_prices_override_builtin_prices() {
        assert_eq!(
            resolve_effective_model_prices("gpt-5.2", 9.9, 19.9, None),
            Some((9.9, 19.9))
        );
    }

    #[test]
    fn runtime_probed_prices_override_builtin_prices_when_config_missing() {
        assert_eq!(
            resolve_effective_model_prices("gpt-5.2", 0.0, 0.0, Some((3.3, 7.7))),
            Some((3.3, 7.7))
        );
    }

    #[test]
    fn cost_is_unavailable_for_unknown_model_without_config() {
        let metrics = with_cost(ChatMetrics::default(), "unknown-model", 0.0, 0.0, None);
        assert_eq!(metrics.estimated_cost_usd, None);
    }

    #[test]
    fn merges_stream_deltas_without_duplicate_suffixes() {
        let mut target = String::new();
        assert_eq!(merge_text_delta(&mut target, "Hello"), "Hello");
        assert_eq!(merge_text_delta(&mut target, "Hello"), "");
        assert_eq!(merge_text_delta(&mut target, "Hello world"), " world");
        assert_eq!(target, "Hello world");
    }

    #[test]
    fn keeps_short_stream_delta_overlaps_without_truncating_code() {
        let mut target = "pub fn ".to_string();
        assert_eq!(
            merge_text_delta(&mut target, "fn merge_sort<T: Ord + Clone>() {"),
            "fn merge_sort<T: Ord + Clone>() {"
        );
        assert_eq!(target, "pub fn fn merge_sort<T: Ord + Clone>() {");
    }

    #[test]
    fn falls_back_when_server_rejects_stream_parameter() {
        assert!(should_fallback_to_non_streaming(
            StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"stream is unsupported\"}}"
        ));
        assert!(!should_fallback_to_non_streaming(
            StatusCode::INTERNAL_SERVER_ERROR,
            "stream is unsupported"
        ));
    }

    #[test]
    fn parses_model_price_probe_response_from_plain_json() {
        assert_eq!(
            parse_model_price_probe_response(
                r#"{"input_price_per_million":1.25,"output_price_per_million":10.0}"#
            ),
            Some((1.25, 10.0))
        );
    }

    #[test]
    fn parses_model_price_probe_response_from_json_code_fence() {
        assert_eq!(
            parse_model_price_probe_response(
                "```json\n{\"input_price_per_million\":1.25,\"output_price_per_million\":10.0}\n```"
            ),
            Some((1.25, 10.0))
        );
    }

    #[test]
    fn rejects_invalid_model_price_probe_response() {
        assert_eq!(
            parse_model_price_probe_response(
                r#"{"input_price_per_million":null,"output_price_per_million":10.0}"#
            ),
            None
        );
        assert_eq!(
            parse_model_price_probe_response(
                r#"{"input_price_per_million":-1,"output_price_per_million":10.0}"#
            ),
            None
        );
    }

    #[test]
    fn parses_model_price_catalog_response_from_mapping_json() {
        let parsed = parse_model_price_catalog_response(
            r#"{"gpt-5.2":{"input_price_per_million":1.75,"output_price_per_million":14.0},"gpt-5-mini":{"input_price_per_million":0.25,"output_price_per_million":2.0}}"#,
            "gpt-5.2",
        );
        assert_eq!(parsed.get("gpt-5.2"), Some(&(1.75, 14.0)));
        assert_eq!(parsed.get("gpt-5-mini"), Some(&(0.25, 2.0)));
    }

    #[test]
    fn parses_model_price_catalog_response_from_single_object_fallback() {
        let parsed = parse_model_price_catalog_response(
            r#"{"input_price_per_million":1.75,"output_price_per_million":14.0}"#,
            "openai/gpt-5.2",
        );
        assert_eq!(parsed.get("gpt-5.2"), Some(&(1.75, 14.0)));
    }

    #[test]
    fn filters_expired_entries_from_persisted_model_price_catalog() {
        let catalog = PersistedModelPriceCatalog {
            version: 1,
            models: [
                (
                    "gpt-5.2".to_string(),
                    PersistedModelPriceEntry {
                        input_price_per_million: 1.75,
                        output_price_per_million: 14.0,
                        checked_at_epoch_secs: 1_700_000_000,
                    },
                ),
                (
                    "gpt-5-mini".to_string(),
                    PersistedModelPriceEntry {
                        input_price_per_million: 0.25,
                        output_price_per_million: 2.0,
                        checked_at_epoch_secs: 1_700_000_000 - (8 * 24 * 60 * 60),
                    },
                ),
            ]
            .into_iter()
            .collect(),
        };
        let fresh = fresh_model_price_catalog(&catalog, 1_700_000_000);
        assert_eq!(fresh.get("gpt-5.2"), Some(&(1.75, 14.0)));
        assert!(!fresh.contains_key("gpt-5-mini"));
    }

    #[test]
    fn prepare_model_pricing_uses_fresh_local_cache_before_online_probe() {
        let mut client = test_ai_client();
        client.model = "gpt-5.2".to_string();
        let cache_path = env::temp_dir().join(format!(
            "machineclaw-model-price-cache-{}.json",
            uuid::Uuid::new_v4()
        ));
        client.model_price_cache_path = cache_path.clone();
        let catalog = PersistedModelPriceCatalog {
            version: 1,
            models: [(
                "gpt-5.2".to_string(),
                PersistedModelPriceEntry {
                    input_price_per_million: 1.75,
                    output_price_per_million: 14.0,
                    checked_at_epoch_secs: super::now_epoch_secs(),
                },
            )]
            .into_iter()
            .collect(),
        };
        fs::write(
            &cache_path,
            serde_json::to_string_pretty(&catalog).expect("serialize catalog"),
        )
        .expect("write cache");
        let result = client.prepare_model_pricing(false);
        assert_eq!(result.source, ModelPriceSource::LocalCache);
        assert_eq!(result.prices, Some((1.75, 14.0)));
        let _ = fs::remove_file(cache_path);
    }

    #[test]
    fn unknown_model_price_probe_candidates_do_not_expand_to_all_known_models() {
        let candidates = price_probe_candidate_models("vendor/custom-model");
        assert_eq!(
            candidates,
            vec![
                "vendor/custom-model".to_string(),
                "custom-model".to_string()
            ]
        );
    }

    #[test]
    fn recoverable_chat_response_prefers_archived_visible_content() {
        let client = test_ai_client();
        let response = client.build_recoverable_chat_response(
            FinalizeWithoutToolsState {
                thinking_chunks: vec!["thought".to_string()],
                visible_assistant_chunks: vec!["alpha".to_string(), "beta".to_string()],
                metrics: ChatMetrics::default(),
                tool_rounds_used: 3,
                total_tool_calls: 2,
            },
            None,
        );
        assert_eq!(response.content, "alpha\n\nbeta");
        assert_eq!(response.archived_content, "alpha\n\nbeta");
        assert_eq!(response.thinking.as_deref(), Some("thought"));
    }

    #[test]
    fn recoverable_chat_response_uses_fallback_when_no_visible_content_exists() {
        let client = test_ai_client();
        let response = client.build_recoverable_chat_response(
            FinalizeWithoutToolsState {
                thinking_chunks: vec!["thought".to_string()],
                visible_assistant_chunks: Vec::new(),
                metrics: ChatMetrics::default(),
                tool_rounds_used: 5,
                total_tool_calls: 7,
            },
            Some(ChatStopReason::MaxToolRoundsReached),
        );
        assert!(response.content.contains("AI 在收尾阶段未返回可展示文本"));
        assert!(response.content.contains("reason=max_tool_rounds_reached"));
        assert_eq!(response.archived_content, response.content);
        assert_eq!(response.thinking.as_deref(), Some("thought"));
    }

    #[test]
    fn streaming_chunk_accepts_null_usage() {
        let parsed: ChatCompletionStreamResponse =
            serde_json::from_str(r#"{"choices":[{"delta":{"content":"hello"}}],"usage":null}"#)
                .expect("null usage should deserialize");
        assert_eq!(parsed.usage.prompt_tokens, 0);
        assert_eq!(parsed.usage.completion_tokens, 0);
        assert_eq!(parsed.usage.total_tokens, 0);
    }

    #[test]
    fn streaming_chunk_accepts_missing_usage() {
        let parsed: ChatCompletionStreamResponse =
            serde_json::from_str(r#"{"choices":[{"delta":{"content":"hello"}}]}"#)
                .expect("missing usage should deserialize");
        assert_eq!(parsed.usage.prompt_tokens, 0);
        assert_eq!(parsed.usage.completion_tokens, 0);
        assert_eq!(parsed.usage.total_tokens, 0);
    }

    #[test]
    fn streaming_chunk_keeps_usage_object_values() {
        let parsed: ChatCompletionStreamResponse = serde_json::from_str(
            r#"{"choices":[{"delta":{"content":"hello"}}],"usage":{"prompt_tokens":12,"completion_tokens":34,"total_tokens":46}}"#,
        )
        .expect("usage object should deserialize");
        assert_eq!(parsed.usage.prompt_tokens, 12);
        assert_eq!(parsed.usage.completion_tokens, 34);
        assert_eq!(parsed.usage.total_tokens, 46);
    }

    #[test]
    fn strips_reasoning_content_from_request_messages() {
        let mut request = ChatCompletionRequest {
            model: "step-3.5-flash".to_string(),
            messages: vec![
                ApiMessage {
                    role: "assistant".to_string(),
                    content: Some("planning".to_string()),
                    reasoning_content: Some("hidden chain".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                ApiMessage {
                    role: "tool".to_string(),
                    content: Some("ok".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: Some("call_1".to_string()),
                },
            ],
            temperature: 0.2,
            tools: None,
            tool_choice: None,
            stream: None,
        };
        assert!(request_contains_reasoning_content(&request));
        strip_reasoning_content_from_request(&mut request);
        assert!(!request_contains_reasoning_content(&request));
    }

    #[test]
    fn detects_stepfun_reasoning_content_compatibility_mode() {
        assert!(provider_requires_reasoning_content_omission(
            "https://api.stepfun.com/v1/chat/completions"
        ));
        assert!(!provider_requires_reasoning_content_omission(
            "https://api.openai.com/v1/chat/completions"
        ));
    }

    #[test]
    fn retries_bad_request_for_unrecognized_chat_message() {
        assert!(should_retry_without_reasoning_content(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"Unrecognized chat message.","type":"request_params_invalid"}}"#
        ));
        assert!(!should_retry_without_reasoning_content(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"invalid api key"}}"#
        ));
    }

    #[test]
    fn detects_rate_limited_errors() {
        assert!(is_rate_limited_error(&AppError::Ai(
            "AI HTTP status=429 Too Many Requests, body={\"error\":{\"type\":\"rate_limited\"}}"
                .to_string()
        )));
        assert!(!is_rate_limited_error(&AppError::Ai(
            "AI HTTP status=400 Bad Request".to_string()
        )));
    }

    #[test]
    fn parses_short_retry_after_delay_for_rate_limit() {
        assert_eq!(
            parse_rate_limit_retry_delay(StatusCode::TOO_MANY_REQUESTS, Some("5")),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            parse_rate_limit_retry_delay(StatusCode::TOO_MANY_REQUESTS, Some("30")),
            None
        );
        assert_eq!(
            parse_rate_limit_retry_delay(StatusCode::BAD_REQUEST, Some("5")),
            None
        );
    }

    #[test]
    fn detects_transient_ai_errors() {
        assert!(super::is_transient_ai_error(&AppError::Ai(
            "AI request failed: kind=timeout, error=operation timed out".to_string()
        )));
        assert!(super::is_transient_ai_error(&AppError::Ai(
            "AI HTTP status=503 Service Unavailable".to_string()
        )));
        assert!(!super::is_transient_ai_error(&AppError::Ai(
            "AI HTTP status=400 Bad Request".to_string()
        )));
    }

    #[test]
    fn refreshes_http_client_for_transient_transport_errors() {
        ensure_rustls_crypto_provider();
        let connect_err = reqwest::blocking::Client::builder()
            .build()
            .expect("test client")
            .get("http://127.0.0.1:1")
            .send()
            .expect_err("request should fail");
        assert!(should_refresh_http_client_after_reqwest_error(&connect_err));
    }

    #[test]
    fn does_not_refresh_http_client_for_builder_validation_errors() {
        ensure_rustls_crypto_provider();
        let invalid_url_err = reqwest::blocking::Client::builder()
            .build()
            .expect("test client")
            .get("://bad-url")
            .build()
            .expect_err("request build should fail");
        assert!(!should_refresh_http_client_after_reqwest_error(
            &invalid_url_err
        ));
    }

    #[test]
    fn stepfun_normalization_fills_empty_assistant_content_and_uniquifies_tool_call_ids() {
        let mut request = ChatCompletionRequest {
            model: "step-3.5-flash".to_string(),
            messages: vec![
                ApiMessage {
                    role: "assistant".to_string(),
                    content: None,
                    reasoning_content: Some("hidden".to_string()),
                    tool_calls: Some(vec![ApiToolCall {
                        id: "call_1".to_string(),
                        r#type: "function".to_string(),
                        function: ApiToolFunction {
                            name: "run_shell_command".to_string(),
                            arguments: "{\"command\":\"echo 1\"}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                ApiMessage {
                    role: "tool".to_string(),
                    content: Some("{\"ok\":true}".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: Some("call_1".to_string()),
                },
                ApiMessage {
                    role: "assistant".to_string(),
                    content: None,
                    reasoning_content: Some("hidden".to_string()),
                    tool_calls: Some(vec![ApiToolCall {
                        id: "call_1".to_string(),
                        r#type: "function".to_string(),
                        function: ApiToolFunction {
                            name: "run_shell_command".to_string(),
                            arguments: "{\"command\":\"echo 2\"}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                ApiMessage {
                    role: "tool".to_string(),
                    content: Some("{\"ok\":false}".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: Some("call_1".to_string()),
                },
            ],
            temperature: 0.2,
            tools: None,
            tool_choice: None,
            stream: None,
        };

        normalize_stepfun_chat_request(&mut request);

        assert_eq!(request.messages[0].content.as_deref(), Some(""));
        assert_eq!(request.messages[2].content.as_deref(), Some(""));
        assert!(request.messages[0].reasoning_content.is_none());
        assert!(request.messages[2].reasoning_content.is_none());

        let first_id = request.messages[0]
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .map(|call| call.id.clone())
            .expect("first assistant tool call id");
        let second_id = request.messages[2]
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .map(|call| call.id.clone())
            .expect("second assistant tool call id");

        assert_eq!(first_id, "call_1");
        assert_ne!(second_id, "call_1");
        assert_ne!(second_id, first_id);
        assert_eq!(
            request.messages[1].tool_call_id.as_deref(),
            Some(first_id.as_str())
        );
        assert_eq!(
            request.messages[3].tool_call_id.as_deref(),
            Some(second_id.as_str())
        );
    }
}
