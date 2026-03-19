use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    error::Error as StdError,
    fs,
    io::{BufRead, BufReader, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crossterm::terminal::is_raw_mode_enabled;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::RETRY_AFTER;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    config::{AiConfig, normalize_ai_provider_type},
    error::AppError,
    i18n, logging,
    mask::mask_sensitive,
    render,
    shell::take_interactive_input_refresh_hint,
    tls::ensure_rustls_crypto_provider,
};

const MODEL_PRICE_CACHE_TTL_SECS: u64 = 7 * 24 * 60 * 60;
const MODEL_PRICE_CACHE_VERSION: u32 = 1;
const MAX_AUTO_RATE_LIMIT_WAIT_SECS: u64 = 15;
const AI_HTTP_CONNECT_TIMEOUT_SECS: u64 = 8;
const AI_HTTP_REQUEST_TIMEOUT_SECS: u64 = 60;
const AI_HTTP_POOL_IDLE_TIMEOUT_SECS: u64 = 30;
const AI_HTTP_TCP_KEEPALIVE_SECS: u64 = 30;
const REQUEST_TRACE_FILE_VERSION: u32 = 1;
const REQUEST_TRACE_MAX_ENTRIES: usize = 1000;
const CHAT_CANCELLED_ERROR: &str = "chat request cancelled by user";
const CLAUDE_MAX_TOKENS: u32 = 4096;
const CLAUDE_API_VERSION: &str = "2023-06-01";

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
    TaskDecompositionRequired,
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
            ChatStopReason::TaskDecompositionRequired => "task_decomposition_required",
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

pub fn is_chat_cancelled_error(err: &AppError) -> bool {
    matches!(err, AppError::Ai(detail) if detail == CHAT_CANCELLED_ERROR)
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
    provider: AiProviderProtocol,
    base_url: String,
    token: String,
    model: String,
    colorful: bool,
    model_price_cache_path: PathBuf,
    request_trace_dir: PathBuf,
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
                Client::builder().build().unwrap_or_else(|_| Client::new())
            });
        let cached_prices = self
            .runtime_model_prices
            .read()
            .ok()
            .and_then(|guard| *guard);
        Self {
            client: RwLock::new(client),
            provider: self.provider,
            base_url: self.base_url.clone(),
            token: self.token.clone(),
            model: self.model.clone(),
            colorful: self.colorful,
            model_price_cache_path: self.model_price_cache_path.clone(),
            request_trace_dir: self.request_trace_dir.clone(),
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
enum AiProviderProtocol {
    OpenAiCompatible,
    Claude,
    Gemini,
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
    #[serde(default, deserialize_with = "deserialize_vec_or_default")]
    choices: Vec<Choice>,
    #[serde(default, deserialize_with = "deserialize_usage_or_default")]
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionStreamResponse {
    #[serde(default, deserialize_with = "deserialize_vec_or_default")]
    choices: Vec<StreamChoice>,
    #[serde(default, deserialize_with = "deserialize_usage_or_default")]
    usage: Usage,
}

#[derive(Debug, Serialize, Deserialize, Default)]
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

fn deserialize_vec_or_default<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(Option::<Vec<T>>::deserialize(deserializer)?.unwrap_or_default())
}

fn deserialize_struct_or_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default, deserialize_with = "deserialize_struct_or_default")]
    delta: AssistantDeltaMessage,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: Option<serde_json::Value>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "deserialize_vec_or_default")]
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
    #[serde(default, deserialize_with = "deserialize_vec_or_default")]
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

#[derive(Debug, Serialize, Deserialize, Default)]
struct SessionRequestTraceFile {
    #[serde(default = "default_request_trace_file_version")]
    version: u32,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    entries: Vec<RequestTraceEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RequestTraceEntry {
    timestamp_epoch_ms: u128,
    event: String,
    stream: bool,
    attempt: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    status_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_packets: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_packets: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
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
        let provider = provider_protocol_from_config_type(cfg.r#type.as_str());
        let request_trace_dir = model_price_cache_path
            .parent()
            .unwrap_or_else(|| Path::new(".machineclaw"))
            .join("requests");
        Ok(Self {
            client: RwLock::new(client),
            provider,
            base_url: build_provider_chat_url(&cfg.base_url, provider, &cfg.model),
            token: cfg.token.clone(),
            model: cfg.model.clone(),
            colorful,
            model_price_cache_path,
            request_trace_dir,
            debug: cfg.debug,
            max_retries: cfg.retry.max_retries,
            backoff_millis: cfg.retry.backoff_millis,
            input_price_per_million: cfg.input_price_per_million,
            output_price_per_million: cfg.output_price_per_million,
            runtime_model_prices: RwLock::new(None),
        })
    }

    fn build_provider_request_body(&self, body: &ChatCompletionRequest) -> serde_json::Value {
        match self.provider {
            AiProviderProtocol::OpenAiCompatible => {
                serde_json::to_value(body).unwrap_or_else(|_| json!({}))
            }
            AiProviderProtocol::Claude => build_claude_request_body(body),
            AiProviderProtocol::Gemini => build_gemini_request_body(body),
        }
    }

    fn parse_provider_response_text(&self, body: &str) -> Result<ParsedChatResponse, AppError> {
        match self.provider {
            AiProviderProtocol::OpenAiCompatible => parse_chat_completion_response_text(body),
            AiProviderProtocol::Claude => parse_claude_response_text(body),
            AiProviderProtocol::Gemini => parse_gemini_response_text(body),
        }
    }

    fn apply_provider_auth(
        &self,
        builder: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        match self.provider {
            AiProviderProtocol::OpenAiCompatible => builder.bearer_auth(&self.token),
            AiProviderProtocol::Claude => builder
                .header("x-api-key", &self.token)
                .header("anthropic-version", CLAUDE_API_VERSION),
            AiProviderProtocol::Gemini => builder.header("x-goog-api-key", &self.token),
        }
    }

    pub fn validate_connectivity_with_response(&self) -> Result<String, AppError> {
        let system_prompt = "You are a connectivity checker. Reply with one word: OK.";
        let user_prompt = "Respond with OK only.";
        let response = self.chat(&[], system_prompt, user_prompt)?;
        if response.trim().is_empty() {
            return Err(AppError::Ai(
                "AI validation returned empty response".to_string(),
            ));
        }
        Ok(response)
    }

    pub fn validate_connectivity(&self) -> Result<(), AppError> {
        self.validate_connectivity_with_response().map(|_| ())
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
        self.chat_with_debug_session(history, system_prompt, user_prompt, None)
    }

    pub fn chat_with_debug_session(
        &self,
        history: &[ChatMessage],
        system_prompt: &str,
        user_prompt: &str,
        debug_session_id: Option<&str>,
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
        let call = self.send_chat_completion(&request, None, debug_session_id)?;
        let assistant = call.assistant;

        let content = assistant_content_text(&assistant);
        if content.trim().is_empty() {
            return Err(AppError::Ai("AI returned empty content".to_string()));
        }
        Ok(content)
    }

    #[allow(dead_code, clippy::too_many_arguments)]
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
        cancel_requested: Option<&AtomicBool>,
        execute_tool: F,
        on_round_event: impl FnMut(ChatRoundEvent),
        on_stream_event: G,
    ) -> Result<ChatToolResponse, AppError>
    where
        F: FnMut(&ToolCallRequest) -> String,
        G: FnMut(ChatStreamEvent),
    {
        self.chat_with_shell_tool_with_debug_session(
            history,
            system_prompt,
            user_prompt,
            policy,
            max_tool_rounds,
            max_total_tool_calls,
            stream_output,
            extra_tools,
            cancel_requested,
            None,
            execute_tool,
            on_round_event,
            on_stream_event,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chat_with_shell_tool_with_debug_session<F, G>(
        &self,
        history: &[ChatMessage],
        system_prompt: &str,
        user_prompt: &str,
        policy: ToolUsePolicy,
        max_tool_rounds: usize,
        max_total_tool_calls: usize,
        stream_output: bool,
        extra_tools: &[ExternalToolDefinition],
        cancel_requested: Option<&AtomicBool>,
        debug_session_id: Option<&str>,
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
        const MAX_REPEAT_TASK_DECOMPOSITION_GUARD: usize = 1;
        const MAX_FORCE_CONTINUE_HINT_ROUNDS: usize = 2;

        let mut messages = build_base_messages(history, system_prompt, user_prompt);
        let mut tools = vec![shell_tool_definition()];
        let mut seen_tool_names = HashSet::<String>::from(["run_shell_command".to_string()]);
        for tool in extra_tools {
            if seen_tool_names.insert(tool.name.clone()) {
                tools.push(external_tool_definition(tool));
            }
        }
        let mut forced_retry_used = false;
        let mut total_tool_calls: usize = 0;
        let mut tool_result_cache: HashMap<String, String> = HashMap::new();
        let mut same_tool_counter: HashMap<String, usize> = HashMap::new();
        let mut task_decomposition_guard_counter: HashMap<String, usize> = HashMap::new();
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
            return_if_chat_cancelled(cancel_requested)?;
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
                cancel_requested,
                debug_session_id,
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
            return_if_chat_cancelled(cancel_requested)?;
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
                    return_if_chat_cancelled(cancel_requested)?;
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
                        let cacheable_request = is_cacheable_tool_request(&request);
                        let result = if cacheable_request {
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
                                tool_result_cache.insert(signature.clone(), result.clone());
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
                        };
                        if tool_result_requires_task_decomposition(result.as_str()) {
                            same_tool_counter.remove(&signature);
                            let guard_count = task_decomposition_guard_counter
                                .entry(signature.clone())
                                .or_insert(0);
                            *guard_count += 1;
                            if *guard_count > MAX_REPEAT_TASK_DECOMPOSITION_GUARD {
                                finalize_reason = Some(ChatStopReason::TaskDecompositionRequired);
                                build_guard_tool_result("task_decomposition_required")
                            } else {
                                result
                            }
                        } else {
                            task_decomposition_guard_counter.remove(&signature);
                            let count_entry = same_tool_counter.entry(signature.clone()).or_insert(0);
                            *count_entry += 1;
                            if *count_entry > MAX_REPEAT_SAME_TOOL {
                                finalize_reason = Some(ChatStopReason::RepeatedSameToolCall);
                                build_guard_tool_result("repeated_same_tool_call")
                            } else {
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
                        cancel_requested,
                        debug_session_id,
                    );
                }
                continue;
            }

            if matches!(policy, ToolUsePolicy::RequireAtLeastOne) && !forced_retry_used {
                forced_retry_used = true;
                messages.push(ApiMessage {
                    role: "system".to_string(),
                    content: Some("You are running locally with direct tool access. You MUST call at least one relevant tool before giving a final answer for this request. Prefer matching MCP tools first, then built-in tools (View/LS/GlobTool/GrepTool/WebSearch), and use run_shell_command only when needed.".to_string()),
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
                    content: Some("Policy requires at least one tool call in this round. Choose the most relevant available tool now (MCP when it clearly matches, otherwise built-in tools, otherwise run_shell_command) instead of giving a final text-only response.".to_string()),
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

            logging::warn(
                "assistant returned empty content and no tool calls; using local recoverable fallback response",
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
            cancel_requested,
            debug_session_id,
        )
    }

    fn send_chat_completion(
        &self,
        body: &ChatCompletionRequest,
        cancel_requested: Option<&AtomicBool>,
        debug_session_id: Option<&str>,
    ) -> Result<ApiChatCallResult, AppError> {
        let mut effective_body = normalize_request_for_provider(body, &self.base_url);
        let mut stripped_reasoning_retry_used = false;
        let mut retry_with_fresh_client = false;
        let mut refreshed_for_idle_hint = false;
        let mut attempt: u32 = 0;
        loop {
            return_if_chat_cancelled(cancel_requested)?;
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
                &format!(
                    "AI request body: {}",
                    serialize_debug_json(&self.build_provider_request_body(&effective_body))
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
            let request_body = self.build_provider_request_body(&effective_body);
            let request_body_text = mask_sensitive(&serialize_debug_json(&request_body));
            let user_packets = collect_role_packets(&effective_body.messages, "user");
            let tool_packets = collect_role_packets(&effective_body.messages, "tool");
            self.append_request_trace_entry(
                debug_session_id,
                RequestTraceEntry {
                    timestamp_epoch_ms: now_epoch_ms(),
                    event: "request".to_string(),
                    stream: false,
                    attempt,
                    status_code: None,
                    request_body: Some(request_body_text),
                    response_body: None,
                    user_packets: (!user_packets.is_empty()).then_some(user_packets),
                    tool_packets: (!tool_packets.is_empty()).then_some(tool_packets),
                    error: None,
                },
            );
            let request_builder = self.apply_provider_auth(client.post(&self.base_url));
            let resp = request_builder.json(&request_body).send();

            let resp = match resp {
                Ok(resp) => resp,
                Err(err) => {
                    let err_msg = format!("AI request failed: {}", format_reqwest_error(&err));
                    self.append_request_trace_entry(
                        debug_session_id,
                        RequestTraceEntry {
                            timestamp_epoch_ms: now_epoch_ms(),
                            event: "response".to_string(),
                            stream: false,
                            attempt,
                            status_code: None,
                            request_body: None,
                            response_body: None,
                            user_packets: None,
                            tool_packets: None,
                            error: Some(mask_sensitive(&err_msg)),
                        },
                    );
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
                        sleep_with_chat_cancel(
                            Duration::from_millis(self.backoff_millis),
                            cancel_requested,
                        )?;
                        continue;
                    }
                    return Err(AppError::Ai(err_msg));
                }
            };

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
                self.append_request_trace_entry(
                    debug_session_id,
                    RequestTraceEntry {
                        timestamp_epoch_ms: now_epoch_ms(),
                        event: "response".to_string(),
                        stream: false,
                        attempt,
                        status_code: Some(status.as_u16()),
                        request_body: None,
                        response_body: Some(safe_body.clone()),
                        user_packets: None,
                        tool_packets: None,
                        error: None,
                    },
                );
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
                        sleep_with_chat_cancel(delay, cancel_requested)?;
                        continue;
                    }
                    if should_retry_status(status) {
                        sleep_with_chat_cancel(
                            Duration::from_millis(self.backoff_millis),
                            cancel_requested,
                        )?;
                        continue;
                    }
                }
                return Err(AppError::Ai(err_msg));
            }

            return_if_chat_cancelled(cancel_requested)?;
            let response_text = match resp.text() {
                Ok(text) => text,
                Err(err) => {
                    let err_msg = format!(
                        "failed to read AI response body: {}",
                        format_reqwest_error(&err)
                    );
                    self.append_request_trace_entry(
                        debug_session_id,
                        RequestTraceEntry {
                            timestamp_epoch_ms: now_epoch_ms(),
                            event: "response".to_string(),
                            stream: false,
                            attempt,
                            status_code: Some(status.as_u16()),
                            request_body: None,
                            response_body: None,
                            user_packets: None,
                            tool_packets: None,
                            error: Some(mask_sensitive(&err_msg)),
                        },
                    );
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
                        sleep_with_chat_cancel(
                            Duration::from_millis(self.backoff_millis),
                            cancel_requested,
                        )?;
                        continue;
                    }
                    return Err(AppError::Ai(err_msg));
                }
            };
            self.debug_emit(
                AiDebugLevel::Debug,
                &format!("AI response body: {}", mask_sensitive(&response_text)),
            );
            let safe_response = mask_sensitive(&response_text);
            self.append_request_trace_entry(
                debug_session_id,
                RequestTraceEntry {
                    timestamp_epoch_ms: now_epoch_ms(),
                    event: "response".to_string(),
                    stream: false,
                    attempt,
                    status_code: Some(status.as_u16()),
                    request_body: None,
                    response_body: Some(safe_response),
                    user_packets: None,
                    tool_packets: None,
                    error: None,
                },
            );
            let parsed = self.parse_provider_response_text(&response_text)?;
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
    }

    fn send_chat_completion_with_optional_streaming<G>(
        &self,
        body: &ChatCompletionRequest,
        stream_output: bool,
        on_stream_event: &mut G,
        cancel_requested: Option<&AtomicBool>,
        debug_session_id: Option<&str>,
    ) -> Result<ApiChatCallResult, AppError>
    where
        G: FnMut(ChatStreamEvent),
    {
        if !stream_output {
            return self.send_chat_completion(body, cancel_requested, debug_session_id);
        }
        let allow_tool_round_compat_fallback =
            body.tools.is_some() && model_prefers_non_streaming_tool_rounds(&self.model);
        if allow_tool_round_compat_fallback {
            logging::warn(
                "model may have weaker streaming tool-round compatibility; trying streaming first and keeping non-streaming fallback enabled",
            );
            self.debug_emit(
                AiDebugLevel::Warn,
                "AI streaming tool round compatibility mode: try streaming first, fallback to non-streaming only if needed",
            );
        }
        if self.provider != AiProviderProtocol::OpenAiCompatible {
            logging::warn(
                "AI provider is using non-OpenAI protocol; streaming mode falls back to non-streaming request",
            );
            self.debug_emit(
                AiDebugLevel::Warn,
                "AI streaming is not enabled for current provider protocol; fallback to non-streaming mode",
            );
            return self.send_chat_completion(body, cancel_requested, debug_session_id);
        }

        let mut effective_body = normalize_request_for_provider(body, &self.base_url);
        let mut stripped_reasoning_retry_used = false;
        let mut retry_with_fresh_client = false;
        let mut refreshed_for_idle_hint = false;
        let mut attempt: u32 = 0;
        loop {
            return_if_chat_cancelled(cancel_requested)?;
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
            let stream_request_body = self.build_provider_request_body(&stream_body);
            self.debug_emit(
                AiDebugLevel::Debug,
                &format!(
                    "AI streaming request body: {}",
                    serialize_debug_json(&stream_request_body)
                ),
            );
            let user_packets = collect_role_packets(&effective_body.messages, "user");
            let tool_packets = collect_role_packets(&effective_body.messages, "tool");
            self.append_request_trace_entry(
                debug_session_id,
                RequestTraceEntry {
                    timestamp_epoch_ms: now_epoch_ms(),
                    event: "request".to_string(),
                    stream: true,
                    attempt,
                    status_code: None,
                    request_body: Some(mask_sensitive(&serialize_debug_json(&stream_request_body))),
                    response_body: None,
                    user_packets: (!user_packets.is_empty()).then_some(user_packets),
                    tool_packets: (!tool_packets.is_empty()).then_some(tool_packets),
                    error: None,
                },
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
            let request_builder = self.apply_provider_auth(client.post(&self.base_url));
            let resp = request_builder.json(&stream_body).send();

            let resp = match resp {
                Ok(resp) => resp,
                Err(err) => {
                    let err_msg = format!("AI request failed: {}", format_reqwest_error(&err));
                    self.append_request_trace_entry(
                        debug_session_id,
                        RequestTraceEntry {
                            timestamp_epoch_ms: now_epoch_ms(),
                            event: "response".to_string(),
                            stream: true,
                            attempt,
                            status_code: None,
                            request_body: None,
                            response_body: None,
                            user_packets: None,
                            tool_packets: None,
                            error: Some(mask_sensitive(&err_msg)),
                        },
                    );
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
                        sleep_with_chat_cancel(
                            Duration::from_millis(self.backoff_millis),
                            cancel_requested,
                        )?;
                        continue;
                    }
                    if allow_tool_round_compat_fallback {
                        logging::warn(
                            "AI streaming transport retries exhausted on compatibility model; fallback to non-streaming mode",
                        );
                        self.debug_emit(
                            AiDebugLevel::Warn,
                            "AI streaming transport retries exhausted on compatibility model; fallback to non-streaming mode",
                        );
                        return self.send_chat_completion(body, cancel_requested, debug_session_id);
                    }
                    return Err(AppError::Ai(err_msg));
                }
            };

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
                self.append_request_trace_entry(
                    debug_session_id,
                    RequestTraceEntry {
                        timestamp_epoch_ms: now_epoch_ms(),
                        event: "response".to_string(),
                        stream: true,
                        attempt,
                        status_code: Some(status.as_u16()),
                        request_body: None,
                        response_body: Some(safe_body.clone()),
                        user_packets: None,
                        tool_packets: None,
                        error: None,
                    },
                );
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
                    return self.send_chat_completion(body, cancel_requested, debug_session_id);
                }
                let err_msg = format!("AI HTTP status={status}, body={safe_body}");
                logging::warn(&err_msg);
                if attempt <= self.max_retries {
                    if let Some(delay) =
                        parse_rate_limit_retry_delay(status, retry_after.as_deref())
                    {
                        sleep_with_chat_cancel(delay, cancel_requested)?;
                        continue;
                    }
                    if should_retry_status(status) {
                        sleep_with_chat_cancel(
                            Duration::from_millis(self.backoff_millis),
                            cancel_requested,
                        )?;
                        continue;
                    }
                }
                return Err(AppError::Ai(err_msg));
            }

            return match parse_streaming_chat_response(
                resp,
                on_stream_event,
                self.debug,
                self.debug && debug_session_id.is_some(),
                cancel_requested,
            ) {
                Ok(parsed) => {
                    self.append_request_trace_entry(
                        debug_session_id,
                        RequestTraceEntry {
                            timestamp_epoch_ms: now_epoch_ms(),
                            event: "response".to_string(),
                            stream: true,
                            attempt,
                            status_code: Some(status.as_u16()),
                            request_body: None,
                            response_body: Some(build_stream_response_trace_summary(&parsed)),
                            user_packets: None,
                            tool_packets: None,
                            error: None,
                        },
                    );
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
                    Ok(ApiChatCallResult {
                        assistant: parsed.assistant,
                        usage: parsed.usage,
                        elapsed_ms: started.elapsed().as_millis(),
                        streamed_content: parsed.streamed_content,
                        streamed_thinking: parsed.streamed_thinking,
                    })
                }
                Err(StreamParseResult::FallbackJson(body_text)) => {
                    self.append_request_trace_entry(
                        debug_session_id,
                        RequestTraceEntry {
                            timestamp_epoch_ms: now_epoch_ms(),
                            event: "response".to_string(),
                            stream: true,
                            attempt,
                            status_code: Some(status.as_u16()),
                            request_body: None,
                            response_body: Some(mask_sensitive(&body_text)),
                            user_packets: None,
                            tool_packets: None,
                            error: None,
                        },
                    );
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
                    Ok(ApiChatCallResult {
                        assistant: parsed.message,
                        usage: parsed.usage,
                        elapsed_ms: started.elapsed().as_millis(),
                        streamed_content: false,
                        streamed_thinking: false,
                    })
                }
                Err(StreamParseResult::Retryable(err_msg)) => {
                    self.append_request_trace_entry(
                        debug_session_id,
                        RequestTraceEntry {
                            timestamp_epoch_ms: now_epoch_ms(),
                            event: "response".to_string(),
                            stream: true,
                            attempt,
                            status_code: Some(status.as_u16()),
                            request_body: None,
                            response_body: None,
                            user_packets: None,
                            tool_packets: None,
                            error: Some(mask_sensitive(&err_msg)),
                        },
                    );
                    self.debug_emit(AiDebugLevel::Warn, &err_msg);
                    logging::warn(&err_msg);
                    if attempt <= self.max_retries {
                        retry_with_fresh_client = true;
                        maybe_print_ai_reconnect_notice(
                            &i18n::chat_ai_reconnecting(attempt, self.max_retries),
                            self.colorful,
                        );
                        sleep_with_chat_cancel(
                            Duration::from_millis(self.backoff_millis),
                            cancel_requested,
                        )?;
                        continue;
                    }
                    if allow_tool_round_compat_fallback {
                        logging::warn(
                            "AI streaming tool round retries exhausted; fallback to non-streaming compatibility mode",
                        );
                        self.debug_emit(
                            AiDebugLevel::Warn,
                            "AI streaming retries exhausted on compatibility model; fallback to non-streaming mode",
                        );
                        return self.send_chat_completion(body, cancel_requested, debug_session_id);
                    }
                    Err(AppError::Ai(err_msg))
                }
                Err(StreamParseResult::Cancelled) => {
                    self.append_request_trace_entry(
                        debug_session_id,
                        RequestTraceEntry {
                            timestamp_epoch_ms: now_epoch_ms(),
                            event: "response".to_string(),
                            stream: true,
                            attempt,
                            status_code: Some(status.as_u16()),
                            request_body: None,
                            response_body: None,
                            user_packets: None,
                            tool_packets: None,
                            error: Some(CHAT_CANCELLED_ERROR.to_string()),
                        },
                    );
                    Err(AppError::Ai(CHAT_CANCELLED_ERROR.to_string()))
                }
                Err(StreamParseResult::Fatal(err_msg)) => {
                    self.append_request_trace_entry(
                        debug_session_id,
                        RequestTraceEntry {
                            timestamp_epoch_ms: now_epoch_ms(),
                            event: "response".to_string(),
                            stream: true,
                            attempt,
                            status_code: Some(status.as_u16()),
                            request_body: None,
                            response_body: None,
                            user_packets: None,
                            tool_packets: None,
                            error: Some(mask_sensitive(&err_msg)),
                        },
                    );
                    self.debug_emit(AiDebugLevel::Error, &err_msg);
                    logging::warn(&err_msg);
                    if allow_tool_round_compat_fallback {
                        logging::warn(
                            "AI streaming parse failed on compatibility model; fallback to non-streaming mode",
                        );
                        self.debug_emit(
                            AiDebugLevel::Warn,
                            "AI streaming parse failed on compatibility model; fallback to non-streaming mode",
                        );
                        return self.send_chat_completion(body, cancel_requested, debug_session_id);
                    }
                    Err(AppError::Ai(err_msg))
                }
            };
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_without_tools(
        &self,
        mut messages: Vec<ApiMessage>,
        mut state: FinalizeWithoutToolsState,
        stop_reason: Option<ChatStopReason>,
        stream_output: bool,
        on_stream_event: &mut impl FnMut(ChatStreamEvent),
        cancel_requested: Option<&AtomicBool>,
        debug_session_id: Option<&str>,
    ) -> Result<ChatToolResponse, AppError> {
        return_if_chat_cancelled(cancel_requested)?;
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
            cancel_requested,
            debug_session_id,
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
        return_if_chat_cancelled(cancel_requested)?;
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

    fn append_request_trace_entry(&self, session_id: Option<&str>, entry: RequestTraceEntry) {
        if !self.debug {
            return;
        }
        let Some(raw_id) = session_id.map(str::trim).filter(|value| !value.is_empty()) else {
            return;
        };
        let session_id = sanitize_trace_session_id(raw_id);
        if session_id.is_empty() {
            return;
        }
        if let Err(err) = self.try_append_request_trace_entry(&session_id, entry) {
            logging::warn(&format!(
                "failed to append AI request trace for session {}: {}",
                session_id,
                mask_sensitive(&err.to_string())
            ));
        }
    }

    fn try_append_request_trace_entry(
        &self,
        session_id: &str,
        entry: RequestTraceEntry,
    ) -> Result<(), AppError> {
        let _guard = AI_REQUEST_TRACE_FILE_LOCK
            .lock()
            .map_err(|_| AppError::Runtime("failed to lock AI request trace file".to_string()))?;
        fs::create_dir_all(&self.request_trace_dir).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create AI request trace directory {}: {err}",
                self.request_trace_dir.display()
            ))
        })?;
        let path = self
            .request_trace_dir
            .join(format!("request-{session_id}.json"));
        let mut file = if path.exists() {
            let raw = fs::read_to_string(&path).map_err(|err| {
                AppError::Runtime(format!(
                    "failed to read AI request trace file {}: {err}",
                    path.display()
                ))
            })?;
            serde_json::from_str::<SessionRequestTraceFile>(&raw).unwrap_or_default()
        } else {
            SessionRequestTraceFile::default()
        };
        if file.version == 0 {
            file.version = REQUEST_TRACE_FILE_VERSION;
        }
        if file.session_id.trim().is_empty() {
            file.session_id = session_id.to_string();
        }
        file.entries.push(entry);
        if file.entries.len() > REQUEST_TRACE_MAX_ENTRIES {
            let drop_count = file.entries.len().saturating_sub(REQUEST_TRACE_MAX_ENTRIES);
            if drop_count > 0 {
                file.entries.drain(0..drop_count);
            }
        }
        let encoded = serde_json::to_string_pretty(&file).map_err(|err| {
            AppError::Runtime(format!("failed to serialize AI request trace file: {err}"))
        })?;
        write_string_atomically(&path, &encoded)?;
        Ok(())
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
        let call = self.send_chat_completion(&request, None, None)?;
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
    if allows_direct_stdout_printing() {
        println!("{} {}", level.localized_tag(), sanitized);
    }
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
    Regex::new(
        r#"(?is)<\s*[|｜]\s*dsml\s*[|｜]\s*function_calls\s*>.*?<\s*/\s*[|｜]\s*dsml\s*[|｜]\s*function_calls\s*>"#,
    )
    .expect("valid dsml function call block regex")
});
static DSML_INVOKE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<\s*[|｜]\s*dsml\s*[|｜]\s*invoke\b[^>]*\bname\s*=\s*["']([^"']+)["'][^>]*>(.*?)<\s*/\s*[|｜]\s*dsml\s*[|｜]\s*invoke\s*>"#,
    )
    .expect("valid dsml invoke regex")
});
static DSML_PARAMETER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<\s*[|｜]\s*dsml\s*[|｜]\s*parameter\b[^>]*\bname\s*=\s*["']([^"']+)["'][^>]*>(.*?)<\s*/\s*[|｜]\s*dsml\s*[|｜]\s*parameter\s*>"#,
    )
    .expect("valid dsml parameter regex")
});
static ANSI_ESCAPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\x1b\[[0-?]*[ -/]*[@-~]").expect("valid ansi escape regex")
});
static AI_REQUEST_TRACE_FILE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

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
    sse_packets: Vec<String>,
}

enum StreamParseResult {
    FallbackJson(String),
    Retryable(String),
    Cancelled,
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

fn chat_cancel_requested(cancel_requested: Option<&AtomicBool>) -> bool {
    cancel_requested.is_some_and(|flag| flag.load(Ordering::SeqCst))
}

fn return_if_chat_cancelled(cancel_requested: Option<&AtomicBool>) -> Result<(), AppError> {
    if chat_cancel_requested(cancel_requested) {
        return Err(AppError::Ai(CHAT_CANCELLED_ERROR.to_string()));
    }
    Ok(())
}

fn sleep_with_chat_cancel(
    duration: Duration,
    cancel_requested: Option<&AtomicBool>,
) -> Result<(), AppError> {
    if duration.is_zero() {
        return Ok(());
    }
    let mut elapsed = Duration::ZERO;
    while elapsed < duration {
        return_if_chat_cancelled(cancel_requested)?;
        let step = (duration - elapsed).min(Duration::from_millis(50));
        thread::sleep(step);
        elapsed += step;
    }
    Ok(())
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
    capture_sse_packets: bool,
    cancel_requested: Option<&AtomicBool>,
) -> Result<ParsedStreamingChatResponse, StreamParseResult>
where
    G: FnMut(ChatStreamEvent),
{
    let mut reader = BufReader::new(response);
    let mut line = String::new();
    let mut buffered_body = String::new();
    let mut pending_sse_payload = String::new();
    let mut saw_sse = false;
    let mut usage = Usage::default();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::<PartialToolCall>::new();
    let mut streamed_content = false;
    let mut streamed_thinking = false;
    let mut sse_packets = Vec::<String>::new();

    loop {
        if chat_cancel_requested(cancel_requested) {
            return Err(StreamParseResult::Cancelled);
        }
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
            if !pending_sse_payload.is_empty() {
                if capture_sse_packets {
                    if sse_packets.len() < 1000 {
                        sse_packets.push(pending_sse_payload.clone());
                    } else if sse_packets.len() == 1000 {
                        sse_packets.push("{\"truncated\":true}".to_string());
                    }
                }
                if process_streaming_sse_payload(
                    pending_sse_payload.as_str(),
                    on_stream_event,
                    debug_enabled,
                    &mut usage,
                    &mut content,
                    &mut reasoning,
                    &mut tool_calls,
                    &mut streamed_content,
                    &mut streamed_thinking,
                )? {
                    pending_sse_payload.clear();
                    break;
                }
                pending_sse_payload.clear();
            }
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
        if payload.is_empty() {
            continue;
        }
        if !pending_sse_payload.is_empty() {
            pending_sse_payload.push('\n');
        }
        pending_sse_payload.push_str(payload);
        if streaming_sse_payload_is_complete(pending_sse_payload.as_str()) {
            if capture_sse_packets {
                if sse_packets.len() < 1000 {
                    sse_packets.push(pending_sse_payload.clone());
                } else if sse_packets.len() == 1000 {
                    sse_packets.push("{\"truncated\":true}".to_string());
                }
            }
            if process_streaming_sse_payload(
                pending_sse_payload.as_str(),
                on_stream_event,
                debug_enabled,
                &mut usage,
                &mut content,
                &mut reasoning,
                &mut tool_calls,
                &mut streamed_content,
                &mut streamed_thinking,
            )? {
                pending_sse_payload.clear();
                break;
            }
            pending_sse_payload.clear();
        }
    }

    if !pending_sse_payload.is_empty() {
        if capture_sse_packets {
            if sse_packets.len() < 1000 {
                sse_packets.push(pending_sse_payload.clone());
            } else if sse_packets.len() == 1000 {
                sse_packets.push("{\"truncated\":true}".to_string());
            }
        }
        if process_streaming_sse_payload(
            pending_sse_payload.as_str(),
            on_stream_event,
            debug_enabled,
            &mut usage,
            &mut content,
            &mut reasoning,
            &mut tool_calls,
            &mut streamed_content,
            &mut streamed_thinking,
        )? {}
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
        sse_packets,
    })
}

fn streaming_sse_payload_is_complete(payload: &str) -> bool {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed == "[DONE]" {
        return true;
    }
    (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
}

#[allow(clippy::too_many_arguments)]
fn process_streaming_sse_payload<G>(
    payload: &str,
    on_stream_event: &mut G,
    debug_enabled: bool,
    usage: &mut Usage,
    content: &mut String,
    reasoning: &mut String,
    tool_calls: &mut Vec<PartialToolCall>,
    streamed_content: &mut bool,
    streamed_thinking: &mut bool,
) -> Result<bool, StreamParseResult>
where
    G: FnMut(ChatStreamEvent),
{
    let normalized = payload.trim();
    if normalized.is_empty() {
        return Ok(false);
    }
    if normalized == "[DONE]" {
        return Ok(true);
    }
    emit_ai_debug(
        debug_enabled,
        AiDebugLevel::Debug,
        &format!("AI streaming chunk: {normalized}"),
    );
    let parsed: ChatCompletionStreamResponse = serde_json::from_str(normalized).map_err(|err| {
        emit_ai_debug(
            debug_enabled,
            AiDebugLevel::Error,
            &format!("AI streaming chunk parse error: {err}; payload={normalized}"),
        );
        if *streamed_content || *streamed_thinking {
            StreamParseResult::Fatal(format!("failed to parse AI streaming chunk: {err}"))
        } else {
            StreamParseResult::Retryable(format!("failed to parse AI streaming chunk: {err}"))
        }
    })?;
    let previous_usage = std::mem::take(usage);
    *usage = merge_usage(previous_usage, parsed.usage);
    for choice in parsed.choices {
        let thinking_delta = assistant_delta_reasoning_text(&choice.delta);
        let added_thinking = merge_text_delta(reasoning, &thinking_delta);
        if !added_thinking.is_empty() {
            *streamed_thinking = true;
            on_stream_event(ChatStreamEvent {
                kind: ChatStreamEventKind::Thinking,
                text: added_thinking,
            });
        }

        let content_delta = assistant_delta_content_text(&choice.delta);
        let added_content = merge_text_delta(content, &content_delta);
        if !added_content.is_empty() {
            *streamed_content = true;
            on_stream_event(ChatStreamEvent {
                kind: ChatStreamEventKind::Content,
                text: added_content,
            });
        }

        for tool_call_delta in choice.delta.tool_calls {
            merge_tool_call_delta(tool_calls, &tool_call_delta);
        }
    }
    Ok(false)
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
    if incoming.len() > target.len() && incoming.starts_with(target.as_str()) {
        let suffix = &incoming[target.len()..];
        if !suffix.is_empty() {
            target.push_str(suffix);
        }
        return suffix.to_string();
    }
    let overlap = stream_text_overlap_bytes(target.as_str(), incoming);
    if overlap > 0 {
        if overlap == incoming.len() && incoming.chars().count() == 1 {
            target.push_str(incoming);
            return incoming.to_string();
        }
        let suffix = &incoming[overlap..];
        if !suffix.is_empty() {
            target.push_str(suffix);
        }
        return suffix.to_string();
    }
    target.push_str(incoming);
    incoming.to_string()
}

fn stream_text_overlap_bytes(target: &str, incoming: &str) -> usize {
    let max_overlap = target.len().min(incoming.len());
    for size in (1..=max_overlap).rev() {
        if !incoming.is_char_boundary(size) {
            continue;
        }
        if target.ends_with(&incoming[..size]) {
            return size;
        }
    }
    0
}

fn merge_tool_call_delta(tool_calls: &mut Vec<PartialToolCall>, delta: &StreamToolCallDelta) {
    let index = resolve_tool_call_delta_index(tool_calls, delta);
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

fn resolve_tool_call_delta_index(
    tool_calls: &[PartialToolCall],
    delta: &StreamToolCallDelta,
) -> usize {
    if let Some(index) = delta.index {
        return index;
    }
    let matched_by_id = delta.id.as_deref().and_then(|id| {
        let normalized = id.trim();
        if normalized.is_empty() {
            return None;
        }
        tool_calls
            .iter()
            .position(|item| item.id.trim() == normalized)
    });
    matched_by_id.unwrap_or(tool_calls.len())
}

fn finalize_tool_calls(tool_calls: Vec<PartialToolCall>) -> Vec<ApiToolCall> {
    tool_calls
        .into_iter()
        .enumerate()
        .filter_map(|(idx, item)| {
            let PartialToolCall {
                id,
                r#type,
                name,
                arguments,
            } = item;
            if name.trim().is_empty() || arguments.trim().is_empty() {
                return None;
            }
            let resolved_id = if id.trim().is_empty() {
                format!("stream_tool_call_{}", idx + 1)
            } else {
                id
            };
            Some(ApiToolCall {
                id: resolved_id,
                r#type: if r#type.trim().is_empty() {
                    "function".to_string()
                } else {
                    r#type
                },
                function: ApiToolFunction { name, arguments },
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
            if map_contains_reasoning_type(map) {
                return;
            }
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
            if map_contains_reasoning_type(map) {
                return;
            }
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
            let is_reasoning_block = map_contains_reasoning_type(map);
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
            for key in ["reasoning", "reasoning_content"] {
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

fn map_contains_reasoning_type(map: &serde_json::Map<String, serde_json::Value>) -> bool {
    map.get("type")
        .and_then(|v| v.as_str())
        .map(|t| {
            let normalized = t.trim().to_ascii_lowercase();
            normalized.contains("reasoning") || normalized.contains("thinking")
        })
        .unwrap_or(false)
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
    if is_builtin_read_tool_name(&request.name) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&request.arguments) {
            return !value
                .get("apply")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        }
        return false;
    }
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

fn is_builtin_read_tool_name(name: &str) -> bool {
    let normalized = name.trim().to_ascii_lowercase().replace(['_', '-'], "");
    matches!(
        normalized.as_str(),
        "view"
            | "readfile"
            | "ls"
            | "listfiles"
            | "globtool"
            | "globsearch"
            | "greptool"
            | "grepsearch"
            | "websearch"
            | "think"
            | "task"
            | "architect"
    )
}

fn tool_result_timed_out(payload: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()
        .and_then(|value| value.get("timed_out").and_then(|item| item.as_bool()))
        .unwrap_or(false)
}

fn tool_result_requires_task_decomposition(payload: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return false;
    };
    if value
        .get("reason")
        .and_then(|item| item.as_str())
        .is_some_and(|item| item.eq_ignore_ascii_case("task_decomposition_required"))
    {
        return true;
    }
    value
        .get("error")
        .and_then(|item| item.as_str())
        .is_some_and(|item| item.to_ascii_lowercase().contains("task decomposition is required"))
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
            description: "Execute local shell command on current machine. Prefer read-only commands. Use mode=write only when mutation is necessary. Arguments must be a strict JSON object.".to_string(),
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

fn default_request_trace_file_version() -> u32 {
    REQUEST_TRACE_FILE_VERSION
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn sanitize_trace_session_id(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn collect_role_packets(messages: &[ApiMessage], role: &str) -> Vec<String> {
    messages
        .iter()
        .filter(|item| item.role == role)
        .map(|item| {
            let content = item.content.clone().unwrap_or_default();
            mask_sensitive(&content)
        })
        .collect()
}

fn build_stream_response_trace_summary(parsed: &ParsedStreamingChatResponse) -> String {
    let content = assistant_content_text(&parsed.assistant);
    let thinking = assistant_reasoning_text(&parsed.assistant);
    let tool_calls = parsed
        .assistant
        .tool_calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": call.r#type,
                "name": call.function.name,
                "arguments": mask_sensitive(&call.function.arguments),
            })
        })
        .collect::<Vec<_>>();
    mask_sensitive(&serialize_debug_json(&json!({
        "streamed_content": parsed.streamed_content,
        "streamed_thinking": parsed.streamed_thinking,
        "usage": {
            "prompt_tokens": parsed.usage.prompt_tokens,
            "completion_tokens": parsed.usage.completion_tokens,
            "total_tokens": parsed.usage.total_tokens,
        },
        "assistant_content": content,
        "assistant_thinking": thinking,
        "assistant_tool_calls": tool_calls,
        "sse_packets": parsed.sse_packets,
    })))
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
    if let Err(rename_err) = fs::rename(&temp_path, path) {
        if let Err(copy_err) = fs::copy(&temp_path, path) {
            let _ = fs::remove_file(&temp_path);
            return Err(AppError::Runtime(format!(
                "failed to replace file {} (rename: {}; copy fallback: {})",
                path.display(),
                rename_err,
                copy_err
            )));
        }
        let _ = fs::remove_file(&temp_path);
    }
    Ok(())
}

fn optional_content(content: String) -> Option<String> {
    if content.trim().is_empty() {
        return None;
    }
    Some(content)
}

fn strip_ansi_escape_sequences(raw: &str) -> String {
    ANSI_ESCAPE_RE.replace_all(raw, "").to_string()
}

fn normalize_dsml_markup(raw: &str) -> String {
    raw.replace(['“', '”'], "\"").replace(['‘', '’'], "'")
}

fn sanitize_assistant_content(raw: &str) -> String {
    let no_ansi = strip_ansi_escape_sequences(raw);
    let normalized = normalize_dsml_markup(no_ansi.as_str());
    let cleaned = DSML_FUNCTION_CALLS_BLOCK_RE.replace_all(normalized.as_str(), "");
    cleaned
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

fn parse_dsml_tool_calls(raw: &str, round: usize) -> Vec<ToolCallRequest> {
    let raw_no_ansi = strip_ansi_escape_sequences(raw);
    let normalized = normalize_dsml_markup(raw_no_ansi.as_str());
    if !normalized.to_ascii_lowercase().contains("dsml") {
        return Vec::new();
    }
    let mut output = Vec::new();
    for (idx, invoke_caps) in DSML_INVOKE_RE.captures_iter(normalized.as_str()).enumerate() {
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
        || lowered.contains("failed to read ai response body")
        || lowered.contains("error decoding response body")
        || lowered.contains("failed to read chunk")
        || lowered.contains("connection closed before message completed")
        || lowered.contains("ai returned empty content")
        || lowered.contains("ai returned empty choices")
        || lowered.contains("ai returned empty candidates")
        || lowered.contains("ai returned empty content blocks")
        || lowered.contains("gemini blocked response")
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
    if err.is_timeout() || err.is_connect() || err.is_request() || err.is_body() || err.is_decode()
    {
        return true;
    }
    let lowered = err.to_string().to_ascii_lowercase();
    lowered.contains("connection reset")
        || lowered.contains("broken pipe")
        || lowered.contains("connection closed")
        || lowered.contains("unexpected eof")
        || lowered.contains("channel closed")
        || lowered.contains("tls handshake eof")
        || lowered.contains("error decoding response body")
        || lowered.contains("failed to read chunk")
        || lowered.contains("connection closed before message completed")
}

fn maybe_print_ai_reconnect_notice(message: &str, colorful: bool) {
    if !allows_direct_stdout_printing() {
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

fn allows_direct_stdout_printing() -> bool {
    should_write_direct_stdout(std::io::stdout().is_terminal(), is_raw_mode_enabled().ok())
}

fn should_write_direct_stdout(is_terminal: bool, raw_mode_enabled: Option<bool>) -> bool {
    is_terminal && !raw_mode_enabled.unwrap_or(false)
}

fn model_prefers_omit_tool_choice(model: &str) -> bool {
    let normalized = model.trim().to_ascii_lowercase();
    normalized == "deepseek-reasoner"
}

fn model_prefers_non_streaming_tool_rounds(model: &str) -> bool {
    let normalized = model.trim().to_ascii_lowercase();
    normalized == "deepseek-reasoner" || normalized == "deepseek-r1"
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

fn provider_protocol_from_config_type(raw: &str) -> AiProviderProtocol {
    match normalize_ai_provider_type(raw) {
        "claude" => AiProviderProtocol::Claude,
        "gemini" => AiProviderProtocol::Gemini,
        _ => AiProviderProtocol::OpenAiCompatible,
    }
}

fn build_provider_chat_url(base_url: &str, provider: AiProviderProtocol, model: &str) -> String {
    match provider {
        AiProviderProtocol::OpenAiCompatible => build_chat_url(base_url),
        AiProviderProtocol::Claude => build_claude_chat_url(base_url),
        AiProviderProtocol::Gemini => build_gemini_chat_url(base_url, model),
    }
}

fn build_claude_chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/messages") {
        return trimmed.to_string();
    }
    if trimmed.ends_with("/chat/completions") {
        let prefix = trimmed.trim_end_matches("/chat/completions");
        if prefix.ends_with("/v1") {
            return format!("{prefix}/messages");
        }
        return format!("{prefix}/v1/messages");
    }
    if trimmed.ends_with("/v1") {
        return format!("{trimmed}/messages");
    }
    format!("{trimmed}/messages")
}

fn build_gemini_chat_url(base_url: &str, model: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.to_ascii_lowercase().contains(":generatecontent") {
        return trimmed.to_string();
    }
    if trimmed.ends_with("/chat/completions") {
        let prefix = trimmed.trim_end_matches("/chat/completions");
        return format!(
            "{prefix}/models/{}:generateContent",
            normalize_gemini_model_name(model)
        );
    }
    if trimmed.contains("/models/") {
        return format!("{trimmed}:generateContent");
    }
    format!(
        "{trimmed}/models/{}:generateContent",
        normalize_gemini_model_name(model)
    )
}

fn normalize_gemini_model_name(model: &str) -> String {
    let trimmed = model.trim().trim_matches('/');
    if let Some(stripped) = trimmed.strip_prefix("models/") {
        return stripped.to_string();
    }
    trimmed.to_string()
}

fn build_claude_request_body(body: &ChatCompletionRequest) -> serde_json::Value {
    let (system_prompt, messages) = convert_openai_messages_to_claude(&body.messages);
    let mut payload = json!({
        "model": body.model,
        "temperature": body.temperature,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "messages": messages,
    });
    if let Some(system) = system_prompt {
        payload["system"] = serde_json::Value::String(system);
    }
    if let Some(tools) = body.tools.as_ref() {
        let tool_defs = tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "input_schema": tool.function.parameters,
                })
            })
            .collect::<Vec<_>>();
        if !tool_defs.is_empty() {
            payload["tools"] = serde_json::Value::Array(tool_defs);
            if let Some(tool_choice) = claude_tool_choice_value(body.tool_choice.as_ref()) {
                payload["tool_choice"] = tool_choice;
            }
        }
    }
    payload
}

fn build_gemini_request_body(body: &ChatCompletionRequest) -> serde_json::Value {
    let (system_prompt, mut contents) = convert_openai_messages_to_gemini(&body.messages);
    if contents.is_empty() {
        contents.push(json!({
            "role": "user",
            "parts": [{"text": ""}]
        }));
    }
    let mut payload = json!({
        "contents": contents,
        "generationConfig": {
            "temperature": body.temperature
        }
    });
    if let Some(system) = system_prompt {
        payload["systemInstruction"] = json!({
            "parts": [{"text": system}]
        });
    }
    if let Some(tools) = body.tools.as_ref() {
        let function_declarations = tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                })
            })
            .collect::<Vec<_>>();
        if !function_declarations.is_empty() {
            payload["tools"] = json!([{
                "functionDeclarations": function_declarations
            }]);
            if let Some(tool_config) = gemini_tool_config_value(body.tool_choice.as_ref()) {
                payload["toolConfig"] = tool_config;
            }
        }
    }
    payload
}

fn convert_openai_messages_to_claude(
    messages: &[ApiMessage],
) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system_chunks = Vec::new();
    let mut converted = Vec::new();
    for message in messages {
        match message.role.as_str() {
            "system" => {
                if let Some(text) = message.content.as_deref()
                    && !text.trim().is_empty()
                {
                    system_chunks.push(text.to_string());
                }
            }
            "assistant" => {
                let mut parts = Vec::new();
                if let Some(text) = message.content.as_deref()
                    && !text.trim().is_empty()
                {
                    parts.push(json!({
                        "type": "text",
                        "text": text
                    }));
                }
                if let Some(tool_calls) = message.tool_calls.as_ref() {
                    for tool_call in tool_calls {
                        parts.push(json!({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": parse_tool_arguments_object(&tool_call.function.arguments),
                        }));
                    }
                }
                if !parts.is_empty() {
                    converted.push(json!({
                        "role": "assistant",
                        "content": parts
                    }));
                }
            }
            "tool" => {
                let mut parts = Vec::new();
                if let Some(id) = message.tool_call_id.as_deref()
                    && !id.trim().is_empty()
                {
                    parts.push(json!({
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": message.content.as_deref().unwrap_or_default(),
                    }));
                } else if let Some(text) = message.content.as_deref()
                    && !text.trim().is_empty()
                {
                    parts.push(json!({
                        "type": "text",
                        "text": text
                    }));
                }
                if !parts.is_empty() {
                    converted.push(json!({
                        "role": "user",
                        "content": parts
                    }));
                }
            }
            _ => {
                if let Some(text) = message.content.as_deref()
                    && !text.trim().is_empty()
                {
                    converted.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": text
                        }]
                    }));
                }
            }
        }
    }
    (optional_text(system_chunks.join("\n\n")), converted)
}

fn convert_openai_messages_to_gemini(
    messages: &[ApiMessage],
) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system_chunks = Vec::new();
    let mut converted = Vec::new();
    let mut tool_names = HashMap::<String, String>::new();

    for message in messages {
        match message.role.as_str() {
            "system" => {
                if let Some(text) = message.content.as_deref()
                    && !text.trim().is_empty()
                {
                    system_chunks.push(text.to_string());
                }
            }
            "assistant" => {
                let mut parts = Vec::new();
                if let Some(text) = message.content.as_deref()
                    && !text.trim().is_empty()
                {
                    parts.push(json!({ "text": text }));
                }
                if let Some(tool_calls) = message.tool_calls.as_ref() {
                    for tool_call in tool_calls {
                        tool_names.insert(tool_call.id.clone(), tool_call.function.name.clone());
                        parts.push(json!({
                            "functionCall": {
                                "name": tool_call.function.name,
                                "args": parse_tool_arguments_object(&tool_call.function.arguments),
                            }
                        }));
                    }
                }
                if !parts.is_empty() {
                    converted.push(json!({
                        "role": "model",
                        "parts": parts
                    }));
                }
            }
            "tool" => {
                let mut parts = Vec::new();
                if let Some(id) = message.tool_call_id.as_deref()
                    && !id.trim().is_empty()
                {
                    let tool_name = tool_names
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| "run_shell_command".to_string());
                    parts.push(json!({
                        "functionResponse": {
                            "name": tool_name,
                            "response": parse_tool_result_payload(message.content.as_deref().unwrap_or_default())
                        }
                    }));
                } else if let Some(text) = message.content.as_deref()
                    && !text.trim().is_empty()
                {
                    parts.push(json!({ "text": text }));
                }
                if !parts.is_empty() {
                    converted.push(json!({
                        "role": "user",
                        "parts": parts
                    }));
                }
            }
            _ => {
                if let Some(text) = message.content.as_deref()
                    && !text.trim().is_empty()
                {
                    converted.push(json!({
                        "role": "user",
                        "parts": [{ "text": text }]
                    }));
                }
            }
        }
    }
    (optional_text(system_chunks.join("\n\n")), converted)
}

fn parse_tool_arguments_object(arguments: &str) -> serde_json::Value {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(serde_json::Value::Object(object)) => serde_json::Value::Object(object),
        Ok(value) => json!({ "value": value }),
        Err(_) => json!({ "raw": trimmed }),
    }
}

fn parse_tool_result_payload(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return json!({ "content": "" });
    }
    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(serde_json::Value::Object(object)) => serde_json::Value::Object(object),
        Ok(value) => json!({ "content": value }),
        Err(_) => json!({ "content": trimmed }),
    }
}

fn claude_tool_choice_value(choice: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    let choice = choice?;
    if let Some(raw) = choice.as_str() {
        return Some(match raw.trim().to_ascii_lowercase().as_str() {
            "required" => json!({ "type": "any" }),
            "none" => json!({ "type": "auto" }),
            _ => json!({ "type": "auto" }),
        });
    }
    let Some(object) = choice.as_object() else {
        return Some(json!({ "type": "auto" }));
    };
    let choice_type = object
        .get("type")
        .and_then(|value| value.as_str())
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if choice_type == "function" {
        let function_name = object
            .get("function")
            .and_then(|value| value.get("name"))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())?;
        return Some(json!({
            "type": "tool",
            "name": function_name
        }));
    }
    Some(match choice_type.as_str() {
        "none" => json!({ "type": "auto" }),
        "required" => json!({ "type": "any" }),
        _ => json!({ "type": "auto" }),
    })
}

fn gemini_tool_config_value(choice: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    let choice = choice?;
    let mut function_calling = json!({ "mode": "AUTO" });
    if let Some(raw) = choice.as_str() {
        function_calling["mode"] = serde_json::Value::String(
            match raw.trim().to_ascii_lowercase().as_str() {
                "required" => "ANY",
                "none" => "NONE",
                _ => "AUTO",
            }
            .to_string(),
        );
        return Some(json!({ "functionCallingConfig": function_calling }));
    }
    if let Some(object) = choice.as_object() {
        let choice_type = object
            .get("type")
            .and_then(|value| value.as_str())
            .map(|value| value.to_ascii_lowercase())
            .unwrap_or_default();
        if choice_type == "function"
            && let Some(function_name) = object
                .get("function")
                .and_then(|value| value.get("name"))
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
        {
            function_calling["mode"] = serde_json::Value::String("ANY".to_string());
            function_calling["allowedFunctionNames"] =
                serde_json::Value::Array(vec![serde_json::Value::String(
                    function_name.to_string(),
                )]);
            return Some(json!({ "functionCallingConfig": function_calling }));
        }
        if choice_type == "none" {
            function_calling["mode"] = serde_json::Value::String("NONE".to_string());
        }
    }
    Some(json!({ "functionCallingConfig": function_calling }))
}

fn parse_claude_response_text(body: &str) -> Result<ParsedChatResponse, AppError> {
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|err| AppError::Ai(format!("failed to parse AI response: {err}")))?;
    if let Some(error_message) = extract_provider_error_message(&parsed) {
        return Err(AppError::Ai(error_message));
    }
    let content_items = parsed
        .get("content")
        .and_then(|value| value.as_array())
        .ok_or_else(|| AppError::Ai("AI returned empty content blocks".to_string()))?;
    let mut text_chunks = Vec::new();
    let mut reasoning_chunks = Vec::new();
    let mut tool_calls = Vec::new();
    for (index, item) in content_items.iter().enumerate() {
        let item_type = item
            .get("type")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        match item_type {
            "text" => {
                if let Some(text) = item.get("text").and_then(|value| value.as_str())
                    && !text.trim().is_empty()
                {
                    text_chunks.push(text.to_string());
                }
            }
            "thinking" => {
                if let Some(thinking) = item
                    .get("thinking")
                    .or_else(|| item.get("text"))
                    .and_then(|value| value.as_str())
                    && !thinking.trim().is_empty()
                {
                    reasoning_chunks.push(thinking.to_string());
                }
            }
            "tool_use" => {
                let name = item
                    .get("name")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .unwrap_or_default();
                if name.is_empty() {
                    continue;
                }
                let id = item
                    .get("id")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| format!("claude_call_{}", index + 1));
                let arguments_value = item.get("input").cloned().unwrap_or_else(|| json!({}));
                let arguments =
                    serde_json::to_string(&arguments_value).unwrap_or_else(|_| "{}".to_string());
                tool_calls.push(ApiToolCall {
                    id,
                    r#type: "function".to_string(),
                    function: ApiToolFunction {
                        name: name.to_string(),
                        arguments,
                    },
                });
            }
            _ => {
                if let Some(text) = item.get("text").and_then(|value| value.as_str())
                    && !text.trim().is_empty()
                {
                    text_chunks.push(text.to_string());
                }
            }
        }
    }
    let usage = parse_claude_usage(&parsed);
    Ok(ParsedChatResponse {
        message: AssistantMessage {
            content: optional_json_text(text_chunks.join("\n")),
            reasoning_content: optional_text(reasoning_chunks.join("\n")),
            reasoning: None,
            tool_calls,
        },
        usage,
    })
}

fn parse_gemini_response_text(body: &str) -> Result<ParsedChatResponse, AppError> {
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|err| AppError::Ai(format!("failed to parse AI response: {err}")))?;
    if let Some(error_message) = extract_provider_error_message(&parsed) {
        return Err(AppError::Ai(error_message));
    }
    let candidates = parsed
        .get("candidates")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            AppError::Ai(
                extract_gemini_blocked_reason(&parsed)
                    .unwrap_or_else(|| "AI returned empty candidates".to_string()),
            )
        })?;
    let Some(first_candidate) = candidates.first() else {
        return Err(AppError::Ai(
            extract_gemini_blocked_reason(&parsed)
                .unwrap_or_else(|| "AI returned empty candidates".to_string()),
        ));
    };
    let parts = first_candidate
        .get("content")
        .and_then(|value| value.get("parts"))
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let mut text_chunks = Vec::new();
    let mut tool_calls = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(text) = part.get("text").and_then(|value| value.as_str())
            && !text.trim().is_empty()
        {
            text_chunks.push(text.to_string());
        }
        if let Some(function_call) = part
            .get("functionCall")
            .or_else(|| part.get("function_call"))
        {
            let name = function_call
                .get("name")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let id = function_call
                .get("id")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.to_string())
                .unwrap_or_else(|| format!("gemini_call_{}", index + 1));
            let arguments_value = function_call
                .get("args")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let arguments =
                serde_json::to_string(&arguments_value).unwrap_or_else(|_| "{}".to_string());
            tool_calls.push(ApiToolCall {
                id,
                r#type: "function".to_string(),
                function: ApiToolFunction {
                    name: name.to_string(),
                    arguments,
                },
            });
        }
    }
    let usage = parse_gemini_usage(&parsed);
    Ok(ParsedChatResponse {
        message: AssistantMessage {
            content: optional_json_text(text_chunks.join("\n")),
            reasoning_content: None,
            reasoning: None,
            tool_calls,
        },
        usage,
    })
}

fn parse_claude_usage(value: &serde_json::Value) -> Usage {
    let prompt_tokens = value
        .get("usage")
        .and_then(|usage| usage.get("input_tokens"))
        .map(json_value_to_u64)
        .unwrap_or(0);
    let completion_tokens = value
        .get("usage")
        .and_then(|usage| usage.get("output_tokens"))
        .map(json_value_to_u64)
        .unwrap_or(0);
    let total_tokens = value
        .get("usage")
        .and_then(|usage| usage.get("total_tokens"))
        .map(json_value_to_u64)
        .unwrap_or(prompt_tokens.saturating_add(completion_tokens));
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    }
}

fn parse_gemini_usage(value: &serde_json::Value) -> Usage {
    let prompt_tokens = value
        .get("usageMetadata")
        .and_then(|usage| usage.get("promptTokenCount"))
        .map(json_value_to_u64)
        .unwrap_or(0);
    let completion_tokens = value
        .get("usageMetadata")
        .and_then(|usage| {
            usage
                .get("candidatesTokenCount")
                .or_else(|| usage.get("responseTokenCount"))
        })
        .map(json_value_to_u64)
        .unwrap_or(0);
    let total_tokens = value
        .get("usageMetadata")
        .and_then(|usage| usage.get("totalTokenCount"))
        .map(json_value_to_u64)
        .unwrap_or(prompt_tokens.saturating_add(completion_tokens));
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    }
}

fn extract_gemini_blocked_reason(value: &serde_json::Value) -> Option<String> {
    let prompt_feedback = value.get("promptFeedback")?;
    let block_reason = prompt_feedback
        .get("blockReason")
        .and_then(|item| item.as_str())
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string());
    let safety = prompt_feedback
        .get("safetyRatings")
        .and_then(|item| item.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let category = item
                        .get("category")
                        .and_then(|field| field.as_str())
                        .map(str::trim)
                        .unwrap_or_default();
                    let probability = item
                        .get("probability")
                        .and_then(|field| field.as_str())
                        .map(str::trim)
                        .unwrap_or_default();
                    if category.is_empty() && probability.is_empty() {
                        return None;
                    }
                    if category.is_empty() {
                        return Some(probability.to_string());
                    }
                    if probability.is_empty() {
                        return Some(category.to_string());
                    }
                    Some(format!("{category}:{probability}"))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if block_reason.is_none() && safety.is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    if let Some(reason) = block_reason {
        parts.push(format!("reason={reason}"));
    }
    if !safety.is_empty() {
        parts.push(format!("safety={}", safety.join(",")));
    }
    Some(format!("gemini blocked response ({})", parts.join("; ")))
}

fn json_value_to_u64(value: &serde_json::Value) -> u64 {
    if let Some(integer) = value.as_u64() {
        return integer;
    }
    if let Some(integer) = value.as_i64() {
        return integer.max(0) as u64;
    }
    if let Some(number) = value.as_f64()
        && number.is_finite()
        && number > 0.0
    {
        return number as u64;
    }
    if let Some(text) = value.as_str() {
        return text.trim().parse::<u64>().unwrap_or(0);
    }
    0
}

fn extract_provider_error_message(value: &serde_json::Value) -> Option<String> {
    let error = value.get("error")?;
    if let Some(message) = error.get("message").and_then(|item| item.as_str()) {
        return Some(format!("AI provider error: {}", message.trim()));
    }
    if let Some(message) = error.as_str() {
        return Some(format!("AI provider error: {}", message.trim()));
    }
    Some(format!(
        "AI provider error: {}",
        serde_json::to_string(error).unwrap_or_else(|_| "<unknown error>".to_string())
    ))
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
        AiClient, AiProviderProtocol, ApiMessage, ApiToolCall, ApiToolFunction, AssistantMessage,
        ChatCompletionRequest, ChatCompletionStreamResponse, ChatMetrics, ChatStopReason,
        FinalizeWithoutToolsState, ModelPriceSource, PersistedModelPriceCatalog,
        PersistedModelPriceEntry, REQUEST_TRACE_FILE_VERSION, REQUEST_TRACE_MAX_ENTRIES,
        RequestTraceEntry, SessionRequestTraceFile, StreamToolCallDelta, StreamToolFunctionDelta,
        assistant_reasoning_text, build_claude_request_body, build_provider_chat_url,
        builtin_model_prices, extract_text_delta_from_value, extract_text_from_value,
        finalize_tool_calls, fresh_model_price_catalog, is_cacheable_tool_request,
        is_rate_limited_error, merge_text_delta, merge_tool_call_delta, merge_visible_chunks,
        model_prefers_non_streaming_tool_rounds, normalize_stepfun_chat_request,
        parse_claude_response_text, parse_dsml_tool_calls, parse_gemini_response_text,
        parse_model_price_catalog_response, parse_model_price_probe_response,
        parse_rate_limit_retry_delay, price_probe_candidate_models, provider_protocol_from_config_type,
        provider_requires_reasoning_content_omission, request_contains_reasoning_content,
        resolve_effective_model_prices, should_fallback_to_non_streaming,
        should_refresh_http_client_after_reqwest_error, should_retry_without_reasoning_content,
        should_write_direct_stdout, sanitize_assistant_content, strip_reasoning_content_from_request,
        tool_result_requires_task_decomposition, with_cost,
    };
    use crate::{ai::ToolCallRequest, error::AppError, tls::ensure_rustls_crypto_provider};

    fn test_ai_client() -> AiClient {
        ensure_rustls_crypto_provider();
        AiClient {
            client: RwLock::new(Client::builder().build().expect("test client")),
            provider: AiProviderProtocol::OpenAiCompatible,
            base_url: "https://example.com/chat/completions".to_string(),
            token: "token".to_string(),
            model: "unknown-model".to_string(),
            colorful: true,
            model_price_cache_path: env::temp_dir()
                .join(format!("machineclaw-test-{}.json", uuid::Uuid::new_v4())),
            request_trace_dir: env::temp_dir().join(format!(
                "machineclaw-requests-test-{}",
                uuid::Uuid::new_v4()
            )),
            debug: false,
            max_retries: 0,
            backoff_millis: 0,
            input_price_per_million: 0.0,
            output_price_per_million: 0.0,
            runtime_model_prices: RwLock::new(None),
        }
    }

    #[test]
    fn resolves_provider_protocol_and_provider_urls() {
        assert_eq!(
            provider_protocol_from_config_type("openai"),
            AiProviderProtocol::OpenAiCompatible
        );
        assert_eq!(
            provider_protocol_from_config_type("xiaomi"),
            AiProviderProtocol::OpenAiCompatible
        );
        assert_eq!(
            provider_protocol_from_config_type("mimo"),
            AiProviderProtocol::OpenAiCompatible
        );
        assert_eq!(
            provider_protocol_from_config_type("anthropic"),
            AiProviderProtocol::Claude
        );
        assert_eq!(
            provider_protocol_from_config_type("google"),
            AiProviderProtocol::Gemini
        );
        assert_eq!(
            build_provider_chat_url(
                "https://api.anthropic.com/v1",
                AiProviderProtocol::Claude,
                "claude-sonnet"
            ),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            build_provider_chat_url(
                "https://generativelanguage.googleapis.com/v1beta",
                AiProviderProtocol::Gemini,
                "gemini-2.0-flash"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        );
    }

    #[test]
    fn claude_request_body_converts_tool_choice_and_tool_calls() {
        let request = ChatCompletionRequest {
            model: "claude-3-7-sonnet".to_string(),
            messages: vec![
                ApiMessage {
                    role: "system".to_string(),
                    content: Some("system prompt".to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                ApiMessage {
                    role: "assistant".to_string(),
                    content: Some("I will call a tool".to_string()),
                    reasoning_content: None,
                    tool_calls: Some(vec![ApiToolCall {
                        id: "call_1".to_string(),
                        r#type: "function".to_string(),
                        function: ApiToolFunction {
                            name: "run_shell_command".to_string(),
                            arguments: "{\"command\":\"pwd\"}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
            ],
            temperature: 0.2,
            tools: Some(vec![super::shell_tool_definition()]),
            tool_choice: Some(json!("required")),
            stream: None,
        };
        let payload = build_claude_request_body(&request);
        assert_eq!(
            payload["tool_choice"]["type"].as_str(),
            Some("any"),
            "required should map to claude any mode"
        );
        assert_eq!(
            payload["messages"][0]["content"][1]["type"].as_str(),
            Some("tool_use")
        );
    }

    #[test]
    fn parses_claude_and_gemini_function_calls_into_tool_calls() {
        let claude = parse_claude_response_text(
            r#"{"content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"toolu_1","name":"run_shell_command","input":{"command":"ls"}}],"usage":{"input_tokens":12,"output_tokens":34}}"#,
        )
        .expect("claude parse");
        assert_eq!(claude.message.tool_calls.len(), 1);
        assert_eq!(claude.message.tool_calls[0].id, "toolu_1");
        assert_eq!(claude.usage.total_tokens, 46);

        let gemini = parse_gemini_response_text(
            r#"{"candidates":[{"content":{"parts":[{"text":"ok"},{"functionCall":{"name":"run_shell_command","args":{"command":"ls"}}}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":7,"totalTokenCount":12}}"#,
        )
        .expect("gemini parse");
        assert_eq!(gemini.message.tool_calls.len(), 1);
        assert_eq!(gemini.message.tool_calls[0].id, "gemini_call_2");
        assert_eq!(gemini.usage.total_tokens, 12);
    }

    #[test]
    fn gemini_empty_candidates_reports_block_reason() {
        let result = parse_gemini_response_text(
            r#"{"promptFeedback":{"blockReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"HIGH"}]}}"#,
        );
        assert!(result.is_err());
        let Err(AppError::Ai(message)) = result else {
            panic!("unexpected error variant");
        };
        assert!(message.contains("gemini blocked response"));
        assert!(message.contains("SAFETY"));
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
    fn sanitize_assistant_content_removes_ansi_escape_sequences() {
        let raw = "before \u{1b}[2m</parameter>\u{1b}[0m after";
        let cleaned = sanitize_assistant_content(raw);
        assert_eq!(cleaned, "before </parameter> after");
    }

    #[test]
    fn parse_dsml_tool_calls_tolerates_ansi_wrapped_tags() {
        let raw = "<|dsml|invoke name=\"LS\">\u{1b}[2m<|dsml|parameter name=\"path\">.\u{1b}[0m</|dsml|parameter></|dsml|invoke>";
        let calls = parse_dsml_tool_calls(raw, 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "LS");
        assert!(calls[0].arguments.contains("\"path\":\".\""));
    }

    #[test]
    fn parse_dsml_tool_calls_tolerates_spaced_delimiters_and_smart_quotes() {
        let raw = "< | DSML | function_calls>< | DSML | invoke name=“Task”>< | DSML | parameter name=“task_id” string=“true”>compression< / | DSML | parameter>< / | DSML | invoke>< / | DSML | function_calls>";
        let calls = parse_dsml_tool_calls(raw, 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Task");
        assert!(calls[0].arguments.contains("\"task_id\":\"compression\""));
    }

    #[test]
    fn sanitize_assistant_content_removes_spaced_dsml_function_block() {
        let raw = "prefix< | DSML | function_calls>< | DSML | invoke name=\"Task\">< | DSML | parameter name=\"task_id\">x< / | DSML | parameter>< / | DSML | invoke>< / | DSML | function_calls>suffix";
        let cleaned = sanitize_assistant_content(raw);
        assert_eq!(cleaned, "prefixsuffix");
    }

    #[test]
    fn task_decomposition_guard_payload_is_detected() {
        let payload = json!({
            "ok": false,
            "blocked": true,
            "reason": "task_decomposition_required",
            "error": "task decomposition is required before executing `LS`"
        })
        .to_string();
        assert!(tool_result_requires_task_decomposition(payload.as_str()));
        assert!(!tool_result_requires_task_decomposition(
            r#"{"ok":false,"reason":"repeated_same_tool_call"}"#
        ));
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
    fn excludes_reasoning_blocks_from_visible_text_extraction() {
        let payload = json!([
            {"type":"reasoning","summary":[{"type":"summary_text","text":"hidden chain"}]},
            {"type":"output_text","text":"final answer"}
        ]);
        assert_eq!(extract_text_from_value(&payload), "final answer");
        assert_eq!(extract_text_delta_from_value(&payload), "final answer");
    }

    #[test]
    fn reasoning_extraction_ignores_plain_summary_without_reasoning_type() {
        let message = AssistantMessage {
            content: Some(json!({"summary":"final answer"})),
            reasoning_content: None,
            reasoning: None,
            tool_calls: Vec::new(),
        };
        assert_eq!(assistant_reasoning_text(&message), "");
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
    fn builtin_tool_cacheability_rejects_invalid_json() {
        let request = ToolCallRequest {
            id: "tool_1".to_string(),
            name: "View".to_string(),
            arguments: "{not-json".to_string(),
        };
        assert!(!is_cacheable_tool_request(&request));
    }

    #[test]
    fn builtin_tool_cacheability_respects_apply_flag() {
        let cacheable = ToolCallRequest {
            id: "tool_2".to_string(),
            name: "View".to_string(),
            arguments: "{\"file_path\":\"README.md\"}".to_string(),
        };
        let non_cacheable = ToolCallRequest {
            id: "tool_3".to_string(),
            name: "View".to_string(),
            arguments: "{\"file_path\":\"README.md\",\"apply\":true}".to_string(),
        };
        assert!(is_cacheable_tool_request(&cacheable));
        assert!(!is_cacheable_tool_request(&non_cacheable));
    }

    #[test]
    fn merges_stream_deltas_without_duplicate_suffixes() {
        let mut target = String::new();
        assert_eq!(merge_text_delta(&mut target, "Hello"), "Hello");
        assert_eq!(merge_text_delta(&mut target, "Hello world"), " world");
        assert_eq!(merge_text_delta(&mut target, "!"), "!");
        assert_eq!(target, "Hello world!");
    }

    #[test]
    fn keeps_short_stream_delta_overlaps_without_truncating_code() {
        let mut target = "pub fn ".to_string();
        assert_eq!(
            merge_text_delta(&mut target, "fn merge_sort<T: Ord + Clone>() {"),
            "merge_sort<T: Ord + Clone>() {"
        );
        assert_eq!(target, "pub fn merge_sort<T: Ord + Clone>() {");
    }

    #[test]
    fn preserves_repeated_markdown_markers_and_spaces_in_stream_deltas() {
        let mut target = String::new();
        assert_eq!(merge_text_delta(&mut target, "#"), "#");
        assert_eq!(merge_text_delta(&mut target, "#"), "#");
        assert_eq!(merge_text_delta(&mut target, " "), " ");
        assert_eq!(merge_text_delta(&mut target, "*"), "*");
        assert_eq!(merge_text_delta(&mut target, "*"), "*");
        assert_eq!(target, "## **");
    }

    #[test]
    fn preserves_repeated_quotes_in_stream_json_fragments() {
        let mut target = "{\"command\":\"echo".to_string();
        assert_eq!(merge_text_delta(&mut target, "\""), "\"");
        assert_eq!(merge_text_delta(&mut target, "}"), "}");
        assert_eq!(target, "{\"command\":\"echo\"}");
    }

    #[test]
    fn merges_overlapping_stream_segments_without_duplicate_content() {
        let mut target = "Hello wor".to_string();
        assert_eq!(merge_text_delta(&mut target, "world"), "ld");
        assert_eq!(target, "Hello world");
    }

    #[test]
    fn deduplicates_replayed_snapshot_segment() {
        let mut target = "Hello world".to_string();
        assert_eq!(merge_text_delta(&mut target, "Hello world"), "");
        assert_eq!(target, "Hello world");
    }

    #[test]
    fn stream_tool_call_delta_uses_id_when_index_missing() {
        let mut tool_calls = Vec::new();
        merge_tool_call_delta(
            &mut tool_calls,
            &StreamToolCallDelta {
                index: Some(0),
                id: Some("call_1".to_string()),
                r#type: Some("function".to_string()),
                function: Some(StreamToolFunctionDelta {
                    name: Some("run_shell_command".to_string()),
                    arguments: Some("{\"command\":\"echo".to_string()),
                }),
            },
        );
        merge_tool_call_delta(
            &mut tool_calls,
            &StreamToolCallDelta {
                index: None,
                id: Some("call_1".to_string()),
                r#type: None,
                function: Some(StreamToolFunctionDelta {
                    name: None,
                    arguments: Some(" 1\"}".to_string()),
                }),
            },
        );
        let finalized = finalize_tool_calls(tool_calls);
        assert_eq!(finalized.len(), 1);
        assert_eq!(finalized[0].id, "call_1");
        assert_eq!(finalized[0].function.name, "run_shell_command");
        assert_eq!(finalized[0].function.arguments, "{\"command\":\"echo 1\"}");
    }

    #[test]
    fn stream_tool_call_without_id_receives_stable_generated_id() {
        let finalized = finalize_tool_calls(vec![super::PartialToolCall {
            id: String::new(),
            r#type: "function".to_string(),
            name: "run_shell_command".to_string(),
            arguments: "{\"command\":\"echo hi\"}".to_string(),
        }]);
        assert_eq!(finalized.len(), 1);
        assert_eq!(finalized[0].id, "stream_tool_call_1");
    }

    #[test]
    fn detects_complete_sse_payload_boundaries() {
        assert!(super::streaming_sse_payload_is_complete("[DONE]"));
        assert!(super::streaming_sse_payload_is_complete("{\"choices\":[]}"));
        assert!(!super::streaming_sse_payload_is_complete("{\"choices\":[]"));
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
    fn chat_completion_response_accepts_null_usage_and_tool_calls() {
        let parsed: super::ChatCompletionResponse = serde_json::from_str(
            r#"{"choices":[{"message":{"content":"hello","tool_calls":null}}],"usage":null}"#,
        )
        .expect("null usage and null tool_calls should deserialize");
        assert_eq!(parsed.choices.len(), 1);
        assert!(parsed.choices[0].message.tool_calls.is_empty());
        assert_eq!(parsed.usage.prompt_tokens, 0);
        assert_eq!(parsed.usage.completion_tokens, 0);
        assert_eq!(parsed.usage.total_tokens, 0);
    }

    #[test]
    fn streaming_chunk_accepts_null_tool_calls_and_delta() {
        let parsed: ChatCompletionStreamResponse = serde_json::from_str(
            r#"{"choices":[{"delta":{"content":null,"reasoning_content":null,"tool_calls":null}},{"delta":null}]}"#,
        )
        .expect("null tool_calls and null delta should deserialize");
        assert_eq!(parsed.choices.len(), 2);
        assert!(parsed.choices[0].delta.tool_calls.is_empty());
        assert!(parsed.choices[1].delta.tool_calls.is_empty());
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
    fn deepseek_reasoner_prefers_non_streaming_tool_rounds() {
        assert!(model_prefers_non_streaming_tool_rounds("deepseek-reasoner"));
        assert!(model_prefers_non_streaming_tool_rounds("deepseek-r1"));
        assert!(!model_prefers_non_streaming_tool_rounds("deepseek-chat"));
    }

    #[test]
    fn direct_stdout_print_is_disabled_when_not_terminal_or_raw_mode() {
        assert!(!should_write_direct_stdout(false, Some(false)));
        assert!(!should_write_direct_stdout(true, Some(true)));
    }

    #[test]
    fn direct_stdout_print_is_enabled_for_normal_terminal_mode() {
        assert!(should_write_direct_stdout(true, Some(false)));
        assert!(should_write_direct_stdout(true, None));
    }

    #[test]
    fn debug_trace_file_is_written_per_session_when_debug_enabled() {
        let mut client = test_ai_client();
        client.debug = true;
        client.request_trace_dir = env::temp_dir().join(format!(
            "machineclaw-request-trace-{}",
            uuid::Uuid::new_v4()
        ));
        let session_id = "session-abc-123";
        let entry = RequestTraceEntry {
            timestamp_epoch_ms: 1,
            event: "request".to_string(),
            stream: false,
            attempt: 1,
            status_code: None,
            request_body: Some("{\"hello\":\"world\"}".to_string()),
            response_body: None,
            user_packets: Some(vec!["user packet".to_string()]),
            tool_packets: Some(vec!["tool packet".to_string()]),
            error: None,
        };

        client.append_request_trace_entry(Some(session_id), entry);

        let path = client
            .request_trace_dir
            .join(format!("request-{session_id}.json"));
        assert!(path.exists());
        let raw = fs::read_to_string(path).expect("read request trace");
        let parsed: SessionRequestTraceFile =
            serde_json::from_str(&raw).expect("parse request trace json");
        assert_eq!(parsed.version, REQUEST_TRACE_FILE_VERSION);
        assert_eq!(parsed.session_id, session_id);
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries[0].event, "request");
    }

    #[test]
    fn debug_trace_is_skipped_when_session_id_missing() {
        let mut client = test_ai_client();
        client.debug = true;
        client.request_trace_dir = env::temp_dir().join(format!(
            "machineclaw-request-trace-{}",
            uuid::Uuid::new_v4()
        ));
        client.append_request_trace_entry(
            None,
            RequestTraceEntry {
                timestamp_epoch_ms: 1,
                event: "request".to_string(),
                stream: false,
                attempt: 1,
                status_code: None,
                request_body: Some("{}".to_string()),
                response_body: None,
                user_packets: None,
                tool_packets: None,
                error: None,
            },
        );
        assert!(!client.request_trace_dir.exists());
    }

    #[test]
    fn debug_trace_file_keeps_recent_entries_under_max_limit() {
        let mut client = test_ai_client();
        client.debug = true;
        client.request_trace_dir = env::temp_dir().join(format!(
            "machineclaw-request-trace-{}",
            uuid::Uuid::new_v4()
        ));
        let session_id = "session-cap-test";
        for idx in 0..(REQUEST_TRACE_MAX_ENTRIES + 5) {
            client.append_request_trace_entry(
                Some(session_id),
                RequestTraceEntry {
                    timestamp_epoch_ms: idx as u128,
                    event: "request".to_string(),
                    stream: false,
                    attempt: 1,
                    status_code: None,
                    request_body: Some(format!("{{\"idx\":{idx}}}")),
                    response_body: None,
                    user_packets: None,
                    tool_packets: None,
                    error: None,
                },
            );
        }
        let path = client
            .request_trace_dir
            .join(format!("request-{session_id}.json"));
        let raw = fs::read_to_string(path).expect("read request trace");
        let parsed: SessionRequestTraceFile =
            serde_json::from_str(&raw).expect("parse request trace json");
        assert_eq!(parsed.entries.len(), REQUEST_TRACE_MAX_ENTRIES);
        let first = parsed
            .entries
            .first()
            .and_then(|item| item.request_body.as_deref())
            .unwrap_or_default()
            .to_string();
        assert!(first.contains("\"idx\":5"));
    }

    #[test]
    fn detects_transient_ai_errors() {
        assert!(super::is_transient_ai_error(&AppError::Ai(
            "AI request failed: kind=timeout, error=operation timed out".to_string()
        )));
        assert!(super::is_transient_ai_error(&AppError::Ai(
            "AI HTTP status=503 Service Unavailable".to_string()
        )));
        assert!(super::is_transient_ai_error(&AppError::Ai(
            "failed to read AI response body: error decoding response body".to_string()
        )));
        assert!(super::is_transient_ai_error(&AppError::Ai(
            "AI returned empty content".to_string()
        )));
        assert!(super::is_transient_ai_error(&AppError::Ai(
            "gemini blocked response (reason=SAFETY)".to_string()
        )));
        assert!(!super::is_transient_ai_error(&AppError::Ai(
            "AI HTTP status=400 Bad Request".to_string()
        )));
    }

    #[test]
    fn refreshes_http_client_for_transient_transport_errors() {
        ensure_rustls_crypto_provider();
        let connect_err = Client::builder()
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
        let invalid_url_err = Client::builder()
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
