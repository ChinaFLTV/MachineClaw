use std::{
    collections::BTreeMap,
    fs,
    path::Component,
    path::{Path, PathBuf},
};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::AppError;

const DEFAULT_CMD_TIMEOUT_SECONDS: u64 = 30;
const DEFAULT_CMD_TIMEOUT_KILL_AFTER_SECONDS: u64 = 5;
const DEFAULT_WRITE_CMD_RUN_CONFIRM: bool = true;
const DEFAULT_WRITE_CMD_CONFIRM_MODE: &str = "allow-once";
const DEFAULT_COMMAND_OUTPUT_MAX_BYTES: usize = 262_144;
const DEFAULT_SKILLS_DIR: &str = "~/.skills";
const DEFAULT_SKILLS_ENABLED: bool = false;
const DEFAULT_CONSOLE_COLORFUL: bool = true;
const DEFAULT_AI_MAX_RETRIES: u32 = 2;
const DEFAULT_AI_BACKOFF_MILLIS: u64 = 1500;
const DEFAULT_AI_CONNECTIVITY_CHECK: bool = true;
const DEFAULT_AI_DEBUG: bool = false;
const DEFAULT_AI_TYPE: &str = "openai";
const DEFAULT_AI_INPUT_PRICE_PER_MILLION: f64 = 0.0;
const DEFAULT_AI_OUTPUT_PRICE_PER_MILLION: f64 = 0.0;
const DEFAULT_CHAT_SHOW_TOOL: bool = false;
const DEFAULT_CHAT_SHOW_TOOL_OK: bool = false;
const DEFAULT_CHAT_SHOW_TOOL_ERR: bool = false;
const DEFAULT_CHAT_SHOW_TOOL_TIMEOUT: bool = false;
const DEFAULT_CHAT_SHOW_TIPS: bool = false;
const DEFAULT_CHAT_COMMAND_CACHE_TTL_SECONDS: u64 = 30;
const DEFAULT_CHAT_SHOW_ROUND_METRICS: bool = true;
const DEFAULT_CHAT_SHOW_TOKEN_COST: bool = true;
const DEFAULT_CHAT_CONTEXT_WARN_PERCENT: u8 = 80;
const DEFAULT_CHAT_CONTEXT_CRITICAL_PERCENT: u8 = 95;
const DEFAULT_CHAT_SKIP_MODEL_PRICE_CHECK: bool = false;
const DEFAULT_CHAT_MODEL_PRICE_CHECK_MODE: &str = "sync";
const DEFAULT_CHAT_STREAM_OUTPUT: bool = false;
const DEFAULT_CHAT_OUTPUT_MULTILINES: bool = false;
const DEFAULT_CHAT_SKIP_ENV_PROFILE: bool = true;
const DEFAULT_CHAT_CMD_RUN_TIMOUT_SECONDS: u64 = 30;
const DEFAULT_CHAT_MAX_TOOL_ROUNDS: usize = 16;
const DEFAULT_CHAT_MAX_TOTAL_TOOL_CALLS: usize = 40;
const DEFAULT_CHAT_COMPRESSION_MAX_HISTORY_MESSAGES: usize = 40;
const DEFAULT_CHAT_COMPRESSION_MAX_CHARS_COUNT: usize = 80_000;
const DEFAULT_LOG_DIR: &str = "logs";
const DEFAULT_LOG_FILE_NAME: &str = "session-{session-id}.log";
const DEFAULT_LOG_MAX_FILE_SIZE: &str = "50mb";
const DEFAULT_LOG_MAX_SAVE_TIME: &str = "7d";
const DEFAULT_CONTEXT_RECENT_MESSAGES: usize = 40;
const MAX_CONTEXT_MESSAGES: usize = 80;
const DEFAULT_APP_ENV_MODE: &str = "prod";
const DEFAULT_MCP_AVAILABILITY_CHECK_MODE: &str = "rsync";
const DEFAULT_MCP_DIR: &str = "~/.machineclaw/mcp";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    #[serde(default)]
    pub app: AppSection,
    pub ai: AiConfig,
    #[serde(default)]
    pub console: ConsoleConfig,
    #[serde(default)]
    pub log: LogConfig,
    #[serde(default)]
    pub session: SessionConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppSection {
    #[serde(default)]
    pub language: Option<String>,
    #[serde(rename = "env-mode", default = "default_app_env_mode")]
    pub env_mode: String,
}

impl Default for AppSection {
    fn default() -> Self {
        Self {
            language: None,
            env_mode: default_app_env_mode(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AiConfig {
    #[serde(rename = "type", default = "default_ai_type")]
    pub r#type: String,
    #[serde(rename = "base-url")]
    pub base_url: String,
    pub token: String,
    pub model: String,
    #[serde(default = "default_ai_debug")]
    pub debug: bool,
    #[serde(
        rename = "connectivity-check",
        default = "default_ai_connectivity_check"
    )]
    pub connectivity_check: bool,
    #[serde(default)]
    pub retry: RetryConfig,
    #[serde(default)]
    pub chat: AiChatConfig,
    #[serde(default)]
    pub tools: AiToolsConfig,
    #[serde(
        rename = "input-price-per-million",
        default = "default_ai_input_price_per_million"
    )]
    pub input_price_per_million: f64,
    #[serde(
        rename = "output-price-per-million",
        default = "default_ai_output_price_per_million"
    )]
    pub output_price_per_million: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AiToolsConfig {
    #[serde(default)]
    pub bash: CmdConfig,
    #[serde(default)]
    pub skills: SkillsConfig,
    #[serde(default)]
    pub mcp: McpConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AiChatConfig {
    #[serde(rename = "show-tool", default = "default_chat_show_tool")]
    pub show_tool: bool,
    #[serde(rename = "show-tool-ok", default = "default_chat_show_tool_ok")]
    pub show_tool_ok: bool,
    #[serde(rename = "show-tool-err", default = "default_chat_show_tool_err")]
    pub show_tool_err: bool,
    #[serde(
        rename = "show-tool-timeout",
        default = "default_chat_show_tool_timeout"
    )]
    pub show_tool_timeout: bool,
    #[serde(rename = "show-tips", default = "default_chat_show_tips")]
    pub show_tips: bool,
    #[serde(
        rename = "command-cache-ttl-seconds",
        default = "default_chat_command_cache_ttl_seconds"
    )]
    pub command_cache_ttl_seconds: u64,
    #[serde(
        rename = "show-round-metrics",
        default = "default_chat_show_round_metrics"
    )]
    pub show_round_metrics: bool,
    #[serde(rename = "show-token-cost", default = "default_chat_show_token_cost")]
    pub show_token_cost: bool,
    #[serde(
        rename = "skip-model-price-check",
        default = "default_chat_skip_model_price_check"
    )]
    pub skip_model_price_check: bool,
    #[serde(
        rename = "model-price-check-mode",
        default = "default_chat_model_price_check_mode"
    )]
    pub model_price_check_mode: String,
    #[serde(
        rename = "context-warn-percent",
        default = "default_chat_context_warn_percent"
    )]
    pub context_warn_percent: u8,
    #[serde(
        rename = "context-critical-percent",
        default = "default_chat_context_critical_percent"
    )]
    pub context_critical_percent: u8,
    #[serde(rename = "stream-output", default = "default_chat_stream_output")]
    pub stream_output: bool,
    #[serde(
        rename = "output-multilines",
        default = "default_chat_output_multilines"
    )]
    pub output_multilines: bool,
    #[serde(rename = "skip-env-profile", default = "default_chat_skip_env_profile")]
    pub skip_env_profile: bool,
    #[serde(
        rename = "cmd-run-timout",
        alias = "cmd-run-timeout",
        default = "default_chat_cmd_run_timout_seconds"
    )]
    pub cmd_run_timout: u64,
    #[serde(rename = "max-tool-rounds", default = "default_chat_max_tool_rounds")]
    pub max_tool_rounds: usize,
    #[serde(
        rename = "max-total-tool-calls",
        default = "default_chat_max_total_tool_calls"
    )]
    pub max_total_tool_calls: usize,
    #[serde(default)]
    pub compression: AiChatCompressionConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AiChatCompressionConfig {
    #[serde(
        rename = "max-history-messages",
        default = "default_chat_compression_max_history_messages"
    )]
    pub max_history_messages: usize,
    #[serde(
        rename = "max-chars-count",
        default = "default_chat_compression_max_chars_count"
    )]
    pub max_chars_count: usize,
}

impl Default for AiChatCompressionConfig {
    fn default() -> Self {
        Self {
            max_history_messages: default_chat_compression_max_history_messages(),
            max_chars_count: default_chat_compression_max_chars_count(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RetryConfig {
    #[serde(rename = "max-retries", default = "default_ai_max_retries")]
    pub max_retries: u32,
    #[serde(rename = "backoff-millis", default = "default_ai_backoff_millis")]
    pub backoff_millis: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CmdConfig {
    #[serde(
        rename = "write-cmd-run-confirm",
        default = "default_write_cmd_run_confirm"
    )]
    pub write_cmd_run_confirm: bool,
    #[serde(
        rename = "command-timeout-seconds",
        default = "default_cmd_timeout_seconds"
    )]
    pub command_timeout_seconds: u64,
    #[serde(
        rename = "command-timeout-kill-after-seconds",
        default = "default_cmd_timeout_kill_after_seconds"
    )]
    pub command_timeout_kill_after_seconds: u64,
    #[serde(
        rename = "write-cmd-confirm-mode",
        default = "default_write_cmd_confirm_mode"
    )]
    pub write_cmd_confirm_mode: String,
    #[serde(rename = "allow-cmd-list", default)]
    pub allow_cmd_list: Vec<String>,
    #[serde(rename = "deny-cmd-list", default)]
    pub deny_cmd_list: Vec<String>,
    #[serde(rename = "write-cmd-allow-patterns", default)]
    pub write_cmd_allow_patterns: Vec<String>,
    #[serde(rename = "write-cmd-deny-patterns", default)]
    pub write_cmd_deny_patterns: Vec<String>,
    #[serde(
        rename = "command-output-max-bytes",
        default = "default_command_output_max_bytes"
    )]
    pub command_output_max_bytes: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SkillsConfig {
    #[serde(default = "default_skills_enabled")]
    pub enabled: bool,
    #[serde(default = "default_skills_dir")]
    pub dir: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpConfig {
    #[serde(default = "default_mcp_enabled")]
    pub enabled: bool,
    #[serde(
        rename = "mcp-availability-check-mode",
        default = "default_mcp_availability_check_mode"
    )]
    pub mcp_availability_check_mode: String,
    #[serde(default = "default_mcp_dir")]
    pub dir: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct McpServerConfig {
    #[serde(default = "default_mcp_server_enabled")]
    pub enabled: bool,
    #[serde(
        default,
        rename = "type",
        alias = "transport",
        skip_serializing_if = "Option::is_none"
    )]
    pub transport: Option<String>,
    #[serde(
        rename = "url",
        alias = "server-url",
        alias = "server_url",
        alias = "serverUrl",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub server_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, alias = "cmd", skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(
        rename = "authType",
        alias = "auth-type",
        alias = "auth_type",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub auth_type: Option<String>,
    #[serde(
        rename = "authToken",
        alias = "auth-token",
        alias = "auth_token",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub auth_token: Option<String>,
    #[serde(default)]
    #[serde(
        rename = "timeoutSeconds",
        alias = "timeout-seconds",
        alias = "timeout_seconds",
        alias = "timeoutSeconds",
        skip_serializing_if = "Option::is_none"
    )]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConsoleConfig {
    #[serde(default = "default_console_colorful")]
    pub colorful: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogConfig {
    #[serde(default = "default_log_dir")]
    pub dir: String,
    #[serde(rename = "log-file-name", default = "default_log_file_name")]
    pub log_file_name: String,
    #[serde(rename = "max-file-size", default = "default_log_max_file_size")]
    pub max_file_size: String,
    #[serde(rename = "max-save-time", default = "default_log_max_save_time")]
    pub max_save_time: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionConfig {
    #[serde(default = "default_context_recent_messages")]
    pub recent_messages: usize,
    #[serde(default = "default_context_max_messages")]
    pub max_messages: usize,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_ai_max_retries(),
            backoff_millis: default_ai_backoff_millis(),
        }
    }
}

impl Default for AiChatConfig {
    fn default() -> Self {
        Self {
            show_tool: default_chat_show_tool(),
            show_tool_ok: default_chat_show_tool_ok(),
            show_tool_err: default_chat_show_tool_err(),
            show_tool_timeout: default_chat_show_tool_timeout(),
            show_tips: default_chat_show_tips(),
            command_cache_ttl_seconds: default_chat_command_cache_ttl_seconds(),
            show_round_metrics: default_chat_show_round_metrics(),
            show_token_cost: default_chat_show_token_cost(),
            skip_model_price_check: default_chat_skip_model_price_check(),
            model_price_check_mode: default_chat_model_price_check_mode(),
            context_warn_percent: default_chat_context_warn_percent(),
            context_critical_percent: default_chat_context_critical_percent(),
            stream_output: default_chat_stream_output(),
            output_multilines: default_chat_output_multilines(),
            skip_env_profile: default_chat_skip_env_profile(),
            cmd_run_timout: default_chat_cmd_run_timout_seconds(),
            max_tool_rounds: default_chat_max_tool_rounds(),
            max_total_tool_calls: default_chat_max_total_tool_calls(),
            compression: AiChatCompressionConfig::default(),
        }
    }
}

impl Default for CmdConfig {
    fn default() -> Self {
        Self {
            write_cmd_run_confirm: default_write_cmd_run_confirm(),
            command_timeout_seconds: default_cmd_timeout_seconds(),
            command_timeout_kill_after_seconds: default_cmd_timeout_kill_after_seconds(),
            write_cmd_confirm_mode: default_write_cmd_confirm_mode(),
            allow_cmd_list: Vec::new(),
            deny_cmd_list: Vec::new(),
            write_cmd_allow_patterns: Vec::new(),
            write_cmd_deny_patterns: Vec::new(),
            command_output_max_bytes: default_command_output_max_bytes(),
        }
    }
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            enabled: default_skills_enabled(),
            dir: default_skills_dir(),
        }
    }
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: default_mcp_enabled(),
            mcp_availability_check_mode: default_mcp_availability_check_mode(),
            dir: default_mcp_dir(),
        }
    }
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        Self {
            colorful: default_console_colorful(),
        }
    }
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            dir: default_log_dir(),
            log_file_name: default_log_file_name(),
            max_file_size: default_log_max_file_size(),
            max_save_time: default_log_max_save_time(),
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            recent_messages: default_context_recent_messages(),
            max_messages: default_context_max_messages(),
        }
    }
}

pub fn load_config(path: &Path) -> Result<AppConfig, AppError> {
    let raw = fs::read_to_string(path).map_err(|err| {
        AppError::Config(format!("failed to read config {}: {err}", path.display()))
    })?;
    let cfg = parse_config_text(&raw, &path.display().to_string())?;
    validate_config(&cfg)?;
    Ok(cfg)
}

pub fn parse_config_text(raw: &str, source: &str) -> Result<AppConfig, AppError> {
    let parsed = toml::from_str::<toml::Value>(raw)
        .map_err(|err| AppError::Config(format!("failed to parse config {source}: {err}")))?;
    reject_legacy_root_tables(&parsed)?;
    AppConfig::deserialize(parsed)
        .map_err(|err| AppError::Config(format!("failed to parse config {source}: {err}")))
}

pub fn read_language_hint(path: &Path) -> Option<String> {
    if !path.exists() {
        return None;
    }
    let raw = fs::read_to_string(path).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let parsed = toml::from_str::<toml::Value>(&raw).ok()?;
    parsed
        .get("app")
        .and_then(|item| item.get("language"))
        .and_then(|item| item.as_str())
        .map(|value| value.to_string())
}

pub fn read_console_colorful_hint(path: &Path) -> Option<bool> {
    if !path.exists() {
        return None;
    }
    let raw = fs::read_to_string(path).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let parsed = toml::from_str::<toml::Value>(&raw).ok()?;
    parsed
        .get("console")
        .and_then(|item| item.get("colorful"))
        .and_then(|item| item.as_bool())
}

pub fn config_template_example() -> &'static str {
    r#"# MachineClaw Config Template

## Usage
- Copy this template into `claw.toml` (default), or keep it as `claw-sample.toml` and pass `--conf`.
- Keep required keys complete: `ai.base-url`, `ai.token`, `ai.model`.
- Remove comments if you want a cleaner production config.

## Full Example
```toml
[app]
# language = "zh-CN" # optional: zh-CN, zh-TW, en, fr, de, ja
env-mode = "prod" # optional: prod, test, dev

[ai]
type = "openai" # optional, default openai; accepted: openai, claude(anthropic), gemini(google), deepseek, qwen, ollama, openrouter, zhipu, moonshot, doubao, stepfun, siliconflow, groq, together, mistral, azure-openai
base-url = "https://api.deepseek.com/v1" # required
token = "sk-xxxx" # required
model = "deepseek-chat" # required
debug = false # optional, default false; when true prints masked AI request/response debug details to terminal
connectivity-check = true # optional, default true (chat startup)
input-price-per-million = 0 # optional, 0 means use built-in known model pricing when available, otherwise cost shows N/A
output-price-per-million = 0 # optional, 0 means use built-in known model pricing when available, otherwise cost shows N/A

[ai.retry]
max-retries = 2 # optional, default 2
backoff-millis = 1500 # optional, default 1500

[ai.chat]
show-tool = false # optional
show-tool-ok = false # optional
show-tool-err = false # optional
show-tool-timeout = false # optional
show-tips = false # optional
command-cache-ttl-seconds = 30 # optional
show-round-metrics = true # optional
show-token-cost = true # optional
skip-model-price-check = false # optional, default false; when true skip online model pricing probe and use configured/builtin pricing only
model-price-check-mode = "sync" # optional, default "sync"; "async" probes model pricing in background when skip-model-price-check=false
context-warn-percent = 80 # optional
context-critical-percent = 95 # optional
stream-output = false # optional, default false
output-multilines = false # optional, default false
skip-env-profile = true # optional, default true
cmd-run-timout = 30 # optional, default 30 seconds
max-tool-rounds = 16 # optional, default 16
max-total-tool-calls = 40 # optional, default 40

[ai.chat.compression]
max-history-messages = 40 # optional, default 40
max-chars-count = 80000 # optional, default 80000

[ai.tools]

[ai.tools.bash]
write-cmd-run-confirm = true # optional, default true
write-cmd-confirm-mode = "allow-once" # optional: deny, edit, allow-once, allow-session
allow-cmd-list = [] # optional, regex list; non-empty means allow-list mode
deny-cmd-list = [] # optional, regex list; higher priority than allow-cmd-list
write-cmd-allow-patterns = [] # optional
write-cmd-deny-patterns = [] # optional
command-timeout-seconds = 30 # optional, default 30
command-timeout-kill-after-seconds = 5 # optional, default 5
command-output-max-bytes = 262144 # optional, default 262144

[ai.tools.skills]
enabled = false # optional, default false
dir = "~/.skills" # optional

[ai.tools.mcp]
enabled = false # optional
mcp-availability-check-mode = "rsync" # optional, default "rsync"; "async" runs MCP availability check in background so chat can start sooner
dir = "~/.machineclaw/mcp" # optional, MCP JSON directory (or a single json file path)
# MCP servers are no longer configured in this TOML.
# Place MCP definitions in `${ai.tools.mcp.dir}/servers.json`, for example:
# {
#   "mcpServers": {
#     "amap-maps": {
#       "enabled": true,
#       "type": "streamable_http",
#       "url": "https://example.com/mcp",
#       "headers": { "Authorization": "Bearer <token>" }
#     }
#   }
# }

[console]
colorful = true # optional, default true

[log]
dir = "logs" # optional, default executable_dir/logs
log-file-name = "session-{session-id}.log" # optional, supports strftime and %N
max-file-size = "50mb" # optional, units: b, kb, mb, gb, tb
max-save-time = "7d" # optional, units: s, m, h, d, M, y

[session]
recent_messages = 40 # optional, default 40
max_messages = 80 # optional, default 80
```

## Notes
- `config set` only updates one key and preserves other sections.
- In chat mode, risky write commands still depend on command confirmation policy.
"#
}

pub fn validate_config(cfg: &AppConfig) -> Result<(), AppError> {
    let env_mode = cfg.app.env_mode.trim().to_ascii_lowercase();
    if !matches!(env_mode.as_str(), "prod" | "test" | "dev") {
        return Err(AppError::Config(
            "app.env-mode must be one of: prod, test, dev".to_string(),
        ));
    }
    if cfg.ai.base_url.trim().is_empty() {
        return Err(AppError::Config("ai.base-url is required".to_string()));
    }
    if cfg.ai.token.trim().is_empty() {
        return Err(AppError::Config("ai.token is required".to_string()));
    }
    if cfg.ai.model.trim().is_empty() {
        return Err(AppError::Config("ai.model is required".to_string()));
    }
    let ai_type = normalize_ai_provider_type(cfg.ai.r#type.as_str());
    if !matches!(ai_type, "openai" | "claude" | "gemini") {
        return Err(AppError::Config(
            "ai.type must be one of: openai, claude, gemini, anthropic, google, deepseek, qwen, ollama, openrouter, zhipu, moonshot, doubao, stepfun, siliconflow, groq, together, mistral, azure-openai"
                .to_string(),
        ));
    }
    let bash_cfg = &cfg.ai.tools.bash;
    if bash_cfg.command_timeout_seconds == 0 {
        return Err(AppError::Config(
            "ai.tools.bash.command-timeout-seconds must be greater than 0".to_string(),
        ));
    }
    if bash_cfg.command_timeout_kill_after_seconds == 0 {
        return Err(AppError::Config(
            "ai.tools.bash.command-timeout-kill-after-seconds must be greater than 0".to_string(),
        ));
    }
    if bash_cfg.command_output_max_bytes < 1024 {
        return Err(AppError::Config(
            "ai.tools.bash.command-output-max-bytes must be >= 1024".to_string(),
        ));
    }
    for item in &bash_cfg.allow_cmd_list {
        let pattern = item.trim();
        if pattern.is_empty() {
            continue;
        }
        Regex::new(pattern).map_err(|err| {
            AppError::Config(format!(
                "ai.tools.bash.allow-cmd-list has invalid regex '{pattern}': {err}"
            ))
        })?;
    }
    for item in &bash_cfg.deny_cmd_list {
        let pattern = item.trim();
        if pattern.is_empty() {
            continue;
        }
        Regex::new(pattern).map_err(|err| {
            AppError::Config(format!(
                "ai.tools.bash.deny-cmd-list has invalid regex '{pattern}': {err}"
            ))
        })?;
    }
    let confirm_mode = bash_cfg.write_cmd_confirm_mode.trim().to_ascii_lowercase();
    if !matches!(
        confirm_mode.as_str(),
        "deny" | "edit" | "allow-once" | "allow-session"
    ) {
        return Err(AppError::Config(
            "ai.tools.bash.write-cmd-confirm-mode must be one of: deny, edit, allow-once, allow-session".to_string(),
        ));
    }
    if cfg.ai.input_price_per_million < 0.0 || cfg.ai.output_price_per_million < 0.0 {
        return Err(AppError::Config(
            "ai input/output price cannot be negative".to_string(),
        ));
    }
    if cfg.ai.chat.context_warn_percent == 0
        || cfg.ai.chat.context_warn_percent > 100
        || cfg.ai.chat.context_critical_percent == 0
        || cfg.ai.chat.context_critical_percent > 100
    {
        return Err(AppError::Config(
            "ai.chat context percent must be in 1..=100".to_string(),
        ));
    }
    if cfg.ai.chat.context_warn_percent > cfg.ai.chat.context_critical_percent {
        return Err(AppError::Config(
            "ai.chat context-warn-percent cannot exceed context-critical-percent".to_string(),
        ));
    }
    let model_price_check_mode =
        normalize_chat_model_price_check_mode(cfg.ai.chat.model_price_check_mode.as_str());
    if !matches!(model_price_check_mode, "sync" | "async") {
        return Err(AppError::Config(
            "ai.chat.model-price-check-mode must be one of: sync, async".to_string(),
        ));
    }
    let mcp_cfg = &cfg.ai.tools.mcp;
    let mcp_availability_check_mode =
        normalize_mcp_availability_check_mode(mcp_cfg.mcp_availability_check_mode.as_str());
    if !matches!(mcp_availability_check_mode, "rsync" | "async") {
        return Err(AppError::Config(
            "ai.tools.mcp.mcp-availability-check-mode must be one of: rsync, async".to_string(),
        ));
    }
    if mcp_cfg.dir.trim().is_empty() {
        return Err(AppError::Config(
            "ai.tools.mcp.dir must not be empty".to_string(),
        ));
    }
    if cfg.ai.chat.cmd_run_timout == 0 {
        return Err(AppError::Config(
            "ai.chat.cmd-run-timout must be greater than 0".to_string(),
        ));
    }
    if cfg.ai.chat.max_tool_rounds == 0 {
        return Err(AppError::Config(
            "ai.chat.max-tool-rounds must be greater than 0".to_string(),
        ));
    }
    if cfg.ai.chat.max_total_tool_calls == 0 {
        return Err(AppError::Config(
            "ai.chat.max-total-tool-calls must be greater than 0".to_string(),
        ));
    }
    if cfg.ai.chat.compression.max_history_messages == 0 {
        return Err(AppError::Config(
            "ai.chat.compression.max-history-messages must be greater than 0".to_string(),
        ));
    }
    if cfg.ai.chat.compression.max_chars_count == 0 {
        return Err(AppError::Config(
            "ai.chat.compression.max-chars-count must be greater than 0".to_string(),
        ));
    }
    if cfg.session.recent_messages == 0 {
        return Err(AppError::Config(
            "session.recent_messages must be greater than 0".to_string(),
        ));
    }
    if cfg.session.max_messages == 0 {
        return Err(AppError::Config(
            "session.max_messages must be greater than 0".to_string(),
        ));
    }
    if cfg.session.max_messages > MAX_CONTEXT_MESSAGES {
        return Err(AppError::Config(format!(
            "session.max_messages must be <= {MAX_CONTEXT_MESSAGES}"
        )));
    }
    if cfg.session.recent_messages > cfg.session.max_messages {
        return Err(AppError::Config(
            "session.recent_messages cannot exceed session.max_messages".to_string(),
        ));
    }
    if cfg.log.log_file_name.trim().is_empty() || !has_file_extension(&cfg.log.log_file_name) {
        return Err(AppError::Config(
            "log.log-file-name must include a file extension".to_string(),
        ));
    }
    if cfg.log.log_file_name.contains('/') || cfg.log.log_file_name.contains('\\') {
        return Err(AppError::Config(
            "log.log-file-name must not contain path separators".to_string(),
        ));
    }
    if cfg.log.max_file_size.trim().is_empty() {
        return Err(AppError::Config(
            "log.max-file-size must not be empty".to_string(),
        ));
    }
    if cfg.log.max_save_time.trim().is_empty() {
        return Err(AppError::Config(
            "log.max-save-time must not be empty".to_string(),
        ));
    }
    Ok(())
}

pub fn resolve_config_path(conf: Option<PathBuf>) -> Result<PathBuf, AppError> {
    if let Some(path) = conf {
        return resolve_path(path);
    }
    let exe = std::env::current_exe()
        .map_err(|err| AppError::Runtime(format!("cannot locate current executable: {err}")))?;
    let exe_dir = exe
        .parent()
        .ok_or_else(|| AppError::Runtime("cannot locate executable directory".to_string()))?;
    Ok(exe_dir.join("claw.toml"))
}

pub fn resolve_path(path: PathBuf) -> Result<PathBuf, AppError> {
    let expanded = expand_tilde_path(path);
    let resolved = if expanded.is_absolute() {
        normalize_path(expanded)
    } else {
        let cwd = std::env::current_dir().map_err(|err| {
            AppError::Runtime(format!("failed to resolve current directory: {err}"))
        })?;
        normalize_path(cwd.join(expanded))
    };
    if resolved.exists() {
        return Ok(resolved);
    }

    if let Some(alias) = resolve_docs_alias(&resolved) {
        return Ok(alias);
    }

    Ok(resolved)
}

fn expand_tilde_path(path: PathBuf) -> PathBuf {
    let raw = path.to_string_lossy();
    expand_tilde(&raw)
}

fn normalize_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn resolve_docs_alias(path: &Path) -> Option<PathBuf> {
    let mut alias = PathBuf::new();
    let mut replaced = false;
    for component in path.components() {
        match component {
            Component::Normal(name) if !replaced && name == "docs" => {
                alias.push(".docs");
                replaced = true;
            }
            _ => alias.push(component.as_os_str()),
        }
    }
    if replaced && alias.exists() {
        return Some(alias);
    }
    None
}

fn reject_legacy_root_tables(parsed: &toml::Value) -> Result<(), AppError> {
    let Some(table) = parsed.as_table() else {
        return Ok(());
    };
    if table.contains_key("cmd") {
        return Err(AppError::Config(
            "legacy [cmd] is not supported; use [ai.tools.bash]".to_string(),
        ));
    }
    if table.contains_key("skills") {
        return Err(AppError::Config(
            "legacy [skills] is not supported; use [ai.tools.skills]".to_string(),
        ));
    }
    if table.contains_key("mcp") {
        return Err(AppError::Config(
            "legacy [mcp] is not supported; use [ai.tools.mcp]".to_string(),
        ));
    }
    Ok(())
}

pub fn expand_tilde(raw: &str) -> PathBuf {
    if raw == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from(raw));
    }
    if let Some(stripped) = raw.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(stripped);
    }
    PathBuf::from(raw)
}

fn default_cmd_timeout_seconds() -> u64 {
    DEFAULT_CMD_TIMEOUT_SECONDS
}

fn default_cmd_timeout_kill_after_seconds() -> u64 {
    DEFAULT_CMD_TIMEOUT_KILL_AFTER_SECONDS
}

fn default_write_cmd_run_confirm() -> bool {
    DEFAULT_WRITE_CMD_RUN_CONFIRM
}

fn default_write_cmd_confirm_mode() -> String {
    DEFAULT_WRITE_CMD_CONFIRM_MODE.to_string()
}

fn default_command_output_max_bytes() -> usize {
    DEFAULT_COMMAND_OUTPUT_MAX_BYTES
}

fn default_skills_dir() -> String {
    DEFAULT_SKILLS_DIR.to_string()
}

fn default_skills_enabled() -> bool {
    DEFAULT_SKILLS_ENABLED
}

fn default_mcp_enabled() -> bool {
    false
}

fn default_mcp_availability_check_mode() -> String {
    DEFAULT_MCP_AVAILABILITY_CHECK_MODE.to_string()
}

fn default_mcp_dir() -> String {
    DEFAULT_MCP_DIR.to_string()
}

fn default_mcp_server_enabled() -> bool {
    true
}

fn default_console_colorful() -> bool {
    DEFAULT_CONSOLE_COLORFUL
}

fn default_log_dir() -> String {
    DEFAULT_LOG_DIR.to_string()
}

fn default_log_file_name() -> String {
    DEFAULT_LOG_FILE_NAME.to_string()
}

fn default_log_max_file_size() -> String {
    DEFAULT_LOG_MAX_FILE_SIZE.to_string()
}

fn default_log_max_save_time() -> String {
    DEFAULT_LOG_MAX_SAVE_TIME.to_string()
}

fn default_ai_max_retries() -> u32 {
    DEFAULT_AI_MAX_RETRIES
}

fn default_ai_backoff_millis() -> u64 {
    DEFAULT_AI_BACKOFF_MILLIS
}

fn default_ai_connectivity_check() -> bool {
    DEFAULT_AI_CONNECTIVITY_CHECK
}

fn default_ai_debug() -> bool {
    DEFAULT_AI_DEBUG
}

fn default_ai_type() -> String {
    DEFAULT_AI_TYPE.to_string()
}

fn default_ai_input_price_per_million() -> f64 {
    DEFAULT_AI_INPUT_PRICE_PER_MILLION
}

fn default_ai_output_price_per_million() -> f64 {
    DEFAULT_AI_OUTPUT_PRICE_PER_MILLION
}

fn default_chat_show_tool() -> bool {
    DEFAULT_CHAT_SHOW_TOOL
}

fn default_chat_show_tool_ok() -> bool {
    DEFAULT_CHAT_SHOW_TOOL_OK
}

fn default_chat_show_tool_err() -> bool {
    DEFAULT_CHAT_SHOW_TOOL_ERR
}

fn default_chat_show_tool_timeout() -> bool {
    DEFAULT_CHAT_SHOW_TOOL_TIMEOUT
}

fn default_chat_show_tips() -> bool {
    DEFAULT_CHAT_SHOW_TIPS
}

fn default_chat_command_cache_ttl_seconds() -> u64 {
    DEFAULT_CHAT_COMMAND_CACHE_TTL_SECONDS
}

fn default_chat_show_round_metrics() -> bool {
    DEFAULT_CHAT_SHOW_ROUND_METRICS
}

fn default_chat_show_token_cost() -> bool {
    DEFAULT_CHAT_SHOW_TOKEN_COST
}

fn default_chat_skip_model_price_check() -> bool {
    DEFAULT_CHAT_SKIP_MODEL_PRICE_CHECK
}

fn default_chat_model_price_check_mode() -> String {
    DEFAULT_CHAT_MODEL_PRICE_CHECK_MODE.to_string()
}

fn default_chat_context_warn_percent() -> u8 {
    DEFAULT_CHAT_CONTEXT_WARN_PERCENT
}

fn default_chat_context_critical_percent() -> u8 {
    DEFAULT_CHAT_CONTEXT_CRITICAL_PERCENT
}

fn default_chat_stream_output() -> bool {
    DEFAULT_CHAT_STREAM_OUTPUT
}

fn default_chat_output_multilines() -> bool {
    DEFAULT_CHAT_OUTPUT_MULTILINES
}

fn default_chat_skip_env_profile() -> bool {
    DEFAULT_CHAT_SKIP_ENV_PROFILE
}

fn default_chat_cmd_run_timout_seconds() -> u64 {
    DEFAULT_CHAT_CMD_RUN_TIMOUT_SECONDS
}

fn default_chat_max_tool_rounds() -> usize {
    DEFAULT_CHAT_MAX_TOOL_ROUNDS
}

fn default_chat_max_total_tool_calls() -> usize {
    DEFAULT_CHAT_MAX_TOTAL_TOOL_CALLS
}

fn default_chat_compression_max_history_messages() -> usize {
    DEFAULT_CHAT_COMPRESSION_MAX_HISTORY_MESSAGES
}

fn default_chat_compression_max_chars_count() -> usize {
    DEFAULT_CHAT_COMPRESSION_MAX_CHARS_COUNT
}

fn default_context_recent_messages() -> usize {
    DEFAULT_CONTEXT_RECENT_MESSAGES
}

fn default_context_max_messages() -> usize {
    MAX_CONTEXT_MESSAGES
}

fn default_app_env_mode() -> String {
    DEFAULT_APP_ENV_MODE.to_string()
}

pub(crate) fn normalize_ai_provider_type(raw: &str) -> &str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "openai" | "deepseek" | "qwen" | "ollama" | "openrouter" | "zhipu" | "moonshot"
        | "doubao" | "stepfun" | "siliconflow" | "groq" | "together" | "mistral"
        | "azure-openai" | "azure" => "openai",
        "claude" | "anthropic" => "claude",
        "gemini" | "google" => "gemini",
        _ => "__invalid__",
    }
}

pub(crate) fn normalize_chat_model_price_check_mode(raw: &str) -> &str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "async" => "async",
        "sync" | "rsync" | "" => "sync",
        _ => "__invalid__",
    }
}

pub(crate) fn normalize_mcp_availability_check_mode(raw: &str) -> &str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "async" => "async",
        "rsync" | "sync" | "" => "rsync",
        _ => "__invalid__",
    }
}

fn has_file_extension(file_name: &str) -> bool {
    let trimmed = file_name.trim();
    if trimmed.is_empty() || trimmed.ends_with('.') {
        return false;
    }
    if let Some(idx) = trimmed.rfind('.') {
        return idx > 0 && idx < trimmed.len() - 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{
        McpServerConfig, normalize_ai_provider_type, normalize_chat_model_price_check_mode,
        normalize_mcp_availability_check_mode, parse_config_text, resolve_config_path,
    };

    #[test]
    fn normalize_ai_provider_type_supports_aliases() {
        assert_eq!(normalize_ai_provider_type("openai"), "openai");
        assert_eq!(normalize_ai_provider_type("deepseek"), "openai");
        assert_eq!(normalize_ai_provider_type("claude"), "claude");
        assert_eq!(normalize_ai_provider_type("anthropic"), "claude");
        assert_eq!(normalize_ai_provider_type("gemini"), "gemini");
        assert_eq!(normalize_ai_provider_type("google"), "gemini");
        assert_eq!(normalize_ai_provider_type("invalid"), "__invalid__");
    }

    #[test]
    fn normalize_model_price_check_mode_accepts_sync_async_and_rsync_alias() {
        assert_eq!(normalize_chat_model_price_check_mode("sync"), "sync");
        assert_eq!(normalize_chat_model_price_check_mode("async"), "async");
        assert_eq!(normalize_chat_model_price_check_mode("rsync"), "sync");
        assert_eq!(
            normalize_chat_model_price_check_mode("invalid"),
            "__invalid__"
        );
    }

    #[test]
    fn normalize_mcp_availability_check_mode_accepts_rsync_async_and_sync_alias() {
        assert_eq!(normalize_mcp_availability_check_mode("rsync"), "rsync");
        assert_eq!(normalize_mcp_availability_check_mode("async"), "async");
        assert_eq!(normalize_mcp_availability_check_mode("sync"), "rsync");
        assert_eq!(normalize_mcp_availability_check_mode(""), "rsync");
        assert_eq!(
            normalize_mcp_availability_check_mode("invalid"),
            "__invalid__"
        );
    }

    #[test]
    fn resolve_default_config_file_name_is_claw_toml() {
        let path = resolve_config_path(None).expect("default config path should resolve");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("claw.toml")
        );
    }

    #[test]
    fn mcp_server_config_supports_smithery_style_type_and_url_alias() {
        let server: McpServerConfig = toml::from_str(
            r#"
enabled = true
type = "sse"
url = "https://example.com/sse"
"#,
        )
        .expect("toml should parse");
        assert_eq!(server.transport.as_deref(), Some("sse"));
        assert_eq!(
            server.server_url.as_deref(),
            Some("https://example.com/sse")
        );
    }

    #[test]
    fn parse_config_text_reads_ai_tools_bash_section() {
        let cfg = parse_config_text(
            r#"
[ai]
base-url = "https://example.com/v1"
token = "sk-test"
model = "test-model"

[ai.tools.bash]
command-timeout-seconds = 12
"#,
            "inline",
        )
        .expect("config should parse");
        assert_eq!(cfg.ai.tools.bash.command_timeout_seconds, 12);
    }

    #[test]
    fn parse_config_text_rejects_legacy_skills_root_table() {
        let err = parse_config_text(
            r#"
[ai]
base-url = "https://example.com/v1"
token = "sk-test"
model = "test-model"

[skills]
enabled = true
"#,
            "inline",
        )
        .expect_err("legacy [skills] must be rejected");
        assert!(err.to_string().contains("legacy [skills] is not supported"));
    }

    #[test]
    fn parse_config_text_rejects_legacy_mcp_root_table() {
        let err = parse_config_text(
            r#"
[ai]
base-url = "https://example.com/v1"
token = "sk-test"
model = "test-model"

[mcp]
enabled = true
"#,
            "inline",
        )
        .expect_err("legacy [mcp] must be rejected");
        assert!(err.to_string().contains("legacy [mcp] is not supported"));
    }
}
