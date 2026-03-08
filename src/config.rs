use std::{
    collections::BTreeMap,
    fs,
    path::Component,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::error::AppError;

const DEFAULT_CMD_TIMEOUT_SECONDS: u64 = 30;
const DEFAULT_CMD_TIMEOUT_KILL_AFTER_SECONDS: u64 = 5;
const DEFAULT_WRITE_CMD_RUN_CONFIRM: bool = true;
const DEFAULT_WRITE_CMD_CONFIRM_MODE: &str = "allow-once";
const DEFAULT_COMMAND_OUTPUT_MAX_BYTES: usize = 262_144;
const DEFAULT_SKILLS_DIR: &str = "~/.skills";
const DEFAULT_CONSOLE_COLORFUL: bool = true;
const DEFAULT_AI_MAX_RETRIES: u32 = 2;
const DEFAULT_AI_BACKOFF_MILLIS: u64 = 1500;
const DEFAULT_AI_INPUT_PRICE_PER_MILLION: f64 = 0.0;
const DEFAULT_AI_OUTPUT_PRICE_PER_MILLION: f64 = 0.0;
const DEFAULT_CHAT_SHOW_TOOL: bool = false;
const DEFAULT_CHAT_SHOW_TOOL_OK: bool = false;
const DEFAULT_CHAT_SHOW_TOOL_ERR: bool = false;
const DEFAULT_CHAT_SHOW_TOOL_TIMEOUT: bool = false;
const DEFAULT_CHAT_COMMAND_CACHE_TTL_SECONDS: u64 = 30;
const DEFAULT_CHAT_SHOW_ROUND_METRICS: bool = true;
const DEFAULT_CHAT_SHOW_TOKEN_COST: bool = true;
const DEFAULT_CHAT_CONTEXT_WARN_PERCENT: u8 = 80;
const DEFAULT_CHAT_CONTEXT_CRITICAL_PERCENT: u8 = 95;
const DEFAULT_CONTEXT_RECENT_MESSAGES: usize = 40;
const MAX_CONTEXT_MESSAGES: usize = 80;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    #[serde(default)]
    pub app: AppSection,
    pub ai: AiConfig,
    #[serde(default)]
    pub cmd: CmdConfig,
    #[serde(default)]
    pub skills: SkillsConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub console: ConsoleConfig,
    #[serde(default)]
    pub session: SessionConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AppSection {
    #[serde(default)]
    pub language: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AiConfig {
    #[serde(rename = "base-url")]
    pub base_url: String,
    pub token: String,
    pub model: String,
    #[serde(default)]
    pub retry: RetryConfig,
    #[serde(default)]
    pub chat: AiChatConfig,
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
        rename = "context-warn-percent",
        default = "default_chat_context_warn_percent"
    )]
    pub context_warn_percent: u8,
    #[serde(
        rename = "context-critical-percent",
        default = "default_chat_context_critical_percent"
    )]
    pub context_critical_percent: u8,
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
    #[serde(default = "default_skills_dir")]
    pub dir: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct McpConfig {
    #[serde(default = "default_mcp_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConsoleConfig {
    #[serde(default = "default_console_colorful")]
    pub colorful: bool,
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
            command_cache_ttl_seconds: default_chat_command_cache_ttl_seconds(),
            show_round_metrics: default_chat_show_round_metrics(),
            show_token_cost: default_chat_show_token_cost(),
            context_warn_percent: default_chat_context_warn_percent(),
            context_critical_percent: default_chat_context_critical_percent(),
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
            write_cmd_allow_patterns: Vec::new(),
            write_cmd_deny_patterns: Vec::new(),
            command_output_max_bytes: default_command_output_max_bytes(),
        }
    }
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            dir: default_skills_dir(),
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
    let cfg: AppConfig = toml::from_str(&raw).map_err(|err| {
        AppError::Config(format!("failed to parse config {}: {err}", path.display()))
    })?;
    validate_config(&cfg)?;
    Ok(cfg)
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

pub fn config_template_example() -> &'static str {
    r#"# MachineClaw Config Template

## Usage
- Copy this template into `claw.toml`.
- Keep required keys complete: `ai.base-url`, `ai.token`, `ai.model`.
- Remove comments if you want a cleaner production config.

## Full Example
```toml
[app]
# language = "zh-CN" # optional: zh-CN, zh-TW, en, fr, de, ja

[ai]
base-url = "https://api.deepseek.com/v1" # required
token = "sk-xxxx" # required
model = "deepseek-chat" # required
input-price-per-million = 0 # optional
output-price-per-million = 0 # optional

[ai.retry]
max-retries = 2 # optional, default 2
backoff-millis = 1500 # optional, default 1500

[ai.chat]
show-tool = false # optional
show-tool-ok = false # optional
show-tool-err = false # optional
show-tool-timeout = false # optional
command-cache-ttl-seconds = 30 # optional
show-round-metrics = true # optional
show-token-cost = true # optional
context-warn-percent = 80 # optional
context-critical-percent = 95 # optional

[cmd]
write-cmd-run-confirm = true # optional, default true
write-cmd-confirm-mode = "allow-once" # optional: deny, edit, allow-once, allow-session
write-cmd-allow-patterns = [] # optional
write-cmd-deny-patterns = [] # optional
command-timeout-seconds = 30 # optional, default 30
command-timeout-kill-after-seconds = 5 # optional, default 5
command-output-max-bytes = 262144 # optional, default 262144

[skills]
dir = "~/.skills" # optional

[mcp]
enabled = false # optional
# endpoint = "http://127.0.0.1:8080/mcp" # optional
# command = "python3" # optional
args = [] # optional
# timeout-seconds = 10 # optional
[mcp.env]
# KEY = "VALUE"

[console]
colorful = true # optional, default true

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
    if cfg.ai.base_url.trim().is_empty() {
        return Err(AppError::Config("ai.base-url is required".to_string()));
    }
    if cfg.ai.token.trim().is_empty() {
        return Err(AppError::Config("ai.token is required".to_string()));
    }
    if cfg.ai.model.trim().is_empty() {
        return Err(AppError::Config("ai.model is required".to_string()));
    }
    if cfg.cmd.command_timeout_seconds == 0 {
        return Err(AppError::Config(
            "cmd.command-timeout-seconds must be greater than 0".to_string(),
        ));
    }
    if cfg.cmd.command_timeout_kill_after_seconds == 0 {
        return Err(AppError::Config(
            "cmd.command-timeout-kill-after-seconds must be greater than 0".to_string(),
        ));
    }
    if cfg.cmd.command_output_max_bytes < 1024 {
        return Err(AppError::Config(
            "cmd.command-output-max-bytes must be >= 1024".to_string(),
        ));
    }
    let confirm_mode = cfg.cmd.write_cmd_confirm_mode.trim().to_ascii_lowercase();
    if !matches!(
        confirm_mode.as_str(),
        "deny" | "edit" | "allow-once" | "allow-session"
    ) {
        return Err(AppError::Config(
            "cmd.write-cmd-confirm-mode must be one of: deny, edit, allow-once, allow-session"
                .to_string(),
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
    Ok(())
}

pub fn resolve_config_path(conf: Option<PathBuf>) -> Result<PathBuf, AppError> {
    if let Some(path) = conf {
        return Ok(resolve_path(path)?);
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

fn default_mcp_enabled() -> bool {
    false
}

fn default_console_colorful() -> bool {
    DEFAULT_CONSOLE_COLORFUL
}

fn default_ai_max_retries() -> u32 {
    DEFAULT_AI_MAX_RETRIES
}

fn default_ai_backoff_millis() -> u64 {
    DEFAULT_AI_BACKOFF_MILLIS
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

fn default_chat_command_cache_ttl_seconds() -> u64 {
    DEFAULT_CHAT_COMMAND_CACHE_TTL_SECONDS
}

fn default_chat_show_round_metrics() -> bool {
    DEFAULT_CHAT_SHOW_ROUND_METRICS
}

fn default_chat_show_token_cost() -> bool {
    DEFAULT_CHAT_SHOW_TOKEN_COST
}

fn default_chat_context_warn_percent() -> u8 {
    DEFAULT_CHAT_CONTEXT_WARN_PERCENT
}

fn default_chat_context_critical_percent() -> u8 {
    DEFAULT_CHAT_CONTEXT_CRITICAL_PERCENT
}

fn default_context_recent_messages() -> usize {
    DEFAULT_CONTEXT_RECENT_MESSAGES
}

fn default_context_max_messages() -> usize {
    MAX_CONTEXT_MESSAGES
}
