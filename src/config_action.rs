use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use toml_edit::{DocumentMut, Item, Table, Value};

use crate::{
    cli::ConfigCommands,
    config::{normalize_chat_interaction_mode, parse_config_text},
    error::{AppError, ExitCode},
    i18n::{self, Language},
};

pub struct ConfigActionOutcome {
    pub rendered: String,
    pub exit_code: ExitCode,
}

#[derive(Debug, Clone, Copy)]
enum ValueSource {
    File,
    Default,
    AutoDetected,
    RequiredUnset,
    OptionalUnset,
}

pub fn run_config_command(
    config_path: &Path,
    command: &ConfigCommands,
) -> Result<ConfigActionOutcome, AppError> {
    match command {
        ConfigCommands::Get { key } => run_get(config_path, key),
        ConfigCommands::Set { key, value } => run_set(config_path, key, value),
    }
}

fn run_get(config_path: &Path, key: &str) -> Result<ConfigActionOutcome, AppError> {
    validate_known_key(key)?;
    let doc = read_document_if_exists(config_path)?;

    let (value, source) = if let Some(item) = get_item_by_path(&doc, key) {
        (display_item(&item), ValueSource::File)
    } else if key == "app.language" {
        (
            i18n::language_code(i18n::resolve_language(None)).to_string(),
            ValueSource::AutoDetected,
        )
    } else if let Some(default_literal) = default_config_value_literal(key) {
        (
            parse_literal_to_item(default_literal)
                .map(|item| display_item(&item))
                .unwrap_or_else(|_| default_literal.to_string()),
            ValueSource::Default,
        )
    } else if is_required_key(key) {
        (unset_text(), ValueSource::RequiredUnset)
    } else {
        (unset_text(), ValueSource::OptionalUnset)
    };

    let display_value = format_value_for_display(key, &value);
    Ok(ConfigActionOutcome {
        rendered: render_get_output(key, &display_value, source, config_path),
        exit_code: ExitCode::Success,
    })
}

fn run_set(
    config_path: &Path,
    key: &str,
    value_raw: &str,
) -> Result<ConfigActionOutcome, AppError> {
    validate_known_key(key)?;

    let (mut doc, file_exists) = read_document_with_state(config_path)?;
    let existed_before = get_item_by_path(&doc, key).is_some();
    let value_item = parse_user_value(value_raw)?;
    validate_key_value_before_write(key, &value_item)?;
    set_item_by_path(&mut doc, key, value_item.clone())?;
    ensure_config_document_parseable(config_path, &doc)?;

    write_document(config_path, &doc)?;

    let display_value = format_value_for_display(key, &display_item(&value_item));
    Ok(ConfigActionOutcome {
        rendered: render_set_output(
            key,
            &display_value,
            config_path,
            !file_exists,
            existed_before,
        ),
        exit_code: ExitCode::Success,
    })
}

fn ensure_config_document_parseable(config_path: &Path, doc: &DocumentMut) -> Result<(), AppError> {
    let raw = doc.to_string();
    parse_config_text(&raw, &config_path.display().to_string())
        .map(|_| ())
        .map_err(|err| AppError::Config(format!("failed to parse updated config: {err}")))
}

fn validate_key_value_before_write(key: &str, value_item: &Item) -> Result<(), AppError> {
    if key != "ai.chat.mode" {
        return Ok(());
    }
    let raw = value_item
        .as_value()
        .and_then(|value| value.as_str())
        .ok_or_else(|| AppError::Config("ai.chat.mode must be string: chat|task".to_string()))?;
    let normalized = normalize_chat_interaction_mode(raw);
    if matches!(normalized, "chat" | "task") {
        return Ok(());
    }
    Err(AppError::Config(
        "ai.chat.mode must be one of: chat, task".to_string(),
    ))
}

fn read_document_if_exists(path: &Path) -> Result<DocumentMut, AppError> {
    if !path.exists() {
        return Ok(DocumentMut::new());
    }
    let raw = fs::read_to_string(path).map_err(|err| {
        AppError::Config(format!(
            "failed to read config file {}: {err}",
            path.display()
        ))
    })?;
    if raw.trim().is_empty() {
        return Ok(DocumentMut::new());
    }
    raw.parse::<DocumentMut>().map_err(|err| {
        AppError::Config(format!(
            "failed to parse config file {}: {err}",
            path.display()
        ))
    })
}

fn read_document_with_state(path: &Path) -> Result<(DocumentMut, bool), AppError> {
    let exists = path.exists();
    let doc = read_document_if_exists(path)?;
    Ok((doc, exists))
}

fn write_document(path: &Path, doc: &DocumentMut) -> Result<(), AppError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            AppError::Config(format!(
                "failed to create config directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    write_config_atomically(path, &doc.to_string())
}

fn get_item_by_path(doc: &DocumentMut, key: &str) -> Option<Item> {
    let mut current = doc.as_item();
    for segment in split_key_path(key) {
        let table = current.as_table_like()?;
        current = table.get(segment)?;
    }
    Some(current.clone())
}

fn set_item_by_path(doc: &mut DocumentMut, key: &str, value: Item) -> Result<(), AppError> {
    let segments = split_key_path(key);
    if segments.is_empty() {
        return Err(AppError::Config("config key is empty".to_string()));
    }

    let mut current = doc.as_item_mut();
    for segment in &segments[..segments.len() - 1] {
        let table = current.as_table_like_mut().ok_or_else(|| {
            AppError::Config(format!(
                "config path conflict at '{}': parent is not a table",
                segment
            ))
        })?;
        if table.get(segment).is_none() {
            table.insert(segment, Item::Table(Table::new()));
        }
        let next = table.get_mut(segment).ok_or_else(|| {
            AppError::Config(format!(
                "failed to navigate config key at '{}': missing node",
                segment
            ))
        })?;
        if next.is_none() {
            *next = Item::Table(Table::new());
        }
        if !next.is_table() {
            return Err(AppError::Config(format!(
                "config path conflict at '{}': target is not a table",
                segment
            )));
        }
        current = next;
    }

    let leaf = segments[segments.len() - 1];
    let table = current.as_table_like_mut().ok_or_else(|| {
        AppError::Config(format!(
            "failed to set config key '{}': parent is not a table",
            key
        ))
    })?;
    table.insert(leaf, value);
    Ok(())
}

fn parse_user_value(raw: &str) -> Result<Item, AppError> {
    let trimmed = raw.trim();
    let literal = if trimmed.is_empty() { "\"\"" } else { trimmed };
    parse_literal_to_item(literal).or_else(|_| Ok(Item::Value(Value::from(raw.to_string()))))
}

fn parse_literal_to_item(literal: &str) -> Result<Item, AppError> {
    let snippet = format!("value = {literal}");
    let doc = snippet
        .parse::<DocumentMut>()
        .map_err(|err| AppError::Config(format!("failed to parse value '{}': {err}", literal)))?;
    doc.get("value")
        .cloned()
        .ok_or_else(|| AppError::Config("parsed value not found".to_string()))
}

fn display_item(item: &Item) -> String {
    if item.is_none() {
        return unset_text();
    }
    if let Some(value) = item.as_value() {
        return display_value(value);
    }
    item.to_string().trim().to_string()
}

fn display_value(value: &Value) -> String {
    if let Some(str_value) = value.as_str() {
        return str_value.to_string();
    }
    value.to_string().trim().to_string()
}

fn write_config_atomically(path: &Path, content: &str) -> Result<(), AppError> {
    let parent = path.parent().ok_or_else(|| {
        AppError::Config(format!(
            "failed to resolve config parent directory for {}",
            path.display()
        ))
    })?;
    let file_name = path
        .file_name()
        .and_then(|item| item.to_str())
        .unwrap_or("config");
    let temp_path = parent.join(format!(
        ".{}.{}.{}.tmp",
        file_name,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::write(&temp_path, content).map_err(|err| {
        AppError::Config(format!(
            "failed to write temporary config file {}: {err}",
            temp_path.display()
        ))
    })?;
    if let Err(rename_err) = fs::rename(&temp_path, path) {
        if let Err(copy_err) = fs::copy(&temp_path, path) {
            let _ = fs::remove_file(&temp_path);
            return Err(AppError::Config(format!(
                "failed to replace config file {} (rename: {}; copy fallback: {})",
                path.display(),
                rename_err,
                copy_err
            )));
        }
        let _ = fs::remove_file(&temp_path);
    }
    Ok(())
}

fn split_key_path(key: &str) -> Vec<&str> {
    key.split('.')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect()
}

fn validate_known_key(key: &str) -> Result<(), AppError> {
    if is_known_config_key(key) {
        return Ok(());
    }
    let samples = known_config_keys()
        .iter()
        .copied()
        .take(10)
        .collect::<Vec<&str>>()
        .join(", ");
    Err(AppError::Config(localized_unknown_key(key, &samples)))
}

pub(crate) fn is_known_config_key(key: &str) -> bool {
    known_config_keys().contains(&key)
}

fn is_required_key(key: &str) -> bool {
    matches!(key, "ai.base-url" | "ai.token" | "ai.model")
}

pub(crate) fn default_config_value_literal(key: &str) -> Option<&'static str> {
    match key {
        "app.theme" => Some("\"default\""),
        "app.env-mode" => Some("\"prod\""),
        "ai.type" => Some("\"openai\""),
        "ai.debug" => Some("false"),
        "ai.connectivity-check" => Some("true"),
        "ai.retry.max-retries" => Some("2"),
        "ai.retry.backoff-millis" => Some("1500"),
        "ai.chat.show-tool" => Some("false"),
        "ai.chat.mode" => Some("\"chat\""),
        "ai.chat.show-tool-ok" => Some("false"),
        "ai.chat.show-tool-err" => Some("false"),
        "ai.chat.show-tool-timeout" => Some("false"),
        "ai.chat.show-tips" => Some("false"),
        "ai.chat.command-cache-ttl-seconds" => Some("30"),
        "ai.chat.show-round-metrics" => Some("true"),
        "ai.chat.show-token-cost" => Some("true"),
        "ai.chat.skip-model-price-check" => Some("false"),
        "ai.chat.model-price-check-mode" => Some("\"sync\""),
        "ai.chat.context-warn-percent" => Some("80"),
        "ai.chat.context-critical-percent" => Some("95"),
        "ai.chat.stream-output" => Some("false"),
        "ai.chat.output-multilines" => Some("false"),
        "ai.chat.skip-env-profile" => Some("true"),
        "ai.chat.cmd-run-timout" => Some("30"),
        "ai.chat.max-tool-rounds" => Some("16"),
        "ai.chat.max-total-tool-calls" => Some("40"),
        "ai.chat.compression.max-history-messages" => Some("40"),
        "ai.chat.compression.max-chars-count" => Some("80000"),
        "ai.input-price-per-million" => Some("0"),
        "ai.output-price-per-million" => Some("0"),
        "ai.tools.bash.write-cmd-run-confirm" => Some("true"),
        "ai.tools.bash.command-timeout-seconds" => Some("30"),
        "ai.tools.bash.command-timeout-kill-after-seconds" => Some("5"),
        "ai.tools.bash.write-cmd-confirm-mode" => Some("\"allow-once\""),
        "ai.tools.bash.allow-cmd-list" => Some("[]"),
        "ai.tools.bash.deny-cmd-list" => Some("[]"),
        "ai.tools.bash.write-cmd-allow-patterns" => Some("[]"),
        "ai.tools.bash.write-cmd-deny-patterns" => Some("[]"),
        "ai.tools.bash.command-output-max-bytes" => Some("0"),
        "ai.tools.builtin.enabled" => Some("true"),
        "ai.tools.builtin.web-search-enabled" => Some("true"),
        "ai.tools.builtin.web-search-timeout-seconds" => Some("10"),
        "ai.tools.builtin.web-search-max-results" => Some("5"),
        "ai.tools.builtin.max-read-bytes" => Some("131072"),
        "ai.tools.builtin.max-search-results" => Some("100"),
        "ai.tools.builtin.write-tools-enabled" => Some("false"),
        "ai.tools.builtin.workspace-only" => Some("true"),
        "ai.tools.skills.enabled" => Some("false"),
        "ai.tools.skills.dir" => Some("\"~/.skills\""),
        "ai.tools.mcp.enabled" => Some("false"),
        "ai.tools.mcp.mcp-availability-check-mode" => Some("\"rsync\""),
        "ai.tools.mcp.dir" => Some("\"~/.machineclaw/mcp\""),
        "console.colorful" => Some("true"),
        "log.dir" => Some("\"logs\""),
        "log.log-file-name" => Some("\"session-{session-id}.log\""),
        "log.max-file-size" => Some("\"50mb\""),
        "log.max-save-time" => Some("\"7d\""),
        "session.recent_messages" => Some("40"),
        "session.max_messages" => Some("80"),
        _ => None,
    }
}

pub(crate) fn known_config_keys() -> &'static [&'static str] {
    &[
        "app.language",
        "app.theme",
        "app.env-mode",
        "ai.type",
        "ai.base-url",
        "ai.token",
        "ai.model",
        "ai.debug",
        "ai.connectivity-check",
        "ai.retry.max-retries",
        "ai.retry.backoff-millis",
        "ai.chat.show-tool",
        "ai.chat.mode",
        "ai.chat.show-tool-ok",
        "ai.chat.show-tool-err",
        "ai.chat.show-tool-timeout",
        "ai.chat.show-tips",
        "ai.chat.command-cache-ttl-seconds",
        "ai.chat.show-round-metrics",
        "ai.chat.show-token-cost",
        "ai.chat.skip-model-price-check",
        "ai.chat.model-price-check-mode",
        "ai.chat.context-warn-percent",
        "ai.chat.context-critical-percent",
        "ai.chat.stream-output",
        "ai.chat.output-multilines",
        "ai.chat.skip-env-profile",
        "ai.chat.cmd-run-timout",
        "ai.chat.max-tool-rounds",
        "ai.chat.max-total-tool-calls",
        "ai.chat.compression.max-history-messages",
        "ai.chat.compression.max-chars-count",
        "ai.input-price-per-million",
        "ai.output-price-per-million",
        "ai.tools.bash.write-cmd-run-confirm",
        "ai.tools.bash.command-timeout-seconds",
        "ai.tools.bash.command-timeout-kill-after-seconds",
        "ai.tools.bash.write-cmd-confirm-mode",
        "ai.tools.bash.allow-cmd-list",
        "ai.tools.bash.deny-cmd-list",
        "ai.tools.bash.write-cmd-allow-patterns",
        "ai.tools.bash.write-cmd-deny-patterns",
        "ai.tools.bash.command-output-max-bytes",
        "ai.tools.builtin.enabled",
        "ai.tools.builtin.web-search-enabled",
        "ai.tools.builtin.web-search-timeout-seconds",
        "ai.tools.builtin.web-search-max-results",
        "ai.tools.builtin.max-read-bytes",
        "ai.tools.builtin.max-search-results",
        "ai.tools.builtin.write-tools-enabled",
        "ai.tools.builtin.workspace-only",
        "ai.tools.skills.enabled",
        "ai.tools.skills.dir",
        "ai.tools.mcp.enabled",
        "ai.tools.mcp.mcp-availability-check-mode",
        "ai.tools.mcp.dir",
        "console.colorful",
        "log.dir",
        "log.log-file-name",
        "log.max-file-size",
        "log.max-save-time",
        "session.recent_messages",
        "session.max_messages",
    ]
}

fn render_get_output(key: &str, value: &str, source: ValueSource, path: &Path) -> String {
    match i18n::current_language() {
        Language::ZhCn => format!(
            "# 配置查询结果\n\n- 字段: {key}\n- 当前值: {value}\n- 值来源: {}\n- 配置文件: {}",
            source_text(source),
            path.display()
        ),
        Language::ZhTw => format!(
            "# 配置查詢結果\n\n- 欄位: {key}\n- 目前值: {value}\n- 值來源: {}\n- 設定檔: {}",
            source_text(source),
            path.display()
        ),
        _ => format!(
            "# Config Value\n\n- Key: {key}\n- Value: {value}\n- Source: {}\n- Config File: {}",
            source_text(source),
            path.display()
        ),
    }
}

fn render_set_output(
    key: &str,
    value: &str,
    path: &Path,
    created_file: bool,
    replaced_existing: bool,
) -> String {
    let change = if replaced_existing {
        match i18n::current_language() {
            Language::ZhCn => "更新已有字段",
            Language::ZhTw => "更新既有欄位",
            _ => "updated existing key",
        }
    } else {
        match i18n::current_language() {
            Language::ZhCn => "新增字段",
            Language::ZhTw => "新增欄位",
            _ => "added new key",
        }
    };

    let file_state = if created_file {
        match i18n::current_language() {
            Language::ZhCn => "已自动创建配置文件",
            Language::ZhTw => "已自動建立設定檔",
            _ => "config file created automatically",
        }
    } else {
        match i18n::current_language() {
            Language::ZhCn => "已写入现有配置文件",
            Language::ZhTw => "已寫入既有設定檔",
            _ => "written into existing config file",
        }
    };

    match i18n::current_language() {
        Language::ZhCn => format!(
            "# 配置更新完成\n\n- 字段: {key}\n- 新值: {value}\n- 变更类型: {change}\n- 配置文件: {}\n- 文件状态: {file_state}",
            path.display()
        ),
        Language::ZhTw => format!(
            "# 配置更新完成\n\n- 欄位: {key}\n- 新值: {value}\n- 變更類型: {change}\n- 設定檔: {}\n- 檔案狀態: {file_state}",
            path.display()
        ),
        _ => format!(
            "# Config Updated\n\n- Key: {key}\n- New Value: {value}\n- Change: {change}\n- Config File: {}\n- File State: {file_state}",
            path.display()
        ),
    }
}

fn source_text(source: ValueSource) -> &'static str {
    match i18n::current_language() {
        Language::ZhCn => match source {
            ValueSource::File => "来自配置文件",
            ValueSource::Default => "来自默认值",
            ValueSource::AutoDetected => "来自环境自动探测",
            ValueSource::RequiredUnset => "未设置（必填字段）",
            ValueSource::OptionalUnset => "未设置（可选字段）",
        },
        Language::ZhTw => match source {
            ValueSource::File => "來自設定檔",
            ValueSource::Default => "來自預設值",
            ValueSource::AutoDetected => "來自環境自動偵測",
            ValueSource::RequiredUnset => "未設定（必填欄位）",
            ValueSource::OptionalUnset => "未設定（選填欄位）",
        },
        _ => match source {
            ValueSource::File => "from config file",
            ValueSource::Default => "from default",
            ValueSource::AutoDetected => "auto-detected from environment",
            ValueSource::RequiredUnset => "unset (required key)",
            ValueSource::OptionalUnset => "unset (optional key)",
        },
    }
}

fn unset_text() -> String {
    match i18n::current_language() {
        Language::ZhCn => "(未设置)".to_string(),
        Language::ZhTw => "(未設定)".to_string(),
        _ => "(unset)".to_string(),
    }
}

fn localized_unknown_key(key: &str, samples: &str) -> String {
    match i18n::current_language() {
        Language::ZhCn => {
            format!("配置字段不存在: {key}。请检查拼写。可用字段示例: {samples}")
        }
        Language::ZhTw => {
            format!("配置欄位不存在: {key}。請檢查拼寫。可用欄位範例: {samples}")
        }
        _ => format!("config key not found: {key}. Please check spelling. Sample keys: {samples}"),
    }
}

fn format_value_for_display(key: &str, value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return value.to_string();
    }
    let Some(parsed) = trimmed.parse::<u128>().ok() else {
        return value.to_string();
    };
    let grouped = i18n::human_count_u128(parsed);

    if key.ends_with("-millis") {
        return format!(
            "{grouped} ({} / {grouped} ms)",
            i18n::human_duration_ms(parsed)
        );
    }
    if key.ends_with("-seconds") {
        return format!(
            "{grouped} ({} / {grouped} s)",
            i18n::human_duration_ms(parsed.saturating_mul(1_000))
        );
    }
    if key == "ai.chat.cmd-run-timout" {
        return format!(
            "{grouped} ({} / {grouped} s)",
            i18n::human_duration_ms(parsed.saturating_mul(1_000))
        );
    }
    if key.ends_with("-bytes") {
        return format!("{grouped} ({})", i18n::human_bytes(parsed));
    }
    grouped
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use toml_edit::DocumentMut;

    use super::{
        ensure_config_document_parseable, is_known_config_key, parse_literal_to_item,
        validate_key_value_before_write,
    };

    #[test]
    fn known_config_keys_include_ai_tools_mcp_fields() {
        assert!(is_known_config_key(
            "ai.tools.mcp.mcp-availability-check-mode"
        ));
        assert!(is_known_config_key("ai.tools.mcp.enabled"));
        assert!(is_known_config_key("ai.tools.mcp.dir"));
        assert!(is_known_config_key("ai.chat.mode"));
        assert!(is_known_config_key("app.theme"));
    }

    #[test]
    fn legacy_mcp_map_keys_are_not_supported_anymore() {
        assert!(!is_known_config_key("mcp.headers.Authorization"));
        assert!(!is_known_config_key("mcp.servers.local.server-url"));
    }

    #[test]
    fn legacy_cmd_keys_are_not_supported_anymore() {
        assert!(!is_known_config_key("cmd.command-timeout-seconds"));
        assert!(!is_known_config_key("cmd.write-cmd-run-confirm"));
    }

    #[test]
    fn validate_ai_chat_mode_before_write_accepts_chat_and_task() {
        let chat = parse_literal_to_item("\"chat\"").expect("chat literal");
        let task = parse_literal_to_item("\"task\"").expect("task literal");
        assert!(validate_key_value_before_write("ai.chat.mode", &chat).is_ok());
        assert!(validate_key_value_before_write("ai.chat.mode", &task).is_ok());
    }

    #[test]
    fn validate_ai_chat_mode_before_write_rejects_invalid_values() {
        let invalid = parse_literal_to_item("\"invalid\"").expect("invalid literal");
        let non_string = parse_literal_to_item("123").expect("number literal");
        assert!(validate_key_value_before_write("ai.chat.mode", &invalid).is_err());
        assert!(validate_key_value_before_write("ai.chat.mode", &non_string).is_err());
    }

    #[test]
    fn config_set_parse_guard_rejects_invalid_scalar_types() {
        let doc = r#"
[ai]
base-url = "https://example.com/v1"
token = "sk-test"
model = "test-model"

[ai.retry]
max-retries = "invalid"
"#
        .parse::<DocumentMut>()
        .expect("parse test doc");
        let err = ensure_config_document_parseable(Path::new("inline"), &doc)
            .expect_err("invalid type must fail parse guard");
        assert!(err.to_string().contains("failed to parse updated config"));
    }
}
