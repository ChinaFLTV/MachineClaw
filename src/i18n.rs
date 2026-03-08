use std::{env, path::Path, process::Command};

use once_cell::sync::Lazy;
use std::sync::RwLock;

use crate::{config::AiChatConfig, error::AppError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    ZhCn,
    ZhTw,
    En,
    Fr,
    De,
    Ja,
}

static ACTIVE_LANGUAGE: Lazy<RwLock<Language>> =
    Lazy::new(|| RwLock::new(detect_system_language()));

pub fn resolve_language(configured: Option<&str>) -> Language {
    if let Some(raw) = configured {
        return parse_language(raw).unwrap_or(Language::En);
    }
    detect_system_language()
}

pub fn parse_language(raw: &str) -> Option<Language> {
    let normalized = raw.trim().replace('_', "-").to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if normalized.starts_with("zh-tw") || normalized.contains("hant") {
        return Some(Language::ZhTw);
    }
    if normalized == "zh" || normalized.starts_with("zh-cn") || normalized.contains("hans") {
        return Some(Language::ZhCn);
    }
    if normalized == "en" || normalized.starts_with("en-") {
        return Some(Language::En);
    }
    if normalized == "fr" || normalized.starts_with("fr-") {
        return Some(Language::Fr);
    }
    if normalized == "de" || normalized.starts_with("de-") {
        return Some(Language::De);
    }
    if normalized == "ja" || normalized.starts_with("ja-") {
        return Some(Language::Ja);
    }
    None
}

pub fn detect_system_language() -> Language {
    for key in ["LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"] {
        if let Ok(value) = env::var(key)
            && let Some(language) = parse_language(&value)
        {
            return language;
        }
    }
    if let Some(language) = detect_language_by_system_command() {
        return language;
    }
    Language::En
}

pub fn set_language(language: Language) {
    if let Ok(mut active) = ACTIVE_LANGUAGE.write() {
        *active = language;
    }
}

pub fn current_language() -> Language {
    if let Ok(active) = ACTIVE_LANGUAGE.read() {
        return *active;
    }
    detect_system_language()
}

pub fn language_code(language: Language) -> &'static str {
    match language {
        Language::ZhCn => "zh-CN",
        Language::ZhTw => "zh-TW",
        Language::En => "en",
        Language::Fr => "fr",
        Language::De => "de",
        Language::Ja => "ja",
    }
}

pub fn unsupported_language_notice(raw: &str) -> String {
    match current_language() {
        Language::ZhCn => format!(
            "配置的语言 '{}' 不受支持，已回退到 English (en)。支持值: zh-CN, zh-TW, en, fr, de, ja",
            raw
        ),
        Language::ZhTw => format!(
            "設定的語言 '{}' 不受支援，已回退到 English (en)。支援值: zh-CN, zh-TW, en, fr, de, ja",
            raw
        ),
        Language::Fr => format!(
            "La langue configurée '{}' n'est pas prise en charge. Retour à English (en). Valeurs prises en charge: zh-CN, zh-TW, en, fr, de, ja",
            raw
        ),
        Language::De => format!(
            "Die konfigurierte Sprache '{}' wird nicht unterstützt. Fallback auf English (en). Unterstützte Werte: zh-CN, zh-TW, en, fr, de, ja",
            raw
        ),
        Language::Ja => format!(
            "設定された言語 '{}' は未対応のため、English (en) にフォールバックしました。対応値: zh-CN, zh-TW, en, fr, de, ja",
            raw
        ),
        Language::En => format!(
            "Configured language '{}' is not supported; falling back to English (en). Supported values: zh-CN, zh-TW, en, fr, de, ja",
            raw
        ),
    }
}

pub fn localize_error(err: &AppError) -> String {
    match err {
        AppError::Config(detail) => {
            format!("{}: {}", error_prefix_config(), localize_detail(detail))
        }
        AppError::Permission(detail) => {
            format!("{}: {}", error_prefix_permission(), localize_detail(detail))
        }
        AppError::Ai(detail) => format!("{}: {}", error_prefix_ai(), localize_detail(detail)),
        AppError::Command(detail) => {
            format!("{}: {}", error_prefix_command(), localize_detail(detail))
        }
        AppError::Runtime(detail) => {
            format!("{}: {}", error_prefix_runtime(), localize_detail(detail))
        }
    }
}

pub fn notice_assets_dir_created(path: &Path) -> String {
    match current_language() {
        Language::ZhCn => format!("assets 目录不存在，已自动创建: {}", path.display()),
        Language::ZhTw => format!("assets 目錄不存在，已自動建立: {}", path.display()),
        Language::Fr => format!(
            "Le répertoire assets est introuvable et a été créé automatiquement: {}",
            path.display()
        ),
        Language::De => format!(
            "Das Verzeichnis assets fehlte und wurde automatisch erstellt: {}",
            path.display()
        ),
        Language::Ja => format!(
            "assets ディレクトリが存在しないため自動作成しました: {}",
            path.display()
        ),
        Language::En => format!(
            "assets directory not found and has been created automatically: {}",
            path.display()
        ),
    }
}

pub fn notice_prompts_dir_created(path: &Path) -> String {
    match current_language() {
        Language::ZhCn => format!("prompts 目录不存在，已自动创建: {}", path.display()),
        Language::ZhTw => format!("prompts 目錄不存在，已自動建立: {}", path.display()),
        Language::Fr => format!(
            "Le répertoire prompts est introuvable et a été créé automatiquement: {}",
            path.display()
        ),
        Language::De => format!(
            "Das Verzeichnis prompts fehlte und wurde automatisch erstellt: {}",
            path.display()
        ),
        Language::Ja => format!(
            "prompts ディレクトリが存在しないため自動作成しました: {}",
            path.display()
        ),
        Language::En => format!(
            "prompts directory not found and has been created automatically: {}",
            path.display()
        ),
    }
}

pub fn notice_output_templates_dir_created(path: &Path) -> String {
    match current_language() {
        Language::ZhCn => format!(
            "output_templates 目录不存在，已自动创建: {}",
            path.display()
        ),
        Language::ZhTw => format!(
            "output_templates 目錄不存在，已自動建立: {}",
            path.display()
        ),
        Language::Fr => format!(
            "Le répertoire output_templates est introuvable et a été créé automatiquement: {}",
            path.display()
        ),
        Language::De => format!(
            "Das Verzeichnis output_templates fehlte und wurde automatisch erstellt: {}",
            path.display()
        ),
        Language::Ja => format!(
            "output_templates ディレクトリが存在しないため自動作成しました: {}",
            path.display()
        ),
        Language::En => format!(
            "output_templates directory not found and has been created automatically: {}",
            path.display()
        ),
    }
}

pub fn notice_asset_file_created(path: &Path) -> String {
    match current_language() {
        Language::ZhCn => format!("资源文件缺失，已自动创建默认文件: {}", path.display()),
        Language::ZhTw => format!("資源檔案缺失，已自動建立預設檔案: {}", path.display()),
        Language::Fr => format!(
            "Fichier de ressource manquant, fichier par défaut créé automatiquement: {}",
            path.display()
        ),
        Language::De => format!(
            "Ressourcendatei fehlte; Standarddatei wurde automatisch erstellt: {}",
            path.display()
        ),
        Language::Ja => format!(
            "リソースファイルが不足していたため、デフォルトファイルを自動作成しました: {}",
            path.display()
        ),
        Language::En => format!(
            "resource file missing; default file created automatically: {}",
            path.display()
        ),
    }
}

pub fn prefix_error() -> &'static str {
    match current_language() {
        Language::ZhCn => "错误",
        Language::ZhTw => "錯誤",
        Language::Fr => "Erreur",
        Language::De => "Fehler",
        Language::Ja => "エラー",
        Language::En => "ERROR",
    }
}

pub fn prefix_warn() -> &'static str {
    match current_language() {
        Language::ZhCn => "警告",
        Language::ZhTw => "警告",
        Language::Fr => "Avertissement",
        Language::De => "Warnung",
        Language::Ja => "警告",
        Language::En => "WARN",
    }
}

pub fn prefix_info() -> &'static str {
    match current_language() {
        Language::ZhCn => "提示",
        Language::ZhTw => "提示",
        Language::Fr => "Info",
        Language::De => "Info",
        Language::Ja => "情報",
        Language::En => "INFO",
    }
}

pub fn output_label_action() -> &'static str {
    match current_language() {
        Language::ZhCn => "动作",
        Language::ZhTw => "動作",
        Language::Fr => "Action",
        Language::De => "Aktion",
        Language::Ja => "アクション",
        Language::En => "Action",
    }
}

pub fn output_label_status() -> &'static str {
    match current_language() {
        Language::ZhCn => "状态",
        Language::ZhTw => "狀態",
        Language::Fr => "Statut",
        Language::De => "Status",
        Language::Ja => "状態",
        Language::En => "Status",
    }
}

pub fn output_label_key_metrics() -> &'static str {
    match current_language() {
        Language::ZhCn => "关键指标",
        Language::ZhTw => "關鍵指標",
        Language::Fr => "Indicateurs clés",
        Language::De => "Kennzahlen",
        Language::Ja => "主要指標",
        Language::En => "KeyMetrics",
    }
}

pub fn output_label_risk_summary() -> &'static str {
    match current_language() {
        Language::ZhCn => "风险摘要",
        Language::ZhTw => "風險摘要",
        Language::Fr => "Résumé des risques",
        Language::De => "Risikozusammenfassung",
        Language::Ja => "リスク要約",
        Language::En => "RiskSummary",
    }
}

pub fn output_label_ai_summary() -> &'static str {
    match current_language() {
        Language::ZhCn => "AI 总结",
        Language::ZhTw => "AI 總結",
        Language::Fr => "Résumé IA",
        Language::De => "KI-Zusammenfassung",
        Language::Ja => "AI 要約",
        Language::En => "AISummary",
    }
}

pub fn output_label_command_summary() -> &'static str {
    match current_language() {
        Language::ZhCn => "命令摘要",
        Language::ZhTw => "命令摘要",
        Language::Fr => "Résumé des commandes",
        Language::De => "Befehlszusammenfassung",
        Language::Ja => "コマンド要約",
        Language::En => "CommandSummary",
    }
}

pub fn output_label_elapsed() -> &'static str {
    match current_language() {
        Language::ZhCn => "耗时",
        Language::ZhTw => "耗時",
        Language::Fr => "Durée",
        Language::De => "Dauer",
        Language::Ja => "経過時間",
        Language::En => "Elapsed",
    }
}

pub fn status_success() -> &'static str {
    match current_language() {
        Language::ZhCn => "成功",
        Language::ZhTw => "成功",
        Language::Fr => "succès",
        Language::De => "erfolgreich",
        Language::Ja => "成功",
        Language::En => "success",
    }
}

pub fn status_failed() -> &'static str {
    match current_language() {
        Language::ZhCn => "失败",
        Language::ZhTw => "失敗",
        Language::Fr => "échec",
        Language::De => "fehlgeschlagen",
        Language::Ja => "失敗",
        Language::En => "failed",
    }
}

pub fn status_ok() -> &'static str {
    match current_language() {
        Language::ZhCn => "成功",
        Language::ZhTw => "成功",
        Language::Fr => "OK",
        Language::De => "OK",
        Language::Ja => "成功",
        Language::En => "OK",
    }
}

pub fn status_fail_short() -> &'static str {
    match current_language() {
        Language::ZhCn => "失败",
        Language::ZhTw => "失敗",
        Language::Fr => "KO",
        Language::De => "FEHLER",
        Language::Ja => "失敗",
        Language::En => "FAIL",
    }
}

pub fn risk_no_obvious() -> &'static str {
    match current_language() {
        Language::ZhCn => "未发现明显风险",
        Language::ZhTw => "未發現明顯風險",
        Language::Fr => "aucun risque évident détecté",
        Language::De => "kein offensichtliches Risiko erkannt",
        Language::Ja => "明らかなリスクは検出されませんでした",
        Language::En => "no obvious risk detected",
    }
}

pub fn prepare_metrics_overview(
    os_name: &str,
    commands_total: usize,
    commands_success: usize,
    commands_failed: usize,
    skills_count: usize,
    mcp_summary: &str,
    recent_messages: usize,
    max_messages: usize,
) -> String {
    match current_language() {
        Language::ZhCn => format!(
            "1. 当前系统平台：{os_name}\n2. 预检命令执行：共 {commands_total} 条，成功 {commands_success} 条，失败 {commands_failed} 条\n3. 能力概览：Skills {skills_count} 个，MCP {mcp_summary}\n4. 会话上下文：最近保留 {recent_messages} 条，最大 {max_messages} 条"
        ),
        Language::ZhTw => format!(
            "1. 目前系統平台：{os_name}\n2. 預檢命令執行：共 {commands_total} 條，成功 {commands_success} 條，失敗 {commands_failed} 條\n3. 能力概覽：Skills {skills_count} 個，MCP {mcp_summary}\n4. 會話上下文：最近保留 {recent_messages} 條，最大 {max_messages} 條"
        ),
        Language::Fr => format!(
            "1. Plateforme système: {os_name}\n2. Commandes de pré-vérification: total {commands_total}, succès {commands_success}, échec {commands_failed}\n3. Capacités: skills {skills_count}, mcp {mcp_summary}\n4. Contexte de session: recent {recent_messages}, max {max_messages}"
        ),
        Language::De => format!(
            "1. Systemplattform: {os_name}\n2. Preflight-Befehle: gesamt {commands_total}, erfolgreich {commands_success}, fehlgeschlagen {commands_failed}\n3. Fähigkeiten: skills {skills_count}, mcp {mcp_summary}\n4. Sitzungs-Kontext: recent {recent_messages}, max {max_messages}"
        ),
        Language::Ja => format!(
            "1. 現在のシステム: {os_name}\n2. 事前チェックコマンド: 合計 {commands_total}、成功 {commands_success}、失敗 {commands_failed}\n3. 機能概要: skills {skills_count}、mcp {mcp_summary}\n4. セッション文脈: recent {recent_messages}、max {max_messages}"
        ),
        Language::En => format!(
            "1. Current platform: {os_name}\n2. Preflight commands: total {commands_total}, success {commands_success}, failed {commands_failed}\n3. Capability: skills {skills_count}, mcp {mcp_summary}\n4. Session context: recent {recent_messages}, max {max_messages}"
        ),
    }
}

pub fn prepare_risk_timeout() -> &'static str {
    match current_language() {
        Language::ZhCn => "命令执行超时",
        Language::ZhTw => "命令執行逾時",
        Language::Fr => "délai de commande dépassé",
        Language::De => "Befehl-Timeout",
        Language::Ja => "コマンドがタイムアウトしました",
        Language::En => "command timed out",
    }
}

pub fn prepare_risk_interrupted() -> &'static str {
    match current_language() {
        Language::ZhCn => "命令执行被中断",
        Language::ZhTw => "命令執行被中斷",
        Language::Fr => "commande interrompue",
        Language::De => "Befehl unterbrochen",
        Language::Ja => "コマンドが中断されました",
        Language::En => "command interrupted",
    }
}

pub fn command_mode_read() -> &'static str {
    match current_language() {
        Language::ZhCn => "读",
        Language::ZhTw => "讀",
        Language::Fr => "lecture",
        Language::De => "lesen",
        Language::Ja => "読み取り",
        Language::En => "read",
    }
}

pub fn command_mode_write() -> &'static str {
    match current_language() {
        Language::ZhCn => "写",
        Language::ZhTw => "寫",
        Language::Fr => "écriture",
        Language::De => "schreiben",
        Language::Ja => "書き込み",
        Language::En => "write",
    }
}

pub fn chat_tag_info() -> &'static str {
    match current_language() {
        Language::ZhCn => "[提示]",
        Language::ZhTw => "[提示]",
        Language::Fr => "[info]",
        Language::De => "[info]",
        Language::Ja => "[情報]",
        Language::En => "[info]",
    }
}

pub fn chat_tag_tool() -> &'static str {
    match current_language() {
        Language::ZhCn => "[工具]",
        Language::ZhTw => "[工具]",
        Language::Fr => "[outil]",
        Language::De => "[tool]",
        Language::Ja => "[ツール]",
        Language::En => "[tool]",
    }
}

pub fn chat_tag_tool_ok() -> &'static str {
    match current_language() {
        Language::ZhCn => "[工具-成功]",
        Language::ZhTw => "[工具-成功]",
        Language::Fr => "[outil-ok]",
        Language::De => "[tool-ok]",
        Language::Ja => "[ツール-成功]",
        Language::En => "[tool-ok]",
    }
}

pub fn chat_tag_tool_err() -> &'static str {
    match current_language() {
        Language::ZhCn => "[工具-失败]",
        Language::ZhTw => "[工具-失敗]",
        Language::Fr => "[outil-err]",
        Language::De => "[tool-err]",
        Language::Ja => "[ツール-失敗]",
        Language::En => "[tool-err]",
    }
}

pub fn chat_tag_tool_timeout() -> &'static str {
    match current_language() {
        Language::ZhCn => "[工具-超时]",
        Language::ZhTw => "[工具-超時]",
        Language::Fr => "[outil-timeout]",
        Language::De => "[tool-timeout]",
        Language::Ja => "[ツール-タイムアウト]",
        Language::En => "[tool-timeout]",
    }
}

pub fn prompt_write_command_confirmation(command: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("写命令需要确认: {}", command),
        Language::ZhTw => format!("寫命令需要確認: {}", command),
        Language::Fr => format!(
            "La commande d'écriture nécessite une confirmation: {}",
            command
        ),
        Language::De => format!("Der Schreibbefehl erfordert eine Bestätigung: {}", command),
        Language::Ja => format!("書き込みコマンドは確認が必要です: {}", command),
        Language::En => format!("Write command needs confirmation: {}", command),
    }
}

pub fn prompt_write_command_proceed() -> &'static str {
    match current_language() {
        Language::ZhCn => "是否继续? [y/N]: ",
        Language::ZhTw => "是否繼續? [y/N]: ",
        Language::Fr => "Continuer ? [y/N]: ",
        Language::De => "Fortfahren? [y/N]: ",
        Language::Ja => "続行しますか? [y/N]: ",
        Language::En => "Proceed? [y/N]: ",
    }
}

pub fn prompt_write_command_proceed_with_session() -> &'static str {
    match current_language() {
        Language::ZhCn => "是否继续? [y=本次允许/a=本会话允许/e=编辑命令/N=拒绝]: ",
        Language::ZhTw => "是否繼續? [y=本次允許/a=本會話允許/e=編輯命令/N=拒絕]: ",
        Language::Fr => {
            "Continuer ? [y=autoriser une fois/a=autoriser la session/e=éditer/N=refuser]: "
        }
        Language::De => {
            "Fortfahren? [y=einmal erlauben/a=Sitzung erlauben/e=bearbeiten/N=ablehnen]: "
        }
        Language::Ja => "続行しますか? [y=今回のみ許可/a=このセッションを許可/e=編集/N=拒否]: ",
        Language::En => "Proceed? [y=allow once/a=allow session/e=edit/N=deny]: ",
    }
}

pub fn prompt_write_command_edit_title(command: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("准备编辑写命令: {command}"),
        Language::ZhTw => format!("準備編輯寫命令: {command}"),
        Language::Fr => format!("Modification de la commande d'écriture: {command}"),
        Language::De => format!("Schreibbefehl bearbeiten: {command}"),
        Language::Ja => format!("書き込みコマンドを編集します: {command}"),
        Language::En => format!("Edit write command: {command}"),
    }
}

pub fn prompt_write_command_edit_hint() -> &'static str {
    match current_language() {
        Language::ZhCn => "请输入修改后的完整命令；直接回车表示取消执行。",
        Language::ZhTw => "請輸入修改後的完整命令；直接 Enter 表示取消執行。",
        Language::Fr => "Saisissez la commande complète modifiée. Entrée vide pour annuler.",
        Language::De => {
            "Geben Sie den vollständigen geänderten Befehl ein. Leere Eingabe = Abbruch."
        }
        Language::Ja => "修正後のコマンド全体を入力してください。空入力で中止します。",
        Language::En => "Enter the edited full command. Empty input cancels execution.",
    }
}

pub fn prompt_write_command_edit_input() -> &'static str {
    match current_language() {
        Language::ZhCn => "新命令> ",
        Language::ZhTw => "新命令> ",
        Language::Fr => "Nouvelle commande> ",
        Language::De => "Neuer Befehl> ",
        Language::Ja => "新しいコマンド> ",
        Language::En => "New command> ",
    }
}

pub fn chat_requires_interactive_terminal() -> String {
    match current_language() {
        Language::ZhCn => "chat 需要在交互式终端中运行".to_string(),
        Language::ZhTw => "chat 需要在互動式終端中執行".to_string(),
        Language::Fr => "chat doit être exécuté dans un terminal interactif".to_string(),
        Language::De => "chat muss in einem interaktiven Terminal ausgeführt werden".to_string(),
        Language::Ja => "chat は対話型ターミナルで実行する必要があります".to_string(),
        Language::En => "chat requires an interactive terminal".to_string(),
    }
}

pub fn chat_welcome(
    session_id: &str,
    session_file: &Path,
    message_count: usize,
    summary_len: usize,
    recent_limit: usize,
    max_limit: usize,
    os_name: &str,
    model: &str,
    skills_count: usize,
    mcp_summary: &str,
    chat_cfg: &AiChatConfig,
) -> String {
    let tool_flags = format!(
        "tool={}, tool_ok={}, tool_err={}, tool_timeout={}",
        chat_cfg.show_tool,
        chat_cfg.show_tool_ok,
        chat_cfg.show_tool_err,
        chat_cfg.show_tool_timeout
    );
    match current_language() {
        Language::ZhCn => format!(
            "chat 模式已启动\n会话: {session_id}\n会话文件: {}\n模型: {model}\n系统: {os_name}\n消息: {message_count} 条, 摘要 {summary_len} 字符\n上下文窗口: recent={recent_limit}, max={max_limit}\n能力: skills={skills_count}, mcp={mcp_summary}\n工具消息显示: {tool_flags}",
            session_file.display()
        ),
        Language::ZhTw => format!(
            "chat 模式已啟動\n會話: {session_id}\n會話檔案: {}\n模型: {model}\n系統: {os_name}\n訊息: {message_count} 筆, 摘要 {summary_len} 字元\n上下文視窗: recent={recent_limit}, max={max_limit}\n能力: skills={skills_count}, mcp={mcp_summary}\n工具訊息顯示: {tool_flags}",
            session_file.display()
        ),
        Language::Fr => format!(
            "Mode chat démarré\nSession: {session_id}\nFichier de session: {}\nModèle: {model}\nSystème: {os_name}\nMessages: {message_count}, résumé {summary_len} caractères\nFenêtre de contexte: recent={recent_limit}, max={max_limit}\nCapacités: skills={skills_count}, mcp={mcp_summary}\nAffichage outils: {tool_flags}",
            session_file.display()
        ),
        Language::De => format!(
            "Chat-Modus gestartet\nSitzung: {session_id}\nSitzungsdatei: {}\nModell: {model}\nSystem: {os_name}\nNachrichten: {message_count}, Zusammenfassung {summary_len} Zeichen\nKontextfenster: recent={recent_limit}, max={max_limit}\nFähigkeiten: skills={skills_count}, mcp={mcp_summary}\nTool-Anzeige: {tool_flags}",
            session_file.display()
        ),
        Language::Ja => format!(
            "chat モードを開始しました\nセッション: {session_id}\nセッションファイル: {}\nモデル: {model}\nシステム: {os_name}\nメッセージ: {message_count} 件, 要約 {summary_len} 文字\nコンテキスト枠: recent={recent_limit}, max={max_limit}\n機能: skills={skills_count}, mcp={mcp_summary}\nツール表示: {tool_flags}",
            session_file.display()
        ),
        Language::En => format!(
            "chat mode started\nsession: {session_id}\nsession file: {}\nmodel: {model}\nos: {os_name}\nmessages: {message_count}, summary {summary_len} chars\ncontext window: recent={recent_limit}, max={max_limit}\ncapability: skills={skills_count}, mcp={mcp_summary}\ntool event visibility: {tool_flags}",
            session_file.display()
        ),
    }
}

pub fn chat_hint() -> &'static str {
    match current_language() {
        Language::ZhCn => "输入问题开始对话。命令: /help, /stats, /new, /clear, /exit",
        Language::ZhTw => "輸入問題開始對話。命令: /help, /stats, /new, /clear, /exit",
        Language::Fr => {
            "Saisissez votre question pour commencer. Commandes: /help, /stats, /new, /clear, /exit"
        }
        Language::De => {
            "Geben Sie eine Frage ein, um zu starten. Befehle: /help, /stats, /new, /clear, /exit"
        }
        Language::Ja => "質問を入力して開始します。コマンド: /help, /stats, /new, /clear, /exit",
        Language::En => "Type a question to start. Commands: /help, /stats, /new, /clear, /exit",
    }
}

pub fn chat_help_text() -> &'static str {
    match current_language() {
        Language::ZhCn => {
            "/help 显示帮助, /stats 查看上下文统计, /new 新建会话, /clear 清屏(不清历史), /exit 退出 chat"
        }
        Language::ZhTw => {
            "/help 顯示說明, /stats 查看上下文統計, /new 建立新會話, /clear 清屏(不清歷史), /exit 離開 chat"
        }
        Language::Fr => {
            "/help aide, /stats statistiques du contexte, /new nouvelle session, /clear effacer l'écran (garde l'historique), /exit quitter chat"
        }
        Language::De => {
            "/help Hilfe, /stats Kontextstatistik, /new neue Sitzung, /clear Bildschirm leeren (Verlauf bleibt), /exit chat beenden"
        }
        Language::Ja => {
            "/help ヘルプ, /stats コンテキスト統計, /new 新規セッション, /clear 画面クリア(履歴保持), /exit chat 終了"
        }
        Language::En => {
            "/help help, /stats context stats, /new new session, /clear clear screen (keep history), /exit leave chat"
        }
    }
}

pub fn chat_cleared() -> &'static str {
    match current_language() {
        Language::ZhCn => "屏幕已清空，历史消息仍保留。",
        Language::ZhTw => "畫面已清空，歷史訊息仍保留。",
        Language::Fr => "Écran effacé, l'historique est conservé.",
        Language::De => "Bildschirm geleert, Verlauf bleibt erhalten.",
        Language::Ja => "画面をクリアしました。履歴は保持されています。",
        Language::En => "Screen cleared, history is kept.",
    }
}

pub fn chat_prompt_user() -> &'static str {
    match current_language() {
        Language::ZhCn => "你> ",
        Language::ZhTw => "你> ",
        Language::Fr => "Vous> ",
        Language::De => "Du> ",
        Language::Ja => "あなた> ",
        Language::En => "You> ",
    }
}

pub fn chat_prompt_assistant() -> &'static str {
    match current_language() {
        Language::ZhCn => "MachineClaw> ",
        Language::ZhTw => "MachineClaw> ",
        Language::Fr => "MachineClaw> ",
        Language::De => "MachineClaw> ",
        Language::Ja => "MachineClaw> ",
        Language::En => "MachineClaw> ",
    }
}

pub fn chat_prompt_thinking() -> &'static str {
    match current_language() {
        Language::ZhCn => "MachineClaw(思考)> ",
        Language::ZhTw => "MachineClaw(思考)> ",
        Language::Fr => "MachineClaw(thinking)> ",
        Language::De => "MachineClaw(thinking)> ",
        Language::Ja => "MachineClaw(thinking)> ",
        Language::En => "MachineClaw(thinking)> ",
    }
}

pub fn chat_option_detected() -> &'static str {
    match current_language() {
        Language::ZhCn => "检测到可选操作，请用方向键移动，空格勾选，回车确认。",
        Language::ZhTw => "偵測到可選操作，請用方向鍵移動、空白鍵勾選、Enter 確認。",
        Language::Fr => {
            "Options détectées. Flèches pour naviguer, Espace pour cocher, Entrée pour confirmer."
        }
        Language::De => {
            "Optionen erkannt. Pfeile zum Navigieren, Leertaste zum Auswählen, Enter zum Bestätigen."
        }
        Language::Ja => "選択肢を検出しました。矢印で移動、スペースで選択、Enter で確定します。",
        Language::En => {
            "Options detected. Use arrows to move, Space to select, and Enter to confirm."
        }
    }
}

pub fn chat_option_prompt() -> &'static str {
    match current_language() {
        Language::ZhCn => "请选择下一步操作",
        Language::ZhTw => "請選擇下一步操作",
        Language::Fr => "Sélectionnez l'action suivante",
        Language::De => "Nächste Aktion auswählen",
        Language::Ja => "次の操作を選択",
        Language::En => "Select next action",
    }
}

pub fn chat_option_selected(option: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("已选择: {option}"),
        Language::ZhTw => format!("已選擇: {option}"),
        Language::Fr => format!("Sélectionné: {option}"),
        Language::De => format!("Ausgewählt: {option}"),
        Language::Ja => format!("選択: {option}"),
        Language::En => format!("Selected: {option}"),
    }
}

pub fn chat_tool_output_preview(preview: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("输出预览: {preview}"),
        Language::ZhTw => format!("輸出預覽: {preview}"),
        Language::Fr => format!("Aperçu de sortie: {preview}"),
        Language::De => format!("Ausgabevorschau: {preview}"),
        Language::Ja => format!("出力プレビュー: {preview}"),
        Language::En => format!("Output preview: {preview}"),
    }
}

pub fn chat_session_switched(session_id: &str, session_file: &Path) -> String {
    match current_language() {
        Language::ZhCn => format!(
            "已创建新会话\n会话: {session_id}\n会话文件: {}",
            session_file.display()
        ),
        Language::ZhTw => format!(
            "已建立新會話\n會話: {session_id}\n會話檔案: {}",
            session_file.display()
        ),
        Language::Fr => format!(
            "Nouvelle session créée\nSession: {session_id}\nFichier: {}",
            session_file.display()
        ),
        Language::De => format!(
            "Neue Sitzung erstellt\nSitzung: {session_id}\nDatei: {}",
            session_file.display()
        ),
        Language::Ja => format!(
            "新しいセッションを作成しました\nセッション: {session_id}\nファイル: {}",
            session_file.display()
        ),
        Language::En => format!(
            "new session created\nsession: {session_id}\nfile: {}",
            session_file.display()
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn chat_stats(
    session_id: &str,
    session_file: &Path,
    message_count: usize,
    summary_len: usize,
    recent_limit: usize,
    max_limit: usize,
    chat_turns: usize,
    os_name: &str,
    model: &str,
    skills_count: usize,
    mcp_summary: &str,
    user_count: usize,
    assistant_count: usize,
    tool_count: usize,
    system_count: usize,
) -> String {
    match current_language() {
        Language::ZhCn => format!(
            "会话统计\n会话: {session_id}\n会话文件: {}\n系统/模型: {os_name} / {model}\n消息总数: {message_count} (用户 {user_count}, 助手 {assistant_count}, 工具 {tool_count}, 系统 {system_count})\n摘要长度: {summary_len} 字符\n本次 chat 轮次: {chat_turns}\n上下文窗口: recent={recent_limit}, max={max_limit}\n能力状态: skills={skills_count}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::ZhTw => format!(
            "會話統計\n會話: {session_id}\n會話檔案: {}\n系統/模型: {os_name} / {model}\n訊息總數: {message_count} (使用者 {user_count}, 助手 {assistant_count}, 工具 {tool_count}, 系統 {system_count})\n摘要長度: {summary_len} 字元\n本次 chat 輪次: {chat_turns}\n上下文視窗: recent={recent_limit}, max={max_limit}\n能力狀態: skills={skills_count}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::Fr => format!(
            "Statistiques de session\nSession: {session_id}\nFichier: {}\nSystème/Modèle: {os_name} / {model}\nMessages: {message_count} (user {user_count}, assistant {assistant_count}, tool {tool_count}, system {system_count})\nLongueur du résumé: {summary_len}\nTours de chat: {chat_turns}\nFenêtre de contexte: recent={recent_limit}, max={max_limit}\nCapacités: skills={skills_count}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::De => format!(
            "Sitzungsstatistik\nSitzung: {session_id}\nDatei: {}\nSystem/Modell: {os_name} / {model}\nNachrichten: {message_count} (user {user_count}, assistant {assistant_count}, tool {tool_count}, system {system_count})\nZusammenfassungslänge: {summary_len}\nChat-Runden: {chat_turns}\nKontextfenster: recent={recent_limit}, max={max_limit}\nFähigkeiten: skills={skills_count}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::Ja => format!(
            "セッション統計\nセッション: {session_id}\nファイル: {}\nシステム/モデル: {os_name} / {model}\nメッセージ総数: {message_count} (user {user_count}, assistant {assistant_count}, tool {tool_count}, system {system_count})\n要約文字数: {summary_len}\nchat ターン数: {chat_turns}\nコンテキスト枠: recent={recent_limit}, max={max_limit}\n機能: skills={skills_count}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::En => format!(
            "session stats\nsession: {session_id}\nfile: {}\nos/model: {os_name} / {model}\nmessages: {message_count} (user {user_count}, assistant {assistant_count}, tool {tool_count}, system {system_count})\nsummary chars: {summary_len}\nchat turns: {chat_turns}\ncontext window: recent={recent_limit}, max={max_limit}\ncapability: skills={skills_count}, mcp={mcp_summary}",
            session_file.display()
        ),
    }
}

pub fn chat_progress_analyzing() -> &'static str {
    match current_language() {
        Language::ZhCn => "正在分析并按需调用本地命令...",
        Language::ZhTw => "正在分析並按需呼叫本地命令...",
        Language::Fr => "Analyse en cours avec appel local des commandes si nécessaire...",
        Language::De => "Analyse läuft, lokale Befehle werden bei Bedarf ausgeführt...",
        Language::Ja => "解析中です。必要に応じてローカルコマンドを実行します...",
        Language::En => "Analyzing and invoking local commands when needed...",
    }
}

pub fn chat_context_pressure_warning(
    usage_percent: u8,
    message_count: usize,
    max_limit: usize,
    recent_limit: usize,
    summary_chars: usize,
    critical: bool,
) -> String {
    match current_language() {
        Language::ZhCn => {
            if critical {
                return format!(
                    "上下文容量已接近上限({usage_percent}%)：messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}。建议尽快结束当前话题或切换 /new 以降低压缩损耗。"
                );
            }
            format!(
                "上下文容量预警({usage_percent}%)：messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}。后续消息可能触发更激进压缩。"
            )
        }
        Language::ZhTw => {
            if critical {
                return format!(
                    "上下文容量接近上限({usage_percent}%)：messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}。建議盡快結束目前話題或切換 /new 降低壓縮損耗。"
                );
            }
            format!(
                "上下文容量預警({usage_percent}%)：messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}。後續訊息可能觸發更激進壓縮。"
            )
        }
        Language::Fr => {
            if critical {
                return format!(
                    "Contexte presque saturé ({usage_percent}%): messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}. Terminez le sujet ou utilisez /new pour réduire la perte due à la compression."
                );
            }
            format!(
                "Alerte contexte ({usage_percent}%): messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}. Une compression plus agressive peut être appliquée."
            )
        }
        Language::De => {
            if critical {
                return format!(
                    "Kontext fast ausgelastet ({usage_percent}%): messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}. Thema abschließen oder /new verwenden, um Kompressionsverlust zu verringern."
                );
            }
            format!(
                "Kontext-Warnung ({usage_percent}%): messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}. Weitere Nachrichten können stärkere Kompression auslösen."
            )
        }
        Language::Ja => {
            if critical {
                return format!(
                    "コンテキスト容量が上限に近づいています ({usage_percent}%): messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}。圧縮劣化を抑えるため、話題を区切るか /new を使用してください。"
                );
            }
            format!(
                "コンテキスト容量の警告 ({usage_percent}%): messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}。以降のメッセージで圧縮が強くなる可能性があります。"
            )
        }
        Language::En => {
            if critical {
                return format!(
                    "Context is near capacity ({usage_percent}%): messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}. Consider ending the topic or switching to /new to reduce compression loss."
                );
            }
            format!(
                "Context pressure warning ({usage_percent}%): messages={message_count}/{max_limit}, recent={recent_limit}, summary_chars={summary_chars}. Further messages may trigger stronger compression."
            )
        }
    }
}

pub fn chat_tool_running(label: &str, mode: &str, command: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("执行命令: label={label}, mode={mode}, command={command}"),
        Language::ZhTw => format!("執行命令: label={label}, mode={mode}, command={command}"),
        Language::Fr => {
            format!("Exécution de commande: label={label}, mode={mode}, command={command}")
        }
        Language::De => {
            format!("Befehl wird ausgeführt: label={label}, mode={mode}, command={command}")
        }
        Language::Ja => format!("コマンド実行: label={label}, mode={mode}, command={command}"),
        Language::En => format!("Running command: label={label}, mode={mode}, command={command}"),
    }
}

pub fn chat_tool_cache_hit(label: &str, age_ms: u128) -> String {
    match current_language() {
        Language::ZhCn => format!("命中命令缓存: label={label}, age_ms={age_ms}"),
        Language::ZhTw => format!("命中命令快取: label={label}, age_ms={age_ms}"),
        Language::Fr => format!("Cache de commande utilisé: label={label}, age_ms={age_ms}"),
        Language::De => format!("Befehls-Cache-Treffer: label={label}, age_ms={age_ms}"),
        Language::Ja => format!("コマンドキャッシュを使用: label={label}, age_ms={age_ms}"),
        Language::En => format!("Command cache hit: label={label}, age_ms={age_ms}"),
    }
}

pub fn chat_round_metrics(
    api_rounds: usize,
    api_duration_ms: u128,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    estimated_cost_usd: f64,
    show_cost: bool,
) -> String {
    match current_language() {
        Language::ZhCn => {
            if show_cost {
                return format!(
                    "本轮指标：请求轮次 {api_rounds}，接口耗时 {api_duration_ms} ms，Token（输入/输出/总计）{prompt_tokens}/{completion_tokens}/{total_tokens}，预估费用 {estimated_cost_usd:.6} USD"
                );
            }
            format!(
                "本轮指标：请求轮次 {api_rounds}，接口耗时 {api_duration_ms} ms，Token（输入/输出/总计）{prompt_tokens}/{completion_tokens}/{total_tokens}"
            )
        }
        Language::ZhTw => {
            if show_cost {
                return format!(
                    "本輪指標：請求輪次 {api_rounds}，介面耗時 {api_duration_ms} ms，Token（輸入/輸出/總計）{prompt_tokens}/{completion_tokens}/{total_tokens}，預估費用 {estimated_cost_usd:.6} USD"
                );
            }
            format!(
                "本輪指標：請求輪次 {api_rounds}，介面耗時 {api_duration_ms} ms，Token（輸入/輸出/總計）{prompt_tokens}/{completion_tokens}/{total_tokens}"
            )
        }
        Language::Fr => {
            if show_cost {
                return format!(
                    "Métriques du tour: cycles {api_rounds}, durée API {api_duration_ms} ms, tokens (entrée/sortie/total) {prompt_tokens}/{completion_tokens}/{total_tokens}, coût estimé {estimated_cost_usd:.6} USD"
                );
            }
            format!(
                "Métriques du tour: cycles {api_rounds}, durée API {api_duration_ms} ms, tokens (entrée/sortie/total) {prompt_tokens}/{completion_tokens}/{total_tokens}"
            )
        }
        Language::De => {
            if show_cost {
                return format!(
                    "Rundenmetriken: Durchläufe {api_rounds}, API-Dauer {api_duration_ms} ms, Tokens (Eingabe/Ausgabe/Gesamt) {prompt_tokens}/{completion_tokens}/{total_tokens}, geschätzte Kosten {estimated_cost_usd:.6} USD"
                );
            }
            format!(
                "Rundenmetriken: Durchläufe {api_rounds}, API-Dauer {api_duration_ms} ms, Tokens (Eingabe/Ausgabe/Gesamt) {prompt_tokens}/{completion_tokens}/{total_tokens}"
            )
        }
        Language::Ja => {
            if show_cost {
                return format!(
                    "ラウンド指標: リクエスト回数 {api_rounds}、API時間 {api_duration_ms} ms、Token（入力/出力/合計）{prompt_tokens}/{completion_tokens}/{total_tokens}、推定コスト {estimated_cost_usd:.6} USD"
                );
            }
            format!(
                "ラウンド指標: リクエスト回数 {api_rounds}、API時間 {api_duration_ms} ms、Token（入力/出力/合計）{prompt_tokens}/{completion_tokens}/{total_tokens}"
            )
        }
        Language::En => {
            if show_cost {
                return format!(
                    "Round metrics: rounds {api_rounds}, API duration {api_duration_ms} ms, tokens (prompt/completion/total) {prompt_tokens}/{completion_tokens}/{total_tokens}, estimated cost {estimated_cost_usd:.6} USD"
                );
            }
            format!(
                "Round metrics: rounds {api_rounds}, API duration {api_duration_ms} ms, tokens (prompt/completion/total) {prompt_tokens}/{completion_tokens}/{total_tokens}"
            )
        }
    }
}

pub fn chat_tool_finished(
    label: &str,
    success: bool,
    exit_code: Option<i32>,
    duration_ms: u128,
    timed_out: bool,
    interrupted: bool,
    blocked: bool,
) -> String {
    let status = if success {
        match current_language() {
            Language::ZhCn => "成功",
            Language::ZhTw => "成功",
            Language::Fr => "succès",
            Language::De => "erfolgreich",
            Language::Ja => "成功",
            Language::En => "success",
        }
    } else {
        match current_language() {
            Language::ZhCn => "失败",
            Language::ZhTw => "失敗",
            Language::Fr => "échec",
            Language::De => "fehlgeschlagen",
            Language::Ja => "失敗",
            Language::En => "failed",
        }
    };
    match current_language() {
        Language::ZhCn => format!(
            "命令完成: label={label}, status={status}, exit={exit_code:?}, duration_ms={duration_ms}, timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::ZhTw => format!(
            "命令完成: label={label}, status={status}, exit={exit_code:?}, duration_ms={duration_ms}, timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::Fr => format!(
            "Commande terminée: label={label}, status={status}, exit={exit_code:?}, duration_ms={duration_ms}, timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::De => format!(
            "Befehl abgeschlossen: label={label}, status={status}, exit={exit_code:?}, duration_ms={duration_ms}, timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::Ja => format!(
            "コマンド完了: label={label}, status={status}, exit={exit_code:?}, duration_ms={duration_ms}, timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::En => format!(
            "Command finished: label={label}, status={status}, exit={exit_code:?}, duration_ms={duration_ms}, timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
    }
}

pub fn chat_goodbye() -> &'static str {
    match current_language() {
        Language::ZhCn => "chat 已结束。",
        Language::ZhTw => "chat 已結束。",
        Language::Fr => "chat terminé.",
        Language::De => "chat beendet.",
        Language::Ja => "chat を終了しました。",
        Language::En => "chat ended.",
    }
}

pub fn command_write_confirm_non_interactive() -> String {
    match current_language() {
        Language::ZhCn => "写命令需要人工确认，但当前环境是非交互模式".to_string(),
        Language::ZhTw => "寫命令需要人工確認，但目前環境為非互動模式".to_string(),
        Language::Fr => {
            "La commande d'écriture requiert une confirmation, mais l'environnement actuel n'est pas interactif".to_string()
        }
        Language::De => {
            "Für den Schreibbefehl ist eine Bestätigung erforderlich, aber die aktuelle Umgebung ist nicht interaktiv".to_string()
        }
        Language::Ja => {
            "書き込みコマンドには確認が必要ですが、現在の環境は非対話モードです".to_string()
        }
        Language::En => {
            "write command confirmation is required, but current environment is non-interactive"
                .to_string()
        }
    }
}

pub fn command_write_denied_by_user() -> String {
    match current_language() {
        Language::ZhCn => "写命令被用户拒绝".to_string(),
        Language::ZhTw => "寫命令被使用者拒絕".to_string(),
        Language::Fr => "Commande d'écriture refusée par l'utilisateur".to_string(),
        Language::De => "Schreibbefehl wurde vom Benutzer abgelehnt".to_string(),
        Language::Ja => "書き込みコマンドはユーザーにより拒否されました".to_string(),
        Language::En => "write command denied by user".to_string(),
    }
}

pub fn command_empty() -> String {
    match current_language() {
        Language::ZhCn => "命令为空".to_string(),
        Language::ZhTw => "命令為空".to_string(),
        Language::Fr => "la commande est vide".to_string(),
        Language::De => "Befehl ist leer".to_string(),
        Language::Ja => "コマンドが空です".to_string(),
        Language::En => "command is empty".to_string(),
    }
}

pub fn dangerous_command_blocked(pattern: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("危险命令已按规则拦截: {pattern}"),
        Language::ZhTw => format!("危險命令已按規則攔截: {pattern}"),
        Language::Fr => format!("Commande dangereuse bloquée par la règle: {pattern}"),
        Language::De => format!("Gefährlicher Befehl wurde durch Regel blockiert: {pattern}"),
        Language::Ja => format!("危険なコマンドはルールによりブロックされました: {pattern}"),
        Language::En => format!("dangerous command blocked by rule: {pattern}"),
    }
}

pub fn command_blocked_by_deny_pattern(pattern: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("命令命中拒绝策略，已拦截: {pattern}"),
        Language::ZhTw => format!("命令命中拒絕策略，已攔截: {pattern}"),
        Language::Fr => format!("Commande bloquée par la politique deny: {pattern}"),
        Language::De => format!("Befehl durch deny-Richtlinie blockiert: {pattern}"),
        Language::Ja => format!("deny ポリシーに一致したためコマンドをブロックしました: {pattern}"),
        Language::En => format!("command blocked by deny policy: {pattern}"),
    }
}

pub fn command_blocked_by_allow_policy() -> String {
    match current_language() {
        Language::ZhCn => "命令不在允许策略范围内，已拦截".to_string(),
        Language::ZhTw => "命令不在允許策略範圍內，已攔截".to_string(),
        Language::Fr => "Commande bloquée: hors politique allow".to_string(),
        Language::De => "Befehl blockiert: nicht von allow-Richtlinie abgedeckt".to_string(),
        Language::Ja => "allow ポリシー対象外のためコマンドをブロックしました".to_string(),
        Language::En => "command blocked: not allowed by allow policy".to_string(),
    }
}

fn error_prefix_config() -> &'static str {
    match current_language() {
        Language::ZhCn => "配置错误",
        Language::ZhTw => "配置錯誤",
        Language::Fr => "Erreur de configuration",
        Language::De => "Konfigurationsfehler",
        Language::Ja => "設定エラー",
        Language::En => "Configuration error",
    }
}

fn error_prefix_permission() -> &'static str {
    match current_language() {
        Language::ZhCn => "权限错误",
        Language::ZhTw => "權限錯誤",
        Language::Fr => "Erreur d'autorisation",
        Language::De => "Berechtigungsfehler",
        Language::Ja => "権限エラー",
        Language::En => "Permission error",
    }
}

fn error_prefix_ai() -> &'static str {
    match current_language() {
        Language::ZhCn => "AI 错误",
        Language::ZhTw => "AI 錯誤",
        Language::Fr => "Erreur IA",
        Language::De => "KI-Fehler",
        Language::Ja => "AI エラー",
        Language::En => "AI error",
    }
}

fn error_prefix_command() -> &'static str {
    match current_language() {
        Language::ZhCn => "命令执行错误",
        Language::ZhTw => "命令執行錯誤",
        Language::Fr => "Erreur d'exécution de commande",
        Language::De => "Befehlsausführungsfehler",
        Language::Ja => "コマンド実行エラー",
        Language::En => "Command error",
    }
}

fn error_prefix_runtime() -> &'static str {
    match current_language() {
        Language::ZhCn => "运行时错误",
        Language::ZhTw => "執行階段錯誤",
        Language::Fr => "Erreur d'exécution",
        Language::De => "Laufzeitfehler",
        Language::Ja => "実行時エラー",
        Language::En => "Runtime error",
    }
}

fn localize_detail(detail: &str) -> String {
    match detail {
        "ai.base-url is required" => match current_language() {
            Language::ZhCn => "缺少必填配置 ai.base-url".to_string(),
            Language::ZhTw => "缺少必填設定 ai.base-url".to_string(),
            Language::Fr => "Le champ obligatoire ai.base-url est manquant".to_string(),
            Language::De => "Pflichtfeld ai.base-url fehlt".to_string(),
            Language::Ja => "必須設定 ai.base-url が不足しています".to_string(),
            Language::En => detail.to_string(),
        },
        "ai.token is required" => match current_language() {
            Language::ZhCn => "缺少必填配置 ai.token".to_string(),
            Language::ZhTw => "缺少必填設定 ai.token".to_string(),
            Language::Fr => "Le champ obligatoire ai.token est manquant".to_string(),
            Language::De => "Pflichtfeld ai.token fehlt".to_string(),
            Language::Ja => "必須設定 ai.token が不足しています".to_string(),
            Language::En => detail.to_string(),
        },
        "ai.model is required" => match current_language() {
            Language::ZhCn => "缺少必填配置 ai.model".to_string(),
            Language::ZhTw => "缺少必填設定 ai.model".to_string(),
            Language::Fr => "Le champ obligatoire ai.model est manquant".to_string(),
            Language::De => "Pflichtfeld ai.model fehlt".to_string(),
            Language::Ja => "必須設定 ai.model が不足しています".to_string(),
            Language::En => detail.to_string(),
        },
        "cmd.command-timeout-seconds must be greater than 0" => match current_language() {
            Language::ZhCn => "cmd.command-timeout-seconds 必须大于 0".to_string(),
            Language::ZhTw => "cmd.command-timeout-seconds 必須大於 0".to_string(),
            Language::Fr => "cmd.command-timeout-seconds doit être supérieur à 0".to_string(),
            Language::De => "cmd.command-timeout-seconds muss größer als 0 sein".to_string(),
            Language::Ja => "cmd.command-timeout-seconds は 0 より大きい必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "cmd.command-timeout-kill-after-seconds must be greater than 0" => {
            match current_language() {
                Language::ZhCn => "cmd.command-timeout-kill-after-seconds 必须大于 0".to_string(),
                Language::ZhTw => "cmd.command-timeout-kill-after-seconds 必須大於 0".to_string(),
                Language::Fr => {
                    "cmd.command-timeout-kill-after-seconds doit être supérieur à 0".to_string()
                }
                Language::De => {
                    "cmd.command-timeout-kill-after-seconds muss größer als 0 sein".to_string()
                }
                Language::Ja => {
                    "cmd.command-timeout-kill-after-seconds は 0 より大きい必要があります"
                        .to_string()
                }
                Language::En => detail.to_string(),
            }
        }
        "cmd.command-output-max-bytes must be >= 1024" => match current_language() {
            Language::ZhCn => "cmd.command-output-max-bytes 必须大于等于 1024".to_string(),
            Language::ZhTw => "cmd.command-output-max-bytes 必須大於等於 1024".to_string(),
            Language::Fr => "cmd.command-output-max-bytes doit être >= 1024".to_string(),
            Language::De => "cmd.command-output-max-bytes muss >= 1024 sein".to_string(),
            Language::Ja => "cmd.command-output-max-bytes は 1024 以上である必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "cmd.write-cmd-confirm-mode must be one of: deny, edit, allow-once, allow-session" => {
            match current_language() {
                Language::ZhCn => "cmd.write-cmd-confirm-mode 取值必须是 deny、edit、allow-once、allow-session".to_string(),
                Language::ZhTw => "cmd.write-cmd-confirm-mode 取值必須是 deny、edit、allow-once、allow-session".to_string(),
                Language::Fr => "cmd.write-cmd-confirm-mode doit être deny, edit, allow-once ou allow-session".to_string(),
                Language::De => "cmd.write-cmd-confirm-mode muss deny, edit, allow-once oder allow-session sein".to_string(),
                Language::Ja => "cmd.write-cmd-confirm-mode は deny / edit / allow-once / allow-session のいずれかである必要があります".to_string(),
                Language::En => detail.to_string(),
            }
        }
        "ai input/output price cannot be negative" => match current_language() {
            Language::ZhCn => "ai 输入/输出价格不能为负数".to_string(),
            Language::ZhTw => "ai 輸入/輸出價格不能為負數".to_string(),
            Language::Fr => "Le coût d'entrée/sortie AI ne peut pas être négatif".to_string(),
            Language::De => "AI Ein-/Ausgabepreis darf nicht negativ sein".to_string(),
            Language::Ja => "AI の入力/出力単価は負の値にできません".to_string(),
            Language::En => detail.to_string(),
        },
        "ai.chat context percent must be in 1..=100" => match current_language() {
            Language::ZhCn => "ai.chat 的上下文百分比必须在 1..=100 范围内".to_string(),
            Language::ZhTw => "ai.chat 的上下文百分比必須在 1..=100 範圍內".to_string(),
            Language::Fr => "Le pourcentage de contexte ai.chat doit être dans 1..=100".to_string(),
            Language::De => "ai.chat Kontext-Prozent muss im Bereich 1..=100 liegen".to_string(),
            Language::Ja => "ai.chat のコンテキスト割合は 1..=100 の範囲である必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "ai.chat context-warn-percent cannot exceed context-critical-percent" => {
            match current_language() {
                Language::ZhCn => "ai.chat.context-warn-percent 不能大于 context-critical-percent".to_string(),
                Language::ZhTw => "ai.chat.context-warn-percent 不能大於 context-critical-percent".to_string(),
                Language::Fr => "ai.chat.context-warn-percent ne peut pas dépasser context-critical-percent".to_string(),
                Language::De => "ai.chat.context-warn-percent darf context-critical-percent nicht überschreiten".to_string(),
                Language::Ja => "ai.chat.context-warn-percent は context-critical-percent を超えられません".to_string(),
                Language::En => detail.to_string(),
            }
        }
        "session.recent_messages must be greater than 0" => match current_language() {
            Language::ZhCn => "session.recent_messages 必须大于 0".to_string(),
            Language::ZhTw => "session.recent_messages 必須大於 0".to_string(),
            Language::Fr => "session.recent_messages doit être supérieur à 0".to_string(),
            Language::De => "session.recent_messages muss größer als 0 sein".to_string(),
            Language::Ja => "session.recent_messages は 0 より大きい必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "session.max_messages must be greater than 0" => match current_language() {
            Language::ZhCn => "session.max_messages 必须大于 0".to_string(),
            Language::ZhTw => "session.max_messages 必須大於 0".to_string(),
            Language::Fr => "session.max_messages doit être supérieur à 0".to_string(),
            Language::De => "session.max_messages muss größer als 0 sein".to_string(),
            Language::Ja => "session.max_messages は 0 より大きい必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "session.recent_messages cannot exceed session.max_messages" => match current_language() {
            Language::ZhCn => "session.recent_messages 不能大于 session.max_messages".to_string(),
            Language::ZhTw => "session.recent_messages 不能大於 session.max_messages".to_string(),
            Language::Fr => {
                "session.recent_messages ne peut pas dépasser session.max_messages".to_string()
            }
            Language::De => {
                "session.recent_messages darf session.max_messages nicht überschreiten".to_string()
            }
            Language::Ja => {
                "session.recent_messages は session.max_messages を超えることはできません"
                    .to_string()
            }
            Language::En => detail.to_string(),
        },
        "administrator privileges are required on Windows" => match current_language() {
            Language::ZhCn => "Windows 需要管理员权限".to_string(),
            Language::ZhTw => "Windows 需要管理員權限".to_string(),
            Language::Fr => "Des privilèges administrateur sont requis sous Windows".to_string(),
            Language::De => "Unter Windows sind Administratorrechte erforderlich".to_string(),
            Language::Ja => "Windows では管理者権限が必要です".to_string(),
            Language::En => detail.to_string(),
        },
        "root privileges are required on Linux/macOS" => match current_language() {
            Language::ZhCn => "Linux/macOS 需要 root 权限".to_string(),
            Language::ZhTw => "Linux/macOS 需要 root 權限".to_string(),
            Language::Fr => "Les privilèges root sont requis sur Linux/macOS".to_string(),
            Language::De => "Unter Linux/macOS sind Root-Rechte erforderlich".to_string(),
            Language::Ja => "Linux/macOS では root 権限が必要です".to_string(),
            Language::En => detail.to_string(),
        },
        "write command confirmation is required, but current environment is non-interactive" => {
            command_write_confirm_non_interactive()
        }
        "command is empty" => command_empty(),
        _ if detail.starts_with("failed to read config ") => {
            let rest = detail
                .strip_prefix("failed to read config ")
                .unwrap_or(detail);
            match current_language() {
                Language::ZhCn => format!("读取配置文件失败 {}", localize_common_os_error(rest)),
                Language::ZhTw => format!("讀取設定檔失敗 {}", localize_common_os_error(rest)),
                Language::Fr => format!(
                    "Échec de lecture du fichier de configuration {}",
                    localize_common_os_error(rest)
                ),
                Language::De => format!(
                    "Konfigurationsdatei konnte nicht gelesen werden {}",
                    localize_common_os_error(rest)
                ),
                Language::Ja => format!(
                    "設定ファイルの読み込みに失敗しました {}",
                    localize_common_os_error(rest)
                ),
                Language::En => detail.to_string(),
            }
        }
        _ if detail.starts_with("failed to parse config ") => {
            let rest = detail
                .strip_prefix("failed to parse config ")
                .unwrap_or(detail);
            match current_language() {
                Language::ZhCn => format!("解析配置文件失败 {}", rest),
                Language::ZhTw => format!("解析設定檔失敗 {}", rest),
                Language::Fr => format!("Échec de l'analyse du fichier de configuration {}", rest),
                Language::De => format!("Konfigurationsdatei konnte nicht geparst werden {}", rest),
                Language::Ja => format!("設定ファイルの解析に失敗しました {}", rest),
                Language::En => detail.to_string(),
            }
        }
        _ if detail.starts_with("session.max_messages must be <= ") => {
            let max = detail
                .strip_prefix("session.max_messages must be <= ")
                .unwrap_or("80");
            match current_language() {
                Language::ZhCn => format!("session.max_messages 必须小于等于 {max}"),
                Language::ZhTw => format!("session.max_messages 必須小於等於 {max}"),
                Language::Fr => format!("session.max_messages doit être <= {max}"),
                Language::De => format!("session.max_messages muss <= {max} sein"),
                Language::Ja => format!("session.max_messages は {max} 以下である必要があります"),
                Language::En => detail.to_string(),
            }
        }
        _ if detail.starts_with("error: unexpected argument ") => match current_language() {
            Language::ZhCn => "命令参数错误：存在未预期参数，请使用 --help 查看正确用法".to_string(),
            Language::ZhTw => "命令參數錯誤：存在未預期參數，請使用 --help 查看正確用法".to_string(),
            Language::Fr => {
                "Argument de commande invalide : utilisez --help pour la syntaxe correcte"
                    .to_string()
            }
            Language::De => {
                "Ungültiges Befehlsargument: Verwenden Sie --help für die korrekte Syntax"
                    .to_string()
            }
            Language::Ja => {
                "コマンド引数エラー: 想定外の引数です。正しい使い方は --help を参照してください"
                    .to_string()
            }
            Language::En => {
                "invalid command arguments; use --help for correct usage".to_string()
            }
        },
        _ if detail.starts_with("error: invalid value ") => match current_language() {
            Language::ZhCn => "参数值无效，请使用 --help 查看可用取值".to_string(),
            Language::ZhTw => "參數值無效，請使用 --help 查看可用取值".to_string(),
            Language::Fr => "Valeur d'argument invalide, utilisez --help pour les valeurs autorisées"
                .to_string(),
            Language::De => {
                "Ungültiger Argumentwert, verwenden Sie --help für zulässige Werte".to_string()
            }
            Language::Ja => {
                "引数の値が無効です。利用可能な値は --help を参照してください".to_string()
            }
            Language::En => "invalid argument value; use --help for allowed values".to_string(),
        },
        _ if detail.starts_with("error: the following required arguments were not provided") => {
            match current_language() {
                Language::ZhCn => "缺少必填参数，请使用 --help 查看完整用法".to_string(),
                Language::ZhTw => "缺少必填參數，請使用 --help 查看完整用法".to_string(),
                Language::Fr => {
                    "Arguments requis manquants, utilisez --help pour l'utilisation complète"
                        .to_string()
                }
                Language::De => {
                    "Erforderliche Argumente fehlen, verwenden Sie --help für die vollständige Nutzung"
                        .to_string()
                }
                Language::Ja => "必須引数が不足しています。完全な使い方は --help を参照してください".to_string(),
                Language::En => {
                    "required arguments are missing; use --help for full usage".to_string()
                }
            }
        }
        _ => detail.to_string(),
    }
}

fn detect_language_by_system_command() -> Option<Language> {
    #[cfg(target_os = "macos")]
    {
        if let Some(language) = detect_by_command("defaults", &["read", "-g", "AppleLocale"]) {
            return Some(language);
        }
    }

    #[cfg(windows)]
    {
        if let Some(language) = detect_by_command(
            "powershell",
            &["-NoProfile", "-Command", "(Get-WinSystemLocale).Name"],
        ) {
            return Some(language);
        }
    }

    detect_by_command("locale", &[])
}

fn detect_by_command(program: &str, args: &[&str]) -> Option<Language> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    for line in text.lines() {
        if let Some((_, value)) = line.split_once('=')
            && let Some(language) = parse_language(value.trim_matches('"'))
        {
            return Some(language);
        }
        if let Some(language) = parse_language(line.trim_matches('"').trim()) {
            return Some(language);
        }
    }
    None
}

fn localize_common_os_error(input: &str) -> String {
    match current_language() {
        Language::ZhCn => input.replace(
            "No such file or directory (os error 2)",
            "没有那个文件或目录 (os error 2)",
        ),
        Language::ZhTw => input.replace(
            "No such file or directory (os error 2)",
            "找不到該檔案或目錄 (os error 2)",
        ),
        Language::Fr => input.replace(
            "No such file or directory (os error 2)",
            "Aucun fichier ou dossier de ce type (os error 2)",
        ),
        Language::De => input.replace(
            "No such file or directory (os error 2)",
            "Datei oder Verzeichnis nicht gefunden (os error 2)",
        ),
        Language::Ja => input.replace(
            "No such file or directory (os error 2)",
            "そのようなファイルまたはディレクトリはありません (os error 2)",
        ),
        Language::En => input.to_string(),
    }
}
