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

pub fn human_count_u128(value: u128) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    let len = text.len();
    for (idx, ch) in text.chars().enumerate() {
        out.push(ch);
        let left = len.saturating_sub(idx + 1);
        if left > 0 && left.is_multiple_of(3) {
            out.push(',');
        }
    }
    out
}

pub fn human_count_u64(value: u64) -> String {
    human_count_u128(value as u128)
}

pub fn human_duration_ms(ms: u128) -> String {
    if ms < 1_000 {
        return format!("{ms}ms");
    }
    let total_seconds = ms / 1_000;
    let remain_ms = ms % 1_000;
    if total_seconds < 60 {
        return format!("{total_seconds}.{remain_ms:03}s");
    }
    let seconds = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    if total_minutes < 60 {
        return format!("{total_minutes}m{seconds:02}s");
    }
    let minutes = total_minutes % 60;
    let total_hours = total_minutes / 60;
    if total_hours < 24 {
        return format!("{total_hours}h{minutes:02}m{seconds:02}s");
    }
    let hours = total_hours % 24;
    let days = total_hours / 24;
    format!("{days}d{hours:02}h{minutes:02}m{seconds:02}s")
}

pub fn human_bytes(bytes: u128) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const TIB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    let value = bytes as f64;
    if value < KIB {
        return format!("{bytes} B");
    }
    if value < MIB {
        return format!("{:.2} KiB", value / KIB);
    }
    if value < GIB {
        return format!("{:.2} MiB", value / MIB);
    }
    if value < TIB {
        return format!("{:.2} GiB", value / GIB);
    }
    format!("{:.2} TiB", value / TIB)
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

pub fn preflight_notice_start() -> &'static str {
    match current_language() {
        Language::ZhCn => "正在执行启动前检查...",
        Language::ZhTw => "正在執行啟動前檢查...",
        Language::Fr => "Vérifications préalables au démarrage en cours...",
        Language::De => "Vorabprüfungen beim Start werden ausgeführt...",
        Language::Ja => "起動前チェックを実行中です...",
        Language::En => "Running startup preflight checks...",
    }
}

pub fn preflight_notice_config_check() -> &'static str {
    match current_language() {
        Language::ZhCn => "校验配置完整性...",
        Language::ZhTw => "校驗配置完整性...",
        Language::Fr => "Validation de l'intégrité de la configuration...",
        Language::De => "Konfigurationsintegrität wird geprüft...",
        Language::Ja => "設定の整合性を検証中...",
        Language::En => "Validating configuration integrity...",
    }
}

pub fn preflight_notice_permission_check() -> &'static str {
    match current_language() {
        Language::ZhCn => "校验运行权限...",
        Language::ZhTw => "校驗執行權限...",
        Language::Fr => "Vérification des permissions d'exécution...",
        Language::De => "Ausführungsberechtigungen werden geprüft...",
        Language::Ja => "実行権限を確認中...",
        Language::En => "Checking runtime permissions...",
    }
}

pub fn preflight_notice_ai_check() -> &'static str {
    match current_language() {
        Language::ZhCn => "校验 AI 连通性（网络不稳定时可能需要更久）...",
        Language::ZhTw => "校驗 AI 連通性（網路不穩時可能較久）...",
        Language::Fr => {
            "Vérification de la connectivité IA (peut prendre plus de temps si le réseau est instable)..."
        }
        Language::De => {
            "KI-Konnektivität wird geprüft (bei instabilem Netzwerk kann es länger dauern)..."
        }
        Language::Ja => {
            "AI 接続性を確認中です（ネットワークが不安定な場合は時間がかかることがあります）..."
        }
        Language::En => "Checking AI connectivity (may take longer on unstable networks)...",
    }
}

pub fn preflight_notice_ai_check_skipped() -> &'static str {
    match current_language() {
        Language::ZhCn => "已按配置跳过 AI 连通性检测（ai.connectivity-check=false）",
        Language::ZhTw => "已依配置略過 AI 連通性檢測（ai.connectivity-check=false）",
        Language::Fr => {
            "Vérification de connectivité IA ignorée par configuration (ai.connectivity-check=false)"
        }
        Language::De => {
            "KI-Konnektivitätsprüfung per Konfiguration übersprungen (ai.connectivity-check=false)"
        }
        Language::Ja => {
            "設定により AI 接続性チェックをスキップしました（ai.connectivity-check=false）"
        }
        Language::En => "AI connectivity check skipped by config (ai.connectivity-check=false)",
    }
}

pub fn preflight_notice_done(elapsed: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("启动前检查完成，耗时 {elapsed}"),
        Language::ZhTw => format!("啟動前檢查完成，耗時 {elapsed}"),
        Language::Fr => format!("Pré-vérification terminée, durée {elapsed}"),
        Language::De => format!("Vorabprüfung abgeschlossen, Dauer {elapsed}"),
        Language::Ja => format!("起動前チェック完了、所要時間 {elapsed}"),
        Language::En => format!("Startup preflight completed in {elapsed}"),
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

#[allow(clippy::too_many_arguments)]
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

pub fn chat_tag_warn() -> &'static str {
    match current_language() {
        Language::ZhCn => "[警告]",
        Language::ZhTw => "[警告]",
        Language::Fr => "[alerte]",
        Language::De => "[warnung]",
        Language::Ja => "[警告]",
        Language::En => "[warn]",
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

pub fn chat_tag_mcp() -> &'static str {
    match current_language() {
        Language::ZhCn => "[MCP]",
        Language::ZhTw => "[MCP]",
        Language::Fr => "[mcp]",
        Language::De => "[mcp]",
        Language::Ja => "[MCP]",
        Language::En => "[mcp]",
    }
}

pub fn chat_tag_skill() -> &'static str {
    match current_language() {
        Language::ZhCn => "[技能]",
        Language::ZhTw => "[技能]",
        Language::Fr => "[skill]",
        Language::De => "[skill]",
        Language::Ja => "[skill]",
        Language::En => "[skill]",
    }
}

pub fn chat_tag_profile() -> &'static str {
    match current_language() {
        Language::ZhCn => "[环境画像]",
        Language::ZhTw => "[環境画像]",
        Language::Fr => "[profil]",
        Language::De => "[profil]",
        Language::Ja => "[プロファイル]",
        Language::En => "[profile]",
    }
}

pub fn chat_tag_compress() -> &'static str {
    match current_language() {
        Language::ZhCn => "[压缩]",
        Language::ZhTw => "[壓縮]",
        Language::Fr => "[compress]",
        Language::De => "[compress]",
        Language::Ja => "[圧縮]",
        Language::En => "[compress]",
    }
}

pub fn chat_tag_debug_info() -> &'static str {
    match current_language() {
        Language::ZhCn => "[调试-info]",
        Language::ZhTw => "[除錯-info]",
        Language::Fr => "[debug-info]",
        Language::De => "[debug-info]",
        Language::Ja => "[デバッグ-info]",
        Language::En => "[debug-info]",
    }
}

pub fn chat_tag_debug_debug() -> &'static str {
    match current_language() {
        Language::ZhCn => "[调试-debug]",
        Language::ZhTw => "[除錯-debug]",
        Language::Fr => "[debug-debug]",
        Language::De => "[debug-debug]",
        Language::Ja => "[デバッグ-debug]",
        Language::En => "[debug-debug]",
    }
}

pub fn chat_tag_debug_warn() -> &'static str {
    match current_language() {
        Language::ZhCn => "[调试-warn]",
        Language::ZhTw => "[除錯-warn]",
        Language::Fr => "[debug-warn]",
        Language::De => "[debug-warn]",
        Language::Ja => "[デバッグ-warn]",
        Language::En => "[debug-warn]",
    }
}

pub fn chat_tag_debug_error() -> &'static str {
    match current_language() {
        Language::ZhCn => "[调试-error]",
        Language::ZhTw => "[除錯-error]",
        Language::Fr => "[debug-error]",
        Language::De => "[debug-error]",
        Language::Ja => "[デバッグ-error]",
        Language::En => "[debug-error]",
    }
}

pub fn chat_tag_model_price() -> &'static str {
    match current_language() {
        Language::ZhCn => "[查询模型价格]",
        Language::ZhTw => "[查詢模型價格]",
        Language::Fr => "[price-check]",
        Language::De => "[price-check]",
        Language::Ja => "[モデル価格確認]",
        Language::En => "[price-check]",
    }
}

pub fn chat_model_price_check_started(model: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("正在准备当前模型单价信息，model={model}"),
        Language::ZhTw => format!("正在準備目前模型單價資訊，model={model}"),
        Language::Fr => format!("Préparation du tarif actuel du modèle, model={model}"),
        Language::De => format!("Aktuelle Modellpreise werden vorbereitet, model={model}"),
        Language::Ja => format!("現在のモデル単価情報を準備中です。model={model}"),
        Language::En => format!("Preparing current model pricing, model={model}"),
    }
}

pub fn chat_model_price_check_skipped() -> &'static str {
    match current_language() {
        Language::ZhCn => "已按配置跳过模型询价（ai.chat.skip-model-price-check=true）",
        Language::ZhTw => "已依設定略過模型詢價（ai.chat.skip-model-price-check=true）",
        Language::Fr => "Vérification du prix du modèle ignorée par configuration (ai.chat.skip-model-price-check=true)",
        Language::De => "Modellpreisprüfung per Konfiguration übersprungen (ai.chat.skip-model-price-check=true)",
        Language::Ja => "設定によりモデル価格確認をスキップしました（ai.chat.skip-model-price-check=true）",
        Language::En => "Model price check skipped by config (ai.chat.skip-model-price-check=true)",
    }
}

pub fn chat_model_price_check_configured(input: f64, output: f64) -> String {
    match current_language() {
        Language::ZhCn => format!("已使用配置单价：输入 {input:.4} / 输出 {output:.4} USD/百万 tokens"),
        Language::ZhTw => format!("已使用設定單價：輸入 {input:.4} / 輸出 {output:.4} USD/百萬 tokens"),
        Language::Fr => format!("Tarif configuré utilisé : entrée {input:.4} / sortie {output:.4} USD par million de tokens"),
        Language::De => format!("Konfigurierter Preis verwendet: Eingabe {input:.4} / Ausgabe {output:.4} USD pro 1 Mio. Tokens"),
        Language::Ja => format!("設定済み単価を使用します: 入力 {input:.4} / 出力 {output:.4} USD / 100万 tokens"),
        Language::En => format!("Using configured pricing: input {input:.4} / output {output:.4} USD per 1M tokens"),
    }
}

pub fn chat_model_price_check_cached(input: f64, output: f64) -> String {
    match current_language() {
        Language::ZhCn => format!("已命中本地模型价格缓存：输入 {input:.4} / 输出 {output:.4} USD/百万 tokens"),
        Language::ZhTw => format!("已命中本地模型價格快取：輸入 {input:.4} / 輸出 {output:.4} USD/百萬 tokens"),
        Language::Fr => format!("Cache local du prix du modèle utilisé : entrée {input:.4} / sortie {output:.4} USD par million de tokens"),
        Language::De => format!("Lokaler Modellpreis-Cache verwendet: Eingabe {input:.4} / Ausgabe {output:.4} USD pro 1 Mio. Tokens"),
        Language::Ja => format!("ローカルのモデル価格キャッシュを使用しました: 入力 {input:.4} / 出力 {output:.4} USD / 100万 tokens"),
        Language::En => format!("Using local model pricing cache: input {input:.4} / output {output:.4} USD per 1M tokens"),
    }
}

pub fn chat_model_price_check_probed(input: f64, output: f64) -> String {
    match current_language() {
        Language::ZhCn => format!("模型询价成功，并已刷新本地缓存：输入 {input:.4} / 输出 {output:.4} USD/百万 tokens"),
        Language::ZhTw => format!("模型詢價成功，並已刷新本地快取：輸入 {input:.4} / 輸出 {output:.4} USD/百萬 tokens"),
        Language::Fr => format!("Tarif du modèle obtenu et cache local rafraîchi : entrée {input:.4} / sortie {output:.4} USD par million de tokens"),
        Language::De => format!("Modellpreis ermittelt und lokaler Cache aktualisiert: Eingabe {input:.4} / Ausgabe {output:.4} USD pro 1 Mio. Tokens"),
        Language::Ja => format!("モデル単価を取得し、ローカルキャッシュを更新しました: 入力 {input:.4} / 出力 {output:.4} USD / 100万 tokens"),
        Language::En => format!("Model pricing resolved and local cache refreshed: input {input:.4} / output {output:.4} USD per 1M tokens"),
    }
}

pub fn chat_model_price_check_builtin(input: f64, output: f64) -> String {
    match current_language() {
        Language::ZhCn => format!("未获得在线模型单价，已回退到内置单价：输入 {input:.4} / 输出 {output:.4} USD/百万 tokens"),
        Language::ZhTw => format!("未取得線上模型單價，已回退到內建單價：輸入 {input:.4} / 輸出 {output:.4} USD/百萬 tokens"),
        Language::Fr => format!("Tarif en ligne indisponible, repli sur le tarif intégré : entrée {input:.4} / sortie {output:.4} USD par million de tokens"),
        Language::De => format!("Kein Online-Preis verfügbar, integrierter Preis verwendet: Eingabe {input:.4} / Ausgabe {output:.4} USD pro 1 Mio. Tokens"),
        Language::Ja => format!("オンライン価格を取得できなかったため内蔵単価にフォールバックしました: 入力 {input:.4} / 出力 {output:.4} USD / 100万 tokens"),
        Language::En => format!("Online model pricing unavailable, falling back to built-in pricing: input {input:.4} / output {output:.4} USD per 1M tokens"),
    }
}

pub fn chat_model_price_check_unavailable() -> &'static str {
    match current_language() {
        Language::ZhCn => "模型询价失败，且无可用内置单价；本次费用估算可能显示 N/A",
        Language::ZhTw => "模型詢價失敗，且無可用內建單價；本次費用估算可能顯示 N/A",
        Language::Fr => "Échec de la détection du tarif du modèle et aucun tarif intégré n'est disponible ; le coût estimé peut afficher N/A",
        Language::De => "Modellpreis konnte nicht ermittelt werden und es ist kein integrierter Preis verfügbar; die Kostenschätzung kann N/A anzeigen",
        Language::Ja => "モデル価格の取得に失敗し、利用可能な内蔵価格もありません。今回のコスト見積りは N/A になる可能性があります",
        Language::En => "Model pricing probe failed and no built-in pricing is available; estimated cost may show N/A",
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

#[allow(clippy::too_many_arguments)]
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
    let message_count_fmt = human_count_u128(message_count as u128);
    let summary_len_fmt = human_count_u128(summary_len as u128);
    let recent_limit_fmt = human_count_u128(recent_limit as u128);
    let max_limit_fmt = human_count_u128(max_limit as u128);
    let skills_count_fmt = human_count_u128(skills_count as u128);
    let tool_flags = format!(
        "tool={}, tool_ok={}, tool_err={}, tool_timeout={}, tips={}, stream_output={}, output_multilines={}, cmd_run_timout={}s, max_tool_rounds={}, max_total_tool_calls={}",
        chat_cfg.show_tool,
        chat_cfg.show_tool_ok,
        chat_cfg.show_tool_err,
        chat_cfg.show_tool_timeout,
        chat_cfg.show_tips,
        chat_cfg.stream_output,
        chat_cfg.output_multilines,
        chat_cfg.cmd_run_timout,
        human_count_u128(chat_cfg.max_tool_rounds as u128),
        human_count_u128(chat_cfg.max_total_tool_calls as u128),
    );
    match current_language() {
        Language::ZhCn => format!(
            "chat 模式已启动\n会话: {session_id}\n会话文件: {}\n模型: {model}\n系统: {os_name}\n消息: {message_count_fmt} 条, 摘要 {summary_len_fmt} 字符\n上下文窗口: recent={recent_limit_fmt}, max={max_limit_fmt}\n能力: skills={skills_count_fmt}, mcp={mcp_summary}\n工具消息显示: {tool_flags}",
            session_file.display()
        ),
        Language::ZhTw => format!(
            "chat 模式已啟動\n會話: {session_id}\n會話檔案: {}\n模型: {model}\n系統: {os_name}\n訊息: {message_count_fmt} 筆, 摘要 {summary_len_fmt} 字元\n上下文視窗: recent={recent_limit_fmt}, max={max_limit_fmt}\n能力: skills={skills_count_fmt}, mcp={mcp_summary}\n工具訊息顯示: {tool_flags}",
            session_file.display()
        ),
        Language::Fr => format!(
            "Mode chat démarré\nSession: {session_id}\nFichier de session: {}\nModèle: {model}\nSystème: {os_name}\nMessages: {message_count_fmt}, résumé {summary_len_fmt} caractères\nFenêtre de contexte: recent={recent_limit_fmt}, max={max_limit_fmt}\nCapacités: skills={skills_count_fmt}, mcp={mcp_summary}\nAffichage outils: {tool_flags}",
            session_file.display()
        ),
        Language::De => format!(
            "Chat-Modus gestartet\nSitzung: {session_id}\nSitzungsdatei: {}\nModell: {model}\nSystem: {os_name}\nNachrichten: {message_count_fmt}, Zusammenfassung {summary_len_fmt} Zeichen\nKontextfenster: recent={recent_limit_fmt}, max={max_limit_fmt}\nFähigkeiten: skills={skills_count_fmt}, mcp={mcp_summary}\nTool-Anzeige: {tool_flags}",
            session_file.display()
        ),
        Language::Ja => format!(
            "chat モードを開始しました\nセッション: {session_id}\nセッションファイル: {}\nモデル: {model}\nシステム: {os_name}\nメッセージ: {message_count_fmt} 件, 要約 {summary_len_fmt} 文字\nコンテキスト枠: recent={recent_limit_fmt}, max={max_limit_fmt}\n機能: skills={skills_count_fmt}, mcp={mcp_summary}\nツール表示: {tool_flags}",
            session_file.display()
        ),
        Language::En => format!(
            "chat mode started\nsession: {session_id}\nsession file: {}\nmodel: {model}\nos: {os_name}\nmessages: {message_count_fmt}, summary {summary_len_fmt} chars\ncontext window: recent={recent_limit_fmt}, max={max_limit_fmt}\ncapability: skills={skills_count_fmt}, mcp={mcp_summary}\ntool event visibility: {tool_flags}",
            session_file.display()
        ),
    }
}

pub fn chat_hint() -> &'static str {
    match current_language() {
        Language::ZhCn => {
            "输入问题开始对话。命令: /help, /stats, /list, /change <id|name>, /name <new-name>, /new, /clear, /exit"
        }
        Language::ZhTw => {
            "輸入問題開始對話。命令: /help, /stats, /list, /change <id|name>, /name <new-name>, /new, /clear, /exit"
        }
        Language::Fr => {
            "Saisissez votre question pour commencer. Commandes: /help, /stats, /list, /change <id|name>, /name <new-name>, /new, /clear, /exit"
        }
        Language::De => {
            "Geben Sie eine Frage ein, um zu starten. Befehle: /help, /stats, /list, /change <id|name>, /name <new-name>, /new, /clear, /exit"
        }
        Language::Ja => {
            "質問を入力して開始します。コマンド: /help, /stats, /list, /change <id|name>, /name <new-name>, /new, /clear, /exit"
        }
        Language::En => {
            "Type a question to start. Commands: /help, /stats, /list, /change <id|name>, /name <new-name>, /new, /clear, /exit"
        }
    }
}

pub fn chat_help_text() -> &'static str {
    match current_language() {
        Language::ZhCn => {
            "/help 显示帮助, /stats 查看上下文统计, /list 列出会话, /change <id|name> 切换会话, /name <new-name> 重命名当前会话, /new 新建会话, /clear 清屏(不清历史), /exit 退出 chat"
        }
        Language::ZhTw => {
            "/help 顯示說明, /stats 查看上下文統計, /list 列出會話, /change <id|name> 切換會話, /name <new-name> 重新命名目前會話, /new 建立新會話, /clear 清屏(不清歷史), /exit 離開 chat"
        }
        Language::Fr => {
            "/help aide, /stats statistiques du contexte, /list lister les sessions, /change <id|name> changer de session, /name <new-name> renommer la session actuelle, /new nouvelle session, /clear effacer l'écran (garde l'historique), /exit quitter chat"
        }
        Language::De => {
            "/help Hilfe, /stats Kontextstatistik, /list Sitzungen auflisten, /change <id|name> Sitzung wechseln, /name <new-name> aktuelle Sitzung umbenennen, /new neue Sitzung, /clear Bildschirm leeren (Verlauf bleibt), /exit chat beenden"
        }
        Language::Ja => {
            "/help ヘルプ, /stats コンテキスト統計, /list セッション一覧, /change <id|name> セッション切替, /name <new-name> 現在セッション名変更, /new 新規セッション, /clear 画面クリア(履歴保持), /exit chat 終了"
        }
        Language::En => {
            "/help help, /stats context stats, /list list sessions, /change <id|name> switch session, /name <new-name> rename current session, /new new session, /clear clear screen (keep history), /exit leave chat"
        }
    }
}

pub fn chat_unknown_builtin_command(command: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("未知内建命令: /{command}，可用命令请执行 /help"),
        Language::ZhTw => format!("未知內建命令: /{command}，可用命令請執行 /help"),
        Language::Fr => format!("Commande intégrée inconnue: /{command}, utilisez /help"),
        Language::De => format!("Unbekannter Builtin-Befehl: /{command}, bitte /help verwenden"),
        Language::Ja => {
            format!("不明な組み込みコマンド: /{command}。利用可能なコマンドは /help")
        }
        Language::En => format!("unknown builtin command: /{command}; use /help"),
    }
}

pub fn chat_change_usage() -> &'static str {
    match current_language() {
        Language::ZhCn => "用法: /change {完整session-id | session-id前缀 | 会话名称}",
        Language::ZhTw => "用法: /change {完整session-id | session-id前綴 | 會話名稱}",
        Language::Fr => "Usage: /change {session-id complet | préfixe session-id | nom de session}",
        Language::De => {
            "Verwendung: /change {vollständige session-id | session-id-Präfix | Sitzungsname}"
        }
        Language::Ja => "使い方: /change {完全なsession-id | session-idの接頭辞 | セッション名}",
        Language::En => "Usage: /change {full session-id | session-id prefix | session name}",
    }
}

pub fn chat_name_usage() -> &'static str {
    match current_language() {
        Language::ZhCn => "用法: /name {当前会话的新名称}",
        Language::ZhTw => "用法: /name {目前會話的新名稱}",
        Language::Fr => "Usage: /name {nouveau nom de la session courante}",
        Language::De => "Verwendung: /name {neuer Name der aktuellen Sitzung}",
        Language::Ja => "使い方: /name {現在のセッションの新しい名前}",
        Language::En => "Usage: /name {new name for current session}",
    }
}

pub fn chat_session_renamed(name: &str, session_id: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("会话已重命名\n名称: {name}\n会话: {session_id}"),
        Language::ZhTw => format!("會話已重新命名\n名稱: {name}\n會話: {session_id}"),
        Language::Fr => format!("Session renommée\nNom: {name}\nSession: {session_id}"),
        Language::De => format!("Sitzung umbenannt\nName: {name}\nSitzung: {session_id}"),
        Language::Ja => {
            format!("セッション名を変更しました\n名前: {name}\nセッション: {session_id}")
        }
        Language::En => format!("session renamed\nname: {name}\nsession: {session_id}"),
    }
}

pub fn chat_session_changed(name: &str, session_id: &str, session_file: &Path) -> String {
    match current_language() {
        Language::ZhCn => format!(
            "会话已切换\n名称: {name}\n会话: {session_id}\n会话文件: {}",
            session_file.display()
        ),
        Language::ZhTw => format!(
            "會話已切換\n名稱: {name}\n會話: {session_id}\n會話檔案: {}",
            session_file.display()
        ),
        Language::Fr => format!(
            "Session changée\nNom: {name}\nSession: {session_id}\nFichier: {}",
            session_file.display()
        ),
        Language::De => format!(
            "Sitzung gewechselt\nName: {name}\nSitzung: {session_id}\nDatei: {}",
            session_file.display()
        ),
        Language::Ja => format!(
            "セッションを切り替えました\n名前: {name}\nセッション: {session_id}\nファイル: {}",
            session_file.display()
        ),
        Language::En => format!(
            "session switched\nname: {name}\nsession: {session_id}\nfile: {}",
            session_file.display()
        ),
    }
}

pub fn chat_session_list_empty() -> &'static str {
    match current_language() {
        Language::ZhCn => "当前没有可用会话。",
        Language::ZhTw => "目前沒有可用會話。",
        Language::Fr => "Aucune session disponible.",
        Language::De => "Keine verfügbaren Sitzungen.",
        Language::Ja => "利用可能なセッションはありません。",
        Language::En => "No sessions available.",
    }
}

pub fn chat_session_list_title(total: usize) -> String {
    let total_fmt = human_count_u128(total as u128);
    match current_language() {
        Language::ZhCn => format!("已暂存会话列表（共 {total_fmt} 条）"),
        Language::ZhTw => format!("已暫存會話列表（共 {total_fmt} 筆）"),
        Language::Fr => format!("Sessions enregistrées ({total_fmt})"),
        Language::De => format!("Gespeicherte Sitzungen ({total_fmt})"),
        Language::Ja => format!("保存済みセッション一覧（{total_fmt} 件）"),
        Language::En => format!("Stored sessions ({total_fmt})"),
    }
}

pub fn chat_session_list_header_active() -> &'static str {
    match current_language() {
        Language::ZhCn => "当前",
        Language::ZhTw => "目前",
        Language::Fr => "Actif",
        Language::De => "Aktiv",
        Language::Ja => "現在",
        Language::En => "Active",
    }
}

pub fn chat_session_list_header_name() -> &'static str {
    match current_language() {
        Language::ZhCn => "会话名称",
        Language::ZhTw => "會話名稱",
        Language::Fr => "Nom",
        Language::De => "Name",
        Language::Ja => "セッション名",
        Language::En => "Name",
    }
}

pub fn chat_session_list_header_id() -> &'static str {
    match current_language() {
        Language::ZhCn => "会话ID",
        Language::ZhTw => "會話ID",
        Language::Fr => "Session ID",
        Language::De => "Session-ID",
        Language::Ja => "セッションID",
        Language::En => "Session ID",
    }
}

pub fn chat_session_list_header_messages() -> &'static str {
    match current_language() {
        Language::ZhCn => "消息(总/分角色)",
        Language::ZhTw => "訊息(總/分角色)",
        Language::Fr => "Messages(total/rôles)",
        Language::De => "Nachrichten(total/Rollen)",
        Language::Ja => "メッセージ(合計/役割別)",
        Language::En => "Messages(total/roles)",
    }
}

pub fn chat_session_list_header_summary() -> &'static str {
    match current_language() {
        Language::ZhCn => "摘要字符",
        Language::ZhTw => "摘要字元",
        Language::Fr => "Résumé(chars)",
        Language::De => "Zusammenfassung(Zeichen)",
        Language::Ja => "要約文字数",
        Language::En => "Summary(chars)",
    }
}

pub fn chat_session_list_header_updated() -> &'static str {
    match current_language() {
        Language::ZhCn => "更新时间",
        Language::ZhTw => "更新時間",
        Language::Fr => "Mis à jour",
        Language::De => "Aktualisiert",
        Language::Ja => "更新時刻",
        Language::En => "Updated",
    }
}

pub fn chat_session_list_header_created() -> &'static str {
    match current_language() {
        Language::ZhCn => "创建时间",
        Language::ZhTw => "建立時間",
        Language::Fr => "Créée",
        Language::De => "Erstellt",
        Language::Ja => "作成時刻",
        Language::En => "Created",
    }
}

pub fn chat_session_list_header_file() -> &'static str {
    match current_language() {
        Language::ZhCn => "会话文件",
        Language::ZhTw => "會話檔案",
        Language::Fr => "Fichier",
        Language::De => "Datei",
        Language::Ja => "ファイル",
        Language::En => "File",
    }
}

pub fn chat_session_list_active_yes() -> &'static str {
    match current_language() {
        Language::ZhCn => "是",
        Language::ZhTw => "是",
        Language::Fr => "oui",
        Language::De => "ja",
        Language::Ja => "はい",
        Language::En => "yes",
    }
}

pub fn chat_session_list_active_no() -> &'static str {
    match current_language() {
        Language::ZhCn => "否",
        Language::ZhTw => "否",
        Language::Fr => "non",
        Language::De => "nein",
        Language::Ja => "いいえ",
        Language::En => "no",
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
        Language::ZhCn => "[你] ",
        Language::ZhTw => "[你] ",
        Language::Fr => "[Vous] ",
        Language::De => "[Du] ",
        Language::Ja => "[あなた] ",
        Language::En => "[You] ",
    }
}

pub fn chat_prompt_assistant() -> &'static str {
    match current_language() {
        Language::ZhCn => "[MachineClaw] ",
        Language::ZhTw => "[MachineClaw] ",
        Language::Fr => "[MachineClaw] ",
        Language::De => "[MachineClaw] ",
        Language::Ja => "[MachineClaw] ",
        Language::En => "[MachineClaw] ",
    }
}

pub fn chat_prompt_thinking() -> &'static str {
    match current_language() {
        Language::ZhCn => "[MachineClaw-思考] ",
        Language::ZhTw => "[MachineClaw-思考] ",
        Language::Fr => "[MachineClaw-thinking] ",
        Language::De => "[MachineClaw-thinking] ",
        Language::Ja => "[MachineClaw-thinking] ",
        Language::En => "[MachineClaw-thinking] ",
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

pub fn chat_tool_type_shell_command() -> &'static str {
    match current_language() {
        Language::ZhCn => "执行shell命令",
        Language::ZhTw => "執行shell命令",
        Language::Fr => "run_shell_command",
        Language::De => "run_shell_command",
        Language::Ja => "run_shell_command",
        Language::En => "run_shell_command",
    }
}

pub fn chat_tool_type_shell_result() -> &'static str {
    match current_language() {
        Language::ZhCn => "命令执行结果",
        Language::ZhTw => "命令執行結果",
        Language::Fr => "shell_result",
        Language::De => "shell_result",
        Language::Ja => "shell_result",
        Language::En => "shell_result",
    }
}

pub fn chat_tool_type_output_preview() -> &'static str {
    match current_language() {
        Language::ZhCn => "命令输出预览",
        Language::ZhTw => "命令輸出預覽",
        Language::Fr => "output_preview",
        Language::De => "output_preview",
        Language::Ja => "output_preview",
        Language::En => "output_preview",
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
    archived_message_count: usize,
    effective_message_count: usize,
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
    effective_user_count: usize,
    effective_assistant_count: usize,
    effective_tool_count: usize,
    effective_system_count: usize,
) -> String {
    let archived_message_count_fmt = human_count_u128(archived_message_count as u128);
    let effective_message_count_fmt = human_count_u128(effective_message_count as u128);
    let summary_len_fmt = human_count_u128(summary_len as u128);
    let recent_limit_fmt = human_count_u128(recent_limit as u128);
    let max_limit_fmt = human_count_u128(max_limit as u128);
    let chat_turns_fmt = human_count_u128(chat_turns as u128);
    let skills_count_fmt = human_count_u128(skills_count as u128);
    let user_count_fmt = human_count_u128(user_count as u128);
    let assistant_count_fmt = human_count_u128(assistant_count as u128);
    let tool_count_fmt = human_count_u128(tool_count as u128);
    let system_count_fmt = human_count_u128(system_count as u128);
    let effective_user_count_fmt = human_count_u128(effective_user_count as u128);
    let effective_assistant_count_fmt = human_count_u128(effective_assistant_count as u128);
    let effective_tool_count_fmt = human_count_u128(effective_tool_count as u128);
    let effective_system_count_fmt = human_count_u128(effective_system_count as u128);
    match current_language() {
        Language::ZhCn => format!(
            "会话统计\n会话: {session_id}\n会话文件: {}\n系统/模型: {os_name} / {model}\n会话存档消息: {archived_message_count_fmt} (用户 {user_count_fmt}, 助手 {assistant_count_fmt}, 工具 {tool_count_fmt}, 系统 {system_count_fmt})\n当前送模上下文: {effective_message_count_fmt} (用户 {effective_user_count_fmt}, 助手 {effective_assistant_count_fmt}, 工具 {effective_tool_count_fmt}, 系统 {effective_system_count_fmt})\n摘要长度: {summary_len_fmt} 字符\n本次 chat 轮次: {chat_turns_fmt}\n上下文窗口: recent={recent_limit_fmt}, max={max_limit_fmt}\n能力状态: skills={skills_count_fmt}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::ZhTw => format!(
            "會話統計\n會話: {session_id}\n會話檔案: {}\n系統/模型: {os_name} / {model}\n會話封存訊息: {archived_message_count_fmt} (使用者 {user_count_fmt}, 助手 {assistant_count_fmt}, 工具 {tool_count_fmt}, 系統 {system_count_fmt})\n目前送模上下文: {effective_message_count_fmt} (使用者 {effective_user_count_fmt}, 助手 {effective_assistant_count_fmt}, 工具 {effective_tool_count_fmt}, 系統 {effective_system_count_fmt})\n摘要長度: {summary_len_fmt} 字元\n本次 chat 輪次: {chat_turns_fmt}\n上下文視窗: recent={recent_limit_fmt}, max={max_limit_fmt}\n能力狀態: skills={skills_count_fmt}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::Fr => format!(
            "Statistiques de session\nSession: {session_id}\nFichier: {}\nSystème/Modèle: {os_name} / {model}\nMessages archivés: {archived_message_count_fmt} (user {user_count_fmt}, assistant {assistant_count_fmt}, tool {tool_count_fmt}, system {system_count_fmt})\nContexte envoyé au modèle: {effective_message_count_fmt} (user {effective_user_count_fmt}, assistant {effective_assistant_count_fmt}, tool {effective_tool_count_fmt}, system {effective_system_count_fmt})\nLongueur du résumé: {summary_len_fmt}\nTours de chat: {chat_turns_fmt}\nFenêtre de contexte: recent={recent_limit_fmt}, max={max_limit_fmt}\nCapacités: skills={skills_count_fmt}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::De => format!(
            "Sitzungsstatistik\nSitzung: {session_id}\nDatei: {}\nSystem/Modell: {os_name} / {model}\nArchivierte Nachrichten: {archived_message_count_fmt} (user {user_count_fmt}, assistant {assistant_count_fmt}, tool {tool_count_fmt}, system {system_count_fmt})\nAn Modell gesendeter Kontext: {effective_message_count_fmt} (user {effective_user_count_fmt}, assistant {effective_assistant_count_fmt}, tool {effective_tool_count_fmt}, system {effective_system_count_fmt})\nZusammenfassungslänge: {summary_len_fmt}\nChat-Runden: {chat_turns_fmt}\nKontextfenster: recent={recent_limit_fmt}, max={max_limit_fmt}\nFähigkeiten: skills={skills_count_fmt}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::Ja => format!(
            "セッション統計\nセッション: {session_id}\nファイル: {}\nシステム/モデル: {os_name} / {model}\n保存済みメッセージ: {archived_message_count_fmt} (user {user_count_fmt}, assistant {assistant_count_fmt}, tool {tool_count_fmt}, system {system_count_fmt})\nモデル送信コンテキスト: {effective_message_count_fmt} (user {effective_user_count_fmt}, assistant {effective_assistant_count_fmt}, tool {effective_tool_count_fmt}, system {effective_system_count_fmt})\n要約文字数: {summary_len_fmt}\nchat ターン数: {chat_turns_fmt}\nコンテキスト枠: recent={recent_limit_fmt}, max={max_limit_fmt}\n機能: skills={skills_count_fmt}, mcp={mcp_summary}",
            session_file.display()
        ),
        Language::En => format!(
            "session stats\nsession: {session_id}\nfile: {}\nos/model: {os_name} / {model}\narchived messages: {archived_message_count_fmt} (user {user_count_fmt}, assistant {assistant_count_fmt}, tool {tool_count_fmt}, system {system_count_fmt})\neffective model context: {effective_message_count_fmt} (user {effective_user_count_fmt}, assistant {effective_assistant_count_fmt}, tool {effective_tool_count_fmt}, system {effective_system_count_fmt})\nsummary chars: {summary_len_fmt}\nchat turns: {chat_turns_fmt}\ncontext window: recent={recent_limit_fmt}, max={max_limit_fmt}\ncapability: skills={skills_count_fmt}, mcp={mcp_summary}",
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

pub fn progress_ai_summarizing(action: &str, target: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("正在生成 AI 总结: action={action}, target={target}"),
        Language::ZhTw => format!("正在產生 AI 摘要: action={action}, target={target}"),
        Language::Fr => format!("Génération du résumé IA: action={action}, target={target}"),
        Language::De => {
            format!("KI-Zusammenfassung wird erstellt: action={action}, target={target}")
        }
        Language::Ja => format!("AI 要約を生成中: action={action}, target={target}"),
        Language::En => format!("Generating AI summary: action={action}, target={target}"),
    }
}

pub fn chat_profile_started() -> &'static str {
    match current_language() {
        Language::ZhCn => "正在建立当前机器环境画像...",
        Language::ZhTw => "正在建立目前機器環境畫像...",
        Language::Fr => "Construction du profil de l'environnement local...",
        Language::De => "Erstelle Umgebungsprofil der aktuellen Maschine...",
        Language::Ja => "現在のマシン環境プロファイルを構築中...",
        Language::En => "Building environment profile for current machine...",
    }
}

pub fn chat_profile_collecting() -> &'static str {
    match current_language() {
        Language::ZhCn => "正在采集环境画像所需数据...",
        Language::ZhTw => "正在蒐集環境畫像所需資料...",
        Language::Fr => "Collecte des données du profil d'environnement...",
        Language::De => "Erfasse Daten für das Umgebungsprofil...",
        Language::Ja => "環境プロファイル用データを収集中...",
        Language::En => "Collecting data for environment profile...",
    }
}

pub fn chat_profile_analyzing() -> &'static str {
    match current_language() {
        Language::ZhCn => "正在分析环境画像并生成摘要...",
        Language::ZhTw => "正在分析環境畫像並產生摘要...",
        Language::Fr => "Analyse du profil d'environnement et génération du résumé...",
        Language::De => "Umgebungsprofil wird analysiert und zusammengefasst...",
        Language::Ja => "環境プロファイルを分析して要約を生成中...",
        Language::En => "Analyzing environment profile and generating summary...",
    }
}

pub fn chat_profile_completed() -> &'static str {
    match current_language() {
        Language::ZhCn => "环境画像已更新并注入会话上下文。",
        Language::ZhTw => "環境畫像已更新並注入會話上下文。",
        Language::Fr => "Le profil d'environnement a été mis à jour et injecté dans le contexte.",
        Language::De => "Umgebungsprofil wurde aktualisiert und in den Kontext übernommen.",
        Language::Ja => "環境プロファイルを更新し、会話コンテキストに注入しました。",
        Language::En => "Environment profile updated and injected into chat context.",
    }
}

pub fn chat_profile_failed(err: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("环境画像构建失败: {err}"),
        Language::ZhTw => format!("環境畫像建立失敗: {err}"),
        Language::Fr => format!("Échec de création du profil d'environnement: {err}"),
        Language::De => format!("Erstellung des Umgebungsprofils fehlgeschlagen: {err}"),
        Language::Ja => format!("環境プロファイルの作成に失敗しました: {err}"),
        Language::En => format!("Failed to build environment profile: {err}"),
    }
}

pub fn chat_compression_started(target_messages: usize) -> String {
    let target_messages_fmt = human_count_u128(target_messages as u128);
    match current_language() {
        Language::ZhCn => {
            format!("上下文消息超过阈值，开始执行 AI 压缩（候选消息 {target_messages_fmt} 条）...")
        }
        Language::ZhTw => {
            format!("上下文訊息超過閾值，開始執行 AI 壓縮（候選訊息 {target_messages_fmt} 筆）...")
        }
        Language::Fr => format!(
            "Le contexte dépasse le seuil, compression IA en cours (messages candidats: {target_messages_fmt})..."
        ),
        Language::De => format!(
            "Kontext über Schwellwert, starte KI-Kompression (Kandidaten: {target_messages_fmt})..."
        ),
        Language::Ja => format!(
            "コンテキストが閾値を超えたため、AI 圧縮を開始します（候補 {target_messages_fmt} 件）..."
        ),
        Language::En => format!(
            "Context exceeds threshold, starting AI compression (candidate messages: {target_messages_fmt})..."
        ),
    }
}

pub fn chat_compression_completed(removed: usize, total: usize) -> String {
    let removed_fmt = human_count_u128(removed as u128);
    let total_fmt = human_count_u128(total as u128);
    match current_language() {
        Language::ZhCn => {
            format!("AI 压缩完成，已折叠 {removed_fmt} 条旧消息，当前总消息 {total_fmt} 条。")
        }
        Language::ZhTw => {
            format!("AI 壓縮完成，已折疊 {removed_fmt} 筆舊訊息，目前總訊息 {total_fmt} 筆。")
        }
        Language::Fr => format!(
            "Compression IA terminée: {removed_fmt} anciens messages condensés, total actuel {total_fmt}."
        ),
        Language::De => format!(
            "KI-Kompression abgeschlossen: {removed_fmt} alte Nachrichten verdichtet, aktuell {total_fmt} gesamt."
        ),
        Language::Ja => format!(
            "AI 圧縮が完了しました。{removed_fmt} 件の旧メッセージを圧縮し、現在 {total_fmt} 件です。"
        ),
        Language::En => format!(
            "AI compression completed: collapsed {removed_fmt} old messages, current total {total_fmt}."
        ),
    }
}

pub fn chat_compression_failed(err: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("AI 压缩失败，已跳过本轮压缩: {err}"),
        Language::ZhTw => format!("AI 壓縮失敗，已跳過本輪壓縮: {err}"),
        Language::Fr => format!("Échec de compression IA, compression ignorée pour ce tour: {err}"),
        Language::De => {
            format!("KI-Kompression fehlgeschlagen, in dieser Runde übersprungen: {err}")
        }
        Language::Ja => format!("AI 圧縮に失敗したため、このラウンドはスキップしました: {err}"),
        Language::En => format!("AI compression failed; skipped this round: {err}"),
    }
}

pub fn chat_compression_running() -> &'static str {
    match current_language() {
        Language::ZhCn => "正在执行 AI 历史压缩...",
        Language::ZhTw => "正在執行 AI 歷史壓縮...",
        Language::Fr => "Compression IA de l'historique en cours...",
        Language::De => "KI-Verlaufs-Kompression wird ausgeführt...",
        Language::Ja => "AI 履歴圧縮を実行中...",
        Language::En => "Running AI history compression...",
    }
}

pub fn chat_skill_enabled(count: usize) -> String {
    let count_fmt = human_count_u128(count as u128);
    match current_language() {
        Language::ZhCn => format!("已启用 Skills 自动扫描，可用技能目录项 {count_fmt} 个。"),
        Language::ZhTw => format!("已啟用 Skills 自動掃描，可用技能目錄項 {count_fmt} 個。"),
        Language::Fr => {
            format!("Scan automatique des skills activé, entrées disponibles: {count_fmt}.")
        }
        Language::De => {
            format!("Automatischer Skill-Scan aktiviert, verfügbare Einträge: {count_fmt}.")
        }
        Language::Ja => {
            format!("Skills 自動スキャンを有効化しました。利用可能な項目: {count_fmt}。")
        }
        Language::En => format!("Skill auto-scan enabled, available entries: {count_fmt}."),
    }
}

pub fn chat_skill_prepare_started(count: usize) -> String {
    let count_fmt = human_count_u128(count as u128);
    match current_language() {
        Language::ZhCn => format!("本轮将加载 Skills 上下文，可用项 {count_fmt} 个。"),
        Language::ZhTw => format!("本輪將載入 Skills 上下文，可用項 {count_fmt} 個。"),
        Language::Fr => {
            format!("Ce tour chargera le contexte skills, entrées disponibles: {count_fmt}.")
        }
        Language::De => format!(
            "In dieser Runde wird der Skill-Kontext geladen, verfügbare Einträge: {count_fmt}."
        ),
        Language::Ja => {
            format!("このラウンドで Skills コンテキストを読み込みます。利用可能項目: {count_fmt}。")
        }
        Language::En => {
            format!("This round will load skill context, available entries: {count_fmt}.")
        }
    }
}

pub fn chat_skill_workflow_started(name: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("即将执行 Skill 规范逻辑: {name}"),
        Language::ZhTw => format!("即將執行 Skill 規範邏輯: {name}"),
        Language::Fr => format!("Exécution imminente du workflow skill: {name}"),
        Language::De => format!("Skill-Workflow wird gleich ausgeführt: {name}"),
        Language::Ja => format!("Skill ワークフローをこれから実行します: {name}"),
        Language::En => format!("About to execute skill workflow: {name}"),
    }
}

pub fn chat_mcp_service_request_started(server: &str, tool: &str) -> String {
    match current_language() {
        Language::ZhCn => format!("即将请求 MCP 服务: {server}, 工具: {tool}"),
        Language::ZhTw => format!("即將請求 MCP 服務: {server}, 工具: {tool}"),
        Language::Fr => format!("Requête MCP imminente: service {server}, outil {tool}"),
        Language::De => format!("MCP-Anfrage folgt: Dienst {server}, Tool {tool}"),
        Language::Ja => {
            format!("これから MCP サービスへリクエストします: {server}, ツール: {tool}")
        }
        Language::En => format!("About to request MCP service: {server}, tool: {tool}"),
    }
}

pub fn chat_round_received(round: usize, tool_calls: usize) -> String {
    let round_fmt = human_count_u128(round as u128);
    let tool_calls_fmt = human_count_u128(tool_calls as u128);
    match current_language() {
        Language::ZhCn => {
            format!("已收到第 {round_fmt} 轮 AI 响应，准备执行 {tool_calls_fmt} 个工具调用...")
        }
        Language::ZhTw => {
            format!("已收到第 {round_fmt} 輪 AI 回應，準備執行 {tool_calls_fmt} 個工具呼叫...")
        }
        Language::Fr => format!(
            "Réponse IA du tour {round_fmt} reçue, préparation de {tool_calls_fmt} appel(s) d'outil..."
        ),
        Language::De => format!(
            "KI-Antwort Runde {round_fmt} empfangen, {tool_calls_fmt} Tool-Aufruf(e) werden vorbereitet..."
        ),
        Language::Ja => format!(
            "第 {round_fmt} ラウンドの AI 応答を受信しました。{tool_calls_fmt} 件のツール呼び出しを実行します..."
        ),
        Language::En => {
            format!("AI round {round_fmt} received, preparing {tool_calls_fmt} tool call(s)...")
        }
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
    let message_count_fmt = human_count_u128(message_count as u128);
    let max_limit_fmt = human_count_u128(max_limit as u128);
    let recent_limit_fmt = human_count_u128(recent_limit as u128);
    let summary_chars_fmt = human_count_u128(summary_chars as u128);
    match current_language() {
        Language::ZhCn => {
            if critical {
                return format!(
                    "上下文容量已接近上限({usage_percent}%)：messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}。建议尽快结束当前话题或切换 /new 以降低压缩损耗。"
                );
            }
            format!(
                "上下文容量预警({usage_percent}%)：messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}。后续消息可能触发更激进压缩。"
            )
        }
        Language::ZhTw => {
            if critical {
                return format!(
                    "上下文容量接近上限({usage_percent}%)：messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}。建議盡快結束目前話題或切換 /new 降低壓縮損耗。"
                );
            }
            format!(
                "上下文容量預警({usage_percent}%)：messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}。後續訊息可能觸發更激進壓縮。"
            )
        }
        Language::Fr => {
            if critical {
                return format!(
                    "Contexte presque saturé ({usage_percent}%): messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}. Terminez le sujet ou utilisez /new pour réduire la perte due à la compression."
                );
            }
            format!(
                "Alerte contexte ({usage_percent}%): messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}. Une compression plus agressive peut être appliquée."
            )
        }
        Language::De => {
            if critical {
                return format!(
                    "Kontext fast ausgelastet ({usage_percent}%): messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}. Thema abschließen oder /new verwenden, um Kompressionsverlust zu verringern."
                );
            }
            format!(
                "Kontext-Warnung ({usage_percent}%): messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}. Weitere Nachrichten können stärkere Kompression auslösen."
            )
        }
        Language::Ja => {
            if critical {
                return format!(
                    "コンテキスト容量が上限に近づいています ({usage_percent}%): messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}。圧縮劣化を抑えるため、話題を区切るか /new を使用してください。"
                );
            }
            format!(
                "コンテキスト容量の警告 ({usage_percent}%): messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}。以降のメッセージで圧縮が強くなる可能性があります。"
            )
        }
        Language::En => {
            if critical {
                return format!(
                    "Context is near capacity ({usage_percent}%): messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}. Consider ending the topic or switching to /new to reduce compression loss."
                );
            }
            format!(
                "Context pressure warning ({usage_percent}%): messages={message_count_fmt}/{max_limit_fmt}, recent={recent_limit_fmt}, summary_chars={summary_chars_fmt}. Further messages may trigger stronger compression."
            )
        }
    }
}

pub fn chat_tool_guard_warning(
    reason_code: &str,
    tool_rounds_used: usize,
    total_tool_calls: usize,
    max_tool_rounds: usize,
    max_total_tool_calls: usize,
) -> String {
    let reason_text = chat_tool_guard_reason_text(reason_code);
    let rounds_used_fmt = human_count_u128(tool_rounds_used as u128);
    let total_tool_calls_fmt = human_count_u128(total_tool_calls as u128);
    let max_tool_rounds_fmt = human_count_u128(max_tool_rounds as u128);
    let max_total_tool_calls_fmt = human_count_u128(max_total_tool_calls as u128);
    match current_language() {
        Language::ZhCn => format!(
            "工具调用流程已触发保护性收口\n原因: {reason_text}\n当前计数: 工具轮次 {rounds_used_fmt}/{max_tool_rounds_fmt}, 工具调用 {total_tool_calls_fmt}/{max_total_tool_calls_fmt}\n建议: 1) 细化问题范围 2) 指定更明确的目标命令 3) 必要时执行 /new 开启新会话后继续"
        ),
        Language::ZhTw => format!(
            "工具呼叫流程已觸發保護性收口\n原因: {reason_text}\n目前計數: 工具輪次 {rounds_used_fmt}/{max_tool_rounds_fmt}, 工具呼叫 {total_tool_calls_fmt}/{max_total_tool_calls_fmt}\n建議: 1) 縮小問題範圍 2) 指定更明確的目標命令 3) 必要時使用 /new 建立新會話再繼續"
        ),
        Language::Fr => format!(
            "Le flux d'appels d'outils a été clôturé par protection\nCause: {reason_text}\nCompteurs: tours outil {rounds_used_fmt}/{max_tool_rounds_fmt}, appels outil {total_tool_calls_fmt}/{max_total_tool_calls_fmt}\nConseils: 1) affiner la demande 2) préciser la commande cible 3) utiliser /new si nécessaire"
        ),
        Language::De => format!(
            "Der Tool-Aufruf-Workflow wurde aus Schutzgründen beendet\nGrund: {reason_text}\nZähler: Tool-Runden {rounds_used_fmt}/{max_tool_rounds_fmt}, Tool-Aufrufe {total_tool_calls_fmt}/{max_total_tool_calls_fmt}\nEmpfehlung: 1) Anfrage eingrenzen 2) Zielbefehl präzisieren 3) bei Bedarf mit /new neu starten"
        ),
        Language::Ja => format!(
            "ツール呼び出しフローは保護制御により収束しました\n理由: {reason_text}\n現在値: ツールラウンド {rounds_used_fmt}/{max_tool_rounds_fmt}, ツール呼び出し {total_tool_calls_fmt}/{max_total_tool_calls_fmt}\n提案: 1) 問題範囲を絞る 2) 目的コマンドを具体化する 3) 必要なら /new で新規セッションを開始"
        ),
        Language::En => format!(
            "Tool-calling flow was safely finalized\nReason: {reason_text}\nCounters: tool rounds {rounds_used_fmt}/{max_tool_rounds_fmt}, tool calls {total_tool_calls_fmt}/{max_total_tool_calls_fmt}\nNext steps: 1) narrow the question 2) specify target commands 3) use /new and continue if needed"
        ),
    }
}

fn chat_tool_guard_reason_text(reason_code: &str) -> &'static str {
    match reason_code {
        "tool_call_limit_exceeded" => match current_language() {
            Language::ZhCn => "工具调用总次数已达到上限",
            Language::ZhTw => "工具呼叫總次數已達上限",
            Language::Fr => "la limite totale des appels d'outil est atteinte",
            Language::De => "das Gesamtlimit für Tool-Aufrufe wurde erreicht",
            Language::Ja => "ツール呼び出し総数が上限に達しました",
            Language::En => "total tool call limit reached",
        },
        "repeated_same_tool_call" => match current_language() {
            Language::ZhCn => "检测到同一工具调用被重复触发",
            Language::ZhTw => "偵測到同一工具呼叫被重複觸發",
            Language::Fr => "des appels d'outil identiques sont répétés",
            Language::De => "wiederholte identische Tool-Aufrufe erkannt",
            Language::Ja => "同一ツール呼び出しの反復を検出しました",
            Language::En => "repeated identical tool call detected",
        },
        "repeated_tool_timeout" => match current_language() {
            Language::ZhCn => "同一工具调用连续超时",
            Language::ZhTw => "同一工具呼叫連續逾時",
            Language::Fr => "timeouts répétés sur le même outil",
            Language::De => "wiederholte Timeouts beim selben Tool",
            Language::Ja => "同一ツールで連続タイムアウトが発生しました",
            Language::En => "repeated timeout on the same tool",
        },
        "too_many_tool_timeouts" => match current_language() {
            Language::ZhCn => "工具调用超时次数过多",
            Language::ZhTw => "工具呼叫逾時次數過多",
            Language::Fr => "trop de timeouts d'appels d'outil",
            Language::De => "zu viele Tool-Timeouts",
            Language::Ja => "ツール呼び出しのタイムアウト回数が多すぎます",
            Language::En => "too many tool timeouts",
        },
        "max_tool_rounds_reached" => match current_language() {
            Language::ZhCn => "工具调用轮次达到上限",
            Language::ZhTw => "工具呼叫輪次達到上限",
            Language::Fr => "le nombre maximal de tours d'outil est atteint",
            Language::De => "maximale Anzahl von Tool-Runden erreicht",
            Language::Ja => "ツール呼び出しラウンド数が上限に達しました",
            Language::En => "maximum tool-calling rounds reached",
        },
        _ => match current_language() {
            Language::ZhCn => "触发了保护性收口条件",
            Language::ZhTw => "觸發了保護性收口條件",
            Language::Fr => "condition de protection déclenchée",
            Language::De => "Schutzbedingung ausgelöst",
            Language::Ja => "保護条件が発動しました",
            Language::En => "a safety guard condition was triggered",
        },
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
    let age = human_duration_ms(age_ms);
    let age_ms_fmt = human_count_u128(age_ms);
    match current_language() {
        Language::ZhCn => format!("命中命令缓存: label={label}, age={age} ({age_ms_fmt} ms)"),
        Language::ZhTw => format!("命中命令快取: label={label}, age={age} ({age_ms_fmt} ms)"),
        Language::Fr => {
            format!("Cache de commande utilisé: label={label}, age={age} ({age_ms_fmt} ms)")
        }
        Language::De => {
            format!("Befehls-Cache-Treffer: label={label}, age={age} ({age_ms_fmt} ms)")
        }
        Language::Ja => {
            format!("コマンドキャッシュを使用: label={label}, age={age} ({age_ms_fmt} ms)")
        }
        Language::En => format!("Command cache hit: label={label}, age={age} ({age_ms_fmt} ms)"),
    }
}

pub fn chat_round_metrics(
    api_rounds: usize,
    api_duration_ms: u128,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    estimated_cost_usd: Option<f64>,
    show_cost: bool,
) -> String {
    let duration = human_duration_ms(api_duration_ms);
    let api_duration_ms_fmt = human_count_u128(api_duration_ms);
    let api_rounds_fmt = human_count_u128(api_rounds as u128);
    let prompt_tokens_fmt = human_count_u64(prompt_tokens);
    let completion_tokens_fmt = human_count_u64(completion_tokens);
    let total_tokens_fmt = human_count_u64(total_tokens);
    let estimated_cost_text = estimated_cost_usd
        .map(|value| format!("{value:.6} USD"))
        .unwrap_or_else(|| match current_language() {
            Language::ZhCn => "N/A（缺少有效单价）".to_string(),
            Language::ZhTw => "N/A（缺少有效單價）".to_string(),
            Language::Fr => "N/A (tarif indisponible)".to_string(),
            Language::De => "N/A (kein Preis verfügbar)".to_string(),
            Language::Ja => "N/A（有効な単価なし）".to_string(),
            Language::En => "N/A (pricing unavailable)".to_string(),
        });
    match current_language() {
        Language::ZhCn => {
            if show_cost {
                return format!(
                    "本轮指标：请求轮次 {api_rounds_fmt}，接口耗时 {duration}（{api_duration_ms_fmt} ms），Token（输入/输出/总计）{prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}，预估费用 {estimated_cost_text}"
                );
            }
            format!(
                "本轮指标：请求轮次 {api_rounds_fmt}，接口耗时 {duration}（{api_duration_ms_fmt} ms），Token（输入/输出/总计）{prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}"
            )
        }
        Language::ZhTw => {
            if show_cost {
                return format!(
                    "本輪指標：請求輪次 {api_rounds_fmt}，介面耗時 {duration}（{api_duration_ms_fmt} ms），Token（輸入/輸出/總計）{prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}，預估費用 {estimated_cost_text}"
                );
            }
            format!(
                "本輪指標：請求輪次 {api_rounds_fmt}，介面耗時 {duration}（{api_duration_ms_fmt} ms），Token（輸入/輸出/總計）{prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}"
            )
        }
        Language::Fr => {
            if show_cost {
                return format!(
                    "Métriques du tour: cycles {api_rounds_fmt}, durée API {duration} ({api_duration_ms_fmt} ms), tokens (entrée/sortie/total) {prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}, coût estimé {estimated_cost_text}"
                );
            }
            format!(
                "Métriques du tour: cycles {api_rounds_fmt}, durée API {duration} ({api_duration_ms_fmt} ms), tokens (entrée/sortie/total) {prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}"
            )
        }
        Language::De => {
            if show_cost {
                return format!(
                    "Rundenmetriken: Durchläufe {api_rounds_fmt}, API-Dauer {duration} ({api_duration_ms_fmt} ms), Tokens (Eingabe/Ausgabe/Gesamt) {prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}, geschätzte Kosten {estimated_cost_text}"
                );
            }
            format!(
                "Rundenmetriken: Durchläufe {api_rounds_fmt}, API-Dauer {duration} ({api_duration_ms_fmt} ms), Tokens (Eingabe/Ausgabe/Gesamt) {prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}"
            )
        }
        Language::Ja => {
            if show_cost {
                return format!(
                    "ラウンド指標: リクエスト回数 {api_rounds_fmt}、API時間 {duration}（{api_duration_ms_fmt} ms）、Token（入力/出力/合計）{prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}、推定コスト {estimated_cost_text}"
                );
            }
            format!(
                "ラウンド指標: リクエスト回数 {api_rounds_fmt}、API時間 {duration}（{api_duration_ms_fmt} ms）、Token（入力/出力/合計）{prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}"
            )
        }
        Language::En => {
            if show_cost {
                return format!(
                    "Round metrics: rounds {api_rounds_fmt}, API duration {duration} ({api_duration_ms_fmt} ms), tokens (prompt/completion/total) {prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}, estimated cost {estimated_cost_text}"
                );
            }
            format!(
                "Round metrics: rounds {api_rounds_fmt}, API duration {duration} ({api_duration_ms_fmt} ms), tokens (prompt/completion/total) {prompt_tokens_fmt}/{completion_tokens_fmt}/{total_tokens_fmt}"
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
    let duration = human_duration_ms(duration_ms);
    let duration_ms_fmt = human_count_u128(duration_ms);
    match current_language() {
        Language::ZhCn => format!(
            "命令完成: label={label}, status={status}, exit={exit_code:?}, duration={duration} ({duration_ms_fmt} ms), timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::ZhTw => format!(
            "命令完成: label={label}, status={status}, exit={exit_code:?}, duration={duration} ({duration_ms_fmt} ms), timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::Fr => format!(
            "Commande terminée: label={label}, status={status}, exit={exit_code:?}, duration={duration} ({duration_ms_fmt} ms), timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::De => format!(
            "Befehl abgeschlossen: label={label}, status={status}, exit={exit_code:?}, duration={duration} ({duration_ms_fmt} ms), timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::Ja => format!(
            "コマンド完了: label={label}, status={status}, exit={exit_code:?}, duration={duration} ({duration_ms_fmt} ms), timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
        ),
        Language::En => format!(
            "Command finished: label={label}, status={status}, exit={exit_code:?}, duration={duration} ({duration_ms_fmt} ms), timeout={timed_out}, interrupted={interrupted}, blocked={blocked}"
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
        "app.env-mode must be one of: prod, test, dev" => match current_language() {
            Language::ZhCn => "app.env-mode 取值必须是 prod、test、dev".to_string(),
            Language::ZhTw => "app.env-mode 取值必須是 prod、test、dev".to_string(),
            Language::Fr => "app.env-mode doit être prod, test ou dev".to_string(),
            Language::De => "app.env-mode muss prod, test oder dev sein".to_string(),
            Language::Ja => "app.env-mode は prod / test / dev のいずれかである必要があります".to_string(),
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
        "ai.chat.compression.max-history-messages must be greater than 0" => {
            match current_language() {
                Language::ZhCn => "ai.chat.compression.max-history-messages 必须大于 0".to_string(),
                Language::ZhTw => "ai.chat.compression.max-history-messages 必須大於 0".to_string(),
                Language::Fr => "ai.chat.compression.max-history-messages doit être supérieur à 0".to_string(),
                Language::De => "ai.chat.compression.max-history-messages muss größer als 0 sein".to_string(),
                Language::Ja => "ai.chat.compression.max-history-messages は 0 より大きい必要があります".to_string(),
                Language::En => detail.to_string(),
            }
        }
        "ai.chat.compression.max-chars-count must be greater than 0" => {
            match current_language() {
                Language::ZhCn => "ai.chat.compression.max-chars-count 必须大于 0".to_string(),
                Language::ZhTw => "ai.chat.compression.max-chars-count 必須大於 0".to_string(),
                Language::Fr => "ai.chat.compression.max-chars-count doit être supérieur à 0".to_string(),
                Language::De => "ai.chat.compression.max-chars-count muss größer als 0 sein".to_string(),
                Language::Ja => "ai.chat.compression.max-chars-count は 0 より大きい必要があります".to_string(),
                Language::En => detail.to_string(),
            }
        }
        "ai.chat.cmd-run-timout must be greater than 0" => match current_language() {
            Language::ZhCn => "ai.chat.cmd-run-timout 必须大于 0".to_string(),
            Language::ZhTw => "ai.chat.cmd-run-timout 必須大於 0".to_string(),
            Language::Fr => "ai.chat.cmd-run-timout doit être supérieur à 0".to_string(),
            Language::De => "ai.chat.cmd-run-timout muss größer als 0 sein".to_string(),
            Language::Ja => "ai.chat.cmd-run-timout は 0 より大きい必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "ai.chat.max-tool-rounds must be greater than 0" => match current_language() {
            Language::ZhCn => "ai.chat.max-tool-rounds 必须大于 0".to_string(),
            Language::ZhTw => "ai.chat.max-tool-rounds 必須大於 0".to_string(),
            Language::Fr => "ai.chat.max-tool-rounds doit être supérieur à 0".to_string(),
            Language::De => "ai.chat.max-tool-rounds muss größer als 0 sein".to_string(),
            Language::Ja => "ai.chat.max-tool-rounds は 0 より大きい必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "ai.chat.max-total-tool-calls must be greater than 0" => match current_language() {
            Language::ZhCn => "ai.chat.max-total-tool-calls 必须大于 0".to_string(),
            Language::ZhTw => "ai.chat.max-total-tool-calls 必須大於 0".to_string(),
            Language::Fr => "ai.chat.max-total-tool-calls doit être supérieur à 0".to_string(),
            Language::De => "ai.chat.max-total-tool-calls muss größer als 0 sein".to_string(),
            Language::Ja => "ai.chat.max-total-tool-calls は 0 より大きい必要があります".to_string(),
            Language::En => detail.to_string(),
        },
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
        "log.log-file-name must include a file extension" => match current_language() {
            Language::ZhCn => "log.log-file-name 必须包含文件后缀名".to_string(),
            Language::ZhTw => "log.log-file-name 必須包含副檔名".to_string(),
            Language::Fr => "log.log-file-name doit contenir une extension".to_string(),
            Language::De => "log.log-file-name muss eine Dateiendung enthalten".to_string(),
            Language::Ja => "log.log-file-name には拡張子が必要です".to_string(),
            Language::En => detail.to_string(),
        },
        "log.log-file-name must not contain path separators" => match current_language() {
            Language::ZhCn => "log.log-file-name 不能包含路径分隔符".to_string(),
            Language::ZhTw => "log.log-file-name 不能包含路徑分隔符".to_string(),
            Language::Fr => {
                "log.log-file-name ne doit pas contenir de séparateur de chemin".to_string()
            }
            Language::De => {
                "log.log-file-name darf keine Pfad-Trennzeichen enthalten".to_string()
            }
            Language::Ja => "log.log-file-name にパス区切り文字は使用できません".to_string(),
            Language::En => detail.to_string(),
        },
        "log.max-file-size must not be empty" => match current_language() {
            Language::ZhCn => "log.max-file-size 不能为空".to_string(),
            Language::ZhTw => "log.max-file-size 不能為空".to_string(),
            Language::Fr => "log.max-file-size ne peut pas être vide".to_string(),
            Language::De => "log.max-file-size darf nicht leer sein".to_string(),
            Language::Ja => "log.max-file-size は空にできません".to_string(),
            Language::En => detail.to_string(),
        },
        "log.max-save-time must not be empty" => match current_language() {
            Language::ZhCn => "log.max-save-time 不能为空".to_string(),
            Language::ZhTw => "log.max-save-time 不能為空".to_string(),
            Language::Fr => "log.max-save-time ne peut pas être vide".to_string(),
            Language::De => "log.max-save-time darf nicht leer sein".to_string(),
            Language::Ja => "log.max-save-time は空にできません".to_string(),
            Language::En => detail.to_string(),
        },
        "log.max-file-size is invalid" => match current_language() {
            Language::ZhCn => "log.max-file-size 格式无效".to_string(),
            Language::ZhTw => "log.max-file-size 格式無效".to_string(),
            Language::Fr => "format de log.max-file-size invalide".to_string(),
            Language::De => "ungültiges Format für log.max-file-size".to_string(),
            Language::Ja => "log.max-file-size の形式が不正です".to_string(),
            Language::En => detail.to_string(),
        },
        "log.max-save-time is invalid" => match current_language() {
            Language::ZhCn => "log.max-save-time 格式无效".to_string(),
            Language::ZhTw => "log.max-save-time 格式無效".to_string(),
            Language::Fr => "format de log.max-save-time invalide".to_string(),
            Language::De => "ungültiges Format für log.max-save-time".to_string(),
            Language::Ja => "log.max-save-time の形式が不正です".to_string(),
            Language::En => detail.to_string(),
        },
        "log.max-file-size must be greater than 0" => match current_language() {
            Language::ZhCn => "log.max-file-size 必须大于 0".to_string(),
            Language::ZhTw => "log.max-file-size 必須大於 0".to_string(),
            Language::Fr => "log.max-file-size doit être supérieur à 0".to_string(),
            Language::De => "log.max-file-size muss größer als 0 sein".to_string(),
            Language::Ja => "log.max-file-size は 0 より大きい必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "log.max-save-time must be greater than 0" => match current_language() {
            Language::ZhCn => "log.max-save-time 必须大于 0".to_string(),
            Language::ZhTw => "log.max-save-time 必須大於 0".to_string(),
            Language::Fr => "log.max-save-time doit être supérieur à 0".to_string(),
            Language::De => "log.max-save-time muss größer als 0 sein".to_string(),
            Language::Ja => "log.max-save-time は 0 より大きい必要があります".to_string(),
            Language::En => detail.to_string(),
        },
        "mcp.enabled=true requires at least one configured server" => match current_language() {
            Language::ZhCn => "mcp.enabled=true 时至少需要配置一个 MCP 服务".to_string(),
            Language::ZhTw => "mcp.enabled=true 時至少需要設定一個 MCP 服務".to_string(),
            Language::Fr => {
                "Quand mcp.enabled=true, au moins un serveur MCP doit être configuré".to_string()
            }
            Language::De => {
                "Bei mcp.enabled=true muss mindestens ein MCP-Server konfiguriert sein".to_string()
            }
            Language::Ja => {
                "mcp.enabled=true の場合、少なくとも 1 つの MCP サーバー設定が必要です"
                    .to_string()
            }
            Language::En => detail.to_string(),
        },
        _ if detail.starts_with("cmd.allow-cmd-list has invalid regex '") => {
            match current_language() {
                Language::ZhCn => format!("cmd.allow-cmd-list 包含非法正则: {detail}"),
                Language::ZhTw => format!("cmd.allow-cmd-list 包含非法正則: {detail}"),
                Language::Fr => format!("cmd.allow-cmd-list contient une regex invalide: {detail}"),
                Language::De => format!("cmd.allow-cmd-list enthält einen ungültigen Regex: {detail}"),
                Language::Ja => format!("cmd.allow-cmd-list に不正な正規表現があります: {detail}"),
                Language::En => detail.to_string(),
            }
        }
        _ if detail.starts_with("cmd.deny-cmd-list has invalid regex '") => {
            match current_language() {
                Language::ZhCn => format!("cmd.deny-cmd-list 包含非法正则: {detail}"),
                Language::ZhTw => format!("cmd.deny-cmd-list 包含非法正則: {detail}"),
                Language::Fr => format!("cmd.deny-cmd-list contient une regex invalide: {detail}"),
                Language::De => format!("cmd.deny-cmd-list enthält einen ungültigen Regex: {detail}"),
                Language::Ja => format!("cmd.deny-cmd-list に不正な正規表現があります: {detail}"),
                Language::En => detail.to_string(),
            }
        }
        "session query cannot be empty" => match current_language() {
            Language::ZhCn => "会话切换条件不能为空".to_string(),
            Language::ZhTw => "會話切換條件不能為空".to_string(),
            Language::Fr => "Le critère de changement de session ne peut pas être vide".to_string(),
            Language::De => "Die Sitzungsabfrage darf nicht leer sein".to_string(),
            Language::Ja => "セッション切替クエリは空にできません".to_string(),
            Language::En => detail.to_string(),
        },
        _ if detail.starts_with("session not found: ") => match current_language() {
            Language::ZhCn => format!(
                "未找到匹配会话: {}",
                detail.trim_start_matches("session not found: ").trim()
            ),
            Language::ZhTw => format!(
                "找不到匹配會話: {}",
                detail.trim_start_matches("session not found: ").trim()
            ),
            Language::Fr => format!(
                "Session introuvable: {}",
                detail.trim_start_matches("session not found: ").trim()
            ),
            Language::De => format!(
                "Sitzung nicht gefunden: {}",
                detail.trim_start_matches("session not found: ").trim()
            ),
            Language::Ja => format!(
                "一致するセッションが見つかりません: {}",
                detail.trim_start_matches("session not found: ").trim()
            ),
            Language::En => detail.to_string(),
        },
        _ if detail.starts_with("session query is ambiguous: ") => match current_language() {
            Language::ZhCn => format!(
                "会话匹配到多项，请使用更完整的 session-id 或更精确名称: {}",
                detail
                    .trim_start_matches("session query is ambiguous: ")
                    .trim()
            ),
            Language::ZhTw => format!(
                "會話匹配到多項，請使用更完整的 session-id 或更精確名稱: {}",
                detail
                    .trim_start_matches("session query is ambiguous: ")
                    .trim()
            ),
            Language::Fr => format!(
                "Requête de session ambiguë, utilisez un session-id plus précis ou un nom exact: {}",
                detail
                    .trim_start_matches("session query is ambiguous: ")
                    .trim()
            ),
            Language::De => format!(
                "Mehrdeutige Sitzungsabfrage, verwenden Sie eine genauere session-id oder einen exakten Namen: {}",
                detail
                    .trim_start_matches("session query is ambiguous: ")
                    .trim()
            ),
            Language::Ja => format!(
                "セッションが複数一致しました。より完全な session-id か正確な名前を指定してください: {}",
                detail
                    .trim_start_matches("session query is ambiguous: ")
                    .trim()
            ),
            Language::En => detail.to_string(),
        },
        _ if detail.starts_with("mcp server '")
            && detail.ends_with("' requires endpoint or command") =>
        {
            match current_language() {
                Language::ZhCn => format!("MCP 服务配置缺失 endpoint 或 command: {}", detail),
                Language::ZhTw => format!("MCP 服務設定缺少 endpoint 或 command: {}", detail),
                Language::Fr => format!(
                    "Configuration MCP incomplète (endpoint ou command manquant): {}",
                    detail
                ),
                Language::De => format!(
                    "Unvollständige MCP-Konfiguration (endpoint oder command fehlt): {}",
                    detail
                ),
                Language::Ja => format!(
                    "MCP 設定が不完全です（endpoint または command が不足）: {}",
                    detail
                ),
                Language::En => detail.to_string(),
            }
        }
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
