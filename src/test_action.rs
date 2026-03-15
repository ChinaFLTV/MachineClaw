use std::{fs, path::Path, time::Instant};

use toml_edit::{DocumentMut, Item};

use crate::{
    cli::TestTarget,
    config::{self},
    config_action::{default_config_value_literal, is_known_config_key, known_config_keys},
    error::{AppError, ExitCode},
    i18n::{self, Language},
    mcp::validate_mcp_config,
    render::{self, ActionRenderData},
};

pub struct TestActionOutcome {
    pub rendered: String,
    pub exit_code: ExitCode,
}

#[derive(Debug, Default)]
struct ConfigTestReport {
    exists: bool,
    file_size_bytes: u128,
    syntax_errors: Vec<String>,
    semantic_errors: Vec<String>,
    unknown_keys: Vec<String>,
    redundant_keys: Vec<String>,
    empty_tables: Vec<String>,
    duplicate_hints: Vec<String>,
}

pub fn run_test_command(
    config_path: &Path,
    target: TestTarget,
    assets_dir: &Path,
) -> Result<TestActionOutcome, AppError> {
    match target {
        TestTarget::Config => run_test_config(config_path, target.as_str(), assets_dir),
    }
}

fn run_test_config(
    config_path: &Path,
    target: &str,
    assets_dir: &Path,
) -> Result<TestActionOutcome, AppError> {
    let started = Instant::now();
    let mut report = ConfigTestReport::default();
    let mut colorful = true;
    report.exists = config_path.exists();

    let raw = if report.exists {
        match fs::read_to_string(config_path) {
            Ok(content) => {
                report.file_size_bytes = content.len() as u128;
                if let Some(hint) = read_colorful_hint(&content) {
                    colorful = hint;
                }
                Some(content)
            }
            Err(err) => {
                report
                    .syntax_errors
                    .push(localized_file_read_error(config_path, &err.to_string()));
                None
            }
        }
    } else {
        report
            .syntax_errors
            .push(localized_file_not_found(config_path));
        None
    };

    if let Some(raw) = raw.as_deref() {
        if raw.trim().is_empty() {
            report.syntax_errors.push(localized_file_empty());
        } else {
            let syntax_ok = analyze_toml_structure(raw, &mut report);
            if syntax_ok {
                analyze_semantic(raw, config_path, &mut report);
            }
        }
    }

    sort_dedup(&mut report.syntax_errors);
    sort_dedup(&mut report.semantic_errors);
    sort_dedup(&mut report.unknown_keys);
    sort_dedup(&mut report.redundant_keys);
    sort_dedup(&mut report.empty_tables);
    sort_dedup(&mut report.duplicate_hints);

    let blocking_count = report.syntax_errors.len()
        + report.semantic_errors.len()
        + report.unknown_keys.len()
        + report.duplicate_hints.len();
    let warning_count = report.redundant_keys.len() + report.empty_tables.len();
    let status = if blocking_count == 0 {
        i18n::status_success()
    } else {
        i18n::status_failed()
    };

    let data = ActionRenderData {
        action: format!("test {target}"),
        status: status.to_string(),
        key_metrics: build_key_metrics(config_path, &report, blocking_count, warning_count),
        risk_summary: build_risk_summary(&report),
        ai_summary: build_ai_summary(blocking_count, warning_count),
        command_summary: build_command_summary(&report),
        elapsed: i18n::human_duration_ms(started.elapsed().as_millis()),
    };

    let rendered = render::render_action(
        assets_dir,
        "test",
        &data,
        render::resolve_colorful_enabled(colorful),
    )?;

    let exit_code = if blocking_count == 0 {
        ExitCode::Success
    } else {
        ExitCode::ConfigError
    };

    Ok(TestActionOutcome {
        rendered,
        exit_code,
    })
}

fn analyze_toml_structure(raw: &str, report: &mut ConfigTestReport) -> bool {
    match raw.parse::<DocumentMut>() {
        Ok(doc) => {
            let mut leaves = Vec::<(String, Item)>::new();
            collect_leaves(doc.as_item(), "", &mut leaves, &mut report.empty_tables);
            for (key, item) in leaves {
                if !is_allowed_key(&key) {
                    report.unknown_keys.push(key);
                    continue;
                }
                if let Some(default_literal) = default_config_value_literal(&key)
                    && item_matches_literal(&item, default_literal)
                {
                    report.redundant_keys.push(key);
                }
            }
            true
        }
        Err(err) => {
            let detail = err.to_string();
            report
                .syntax_errors
                .push(localized_toml_parse_error(&detail));
            if detail.to_ascii_lowercase().contains("duplicate key") {
                report
                    .duplicate_hints
                    .push(localized_duplicate_hint(&detail));
            }
            false
        }
    }
}

fn analyze_semantic(raw: &str, config_path: &Path, report: &mut ConfigTestReport) {
    let parsed = match config::parse_config_text(raw, &config_path.display().to_string()) {
        Ok(cfg) => cfg,
        Err(err) => {
            report
                .semantic_errors
                .push(localized_schema_parse_error(&err.to_string()));
            return;
        }
    };
    if let Err(err) = config::validate_config(&parsed) {
        report.semantic_errors.push(localized_app_error(err));
    }
    if let Err(err) = validate_mcp_config(&parsed.ai.tools.mcp, config_path) {
        report.semantic_errors.push(localized_app_error(err));
    }
}

fn collect_leaves(
    item: &Item,
    path: &str,
    leaves: &mut Vec<(String, Item)>,
    empty_tables: &mut Vec<String>,
) {
    if let Some(table_like) = item.as_table_like() {
        let mut has_children = false;
        for (key, child) in table_like.iter() {
            has_children = true;
            let next = if path.is_empty() {
                key.to_string()
            } else {
                format!("{path}.{key}")
            };
            collect_leaves(child, &next, leaves, empty_tables);
        }
        if !path.is_empty() && !has_children {
            empty_tables.push(path.to_string());
        }
        return;
    }
    if !path.is_empty() {
        leaves.push((path.to_string(), item.clone()));
    }
}

fn is_allowed_key(key: &str) -> bool {
    is_known_config_key(key)
}

fn item_matches_literal(item: &Item, literal: &str) -> bool {
    let item_literal = item.to_string().trim().to_string();
    match (
        parse_toml_literal_to_value(&item_literal),
        parse_toml_literal_to_value(literal),
    ) {
        (Some(left), Some(right)) => left == right,
        _ => item_literal == literal.trim(),
    }
}

fn parse_toml_literal_to_value(literal: &str) -> Option<toml::Value> {
    let wrapped = format!("value = {literal}");
    let parsed = toml::from_str::<toml::Table>(&wrapped).ok()?;
    parsed.get("value").cloned()
}

fn read_colorful_hint(raw: &str) -> Option<bool> {
    let parsed = toml::from_str::<toml::Value>(raw).ok()?;
    parsed
        .get("console")
        .and_then(|value| value.get("colorful"))
        .and_then(toml::Value::as_bool)
}

fn localized_file_not_found(path: &Path) -> String {
    match i18n::current_language() {
        Language::ZhCn => format!("配置文件不存在: {}", path.display()),
        Language::ZhTw => format!("設定檔不存在: {}", path.display()),
        Language::Fr => format!("fichier de configuration introuvable: {}", path.display()),
        Language::De => format!("Konfigurationsdatei nicht gefunden: {}", path.display()),
        Language::Ja => format!("設定ファイルが存在しません: {}", path.display()),
        Language::En => format!("config file not found: {}", path.display()),
    }
}

fn localized_file_read_error(path: &Path, detail: &str) -> String {
    let masked = detail.trim();
    match i18n::current_language() {
        Language::ZhCn => format!("读取配置文件失败 {}: {masked}", path.display()),
        Language::ZhTw => format!("讀取設定檔失敗 {}: {masked}", path.display()),
        Language::Fr => format!("échec de lecture du fichier {}: {masked}", path.display()),
        Language::De => format!("Lesefehler bei {}: {masked}", path.display()),
        Language::Ja => format!(
            "設定ファイルの読み取りに失敗しました {}: {masked}",
            path.display()
        ),
        Language::En => format!("failed to read config file {}: {masked}", path.display()),
    }
}

fn localized_file_empty() -> String {
    match i18n::current_language() {
        Language::ZhCn => "配置文件为空".to_string(),
        Language::ZhTw => "設定檔為空".to_string(),
        Language::Fr => "fichier de configuration vide".to_string(),
        Language::De => "Konfigurationsdatei ist leer".to_string(),
        Language::Ja => "設定ファイルが空です".to_string(),
        Language::En => "config file is empty".to_string(),
    }
}

fn localized_toml_parse_error(detail: &str) -> String {
    let masked = detail.trim();
    match i18n::current_language() {
        Language::ZhCn => format!("TOML 语法错误: {masked}"),
        Language::ZhTw => format!("TOML 語法錯誤: {masked}"),
        Language::Fr => format!("erreur de syntaxe TOML: {masked}"),
        Language::De => format!("TOML-Syntaxfehler: {masked}"),
        Language::Ja => format!("TOML 構文エラー: {masked}"),
        Language::En => format!("TOML syntax error: {masked}"),
    }
}

fn localized_duplicate_hint(detail: &str) -> String {
    let masked = detail.trim();
    match i18n::current_language() {
        Language::ZhCn => format!("疑似重复字段: {masked}"),
        Language::ZhTw => format!("疑似重複欄位: {masked}"),
        Language::Fr => format!("clé dupliquée suspectée: {masked}"),
        Language::De => format!("vermutlich doppelter Schlüssel: {masked}"),
        Language::Ja => format!("重複キーの可能性: {masked}"),
        Language::En => format!("possible duplicate key: {masked}"),
    }
}

fn localized_schema_parse_error(detail: &str) -> String {
    let masked = detail.trim();
    match i18n::current_language() {
        Language::ZhCn => format!("配置结构解析失败: {masked}"),
        Language::ZhTw => format!("配置結構解析失敗: {masked}"),
        Language::Fr => format!("échec de validation de schéma: {masked}"),
        Language::De => format!("Schema-Prüfung fehlgeschlagen: {masked}"),
        Language::Ja => format!("設定スキーマ解析に失敗しました: {masked}"),
        Language::En => format!("schema parse failed: {masked}"),
    }
}

fn localized_app_error(err: AppError) -> String {
    match err {
        AppError::Config(detail) => detail,
        AppError::Permission(detail) => detail,
        AppError::Ai(detail) => detail,
        AppError::Command(detail) => detail,
        AppError::Runtime(detail) => detail,
    }
}

fn build_key_metrics(
    config_path: &Path,
    report: &ConfigTestReport,
    blocking_count: usize,
    warning_count: usize,
) -> String {
    let checks_total = 8usize;
    let samples = known_config_keys().len();
    let size = if report.exists {
        i18n::human_bytes(report.file_size_bytes)
    } else {
        "-".to_string()
    };
    match i18n::current_language() {
        Language::ZhCn => format!(
            "- 检测目标: config\n- 配置文件: {}\n- 文件大小: {size}\n- 检测项: {}\n- 阻断问题: {}\n- 警告项: {}\n- 未知字段: {}\n- 冗余字段: {}\n- 空表: {}\n- 已知字段样本数: {}",
            config_path.display(),
            i18n::human_count_u128(checks_total as u128),
            i18n::human_count_u128(blocking_count as u128),
            i18n::human_count_u128(warning_count as u128),
            i18n::human_count_u128(report.unknown_keys.len() as u128),
            i18n::human_count_u128(report.redundant_keys.len() as u128),
            i18n::human_count_u128(report.empty_tables.len() as u128),
            i18n::human_count_u128(samples as u128)
        ),
        Language::ZhTw => format!(
            "- 檢測目標: config\n- 設定檔: {}\n- 檔案大小: {size}\n- 檢測項: {}\n- 阻斷問題: {}\n- 警告項: {}\n- 未知欄位: {}\n- 冗餘欄位: {}\n- 空表: {}\n- 已知欄位樣本數: {}",
            config_path.display(),
            i18n::human_count_u128(checks_total as u128),
            i18n::human_count_u128(blocking_count as u128),
            i18n::human_count_u128(warning_count as u128),
            i18n::human_count_u128(report.unknown_keys.len() as u128),
            i18n::human_count_u128(report.redundant_keys.len() as u128),
            i18n::human_count_u128(report.empty_tables.len() as u128),
            i18n::human_count_u128(samples as u128)
        ),
        Language::Fr => format!(
            "- Cible: config\n- Fichier: {}\n- Taille: {size}\n- Contrôles: {}\n- Problèmes bloquants: {}\n- Avertissements: {}\n- Clés inconnues: {}\n- Clés redondantes: {}\n- Tables vides: {}\n- Clés connues (échantillon): {}",
            config_path.display(),
            i18n::human_count_u128(checks_total as u128),
            i18n::human_count_u128(blocking_count as u128),
            i18n::human_count_u128(warning_count as u128),
            i18n::human_count_u128(report.unknown_keys.len() as u128),
            i18n::human_count_u128(report.redundant_keys.len() as u128),
            i18n::human_count_u128(report.empty_tables.len() as u128),
            i18n::human_count_u128(samples as u128)
        ),
        Language::De => format!(
            "- Ziel: config\n- Datei: {}\n- Größe: {size}\n- Prüfungen: {}\n- Blockierende Probleme: {}\n- Warnungen: {}\n- Unbekannte Schlüssel: {}\n- Redundante Schlüssel: {}\n- Leere Tabellen: {}\n- Bekannte Schlüssel (Stichprobe): {}",
            config_path.display(),
            i18n::human_count_u128(checks_total as u128),
            i18n::human_count_u128(blocking_count as u128),
            i18n::human_count_u128(warning_count as u128),
            i18n::human_count_u128(report.unknown_keys.len() as u128),
            i18n::human_count_u128(report.redundant_keys.len() as u128),
            i18n::human_count_u128(report.empty_tables.len() as u128),
            i18n::human_count_u128(samples as u128)
        ),
        Language::Ja => format!(
            "- 対象: config\n- 設定ファイル: {}\n- ファイルサイズ: {size}\n- チェック項目: {}\n- ブロッカー: {}\n- 警告: {}\n- 未知キー: {}\n- 冗長キー: {}\n- 空テーブル: {}\n- 既知キー数(サンプル): {}",
            config_path.display(),
            i18n::human_count_u128(checks_total as u128),
            i18n::human_count_u128(blocking_count as u128),
            i18n::human_count_u128(warning_count as u128),
            i18n::human_count_u128(report.unknown_keys.len() as u128),
            i18n::human_count_u128(report.redundant_keys.len() as u128),
            i18n::human_count_u128(report.empty_tables.len() as u128),
            i18n::human_count_u128(samples as u128)
        ),
        Language::En => format!(
            "- Target: config\n- Config file: {}\n- File size: {size}\n- Checks: {}\n- Blocking issues: {}\n- Warnings: {}\n- Unknown keys: {}\n- Redundant keys: {}\n- Empty tables: {}\n- Known key samples: {}",
            config_path.display(),
            i18n::human_count_u128(checks_total as u128),
            i18n::human_count_u128(blocking_count as u128),
            i18n::human_count_u128(warning_count as u128),
            i18n::human_count_u128(report.unknown_keys.len() as u128),
            i18n::human_count_u128(report.redundant_keys.len() as u128),
            i18n::human_count_u128(report.empty_tables.len() as u128),
            i18n::human_count_u128(samples as u128)
        ),
    }
}

fn build_risk_summary(report: &ConfigTestReport) -> String {
    if report.syntax_errors.is_empty()
        && report.semantic_errors.is_empty()
        && report.unknown_keys.is_empty()
        && report.duplicate_hints.is_empty()
        && report.redundant_keys.is_empty()
        && report.empty_tables.is_empty()
    {
        return i18n::risk_no_obvious().to_string();
    }

    let mut lines = Vec::<String>::new();
    append_issue_section(
        &mut lines,
        localized_group_title(
            "语法错误",
            "語法錯誤",
            "Syntax errors",
            "Erreurs de syntaxe",
            "Syntaxfehler",
            "構文エラー",
        ),
        &report.syntax_errors,
    );
    append_issue_section(
        &mut lines,
        localized_group_title(
            "结构/语义错误",
            "結構/語意錯誤",
            "Schema/Semantic errors",
            "Erreurs de schéma/sémantique",
            "Schema-/Semantikfehler",
            "スキーマ/意味エラー",
        ),
        &report.semantic_errors,
    );
    append_issue_section(
        &mut lines,
        localized_group_title(
            "未知字段",
            "未知欄位",
            "Unknown keys",
            "Clés inconnues",
            "Unbekannte Schlüssel",
            "未知キー",
        ),
        &report.unknown_keys,
    );
    append_issue_section(
        &mut lines,
        localized_group_title(
            "重复字段线索",
            "重複欄位線索",
            "Duplicate key hints",
            "Indices de clé dupliquée",
            "Hinweise auf doppelte Schlüssel",
            "重複キーの手掛かり",
        ),
        &report.duplicate_hints,
    );
    append_issue_section(
        &mut lines,
        localized_group_title(
            "冗余默认值",
            "冗餘預設值",
            "Redundant defaults",
            "Valeurs par défaut redondantes",
            "Redundante Standardwerte",
            "冗長なデフォルト値",
        ),
        &report.redundant_keys,
    );
    append_issue_section(
        &mut lines,
        localized_group_title(
            "空表",
            "空表",
            "Empty tables",
            "Tables vides",
            "Leere Tabellen",
            "空テーブル",
        ),
        &report.empty_tables,
    );
    lines.join("\n")
}

fn build_ai_summary(blocking_count: usize, warning_count: usize) -> String {
    match i18n::current_language() {
        Language::ZhCn => {
            if blocking_count > 0 {
                "配置测试未通过：请先修复阻断问题，再执行 prepare/inspect/chat。".to_string()
            } else if warning_count > 0 {
                "配置可用但存在清理项：建议清理冗余字段和空表，降低后续维护成本。".to_string()
            } else {
                "配置测试通过：结构、语义与字段完整性均正常，可继续执行后续动作。".to_string()
            }
        }
        Language::ZhTw => {
            if blocking_count > 0 {
                "設定測試未通過：請先修復阻斷問題，再執行 prepare/inspect/chat。".to_string()
            } else if warning_count > 0 {
                "設定可用但存在清理項：建議清理冗餘欄位與空表，降低維護成本。".to_string()
            } else {
                "設定測試通過：結構、語意與欄位完整性皆正常，可繼續後續動作。".to_string()
            }
        }
        Language::Fr => {
            if blocking_count > 0 {
                "Validation échouée: corrigez d'abord les problèmes bloquants avant prepare/inspect/chat.".to_string()
            } else if warning_count > 0 {
                "Configuration utilisable avec avertissements: nettoyez les clés redondantes et tables vides.".to_string()
            } else {
                "Validation réussie: structure, sémantique et intégrité des clés sont correctes."
                    .to_string()
            }
        }
        Language::De => {
            if blocking_count > 0 {
                "Validierung fehlgeschlagen: beheben Sie zuerst blockierende Probleme vor prepare/inspect/chat.".to_string()
            } else if warning_count > 0 {
                "Konfiguration nutzbar mit Warnungen: redundante Schlüssel und leere Tabellen bereinigen.".to_string()
            } else {
                "Validierung erfolgreich: Struktur, Semantik und Schlüsselintegrität sind korrekt."
                    .to_string()
            }
        }
        Language::Ja => {
            if blocking_count > 0 {
                "検証に失敗しました。prepare/inspect/chat の前にブロッカーを修正してください。"
                    .to_string()
            } else if warning_count > 0 {
                "設定は利用可能ですが警告があります。冗長キーと空テーブルの整理を推奨します。"
                    .to_string()
            } else {
                "検証に成功しました。構造・意味・キー整合性は正常です。".to_string()
            }
        }
        Language::En => {
            if blocking_count > 0 {
                "Config validation failed: fix blocking issues before running prepare/inspect/chat."
                    .to_string()
            } else if warning_count > 0 {
                "Config is usable with warnings: clean redundant keys and empty tables.".to_string()
            } else {
                "Config validation passed: structure, semantics, and key integrity are healthy."
                    .to_string()
            }
        }
    }
}

fn build_command_summary(report: &ConfigTestReport) -> String {
    let syntax_state = if report.syntax_errors.is_empty() {
        i18n::status_ok()
    } else {
        i18n::status_fail_short()
    };
    let semantic_skipped = !report.syntax_errors.is_empty();
    let semantic_state = if semantic_skipped {
        localized_skipped()
    } else if report.semantic_errors.is_empty() {
        i18n::status_ok()
    } else {
        i18n::status_fail_short()
    };
    let unknown_state = if report.unknown_keys.is_empty() {
        i18n::status_ok()
    } else {
        i18n::status_fail_short()
    };
    let redundant_state = if report.redundant_keys.is_empty() && report.empty_tables.is_empty() {
        i18n::status_ok()
    } else {
        i18n::status_fail_short()
    };

    match i18n::current_language() {
        Language::ZhCn => format!(
            "- [1] TOML 语法解析: {syntax_state}\n- [2] 配置结构解析(AppConfig): {semantic_state}\n- [3] 未知字段扫描: {unknown_state}\n- [4] 冗余默认值/空表扫描: {redundant_state}\n- [5] 语义约束校验(validate_config + validate_mcp_config): {semantic_state}"
        ),
        Language::ZhTw => format!(
            "- [1] TOML 語法解析: {syntax_state}\n- [2] 設定結構解析(AppConfig): {semantic_state}\n- [3] 未知欄位掃描: {unknown_state}\n- [4] 冗餘預設值/空表掃描: {redundant_state}\n- [5] 語意約束校驗(validate_config + validate_mcp_config): {semantic_state}"
        ),
        Language::Fr => format!(
            "- [1] Parse TOML: {syntax_state}\n- [2] Parse schéma (AppConfig): {semantic_state}\n- [3] Scan clés inconnues: {unknown_state}\n- [4] Scan redondance/tables vides: {redundant_state}\n- [5] Validation sémantique (validate_config + validate_mcp_config): {semantic_state}"
        ),
        Language::De => format!(
            "- [1] TOML-Parsing: {syntax_state}\n- [2] Schema-Parsing (AppConfig): {semantic_state}\n- [3] Scan unbekannter Schlüssel: {unknown_state}\n- [4] Scan Redundanz/leere Tabellen: {redundant_state}\n- [5] Semantik-Prüfung (validate_config + validate_mcp_config): {semantic_state}"
        ),
        Language::Ja => format!(
            "- [1] TOML 構文解析: {syntax_state}\n- [2] スキーマ解析(AppConfig): {semantic_state}\n- [3] 未知キー検出: {unknown_state}\n- [4] 冗長デフォルト/空テーブル検出: {redundant_state}\n- [5] 意味チェック(validate_config + validate_mcp_config): {semantic_state}"
        ),
        Language::En => format!(
            "- [1] TOML syntax parse: {syntax_state}\n- [2] Schema parse (AppConfig): {semantic_state}\n- [3] Unknown-key scan: {unknown_state}\n- [4] Redundant-default/empty-table scan: {redundant_state}\n- [5] Semantic validation (validate_config + validate_mcp_config): {semantic_state}"
        ),
    }
}

fn localized_skipped() -> &'static str {
    match i18n::current_language() {
        Language::ZhCn => "跳过",
        Language::ZhTw => "略過",
        Language::Fr => "ignoré",
        Language::De => "übersprungen",
        Language::Ja => "スキップ",
        Language::En => "skipped",
    }
}

fn append_issue_section(lines: &mut Vec<String>, title: &str, entries: &[String]) {
    if entries.is_empty() {
        return;
    }
    lines.push(format!(
        "- {title}: {}",
        i18n::human_count_u128(entries.len() as u128)
    ));
    for entry in entries.iter().take(6) {
        lines.push(format!("  - {}", trim_for_console(entry, 220)));
    }
    if entries.len() > 6 {
        lines.push(format!(
            "  - ... (+{})",
            i18n::human_count_u128((entries.len() - 6) as u128)
        ));
    }
}

fn trim_for_console(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(max_chars) {
        out.push(ch);
    }
    if text.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn sort_dedup(values: &mut Vec<String>) {
    values.sort();
    values.dedup();
}

fn localized_group_title<'a>(
    zh_cn: &'a str,
    zh_tw: &'a str,
    en: &'a str,
    fr: &'a str,
    de: &'a str,
    ja: &'a str,
) -> &'a str {
    match i18n::current_language() {
        Language::ZhCn => zh_cn,
        Language::ZhTw => zh_tw,
        Language::Fr => fr,
        Language::De => de,
        Language::Ja => ja,
        Language::En => en,
    }
}
