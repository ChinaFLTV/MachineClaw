use std::{
    fs,
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use colored::Colorize;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::{error::AppError, i18n};

const DEFAULT_SYSTEM_PROMPT: &str = "你是 MachineClaw 的系统巡检分析助手。\n\n目标：基于用户给出的关键指标与命令结果，输出可执行、可追溯、风险导向的结论。\n\n输出要求：\n1. 先给结论，再给证据。\n2. 风险等级必须明确（低/中/高）并说明触发原因。\n3. 单独列出异常命令（失败/超时/中断/拦截）及影响。\n4. 给出最多 3 条下一步建议，按优先级排序。\n5. 严禁输出敏感信息（token、cookie、密码、私钥、密钥路径等）。\n6. 文本简洁，避免空话，默认中文输出。\n";
const DEFAULT_PREPARE_PROMPT: &str = "请基于以下数据总结本次运行前检查结果。\n\n# action\n{{action}}\n\n# target\n{{target}}\n\n# key_metrics\n{{key_metrics}}\n\n# command_details\n{{command_details}}\n\n请按以下结构输出：\n1. 结论（是否可继续执行）\n2. 关键异常与风险等级\n3. 优先处理建议（最多 3 条）\n";
const DEFAULT_INSPECT_PROMPT: &str = "请基于以下数据总结本次状态检查结果。\n\n# action\n{{action}}\n\n# target\n{{target}}\n\n# key_metrics\n{{key_metrics}}\n\n# command_details\n{{command_details}}\n\n请按以下结构输出：\n1. 当前状态结论\n2. 关键证据与异常项\n3. 风险等级与触发原因\n4. 下一步建议（最多 3 条）\n";
const DEFAULT_CHAT_SYSTEM_PROMPT: &str = "你是 MachineClaw 的本机交互助手，负责系统巡检、诊断、风险分析与必要的本地命令执行。\n\n核心规则：\n1. 用户要求检查/排查/执行时，优先通过 function calling 调用工具获取事实，再回答。\n2. Shell 命令执行、Skill、MCP 是平级能力，必须按任务目标选择最合适的一种或组合，不能对任何一方形成固定偏置。\n3. 读命令优先；写命令仅在必要时使用，并先说明影响与风险。\n4. 对危险或高风险操作，必须提示风险、确认前置条件与回滚建议。\n5. 不得伪造命令结果；信息不足时明确缺失项并继续收集。\n6. 允许连续多轮工具调用，但应避免无意义重复调用；证据足够时要及时用文字总结。\n7. 只要模型任何一轮返回了可展示文本，就必须保留并向用户输出，不得因为随后还有工具调用而丢弃、覆盖或忽略。\n8. 输出结构固定：结论 -> 关键证据 -> 风险评估 -> 下一步。\n9. 严禁泄露敏感信息（token、cookie、密码、私钥、密钥路径等）。\n\n技能流程：\n1. 在处理复杂任务前，先查看可用 skills。\n2. 若匹配到 skill，则优先按对应 SKILL.md 的流程执行。\n3. 若使用了 skill，在回复中明确说明使用了哪个 skill。\n\n风格要求：\n- 简洁、专业、直击要点。\n- 默认中文，与用户语言保持一致。\n";
const DEFAULT_PREPARE_OUTPUT_TEMPLATE: &str = "# Preparation Report\n\nAction: {{action}}\nStatus: {{status}}\n\n## Overview\nKeyMetrics:\n{{key_metrics}}\n\n## Risks\nRiskSummary:\n{{risk_summary}}\n\n## AI Interpretation\nAISummary:\n{{ai_summary}}\n\n## Command Execution\nCommandSummary:\n{{command_summary}}\n\n## Elapsed\nElapsed: {{elapsed}}\n";
const DEFAULT_INSPECT_OUTPUT_TEMPLATE: &str = "Action: {{action}}\nStatus: {{status}}\nKeyMetrics:\n{{key_metrics}}\nRiskSummary:\n{{risk_summary}}\nAISummary:\n{{ai_summary}}\nCommandSummary:\n{{command_summary}}\nElapsed: {{elapsed}}\n";
const DEFAULT_CHAT_OUTPUT_TEMPLATE: &str = "Action: {{action}}\nStatus: {{status}}\nKeyMetrics:\n{{key_metrics}}\nRiskSummary:\n{{risk_summary}}\nAISummary:\n{{ai_summary}}\nCommandSummary:\n{{command_summary}}\nElapsed: {{elapsed}}\n";
const DEFAULT_TEST_OUTPUT_TEMPLATE: &str = "# Configuration Test Report\n\nAction: {{action}}\nStatus: {{status}}\n\n## Overview\nKeyMetrics:\n{{key_metrics}}\n\n## Findings\nRiskSummary:\n{{risk_summary}}\n\n## Assessment\nAISummary:\n{{ai_summary}}\n\n## Checks\nCommandSummary:\n{{command_summary}}\n\n## Elapsed\nElapsed: {{elapsed}}\n";

#[derive(Debug, Clone)]
pub struct ActionRenderData {
    pub action: String,
    pub status: String,
    pub key_metrics: String,
    pub risk_summary: String,
    pub ai_summary: String,
    pub command_summary: String,
    pub elapsed: String,
}

#[derive(Debug, Clone)]
pub struct AssetsSetup {
    pub path: PathBuf,
    pub notices: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatToolEventKind {
    Running,
    Success,
    Error,
    Timeout,
}

static INLINE_CODE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"`([^`\n]+)`").expect("valid inline code regex"));
static INLINE_BOLD_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\*\*([^\*\n]+)\*\*").expect("valid bold regex"));
static INLINE_BOLD_UNDERSCORE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"__([^_\n]+)__").expect("valid bold underscore regex"));
static INLINE_STRIKE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"~~([^~\n]+)~~").expect("valid strike regex"));
static INLINE_ITALIC_ASTERISK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\*([^*\n]+)\*").expect("valid italic asterisk regex"));
static INLINE_ITALIC_UNDERSCORE_SAFE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(^|[^[:alnum:]_])_([^_\n]+)_([^[:alnum:]_]|$)")
        .expect("valid safe italic underscore regex")
});
static INLINE_LINK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").expect("valid link regex"));
static INLINE_IMAGE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"!\[([^\]]*)\]\(([^)]+)\)").expect("valid image regex"));
static ORDERED_LIST_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(\d+)\.\s+(.*)$").expect("valid ordered list regex"));
static HORIZONTAL_RULE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[-*_]{3,}$").expect("valid horizontal rule regex"));

pub fn locate_or_init_assets_dir() -> Result<AssetsSetup, AppError> {
    let cwd = std::env::current_dir()
        .map_err(|err| AppError::Runtime(format!("failed to get current directory: {err}")))?;

    if let Ok(custom) = std::env::var("MACHINECLAW_ASSETS_DIR") {
        let custom_path = PathBuf::from(custom);
        let notices = ensure_default_assets(&custom_path)?;
        return Ok(AssetsSetup {
            path: custom_path,
            notices,
        });
    }

    let cwd_assets = cwd.join("assets");
    if cwd_assets.exists() {
        let notices = ensure_default_assets(&cwd_assets)?;
        return Ok(AssetsSetup {
            path: cwd_assets,
            notices,
        });
    }

    let exe = std::env::current_exe().map_err(|err| {
        AppError::Runtime(format!("failed to get current executable path: {err}"))
    })?;
    if let Some(parent) = exe.parent() {
        let exe_assets = parent.join("assets");
        if exe_assets.exists() {
            let notices = ensure_default_assets(&exe_assets)?;
            return Ok(AssetsSetup {
                path: exe_assets,
                notices,
            });
        }
    }

    let notices = ensure_default_assets(&cwd_assets)?;
    Ok(AssetsSetup {
        path: cwd_assets,
        notices,
    })
}

pub fn load_prompt_template(assets_dir: &Path, name: &str) -> Result<String, AppError> {
    read_asset_with_fallback(&assets_dir.join("prompts"), name)
        .map_err(|err| AppError::Runtime(format!("failed to read prompt template {}: {err}", name)))
}

pub fn render_action(
    assets_dir: &Path,
    template_name: &str,
    data: &ActionRenderData,
    colorful: bool,
) -> Result<String, AppError> {
    let mut raw = read_asset_with_fallback(
        &assets_dir.join("output_templates"),
        &format!("{template_name}.md"),
    )
    .map_err(|err| {
        AppError::Runtime(format!(
            "failed to read output template {template_name}.md: {err}"
        ))
    })?;

    raw = raw.replace("{{action}}", &data.action);
    raw = raw.replace("{{status}}", &data.status);
    raw = raw.replace("{{key_metrics}}", &data.key_metrics);
    raw = raw.replace("{{risk_summary}}", &data.risk_summary);
    raw = raw.replace("{{ai_summary}}", &data.ai_summary);
    raw = raw.replace("{{command_summary}}", &data.command_summary);
    raw = raw.replace("{{elapsed}}", &data.elapsed);
    raw = localize_output_labels(&raw);

    if !supports_color(colorful) {
        return Ok(raw);
    }

    Ok(colorize_output(&raw))
}

pub fn render_chat_notice(text: &str, colorful: bool) -> String {
    if !supports_color(colorful) {
        return text.to_string();
    }
    format!(
        "{} {}",
        i18n::chat_tag_info().bright_cyan().bold(),
        text.white()
    )
}

pub fn render_info_line(text: &str, colorful: bool) -> String {
    if !supports_color(colorful) {
        return format!("{}: {text}", i18n::prefix_info());
    }
    format!(
        "{} {}",
        format!("[{}]", i18n::prefix_info()).bright_cyan().bold(),
        text.white()
    )
}

pub fn render_warn_line(text: &str, colorful: bool) -> String {
    if !supports_color(colorful) {
        return format!("{}: {text}", i18n::prefix_warn());
    }
    format!(
        "{} {}",
        format!("[{}]", i18n::prefix_warn()).bright_yellow().bold(),
        text.bright_yellow()
    )
}

pub fn render_error_line(text: &str, colorful: bool) -> String {
    if !supports_color(colorful) {
        return format!("{}: {text}", i18n::prefix_error());
    }
    format!(
        "{} {}",
        format!("[{}]", i18n::prefix_error()).bright_red().bold(),
        text.bright_red()
    )
}

pub fn render_chat_warning(text: &str, colorful: bool) -> String {
    let multiline = text.contains('\n');
    let tag = i18n::chat_tag_warn();
    if !supports_color(colorful) {
        if multiline {
            return format!("{tag}\n{text}");
        }
        return format!("{tag} {text}");
    }
    if multiline {
        return format!("{}\n{}", tag.bright_red().bold(), text.bright_yellow());
    }
    format!("{} {}", tag.bright_red().bold(), text.bright_yellow())
}

pub fn render_chat_custom_tag_event(tag: &str, text: &str, colorful: bool) -> String {
    let multiline = text.contains('\n');
    if !supports_color(colorful) {
        if multiline {
            return format!("{tag}\n{text}");
        }
        return format!("{tag} {text}");
    }
    if multiline {
        return format!("{}\n{}", tag.bright_cyan().bold(), text.white());
    }
    format!("{} {}", tag.bright_cyan().bold(), text.white())
}

pub fn render_chat_user_prompt(prompt: &str, colorful: bool) -> String {
    if !supports_color(colorful) {
        return prompt.to_string();
    }
    prompt.bright_green().bold().to_string()
}

pub fn render_chat_tool_event(text: &str, kind: ChatToolEventKind, colorful: bool) -> String {
    let multiline = text.contains('\n');
    if !supports_color(colorful) {
        let tag = match kind {
            ChatToolEventKind::Running => i18n::chat_tag_tool(),
            ChatToolEventKind::Success => i18n::chat_tag_tool_ok(),
            ChatToolEventKind::Error => i18n::chat_tag_tool_err(),
            ChatToolEventKind::Timeout => i18n::chat_tag_tool_timeout(),
        };
        if multiline {
            return format!("{tag}\n{text}");
        }
        return format!("{tag} {text}");
    }
    let tag = match kind {
        ChatToolEventKind::Running => i18n::chat_tag_tool().bright_yellow().bold(),
        ChatToolEventKind::Success => i18n::chat_tag_tool_ok().bright_green().bold(),
        ChatToolEventKind::Error => i18n::chat_tag_tool_err().bright_red().bold(),
        ChatToolEventKind::Timeout => i18n::chat_tag_tool_timeout().bright_magenta().bold(),
    };
    if multiline {
        return format!("{tag}\n{}", text.white());
    }
    format!("{tag} {}", text.white())
}

pub fn render_chat_assistant_reply(prefix: &str, content: &str, colorful: bool) -> String {
    let body = render_markdown_for_terminal(content, colorful);
    if !supports_color(colorful) {
        return format!("{prefix}\n{}", indent_block(&body, "  "));
    }
    format!(
        "{}\n{}",
        prefix.bright_blue().bold(),
        indent_block(&body, "  ")
    )
}

pub fn print_chat_assistant_reply_stream(
    prefix: &str,
    content: &str,
    colorful: bool,
) -> Result<(), AppError> {
    let mut stdout = std::io::stdout();
    let body = render_markdown_for_terminal(content, false);
    if supports_color(colorful) {
        writeln!(stdout, "{}", prefix.bright_blue().bold())
            .map_err(|err| AppError::Command(format!("failed to write chat prefix: {err}")))?;
    } else {
        writeln!(stdout, "{prefix}")
            .map_err(|err| AppError::Command(format!("failed to write chat prefix: {err}")))?;
    }
    write!(stdout, "  ")
        .map_err(|err| AppError::Command(format!("failed to write chat indent: {err}")))?;
    stdout
        .flush()
        .map_err(|err| AppError::Command(format!("failed to flush chat output: {err}")))?;
    for ch in body.chars() {
        if ch == '\n' {
            writeln!(stdout)
                .map_err(|err| AppError::Command(format!("failed to stream chat output: {err}")))?;
            write!(stdout, "  ").map_err(|err| {
                AppError::Command(format!("failed to stream chat line indent: {err}"))
            })?;
        } else {
            write!(stdout, "{ch}")
                .map_err(|err| AppError::Command(format!("failed to stream chat output: {err}")))?;
        }
        stdout
            .flush()
            .map_err(|err| AppError::Command(format!("failed to flush stream output: {err}")))?;
        thread::sleep(Duration::from_millis(6));
    }
    writeln!(stdout)
        .map_err(|err| AppError::Command(format!("failed to finalize output: {err}")))?;
    Ok(())
}

pub fn render_chat_thinking(prefix: &str, content: &str, colorful: bool) -> String {
    let body = render_markdown_for_terminal(content, colorful);
    if !supports_color(colorful) {
        return format!("{prefix}\n{}", indent_block(&body, "  "));
    }
    let styled_body = body
        .lines()
        .map(|line| line.dimmed().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{}\n{}",
        prefix.bright_magenta().bold(),
        indent_block(&styled_body, "  ")
    )
}

pub fn render_markdown_for_terminal(text: &str, colorful: bool) -> String {
    if !supports_color(colorful) {
        return strip_markdown_for_plain(text);
    }

    let mut in_code_block = false;
    let mut lines = Vec::new();
    let raw_lines: Vec<&str> = text.lines().collect();
    let mut idx = 0usize;
    while idx < raw_lines.len() {
        let line = raw_lines[idx];
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            idx += 1;
            continue;
        }
        if in_code_block {
            lines.push(line.bright_black().to_string());
            idx += 1;
            continue;
        }
        if is_table_block_start(&raw_lines, idx) {
            let mut table_lines = Vec::new();
            while idx < raw_lines.len() && is_table_candidate(raw_lines[idx]) {
                table_lines.push(raw_lines[idx]);
                idx += 1;
            }
            lines.extend(render_markdown_table_block(&table_lines));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("# ") {
            lines.push(rest.bold().bright_white().to_string());
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("## ") {
            lines.push(rest.bold().bright_cyan().to_string());
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("### ") {
            lines.push(rest.bold().bright_blue().to_string());
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#### ") {
            lines.push(rest.bold().bright_blue().to_string());
            idx += 1;
            continue;
        }
        if HORIZONTAL_RULE_RE.is_match(trimmed) {
            lines.push("─".repeat(24).bright_black().to_string());
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- [ ] ") {
            lines.push(format!(
                "{} {}",
                "☐".bright_black(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- [x] ") {
            lines.push(format!(
                "{} {}",
                "☑".bright_green(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- [X] ") {
            lines.push(format!(
                "{} {}",
                "☑".bright_green(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* [ ] ") {
            lines.push(format!(
                "{} {}",
                "☐".bright_black(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* [x] ") {
            lines.push(format!(
                "{} {}",
                "☑".bright_green(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* [X] ") {
            lines.push(format!(
                "{} {}",
                "☑".bright_green(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- ") {
            lines.push(format!(
                "{} {}",
                "-".bright_black(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* ") {
            lines.push(format!(
                "{} {}",
                "-".bright_black(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(caps) = ORDERED_LIST_RE.captures(trimmed) {
            let order = caps.get(1).map(|m| m.as_str()).unwrap_or("1");
            let rest = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
            lines.push(format!(
                "{} {}",
                format!("{order}.").bright_black(),
                style_inline_markdown(rest)
            ));
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("> ") {
            lines.push(format!(
                "{} {}",
                "|".bright_black(),
                style_inline_markdown(rest).dimmed()
            ));
            idx += 1;
            continue;
        }
        lines.push(style_inline_markdown(line));
        idx += 1;
    }
    lines.join("\n")
}

fn strip_markdown_for_plain(text: &str) -> String {
    let mut in_code_block = false;
    let mut lines = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            lines.push(line.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("# ") {
            lines.push(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("## ") {
            lines.push(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("### ") {
            lines.push(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#### ") {
            lines.push(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- [ ] ") {
            lines.push(format!("[ ] {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- [x] ") {
            lines.push(format!("[x] {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- [X] ") {
            lines.push(format!("[x] {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* [ ] ") {
            lines.push(format!("[ ] {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* [x] ") {
            lines.push(format!("[x] {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* [X] ") {
            lines.push(format!("[x] {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- ") {
            lines.push(format!("- {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("* ") {
            lines.push(format!("- {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(caps) = ORDERED_LIST_RE.captures(trimmed) {
            let order = caps.get(1).map(|m| m.as_str()).unwrap_or("1");
            let rest = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
            lines.push(format!("{order}. {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("> ") {
            lines.push(format!("| {}", style_inline_markdown_plain(rest)));
            continue;
        }
        if HORIZONTAL_RULE_RE.is_match(trimmed) {
            lines.push("-".repeat(24));
            continue;
        }
        lines.push(style_inline_markdown_plain(line));
    }
    lines.join("\n")
}

fn ensure_default_assets(dir: &Path) -> Result<Vec<String>, AppError> {
    let mut notices = Vec::new();

    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create assets directory {}: {err}",
                dir.display()
            ))
        })?;
        notices.push(i18n::notice_assets_dir_created(dir));
    }

    let prompts_dir = dir.join("prompts");
    if !prompts_dir.exists() {
        fs::create_dir_all(&prompts_dir).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create prompts directory {}: {err}",
                prompts_dir.display()
            ))
        })?;
        notices.push(i18n::notice_prompts_dir_created(&prompts_dir));
    }

    let output_templates_dir = dir.join("output_templates");
    if !output_templates_dir.exists() {
        fs::create_dir_all(&output_templates_dir).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create output_templates directory {}: {err}",
                output_templates_dir.display()
            ))
        })?;
        notices.push(i18n::notice_output_templates_dir_created(
            &output_templates_dir,
        ));
    }

    ensure_file(
        &prompts_dir.join("system.md"),
        DEFAULT_SYSTEM_PROMPT,
        &mut notices,
    )?;
    ensure_file(
        &prompts_dir.join("prepare_user.md"),
        DEFAULT_PREPARE_PROMPT,
        &mut notices,
    )?;
    ensure_file(
        &prompts_dir.join("inspect_user.md"),
        DEFAULT_INSPECT_PROMPT,
        &mut notices,
    )?;
    ensure_file(
        &prompts_dir.join("chat_system.md"),
        DEFAULT_CHAT_SYSTEM_PROMPT,
        &mut notices,
    )?;
    ensure_file(
        &output_templates_dir.join("prepare.md"),
        DEFAULT_PREPARE_OUTPUT_TEMPLATE,
        &mut notices,
    )?;
    ensure_file(
        &output_templates_dir.join("inspect.md"),
        DEFAULT_INSPECT_OUTPUT_TEMPLATE,
        &mut notices,
    )?;
    ensure_file(
        &output_templates_dir.join("chat.md"),
        DEFAULT_CHAT_OUTPUT_TEMPLATE,
        &mut notices,
    )?;
    ensure_file(
        &output_templates_dir.join("test.md"),
        DEFAULT_TEST_OUTPUT_TEMPLATE,
        &mut notices,
    )?;

    Ok(notices)
}

fn ensure_file(
    path: &Path,
    default_content: &str,
    notices: &mut Vec<String>,
) -> Result<(), AppError> {
    if path.exists() {
        return Ok(());
    }
    fs::write(path, default_content).map_err(|err| {
        AppError::Runtime(format!(
            "failed to create default asset file {}: {err}",
            path.display()
        ))
    })?;
    notices.push(i18n::notice_asset_file_created(path));
    Ok(())
}

fn colorize_output(text: &str) -> String {
    let mut out = text.to_string();
    let action = format!("{}:", i18n::output_label_action());
    let status = format!("{}:", i18n::output_label_status());
    let key_metrics = format!("{}:", i18n::output_label_key_metrics());
    let risk_summary = format!("{}:", i18n::output_label_risk_summary());
    let ai_summary = format!("{}:", i18n::output_label_ai_summary());
    let command_summary = format!("{}:", i18n::output_label_command_summary());
    let elapsed = format!("{}:", i18n::output_label_elapsed());
    out = out.replace(
        &action,
        &format!("{} {}", "==".bright_black(), action.bright_cyan().bold()),
    );
    out = out.replace(
        &status,
        &format!("{} {}", "==".bright_black(), status.bright_cyan().bold()),
    );
    out = out.replace(
        &key_metrics,
        &format!(
            "{} {}",
            "==".bright_black(),
            key_metrics.bright_cyan().bold()
        ),
    );
    out = out.replace(
        &risk_summary,
        &format!(
            "{} {}",
            "==".bright_black(),
            risk_summary.bright_cyan().bold()
        ),
    );
    out = out.replace(
        &ai_summary,
        &format!(
            "{} {}",
            "==".bright_black(),
            ai_summary.bright_cyan().bold()
        ),
    );
    out = out.replace(
        &command_summary,
        &format!(
            "{} {}",
            "==".bright_black(),
            command_summary.bright_cyan().bold()
        ),
    );
    out = out.replace(
        &elapsed,
        &format!("{} {}", "==".bright_black(), elapsed.bright_cyan().bold()),
    );
    render_markdown_for_terminal(&out, true)
}

fn supports_color(colorful: bool) -> bool {
    resolve_colorful_enabled(colorful)
}

pub fn resolve_colorful_enabled(colorful: bool) -> bool {
    if !colorful {
        return false;
    }
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if let Ok(value) = std::env::var("TERM")
        && value.trim().eq_ignore_ascii_case("dumb")
    {
        return false;
    }
    if let Ok(force) = std::env::var("CLICOLOR_FORCE")
        && (force.trim() == "1" || force.trim().eq_ignore_ascii_case("true"))
    {
        return true;
    }
    std::io::stdout().is_terminal()
}

fn style_inline_markdown(line: &str) -> String {
    let mut text = line.to_string();
    text = INLINE_IMAGE_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            let alt = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            let url = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
            if alt.trim().is_empty() {
                format!("[image] {url}")
            } else {
                format!("{alt} ({url})")
            }
        })
        .to_string();
    text = INLINE_LINK_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            let label = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            let url = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
            format!("{label} ({url})")
        })
        .to_string();
    text = INLINE_BOLD_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            caps[1].bold().to_string()
        })
        .to_string();
    text = INLINE_BOLD_UNDERSCORE_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            caps[1].bold().to_string()
        })
        .to_string();
    text = INLINE_STRIKE_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            caps[1].bright_black().strikethrough().to_string()
        })
        .to_string();
    text = INLINE_ITALIC_ASTERISK_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            caps[1].italic().to_string()
        })
        .to_string();
    text = INLINE_ITALIC_UNDERSCORE_SAFE_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            let pre = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            let body = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
            let post = caps.get(3).map(|m| m.as_str()).unwrap_or_default();
            format!("{pre}{}{post}", body.italic())
        })
        .to_string();
    INLINE_CODE_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            caps[1].bright_yellow().to_string()
        })
        .to_string()
}

fn style_inline_markdown_plain(line: &str) -> String {
    let mut text = line.to_string();
    text = INLINE_IMAGE_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            let alt = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            let url = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
            if alt.trim().is_empty() {
                format!("[image] {url}")
            } else {
                format!("{alt} ({url})")
            }
        })
        .to_string();
    text = INLINE_LINK_RE
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            let label = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            let url = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
            format!("{label} ({url})")
        })
        .to_string();
    text = INLINE_BOLD_RE.replace_all(&text, "$1").to_string();
    text = INLINE_BOLD_UNDERSCORE_RE
        .replace_all(&text, "$1")
        .to_string();
    text = INLINE_STRIKE_RE.replace_all(&text, "$1").to_string();
    text = INLINE_ITALIC_ASTERISK_RE
        .replace_all(&text, "$1")
        .to_string();
    text = INLINE_ITALIC_UNDERSCORE_SAFE_RE
        .replace_all(&text, "$1$2$3")
        .to_string();
    INLINE_CODE_RE.replace_all(&text, "$1").to_string()
}

fn is_table_block_start(lines: &[&str], idx: usize) -> bool {
    if idx + 1 >= lines.len() {
        return false;
    }
    if !is_table_candidate(lines[idx]) || !is_table_candidate(lines[idx + 1]) {
        return false;
    }
    is_table_delimiter_row(lines[idx + 1])
}

fn is_table_candidate(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.ends_with('|') && trimmed.matches('|').count() >= 2
}

fn is_table_delimiter_row(line: &str) -> bool {
    parse_table_cells(line).iter().all(|cell| {
        let value = cell.trim();
        !value.is_empty() && value.chars().all(|ch| matches!(ch, '-' | ':' | ' '))
    })
}

fn parse_table_cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_start_matches('|')
        .trim_end_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn render_markdown_table_block(lines: &[&str]) -> Vec<String> {
    if lines.is_empty() {
        return Vec::new();
    }
    let header = parse_table_cells(lines[0]);
    let mut output = Vec::new();
    output.push(format_table_row(&header, true));
    output.push("─".repeat(32).bright_black().to_string());
    for line in lines.iter().skip(2) {
        let cells = parse_table_cells(line);
        output.push(format_table_row(&cells, false));
    }
    output
}

fn format_table_row(cells: &[String], header: bool) -> String {
    let separator = format!(" {} ", "│".bright_black());
    let row = cells
        .iter()
        .map(|cell| style_inline_markdown(cell))
        .collect::<Vec<_>>()
        .join(&separator);
    if header {
        return row.bold().bright_white().to_string();
    }
    row
}

fn indent_block(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn localize_output_labels(text: &str) -> String {
    let mut out = text.to_string();
    out = out.replace("Action:", &format!("{}:", i18n::output_label_action()));
    out = out.replace("Status:", &format!("{}:", i18n::output_label_status()));
    out = out.replace(
        "KeyMetrics:",
        &format!("{}:", i18n::output_label_key_metrics()),
    );
    out = out.replace(
        "RiskSummary:",
        &format!("{}:", i18n::output_label_risk_summary()),
    );
    out = out.replace(
        "AISummary:",
        &format!("{}:", i18n::output_label_ai_summary()),
    );
    out = out.replace(
        "CommandSummary:",
        &format!("{}:", i18n::output_label_command_summary()),
    );
    out = out.replace("Elapsed:", &format!("{}:", i18n::output_label_elapsed()));
    out
}

fn read_asset_with_fallback(dir: &Path, preferred_name: &str) -> Result<String, std::io::Error> {
    let preferred_path = dir.join(preferred_name);
    if preferred_path.exists() {
        return fs::read_to_string(preferred_path);
    }

    if let Some(fallback_name) = swap_extension(preferred_name) {
        let fallback_path = dir.join(fallback_name);
        if fallback_path.exists() {
            return fs::read_to_string(fallback_path);
        }
    }

    fs::read_to_string(preferred_path)
}

fn swap_extension(name: &str) -> Option<String> {
    if name.ends_with(".md") {
        return Some(name.trim_end_matches(".md").to_string() + ".txt");
    }
    if name.ends_with(".txt") {
        return Some(name.trim_end_matches(".txt").to_string() + ".md");
    }
    None
}
