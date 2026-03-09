use std::{
    fs,
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatStreamBlockKind {
    Assistant,
    Thinking,
}

static INLINE_CODE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"`([^`\n]+)`").expect("valid inline code regex"));
static INLINE_BOLD_ITALIC_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\*\*\*([^\*\n]+)\*\*\*").expect("valid bold italic regex"));
static INLINE_BOLD_ITALIC_UNDERSCORE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"___([^_\n]+)___").expect("valid bold italic underscore regex"));
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
static INLINE_AUTO_LINK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<((?:https?://|mailto:)[^>]+)>").expect("valid autolink regex"));
static INLINE_ESCAPE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"\\([\\`*_{}\[\]()#+\-.!>|])"#).expect("valid markdown escape regex")
});
static ORDERED_LIST_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(\d+)[\.\)]\s+(.*)$").expect("valid ordered list regex"));
static HORIZONTAL_RULE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?:\s*[-*_]\s*){3,}$").expect("valid horizontal rule regex"));

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

pub struct ChatStreamPrinter {
    colorful: bool,
    active_block: Option<ChatStreamBlockKind>,
    active_prefix: String,
    active_line_raw: String,
    frame_index: usize,
    stdout_is_terminal: bool,
    line_has_output: bool,
    in_code_block: bool,
}

impl ChatStreamPrinter {
    pub fn new(colorful: bool) -> Self {
        Self {
            colorful,
            active_block: None,
            active_prefix: String::new(),
            active_line_raw: String::new(),
            frame_index: 0,
            stdout_is_terminal: std::io::stdout().is_terminal(),
            line_has_output: false,
            in_code_block: false,
        }
    }

    pub fn write(
        &mut self,
        block: ChatStreamBlockKind,
        prefix: &str,
        content: &str,
    ) -> Result<(), AppError> {
        if content.is_empty() {
            return Ok(());
        }
        self.ensure_block(block, prefix)?;
        for segment in content.split_inclusive('\n') {
            let has_newline = segment.ends_with('\n');
            let text = if has_newline {
                segment.trim_end_matches('\n')
            } else {
                segment
            };
            if !text.is_empty() {
                if !text.trim_start().starts_with("```") {
                    self.ensure_line_prefix(block)?;
                }
                self.active_line_raw.push_str(text);
                self.flush_stable_prefix(block)?;
            }
            if has_newline {
                self.commit_current_line(block, text.is_empty())?;
            }
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), AppError> {
        if self.active_block.is_none() {
            return Ok(());
        }
        if !self.active_line_raw.is_empty() {
            if let Some(block) = self.active_block {
                self.flush_stable_prefix(block)?;
                self.flush_final_line_fragment(block, &self.active_line_raw.clone())?;
            }
        }
        if self.line_has_output {
            let mut stdout = std::io::stdout();
            writeln!(stdout)
                .map_err(|err| AppError::Command(format!("failed to finalize output: {err}")))?;
            stdout
                .flush()
                .map_err(|err| AppError::Command(format!("failed to flush final output: {err}")))?;
        }
        self.active_block = None;
        self.active_prefix.clear();
        self.active_line_raw.clear();
        self.line_has_output = false;
        self.in_code_block = false;
        Ok(())
    }

    fn ensure_block(&mut self, block: ChatStreamBlockKind, prefix: &str) -> Result<(), AppError> {
        if self.active_block == Some(block) && self.active_prefix == prefix {
            return Ok(());
        }
        if self.active_block.is_some() {
            self.finish()?;
        }
        self.active_block = Some(block);
        self.active_prefix = prefix.to_string();
        self.active_line_raw.clear();
        self.line_has_output = false;
        self.in_code_block = false;
        Ok(())
    }

    fn commit_current_line(
        &mut self,
        block: ChatStreamBlockKind,
        preserve_blank_line: bool,
    ) -> Result<(), AppError> {
        let trimmed = self.active_line_raw.trim_start();
        if trimmed.starts_with("```") {
            self.in_code_block = !self.in_code_block;
            self.active_line_raw.clear();
            self.line_has_output = false;
            return Ok(());
        }
        self.flush_stable_prefix(block)?;
        if !self.active_line_raw.is_empty() {
            self.flush_final_line_fragment(block, &self.active_line_raw.clone())?;
            self.active_line_raw.clear();
        }
        if self.line_has_output || preserve_blank_line {
            let mut stdout = std::io::stdout();
            writeln!(stdout)
                .map_err(|err| AppError::Command(format!("failed to commit streaming line: {err}")))?;
            stdout
                .flush()
                .map_err(|err| AppError::Command(format!("failed to flush committed line: {err}")))?;
        }
        self.active_line_raw.clear();
        self.line_has_output = false;
        Ok(())
    }

    fn flush_stable_prefix(&mut self, block: ChatStreamBlockKind) -> Result<(), AppError> {
        loop {
            let stable_len = find_stream_stable_prefix_len(&self.active_line_raw);
            if stable_len == 0 {
                break;
            }
            let stable = self.active_line_raw[..stable_len].to_string();
            self.active_line_raw.replace_range(..stable_len, "");
            self.flush_rendered_fragment(
                block,
                &render_stream_line_preview(&stable, self.colorful, self.in_code_block, block),
            )?;
        }
        Ok(())
    }

    fn flush_final_line_fragment(
        &mut self,
        block: ChatStreamBlockKind,
        fragment: &str,
    ) -> Result<(), AppError> {
        self.flush_rendered_fragment(
            block,
            &render_stream_line_final(fragment, self.colorful, self.in_code_block, block),
        )
    }

    fn flush_rendered_fragment(
        &mut self,
        block: ChatStreamBlockKind,
        fragment: &str,
    ) -> Result<(), AppError> {
        if fragment.is_empty() {
            return Ok(());
        }
        self.ensure_line_prefix(block)?;
        let mut stdout = std::io::stdout();
        write!(stdout, "{fragment}")
            .map_err(|err| AppError::Command(format!("failed to write stream fragment: {err}")))?;
        stdout
            .flush()
            .map_err(|err| AppError::Command(format!("failed to flush stream fragment: {err}")))?;
        Ok(())
    }

    fn ensure_line_prefix(&mut self, block: ChatStreamBlockKind) -> Result<(), AppError> {
        if self.line_has_output {
            return Ok(());
        }
        let prefix_text = render_stream_prefix(
            &self.active_prefix,
            self.colorful,
            self.stdout_is_terminal,
            block,
            self.frame_index,
        );
        self.frame_index = self.frame_index.wrapping_add(1);
        let mut stdout = std::io::stdout();
        write!(stdout, "{prefix_text} ")
            .map_err(|err| AppError::Command(format!("failed to write stream prefix: {err}")))?;
        stdout
            .flush()
            .map_err(|err| AppError::Command(format!("failed to flush stream prefix: {err}")))?;
        self.line_has_output = true;
        Ok(())
    }
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
    render_markdown_blocks(text, supports_color(colorful))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkdownListKind {
    Unordered,
    Ordered,
    Task(bool),
}

fn render_markdown_blocks(text: &str, colorful: bool) -> String {
    let expanded_lines = expand_compact_markdown_text(text);
    let line_refs = expanded_lines.iter().map(String::as_str).collect::<Vec<_>>();
    let mut lines = Vec::new();
    let mut in_code_block = false;
    let mut idx = 0usize;
    while idx < line_refs.len() {
        let line = line_refs[idx];
        let trimmed = line.trim_start();
        if is_code_fence_line(trimmed) {
            in_code_block = !in_code_block;
            idx += 1;
            continue;
        }
        if in_code_block {
            lines.push(render_code_block_line(line, colorful));
            idx += 1;
            continue;
        }
        if is_table_block_start(&line_refs, idx) {
            let mut table_lines = Vec::new();
            while idx < line_refs.len() && is_table_candidate(line_refs[idx]) {
                table_lines.push(line_refs[idx]);
                idx += 1;
            }
            lines.extend(render_markdown_table_block_mode(&table_lines, colorful));
            continue;
        }
        lines.push(render_markdown_structured_line(line, colorful));
        idx += 1;
    }
    lines.join("\n")
}

fn expand_compact_markdown_text(text: &str) -> Vec<String> {
    let mut output = Vec::new();
    for line in text.lines() {
        output.extend(expand_compact_markdown_line(line));
    }
    if text.ends_with('\n') {
        output.push(String::new());
    }
    output
}

fn expand_compact_markdown_line(line: &str) -> Vec<String> {
    if let Some((prefix, items)) = parse_compact_unordered_items(line)
        && items.len() >= 2
    {
        let mut output = Vec::new();
        if !prefix.trim().is_empty() {
            output.push(prefix.trim_end().to_string());
        }
        output.extend(items.into_iter().map(|item| format!("- {}", item.trim())));
        return output;
    }
    if let Some((prefix, items)) = parse_compact_ordered_items(line)
        && items.len() >= 2
    {
        let mut output = Vec::new();
        if !prefix.trim().is_empty() {
            output.push(prefix.trim_end().to_string());
        }
        output.extend(
            items
                .into_iter()
                .map(|(order, item)| format!("{order}. {}", item.trim())),
        );
        return output;
    }
    vec![line.to_string()]
}

fn render_markdown_structured_line(line: &str, colorful: bool) -> String {
    if line.trim().is_empty() {
        return String::new();
    }
    let normalized_line = normalize_compact_markdown_line(line);
    let (quote_depth, rest_after_quote) = split_blockquote_prefix(&normalized_line);
    let indent_width = count_indent_width(rest_after_quote);
    let nesting_level = indent_width / 2;
    let trimmed = rest_after_quote.trim_start();
    let block_prefix = render_block_prefix(quote_depth, colorful);
    let nested_indent = "  ".repeat(nesting_level);

    if let Some((level, content)) = parse_heading(trimmed) {
        return format!(
            "{block_prefix}{nested_indent}{}",
            render_heading_text(level, content, colorful)
        );
    }
    if is_horizontal_rule(trimmed) {
        return format!("{block_prefix}{nested_indent}{}", render_horizontal_rule(colorful));
    }
    if let Some((kind, marker, content)) = parse_list_line(trimmed) {
        return render_list_line(
            &block_prefix,
            nesting_level,
            kind,
            &marker,
            content,
            colorful,
        );
    }

    let content = render_inline_by_mode(trimmed, colorful);
    if quote_depth > 0 && colorful {
        return format!("{block_prefix}{nested_indent}{}", content.dimmed());
    }
    format!("{block_prefix}{nested_indent}{content}")
}

fn render_code_block_line(line: &str, colorful: bool) -> String {
    if colorful {
        return line.bright_black().to_string();
    }
    line.to_string()
}

fn render_heading_text(level: usize, text: &str, colorful: bool) -> String {
    let content = render_inline_by_mode(text.trim(), colorful);
    if !colorful {
        return content;
    }
    match level {
        1 => content.bold().bright_white().to_string(),
        2 => content.bold().bright_cyan().to_string(),
        3 => content.bold().bright_blue().to_string(),
        4 => content.bold().bright_blue().to_string(),
        5 => content.bold().bright_magenta().to_string(),
        _ => content.bold().bright_black().to_string(),
    }
}

fn render_horizontal_rule(colorful: bool) -> String {
    if colorful {
        return "─".repeat(24).bright_black().to_string();
    }
    "-".repeat(24)
}

fn render_list_line(
    block_prefix: &str,
    nesting_level: usize,
    kind: MarkdownListKind,
    marker: &str,
    content: &str,
    colorful: bool,
) -> String {
    let extra_indent = if nesting_level > 0 {
        "  ".repeat(nesting_level)
    } else {
        String::new()
    };
    let list_prefix = format!("{block_prefix}{extra_indent}");
    let styled_content = render_inline_by_mode(content.trim(), colorful);
    match kind {
        MarkdownListKind::Unordered => {
            if colorful {
                format!("{list_prefix}{} {}", marker.bright_black(), styled_content)
            } else {
                format!("{list_prefix}{marker} {styled_content}")
            }
        }
        MarkdownListKind::Ordered => {
            if colorful {
                format!("{list_prefix}{} {}", marker.bright_black(), styled_content)
            } else {
                format!("{list_prefix}{marker} {styled_content}")
            }
        }
        MarkdownListKind::Task(checked) => {
            if colorful {
                let checkbox = if checked {
                    "☑".bright_green().to_string()
                } else {
                    "☐".bright_black().to_string()
                };
                format!("{list_prefix}{checkbox} {styled_content}")
            } else {
                let checkbox = if checked { "[x]" } else { "[ ]" };
                format!("{list_prefix}{checkbox} {styled_content}")
            }
        }
    }
}

fn render_inline_by_mode(text: &str, colorful: bool) -> String {
    if colorful {
        return style_inline_markdown(text);
    }
    style_inline_markdown_plain(text)
}

fn split_blockquote_prefix(line: &str) -> (usize, &str) {
    let mut rest = line.trim_start();
    let mut depth = 0usize;
    loop {
        let candidate = rest.trim_start();
        if !candidate.starts_with('>') {
            break;
        }
        depth += 1;
        rest = &candidate[1..];
        rest = rest.strip_prefix(' ').unwrap_or(rest);
    }
    (depth, rest)
}

fn count_indent_width(line: &str) -> usize {
    line.chars()
        .take_while(|ch| matches!(ch, ' ' | '\t'))
        .map(|ch| if ch == '\t' { 4 } else { 1 })
        .sum()
}

fn render_block_prefix(quote_depth: usize, colorful: bool) -> String {
    let mut prefix = String::new();
    for _ in 0..quote_depth {
        if colorful {
            prefix.push_str(&format!("{} ", "│".bright_black()));
        } else {
            prefix.push_str("| ");
        }
    }
    prefix
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|ch| *ch == '#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let rest = trimmed[level..].trim_start();
    if rest.is_empty() {
        return None;
    }
    Some((level, rest))
}

fn is_horizontal_rule(line: &str) -> bool {
    HORIZONTAL_RULE_RE.is_match(line.trim())
}

fn parse_list_line(line: &str) -> Option<(MarkdownListKind, String, &str)> {
    let trimmed = line.trim_start();
    if trimmed.len() >= 6 {
        let chars: Vec<char> = trimmed.chars().collect();
        if matches!(chars.first(), Some('-' | '*' | '+'))
            && chars.get(1) == Some(&' ')
            && chars.get(2) == Some(&'[')
            && matches!(chars.get(3), Some(' ' | 'x' | 'X'))
            && chars.get(4) == Some(&']')
            && chars.get(5) == Some(&' ')
        {
            let checked = matches!(chars.get(3), Some('x' | 'X'));
            let content = &trimmed[6..];
            return Some((MarkdownListKind::Task(checked), String::new(), content));
        }
    }
    if let Some(caps) = ORDERED_LIST_RE.captures(trimmed) {
        let order = caps.get(1).map(|m| m.as_str()).unwrap_or("1");
        let content = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
        return Some((MarkdownListKind::Ordered, format!("{order}."), content));
    }
    if let Some(marker) = trimmed.chars().next()
        && matches!(marker, '-' | '*' | '+')
        && trimmed[marker.len_utf8()..].starts_with(' ')
    {
        let content = trimmed[marker.len_utf8() + 1..].trim_start();
        return Some((MarkdownListKind::Unordered, "-".to_string(), content));
    }
    None
}

fn is_code_fence_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
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

fn render_stream_line_preview(
    line: &str,
    colorful: bool,
    in_code_block: bool,
    block: ChatStreamBlockKind,
) -> String {
    if line.is_empty() {
        return String::new();
    }
    if line.trim_start().starts_with("```") {
        return String::new();
    }
    if in_code_block {
        if supports_color(colorful) {
            return line.bright_black().to_string();
        }
        return line.to_string();
    }

    let normalized = close_unfinished_inline_markdown(&normalize_compact_markdown_line(line));
    let rendered = render_single_markdown_line(&normalized, colorful);
    if block == ChatStreamBlockKind::Thinking && supports_color(colorful) && !rendered.is_empty() {
        return format!("\x1b[2m{rendered}\x1b[0m");
    }
    rendered
}

fn render_stream_line_final(
    line: &str,
    colorful: bool,
    in_code_block: bool,
    block: ChatStreamBlockKind,
) -> String {
    if line.is_empty() {
        return String::new();
    }
    if line.trim_start().starts_with("```") {
        return String::new();
    }
    if in_code_block {
        if supports_color(colorful) {
            return apply_stream_block_style(line.bright_black().to_string(), colorful, block);
        }
        return line.to_string();
    }
    apply_stream_block_style(
        render_single_markdown_line(&normalize_compact_markdown_line(line), colorful),
        colorful,
        block,
    )
}

fn apply_stream_block_style(text: String, colorful: bool, block: ChatStreamBlockKind) -> String {
    if block == ChatStreamBlockKind::Thinking && supports_color(colorful) && !text.is_empty() {
        return format!("\x1b[2m{text}\x1b[0m");
    }
    text
}

fn find_stream_stable_prefix_len(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    if text.trim_start().starts_with("```") {
        return 0;
    }

    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut idx = 0usize;
    let mut in_inline_code = false;
    let mut in_bold_star = false;
    let mut in_bold_underscore = false;
    let mut in_strike = false;
    let mut in_italic_star = false;
    let mut in_italic_underscore = false;
    let mut bracket_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut stable_end = 0usize;
    let mut escaped = false;

    while idx < chars.len() {
        let (byte_idx, ch) = chars[idx];
        if escaped {
            escaped = false;
            if stream_safe_emit_char(ch)
                && stream_inline_state_closed(
                    in_inline_code,
                    in_bold_star,
                    in_bold_underscore,
                    in_strike,
                    in_italic_star,
                    in_italic_underscore,
                    bracket_depth,
                    paren_depth,
                )
            {
                stable_end = byte_idx + ch.len_utf8();
            }
            idx += 1;
            continue;
        }

        if ch == '\\' {
            escaped = true;
            idx += 1;
            continue;
        }

        let next = chars.get(idx + 1).map(|(_, item)| *item);
        if ch == '`' {
            in_inline_code = !in_inline_code;
            if !in_inline_code {
                stable_end = byte_idx + ch.len_utf8();
            }
            idx += 1;
            continue;
        }

        if !in_inline_code && ch == '*' && next == Some('*') {
            in_bold_star = !in_bold_star;
            if stream_inline_state_closed(
                in_inline_code,
                in_bold_star,
                in_bold_underscore,
                in_strike,
                in_italic_star,
                in_italic_underscore,
                bracket_depth,
                paren_depth,
            ) {
                stable_end = chars[idx + 1].0 + chars[idx + 1].1.len_utf8();
            }
            idx += 2;
            continue;
        }

        if !in_inline_code && ch == '_' && next == Some('_') {
            in_bold_underscore = !in_bold_underscore;
            if stream_inline_state_closed(
                in_inline_code,
                in_bold_star,
                in_bold_underscore,
                in_strike,
                in_italic_star,
                in_italic_underscore,
                bracket_depth,
                paren_depth,
            ) {
                stable_end = chars[idx + 1].0 + chars[idx + 1].1.len_utf8();
            }
            idx += 2;
            continue;
        }

        if !in_inline_code && ch == '~' && next == Some('~') {
            in_strike = !in_strike;
            if stream_inline_state_closed(
                in_inline_code,
                in_bold_star,
                in_bold_underscore,
                in_strike,
                in_italic_star,
                in_italic_underscore,
                bracket_depth,
                paren_depth,
            ) {
                stable_end = chars[idx + 1].0 + chars[idx + 1].1.len_utf8();
            }
            idx += 2;
            continue;
        }

        if !in_inline_code && ch == '*' {
            in_italic_star = !in_italic_star;
            idx += 1;
            continue;
        }

        if !in_inline_code && ch == '_' {
            in_italic_underscore = !in_italic_underscore;
            idx += 1;
            continue;
        }

        if !in_inline_code && ch == '[' {
            bracket_depth = bracket_depth.saturating_add(1);
            idx += 1;
            continue;
        }

        if !in_inline_code && ch == ']' && bracket_depth > 0 {
            bracket_depth -= 1;
            if next == Some('(') {
                paren_depth = paren_depth.saturating_add(1);
                idx += 2;
                continue;
            }
        }

        if !in_inline_code && ch == ')' && paren_depth > 0 {
            paren_depth -= 1;
            if stream_inline_state_closed(
                in_inline_code,
                in_bold_star,
                in_bold_underscore,
                in_strike,
                in_italic_star,
                in_italic_underscore,
                bracket_depth,
                paren_depth,
            ) {
                stable_end = byte_idx + ch.len_utf8();
            }
            idx += 1;
            continue;
        }

        if stream_safe_emit_char(ch)
            && stream_inline_state_closed(
                in_inline_code,
                in_bold_star,
                in_bold_underscore,
                in_strike,
                in_italic_star,
                in_italic_underscore,
                bracket_depth,
                paren_depth,
            )
        {
            stable_end = byte_idx + ch.len_utf8();
        }
        idx += 1;
    }

    stable_end
}

fn stream_inline_state_closed(
    in_inline_code: bool,
    in_bold_star: bool,
    in_bold_underscore: bool,
    in_strike: bool,
    in_italic_star: bool,
    in_italic_underscore: bool,
    bracket_depth: usize,
    paren_depth: usize,
) -> bool {
    !in_inline_code
        && !in_bold_star
        && !in_bold_underscore
        && !in_strike
        && !in_italic_star
        && !in_italic_underscore
        && bracket_depth == 0
        && paren_depth == 0
}

fn stream_safe_emit_char(ch: char) -> bool {
    !matches!(ch, '\\')
}

fn render_single_markdown_line(line: &str, colorful: bool) -> String {
    render_markdown_structured_line(line, colorful)
}

fn close_unfinished_inline_markdown(line: &str) -> String {
    let mut normalized = line.to_string();
    if count_non_overlapping_token(line, "~~") % 2 == 1 {
        normalized.push_str("~~");
    }
    if count_non_overlapping_token(line, "**") % 2 == 1 {
        normalized.push_str("**");
    }
    if count_non_overlapping_token(line, "__") % 2 == 1 {
        normalized.push_str("__");
    }
    if count_unescaped_marker(line, '`') % 2 == 1 {
        normalized.push('`');
    }
    if count_single_markers(line, '*') % 2 == 1 {
        normalized.push('*');
    }
    if count_single_markers(line, '_') % 2 == 1 {
        normalized.push('_');
    }
    normalized
}

fn count_non_overlapping_token(text: &str, token: &str) -> usize {
    let mut count = 0usize;
    let mut rest = text;
    while let Some(pos) = rest.find(token) {
        count += 1;
        rest = &rest[pos + token.len()..];
    }
    count
}

fn count_unescaped_marker(text: &str, marker: char) -> usize {
    let mut count = 0usize;
    let mut escaped = false;
    for ch in text.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == marker {
            count += 1;
        }
    }
    count
}

fn count_single_markers(text: &str, marker: char) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0usize;
    let mut idx = 0usize;
    while idx < chars.len() {
        if chars[idx] == '\\' {
            idx += 2;
            continue;
        }
        if chars[idx] != marker {
            idx += 1;
            continue;
        }
        let prev_same = idx > 0 && chars[idx - 1] == marker;
        let next_same = idx + 1 < chars.len() && chars[idx + 1] == marker;
        if !prev_same && !next_same {
            count += 1;
        }
        idx += 1;
    }
    count
}

fn render_stream_prefix(
    prefix: &str,
    colorful: bool,
    stdout_is_terminal: bool,
    block: ChatStreamBlockKind,
    frame_index: usize,
) -> String {
    let prefix = prefix.trim_end();
    if !colorful || !stdout_is_terminal {
        return prefix.to_string();
    }
    if std::env::var_os("NO_COLOR").is_some() {
        return prefix.to_string();
    }
    if let Ok(value) = std::env::var("TERM")
        && value.trim().eq_ignore_ascii_case("dumb")
    {
        return prefix.to_string();
    }
    let visible_chars = prefix.chars().count().max(1);
    let highlight_center = frame_index % visible_chars;
    prefix
        .chars()
        .enumerate()
        .map(|(idx, ch)| {
            let distance = idx.abs_diff(highlight_center);
            let text = ch.to_string();
            match distance {
                0 => style_stream_prefix_segment(&text, 255, true),
                1 => style_stream_prefix_segment(&text, 232, true),
                2 => style_stream_prefix_segment(&text, 205, true),
                3 => style_stream_prefix_segment(&text, 165, false),
                _ => match block {
                    ChatStreamBlockKind::Assistant => {
                        style_stream_prefix_segment(&text, 128, false)
                    }
                    ChatStreamBlockKind::Thinking => {
                        style_stream_prefix_segment(&text, 148, false)
                    }
                },
            }
        })
        .collect::<String>()
}

fn style_stream_prefix_segment(text: &str, gray: u8, bold: bool) -> String {
    if bold {
        return format!("\x1b[1;38;2;{gray};{gray};{gray}m{text}\x1b[0m");
    }
    format!("\x1b[38;2;{gray};{gray};{gray}m{text}\x1b[0m")
}

fn normalize_compact_markdown_line(line: &str) -> String {
    if let Some((prefix, items)) = parse_compact_unordered_items(line)
        && items.len() >= 2
    {
        return format!("{prefix}{}", format_compact_unordered_items(&items));
    }
    if let Some((prefix, items)) = parse_compact_ordered_items(line)
        && items.len() >= 2
    {
        return format!("{prefix}{}", format_compact_ordered_items(&items));
    }
    line.to_string()
}

fn parse_compact_unordered_items(line: &str) -> Option<(String, Vec<String>)> {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut start_idx = None;
    for (idx, (byte_idx, ch)) in chars.iter().enumerate() {
        if *ch != '-' {
            continue;
        }
        let next = chars.get(idx + 1).map(|(_, c)| *c);
        if !matches!(next, Some(next_ch) if !next_ch.is_whitespace() && next_ch != '-') {
            continue;
        }
        let prev = if idx == 0 {
            None
        } else {
            Some(chars[idx - 1].1)
        };
        if prev.is_none() || prev.is_some_and(is_compact_list_boundary) {
            start_idx = Some(*byte_idx);
            break;
        }
    }
    let start_idx = start_idx?;
    let suffix = &line[start_idx..];
    let items = split_compact_unordered_suffix(suffix);
    if items.len() < 2 {
        return None;
    }
    Some((line[..start_idx].to_string(), items))
}

fn split_compact_unordered_suffix(suffix: &str) -> Vec<String> {
    let chars: Vec<(usize, char)> = suffix.char_indices().collect();
    let mut items = Vec::new();
    let mut current = String::new();
    let mut idx = 0usize;
    while idx < chars.len() {
        let ch = chars[idx].1;
        if ch == '-' {
            let next = chars.get(idx + 1).map(|(_, c)| *c);
            if !matches!(next, Some(next_ch) if !next_ch.is_whitespace() && next_ch != '-') {
                current.push(ch);
                idx += 1;
                continue;
            }
            if !current.trim().is_empty() {
                items.push(current.trim().to_string());
                current.clear();
            }
            idx += 1;
            continue;
        }
        current.push(ch);
        idx += 1;
    }
    if !current.trim().is_empty() {
        items.push(current.trim().to_string());
    }
    items
}

fn format_compact_unordered_items(items: &[String]) -> String {
    items
        .iter()
        .map(|item| format!(" • {}", item.trim()))
        .collect::<String>()
}

fn parse_compact_ordered_items(line: &str) -> Option<(String, Vec<(usize, String)>)> {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut idx = 0usize;
    while idx < chars.len() {
        if !chars[idx].1.is_ascii_digit() {
            idx += 1;
            continue;
        }
        let prev = if idx == 0 {
            None
        } else {
            Some(chars[idx - 1].1)
        };
        if prev.is_some_and(|ch| !is_compact_list_boundary(ch)) {
            idx += 1;
            continue;
        }
        let number_start = chars[idx].0;
        let mut number_end = idx;
        while number_end + 1 < chars.len() && chars[number_end + 1].1.is_ascii_digit() {
            number_end += 1;
        }
        if chars.get(number_end + 1).map(|(_, ch)| *ch) != Some('.') {
            idx += 1;
            continue;
        }
        if !matches!(
            chars.get(number_end + 2).map(|(_, ch)| *ch),
            Some(next_ch) if !next_ch.is_whitespace()
        ) {
            idx += 1;
            continue;
        }
        let prefix = line[..number_start].to_string();
        let suffix = &line[number_start..];
        let items = split_compact_ordered_suffix(suffix);
        if items.len() < 2 {
            return None;
        }
        return Some((prefix, items));
    }
    None
}

fn split_compact_ordered_suffix(suffix: &str) -> Vec<(usize, String)> {
    let chars: Vec<(usize, char)> = suffix.char_indices().collect();
    let mut items = Vec::new();
    let mut idx = 0usize;
    while idx < chars.len() {
        if !chars[idx].1.is_ascii_digit() {
            break;
        }
        let number_start = idx;
        while idx + 1 < chars.len() && chars[idx + 1].1.is_ascii_digit() {
            idx += 1;
        }
        if chars.get(idx + 1).map(|(_, ch)| *ch) != Some('.') {
            break;
        }
        let Some(order) = suffix[chars[number_start].0..chars[idx].0 + chars[idx].1.len_utf8()]
            .parse::<usize>()
            .ok()
        else {
            break;
        };
        idx += 2;
        let content_start = chars.get(idx).map(|(offset, _)| *offset).unwrap_or(suffix.len());
        let mut content_end = suffix.len();
        let mut probe = idx;
        while probe < chars.len() {
            if chars[probe].1.is_ascii_digit() {
                let mut end = probe;
                while end + 1 < chars.len() && chars[end + 1].1.is_ascii_digit() {
                    end += 1;
                }
                if chars.get(end + 1).map(|(_, ch)| *ch) == Some('.')
                    && matches!(chars.get(end + 2).map(|(_, ch)| *ch), Some(next_ch) if !next_ch.is_whitespace())
                {
                    content_end = chars[probe].0;
                    idx = probe;
                    break;
                }
            }
            probe += 1;
        }
        if probe >= chars.len() {
            idx = chars.len();
        }
        let item = suffix[content_start..content_end].trim().to_string();
        if item.is_empty() {
            break;
        }
        items.push((order, item));
    }
    items
}

fn format_compact_ordered_items(items: &[(usize, String)]) -> String {
    items
        .iter()
        .map(|(order, item)| format!(" {order}. {}", item.trim()))
        .collect::<String>()
}

fn is_compact_list_boundary(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            ':' | '：' | ';' | '；' | ',' | '，' | '、' | '(' | '（' | '[' | '【'
        )
}

fn style_inline_markdown(line: &str) -> String {
    render_inline_markdown_mode(line, true)
}

fn style_inline_markdown_plain(line: &str) -> String {
    render_inline_markdown_mode(line, false)
}

fn render_inline_markdown_mode(line: &str, colorful: bool) -> String {
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
    text = INLINE_AUTO_LINK_RE
        .replace_all(&text, "$1")
        .to_string();
    if colorful {
        text = INLINE_BOLD_ITALIC_RE
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                caps[1].bold().italic().to_string()
            })
            .to_string();
        text = INLINE_BOLD_ITALIC_UNDERSCORE_RE
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                caps[1].bold().italic().to_string()
            })
            .to_string();
        text = INLINE_BOLD_RE
            .replace_all(&text, |caps: &regex::Captures<'_>| caps[1].bold().to_string())
            .to_string();
        text = INLINE_BOLD_UNDERSCORE_RE
            .replace_all(&text, |caps: &regex::Captures<'_>| caps[1].bold().to_string())
            .to_string();
        text = INLINE_STRIKE_RE
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                caps[1].bright_black().strikethrough().to_string()
            })
            .to_string();
        text = INLINE_ITALIC_ASTERISK_RE
            .replace_all(&text, |caps: &regex::Captures<'_>| caps[1].italic().to_string())
            .to_string();
        text = INLINE_ITALIC_UNDERSCORE_SAFE_RE
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                let pre = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
                let body = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
                let post = caps.get(3).map(|m| m.as_str()).unwrap_or_default();
                format!("{pre}{}{post}", body.italic())
            })
            .to_string();
        text = INLINE_CODE_RE
            .replace_all(&text, |caps: &regex::Captures<'_>| {
                caps[1].bright_yellow().to_string()
            })
            .to_string();
    } else {
        text = INLINE_BOLD_ITALIC_RE.replace_all(&text, "$1").to_string();
        text = INLINE_BOLD_ITALIC_UNDERSCORE_RE
            .replace_all(&text, "$1")
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
        text = INLINE_CODE_RE.replace_all(&text, "$1").to_string();
    }
    INLINE_ESCAPE_RE.replace_all(&text, "$1").to_string()
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

fn render_markdown_table_block_mode(lines: &[&str], colorful: bool) -> Vec<String> {
    if lines.is_empty() {
        return Vec::new();
    }
    let header = parse_table_cells(lines[0]);
    let mut output = Vec::new();
    output.push(format_table_row_mode(&header, true, colorful));
    output.push(if colorful {
        "─".repeat(32).bright_black().to_string()
    } else {
        "-".repeat(32)
    });
    for line in lines.iter().skip(2) {
        let cells = parse_table_cells(line);
        output.push(format_table_row_mode(&cells, false, colorful));
    }
    output
}

fn format_table_row_mode(cells: &[String], header: bool, colorful: bool) -> String {
    let separator = if colorful {
        format!(" {} ", "│".bright_black())
    } else {
        " | ".to_string()
    };
    let row = cells
        .iter()
        .map(|cell| render_inline_by_mode(cell, colorful))
        .collect::<Vec<_>>()
        .join(&separator);
    if header && colorful {
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

#[cfg(test)]
mod tests {
    use super::{
        ChatStreamBlockKind, close_unfinished_inline_markdown, find_stream_stable_prefix_len,
        normalize_compact_markdown_line, render_markdown_for_terminal, render_single_markdown_line,
        render_stream_line_preview, render_stream_prefix,
    };
    use regex::Regex;

    fn strip_ansi(text: &str) -> String {
        let ansi_re = Regex::new(r"\x1b\[[0-9;]*m").expect("valid ansi regex");
        ansi_re.replace_all(text, "").into_owned()
    }

    #[test]
    fn closes_unfinished_inline_markdown_for_stream_preview() {
        assert_eq!(close_unfinished_inline_markdown("**bold"), "**bold**");
        assert_eq!(close_unfinished_inline_markdown("`code"), "`code`");
        assert_eq!(close_unfinished_inline_markdown("*italic"), "*italic*");
    }

    #[test]
    fn stream_preview_hides_unfinished_markdown_symbols() {
        let rendered =
            render_stream_line_preview("**结论", false, false, ChatStreamBlockKind::Assistant);
        assert_eq!(rendered, "结论");
    }

    #[test]
    fn stream_prefix_degrades_to_static_when_color_disabled() {
        assert_eq!(
            render_stream_prefix("[MachineClaw]", false, true, ChatStreamBlockKind::Assistant, 3),
            "[MachineClaw]"
        );
    }

    #[test]
    fn stream_prefix_preserves_visible_thinking_text() {
        let rendered = render_stream_prefix(
            "[MachineClaw-思考]",
            true,
            true,
            ChatStreamBlockKind::Thinking,
            4,
        );
        assert_eq!(strip_ansi(&rendered), "[MachineClaw-思考]");
    }

    #[test]
    fn stream_prefix_degrades_to_static_when_terminal_is_unavailable() {
        let rendered =
            render_stream_prefix("[MachineClaw]", true, false, ChatStreamBlockKind::Assistant, 2);
        assert_eq!(rendered, "[MachineClaw]");
    }

    #[test]
    fn single_line_markdown_render_formats_lists() {
        let rendered = render_single_markdown_line("- **item**", false);
        assert_eq!(rendered, "- item");
    }

    #[test]
    fn stable_prefix_waits_for_markdown_closure() {
        assert_eq!(find_stream_stable_prefix_len("**结论"), 0);
        assert_eq!(
            find_stream_stable_prefix_len("**结论** 后续"),
            "**结论** 后续".len()
        );
    }

    #[test]
    fn normalizes_compact_unordered_items() {
        assert_eq!(
            normalize_compact_markdown_line("比如：-系统状态检查-日志分析-性能诊断"),
            "比如： • 系统状态检查 • 日志分析 • 性能诊断"
        );
    }

    #[test]
    fn normalizes_compact_ordered_items() {
        assert_eq!(
            normalize_compact_markdown_line("步骤：1.检查配置2.查看日志3.分析结果"),
            "步骤： 1. 检查配置 2. 查看日志 3. 分析结果"
        );
    }

    #[test]
    fn renders_extended_heading_levels_in_plain_mode() {
        assert_eq!(render_markdown_for_terminal("###### heading", false), "heading");
    }

    #[test]
    fn renders_nested_blockquote_and_list_in_plain_mode() {
        let rendered = render_markdown_for_terminal("> > - item", false);
        assert_eq!(rendered, "| | - item");
    }

    #[test]
    fn renders_plus_list_and_ordered_parenthesis_in_plain_mode() {
        let rendered = render_markdown_for_terminal("+ alpha\n2) beta", false);
        assert_eq!(rendered, "- alpha\n2. beta");
    }

    #[test]
    fn renders_markdown_table_in_plain_mode() {
        let markdown = "| a | b |\n| --- | --- |\n| 1 | 2 |";
        let rendered = render_markdown_for_terminal(markdown, false);
        assert_eq!(rendered, "a | b\n--------------------------------\n1 | 2");
    }
}
