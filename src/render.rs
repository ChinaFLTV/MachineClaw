use std::{
    fs,
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
};

use colored::Colorize;
use mdansi::{RenderOptions, Renderer, TerminalCaps, Theme};

use crate::{error::AppError, i18n};

const DEFAULT_SYSTEM_PROMPT: &str = "你是 MachineClaw 的系统巡检分析助手。\n\n目标：基于用户给出的关键指标与命令结果，输出可执行、可追溯、风险导向的结论。\n\n输出要求：\n1. 先给结论，再给证据。\n2. 风险等级必须明确（低/中/高）并说明触发原因。\n3. 单独列出异常命令（失败/超时/中断/拦截）及影响。\n4. 给出最多 3 条下一步建议，按优先级排序。\n5. 严禁输出敏感信息（token、cookie、密码、私钥、密钥路径等）。\n6. 文本简洁，避免空话，默认中文输出。\n";
const DEFAULT_PREPARE_PROMPT: &str = "请基于以下数据总结本次运行前检查结果。\n\n# action\n{{action}}\n\n# target\n{{target}}\n\n# key_metrics\n{{key_metrics}}\n\n# command_details\n{{command_details}}\n\n请按以下结构输出：\n1. 结论（是否可继续执行）\n2. 关键异常与风险等级\n3. 优先处理建议（最多 3 条）\n";
const DEFAULT_INSPECT_PROMPT: &str = "请基于以下数据总结本次状态检查结果。\n\n# action\n{{action}}\n\n# target\n{{target}}\n\n# key_metrics\n{{key_metrics}}\n\n# command_details\n{{command_details}}\n\n请按以下结构输出：\n1. 当前状态结论\n2. 关键证据与异常项\n3. 风险等级与触发原因\n4. 下一步建议（最多 3 条）\n";
const DEFAULT_CHAT_SYSTEM_PROMPT: &str = "你是 MachineClaw 的本机交互助手，负责系统巡检、诊断、风险分析与必要的本地执行。\n\n核心原则：\n1. 先用工具拿证据，再下结论；严禁伪造结果。\n2. 工具能力平级：Builtin / Bash / Skills / MCP；禁止固定偏置。\n3. 能力路由顺序：匹配 Skill -> 匹配 MCP -> 匹配 Builtin -> Bash 回退。\n4. 本地文件检索优先使用 Builtin 工具：`View`、`LS`、`GlobTool`、`GrepTool`；非必要不要用 shell 的 `cat/head/tail/ls/find/grep`。\n5. 写操作最小化；仅在必要时执行，并明确影响、前置条件与回滚路径。\n6. 工具参数必须是严格 JSON 对象；参数错误先修参数，不要盲重试。\n7. 允许多轮工具调用，但证据足够后立即收敛，不做无意义链式调用。\n8. 任一轮已产出可展示文本时，必须保留并输出，不得被后续工具轮覆盖或丢弃。\n9. 禁止泄露敏感信息（token/cookie/password/private key/secret path 等）。\n10. MCP 失败时给可执行排障项：开关、服务状态、鉴权头、端点路径；HTTP 优先 `/mcp`。\n\n内置工具约定：\n- `View`：读取文件（支持 offset/limit）。\n- `LS`：列目录（可递归）。\n- `GlobTool`：按 glob 查找路径。\n- `GrepTool`：按 regex 检索内容。\n- `WebSearch`：查询公开网页信息。\n- `Edit` / `Replace` / `NotebookEditCell`：写入型工具，必须显式 `apply=true` 且仅在允许时使用。\n- `Think` / `Task` / `Architect`：用于中间推理、拆解与架构权衡，结果要可执行。\n\n输出规范：\n- 结构：结论 -> 关键证据 -> 风险评估 -> 下一步。\n- 简洁专业，直击要点；默认中文并跟随用户语言。\n";
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

pub fn render_chat_reconnect_notice(text: &str, colorful: bool) -> String {
    let multiline = text.contains('\n');
    let tag = i18n::chat_tag_reconnect();
    if !supports_color(colorful) {
        if multiline {
            return format!("{tag}\n{text}");
        }
        return format!("{tag} {text}");
    }
    if multiline {
        return format!("{}\n{}", tag.bright_magenta().bold(), text.bright_yellow());
    }
    format!("{} {}", tag.bright_magenta().bold(), text.bright_yellow())
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
    let prompt = prompt.trim_end();
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
    frame_index: usize,
    stdout_is_terminal: bool,
    stream: Option<mdansi::StreamRenderer<PrefixedStdoutWriter>>,
}

impl ChatStreamPrinter {
    pub fn new(colorful: bool) -> Self {
        Self {
            colorful,
            active_block: None,
            active_prefix: String::new(),
            frame_index: 0,
            stdout_is_terminal: std::io::stdout().is_terminal(),
            stream: None,
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
        if let Some(stream) = self.stream.as_mut() {
            stream.push(content).map_err(|err| {
                AppError::Command(format!("failed to render markdown stream chunk: {err}"))
            })?;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), AppError> {
        self.finish_active_stream()?;
        self.active_block = None;
        self.active_prefix.clear();
        Ok(())
    }

    fn ensure_block(&mut self, block: ChatStreamBlockKind, prefix: &str) -> Result<(), AppError> {
        if self.active_block == Some(block) && self.active_prefix == prefix {
            return Ok(());
        }
        self.finish_active_stream()?;
        self.active_block = Some(block);
        self.active_prefix = prefix.to_string();
        let writer = PrefixedStdoutWriter::new(
            self.active_prefix.clone(),
            self.colorful,
            block,
            self.stdout_is_terminal,
            self.frame_index,
        );
        self.stream = Some(mdansi::StreamRenderer::new(
            writer,
            Theme::default(),
            mdansi_render_options(self.colorful),
        ));
        Ok(())
    }

    fn finish_active_stream(&mut self) -> Result<(), AppError> {
        if let Some(mut stream) = self.stream.take() {
            stream.flush_remaining().map_err(|err| {
                AppError::Command(format!("failed to flush markdown stream renderer: {err}"))
            })?;
            let writer = stream.into_writer();
            if writer.needs_trailing_newline() {
                println!();
            }
            self.frame_index = writer.frame_index();
        }
        Ok(())
    }
}

struct PrefixedStdoutWriter {
    prefix: String,
    colorful: bool,
    block: ChatStreamBlockKind,
    stdout_is_terminal: bool,
    frame_index: usize,
    prefix_written: bool,
    at_line_start: bool,
}

impl PrefixedStdoutWriter {
    fn new(
        prefix: String,
        colorful: bool,
        block: ChatStreamBlockKind,
        stdout_is_terminal: bool,
        frame_index: usize,
    ) -> Self {
        Self {
            prefix,
            colorful,
            block,
            stdout_is_terminal,
            frame_index,
            prefix_written: false,
            at_line_start: true,
        }
    }

    fn frame_index(&self) -> usize {
        self.frame_index
    }

    fn needs_trailing_newline(&self) -> bool {
        self.prefix_written && !self.at_line_start
    }

    fn write_prefix(&mut self, stdout: &mut dyn Write) -> std::io::Result<()> {
        let prefix_text = render_stream_prefix(
            &self.prefix,
            self.colorful,
            self.stdout_is_terminal,
            self.block,
            self.frame_index,
        );
        self.frame_index = self.frame_index.wrapping_add(1);
        stdout.write_all(prefix_text.as_bytes())?;
        stdout.write_all(b"\n")?;
        self.prefix_written = true;
        self.at_line_start = true;
        Ok(())
    }

    fn ensure_prefix_written(&mut self, stdout: &mut dyn Write) -> std::io::Result<()> {
        if self.prefix_written {
            return Ok(());
        }
        self.write_prefix(stdout)?;
        Ok(())
    }
}

impl Write for PrefixedStdoutWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        if !buf.is_empty() {
            self.ensure_prefix_written(&mut out)?;
            out.write_all(buf)?;
            self.at_line_start = buf.last().is_none_or(|byte| *byte == b'\n');
        }
        out.flush()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stdout().flush()
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
    render_markdown_with_mdansi(text, colorful)
}

fn render_markdown_with_mdansi(text: &str, colorful: bool) -> String {
    if text.is_empty() {
        return String::new();
    }
    let renderer = Renderer::new(Theme::default(), mdansi_render_options(colorful));
    renderer.render(text).trim_end_matches('\n').to_string()
}

fn mdansi_render_options(colorful: bool) -> RenderOptions {
    let use_color = supports_color(colorful);
    let caps = if use_color {
        TerminalCaps::detect()
    } else {
        TerminalCaps::pipe(120)
    };
    let mut options = RenderOptions::from_terminal(&caps);
    if !use_color {
        options.plain = true;
        options.highlight = false;
        options.hyperlinks = false;
    }
    options
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
                    ChatStreamBlockKind::Thinking => style_stream_prefix_segment(&text, 148, false),
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
        ChatStreamBlockKind, render_chat_reconnect_notice, render_chat_user_prompt,
        render_markdown_for_terminal, render_stream_prefix,
    };
    use regex::Regex;

    fn strip_ansi(text: &str) -> String {
        let ansi_re = Regex::new(r"\x1b\[[0-9;]*m").expect("valid ansi regex");
        ansi_re.replace_all(text, "").into_owned()
    }

    #[test]
    fn stream_prefix_degrades_to_static_when_color_disabled() {
        assert_eq!(
            render_stream_prefix(
                "[MachineClaw]",
                false,
                true,
                ChatStreamBlockKind::Assistant,
                3
            ),
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
        let rendered = render_stream_prefix(
            "[MachineClaw]",
            true,
            false,
            ChatStreamBlockKind::Assistant,
            2,
        );
        assert_eq!(rendered, "[MachineClaw]");
    }

    #[test]
    fn chat_user_prompt_trims_trailing_space() {
        let rendered = render_chat_user_prompt("[你] ", false);
        assert_eq!(rendered, "[你]");
    }

    #[test]
    fn markdown_renders_list_in_plain_mode() {
        let rendered = render_markdown_for_terminal("- **item**", false);
        assert_eq!(rendered, "• item");
    }

    #[test]
    fn renders_extended_heading_levels_in_plain_mode() {
        let rendered = render_markdown_for_terminal("###### heading", false);
        assert!(rendered.contains("###### heading"));
    }

    #[test]
    fn renders_nested_blockquote_and_list_in_plain_mode() {
        let rendered = render_markdown_for_terminal("> > - item", false);
        assert!(rendered.contains("• item"));
    }

    #[test]
    fn renders_plus_list_and_ordered_parenthesis_in_plain_mode() {
        let rendered = render_markdown_for_terminal("+ alpha\n2) beta", false);
        assert_eq!(rendered, "• alpha\n\n2. beta");
    }

    #[test]
    fn renders_markdown_table_in_plain_mode() {
        let markdown = "| a | b |\n| --- | --- |\n| 1 | 2 |";
        let rendered = render_markdown_for_terminal(markdown, false);
        assert!(rendered.contains("a"));
        assert!(rendered.contains("b"));
        assert!(rendered.contains("1"));
        assert!(rendered.contains("2"));
    }

    #[test]
    fn preserves_code_fence_and_code_lines_in_plain_mode() {
        let markdown = "```rust\nfn main() {}\n```";
        let rendered = render_markdown_for_terminal(markdown, false);
        assert!(rendered.contains("fn main() {}"));
    }

    #[test]
    fn preserves_tilde_code_fence_in_plain_mode() {
        let markdown = "~~~python\nprint(1)\n~~~";
        let rendered = render_markdown_for_terminal(markdown, false);
        assert!(rendered.contains("print(1)"));
    }

    #[test]
    fn reconnect_notice_uses_dedicated_tag_in_plain_mode() {
        let rendered = render_chat_reconnect_notice("AI connection issue", false);
        assert_eq!(
            rendered,
            format!("{} AI connection issue", crate::i18n::chat_tag_reconnect())
        );
    }

    #[test]
    fn reconnect_notice_puts_multiline_text_on_separate_line() {
        let rendered = render_chat_reconnect_notice("line1\nline2", false);
        assert_eq!(
            rendered,
            format!("{}\nline1\nline2", crate::i18n::chat_tag_reconnect())
        );
    }

    #[test]
    fn preserves_indented_code_block_in_plain_mode() {
        let markdown = "    let x = 1;\n    println!(\"{}\", x);";
        let rendered = render_markdown_for_terminal(markdown, false);
        assert!(rendered.contains("let x = 1;"));
        assert!(rendered.contains("println!(\"{}\", x);"));
    }
}
