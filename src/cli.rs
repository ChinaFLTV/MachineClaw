use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::i18n::{self, Language};

#[derive(Debug, Parser)]
#[command(
    name = "MachineClaw",
    version,
    about = "Cross-platform machine inspection CLI"
)]
pub struct Cli {
    #[arg(short = 'c', long = "conf", value_name = "path", global = true)]
    pub conf: Option<PathBuf>,
    #[arg(long = "show-config-template", global = true, default_value_t = false)]
    pub show_config_template: bool,
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run environment checks before formal actions.
    Prepare,
    /// Inspect a target state.
    Inspect { target: InspectTarget },
    /// Test runtime settings and configuration integrity.
    Test { target: TestTarget },
    /// Start interactive chat with MachineClaw AI assistant.
    Chat,
    /// Package current config snapshot into a self-contained executable.
    Snapshot {
        #[arg(short = 'o', long = "output", value_name = "path")]
        output: Option<PathBuf>,
    },
    /// Check latest release and upgrade local MachineClaw binary.
    Upgrade {
        #[arg(long = "check-only", default_value_t = false)]
        check_only: bool,
        #[arg(short = 'o', long = "output", value_name = "path")]
        output: Option<PathBuf>,
        #[arg(long = "allow-prerelease", default_value_t = false)]
        allow_prerelease: bool,
    },
    /// Show effective config snapshot (sensitive fields masked).
    ShowConfig,
    /// Get or set configuration values.
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum ConfigCommands {
    /// Get effective value of one config key.
    Get { key: String },
    /// Set value for one config key.
    Set { key: String, value: String },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum InspectTarget {
    Cpu,
    Memory,
    Disk,
    Os,
    Process,
    Filesystem,
    Hardware,
    Logs,
    Network,
    All,
}

impl InspectTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Memory => "memory",
            Self::Disk => "disk",
            Self::Os => "os",
            Self::Process => "process",
            Self::Filesystem => "filesystem",
            Self::Hardware => "hardware",
            Self::Logs => "logs",
            Self::Network => "network",
            Self::All => "all",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum TestTarget {
    Config,
}

impl TestTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Config => "config",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum HelpTopic {
    Global,
    Prepare,
    Inspect,
    Test,
    Chat,
    Snapshot,
    Upgrade,
    ShowConfig,
    Config,
}

pub fn extract_conf_path_from_args(args: &[String]) -> Option<PathBuf> {
    let mut idx = 1usize;
    while idx < args.len() {
        let arg = args[idx].as_str();
        if arg == "-c" || arg == "--conf" {
            if idx + 1 < args.len() {
                return Some(PathBuf::from(args[idx + 1].clone()));
            }
            return None;
        }
        if let Some(rest) = arg.strip_prefix("--conf=")
            && !rest.trim().is_empty()
        {
            return Some(PathBuf::from(rest));
        }
        if let Some(rest) = arg.strip_prefix("-c=")
            && !rest.trim().is_empty()
        {
            return Some(PathBuf::from(rest));
        }
        idx += 1;
    }
    None
}

pub fn detect_help_topic(args: &[String]) -> Option<HelpTopic> {
    if args.len() <= 1 {
        return Some(HelpTopic::Global);
    }

    if args.get(1).map(|s| s.as_str()) == Some("help") {
        return Some(match args.get(2).map(|s| s.as_str()) {
            Some("prepare") => HelpTopic::Prepare,
            Some("inspect") => HelpTopic::Inspect,
            Some("test") => HelpTopic::Test,
            Some("chat") => HelpTopic::Chat,
            Some("snapshot") => HelpTopic::Snapshot,
            Some("upgrade") => HelpTopic::Upgrade,
            Some("show-config") => HelpTopic::ShowConfig,
            Some("config") => HelpTopic::Config,
            _ => HelpTopic::Global,
        });
    }

    let has_help_flag = args
        .iter()
        .skip(1)
        .any(|arg| arg == "-h" || arg == "--help");
    if !has_help_flag {
        return None;
    }

    let mut skip_next = false;
    for arg in args.iter().skip(1) {
        let value = arg.as_str();
        if skip_next {
            skip_next = false;
            continue;
        }
        if value == "-c" || value == "--conf" {
            skip_next = true;
            continue;
        }
        if value.starts_with("--conf=") || value.starts_with("-c=") {
            continue;
        }
        if value.starts_with('-') {
            continue;
        }
        return Some(match value {
            "prepare" => HelpTopic::Prepare,
            "inspect" => HelpTopic::Inspect,
            "test" => HelpTopic::Test,
            "chat" => HelpTopic::Chat,
            "snapshot" => HelpTopic::Snapshot,
            "upgrade" => HelpTopic::Upgrade,
            "show-config" => HelpTopic::ShowConfig,
            "config" => HelpTopic::Config,
            _ => HelpTopic::Global,
        });
    }
    Some(HelpTopic::Global)
}

pub fn localized_help(topic: HelpTopic) -> String {
    match i18n::current_language() {
        Language::ZhCn => help_zh_cn(topic),
        Language::ZhTw => help_zh_tw(topic),
        Language::Fr => help_fr(topic),
        Language::De => help_de(topic),
        Language::Ja => help_ja(topic),
        Language::En => help_en(topic),
    }
}

pub fn prettify_help_markdown(raw: &str) -> String {
    let mut out = Vec::<String>::new();
    let mut title_written = false;
    let mut current_heading = String::new();

    for line in raw.lines() {
        let trimmed_end = line.trim_end();
        let trimmed = trimmed_end.trim();
        if trimmed.is_empty() {
            if !out.last().is_some_and(|item| item.is_empty()) {
                out.push(String::new());
            }
            continue;
        }

        if !title_written {
            if is_help_heading(trimmed) {
                out.push("# MachineClaw".to_string());
                title_written = true;
            } else {
                out.push(format!("# {trimmed}"));
                title_written = true;
                continue;
            }
        }

        if is_help_heading(trimmed) {
            let heading = trimmed
                .trim_end_matches(':')
                .trim_end_matches('：')
                .trim()
                .to_string();
            if !out.last().is_some_and(|item| item.is_empty()) {
                out.push(String::new());
            }
            out.push(format!("## {heading}"));
            current_heading = heading.to_ascii_lowercase();
            continue;
        }

        if line.starts_with(' ') || line.starts_with('\t') {
            let item = trimmed_end.trim();
            if is_example_heading(current_heading.as_str()) {
                out.push(format!("- `{item}`"));
                continue;
            }
            if let Some((left, right)) = split_help_columns(item) {
                out.push(format!("- `{left}` {right}"));
            } else if is_option_heading(current_heading.as_str()) && item.starts_with('-') {
                out.push(format!("- `{item}`"));
            } else {
                out.push(format!("- {item}"));
            }
            continue;
        }

        out.push(trimmed.to_string());
    }

    out.join("\n")
}

fn is_help_heading(line: &str) -> bool {
    if line.starts_with(' ') || line.starts_with('\t') {
        return false;
    }
    line.ends_with(':') || line.ends_with('：')
}

fn is_example_heading(heading: &str) -> bool {
    let lowered = heading.to_ascii_lowercase();
    lowered.contains("example")
        || lowered.contains("示例")
        || lowered.contains("例")
        || lowered.contains("beispiel")
        || lowered.contains("exemple")
}

fn is_option_heading(heading: &str) -> bool {
    let lowered = heading.to_ascii_lowercase();
    lowered.contains("option")
        || lowered.contains("选项")
        || lowered.contains("選項")
        || lowered.contains("オプション")
        || lowered.contains("optionen")
}

fn split_help_columns(item: &str) -> Option<(String, String)> {
    let chars: Vec<char> = item.chars().collect();
    let mut run = 0usize;
    let mut split_idx = None;
    for (idx, ch) in chars.iter().enumerate() {
        if *ch == ' ' {
            run = run.saturating_add(1);
            if run >= 2 {
                split_idx = Some(idx + 1 - run);
                break;
            }
        } else {
            run = 0;
        }
    }
    let idx = split_idx?;
    let left: String = chars[..idx].iter().collect::<String>().trim().to_string();
    let right: String = chars[idx..].iter().collect::<String>().trim().to_string();
    if left.is_empty() || right.is_empty() {
        return None;
    }
    Some((left, right))
}

pub fn render_help_panel(markdown: &str, colorful: bool) -> String {
    let parsed = parse_prettified_help(markdown);
    let width = terminal_render_width();
    let inner_width = width.saturating_sub(2);
    let mut lines = Vec::<String>::new();

    lines.push(top_border(inner_width));
    lines.push(box_line(
        inner_width,
        format!("✦ {}", parsed.title).as_str(),
    ));
    if !parsed.intro.is_empty() {
        for intro in parsed.intro {
            let wrapped = wrap_text_by_display_width(intro.as_str(), inner_width.saturating_sub(2));
            for item in wrapped {
                lines.push(box_line(inner_width, format!(" {item}").as_str()));
            }
        }
    }
    for section in parsed.sections {
        lines.push(mid_border(inner_width));
        lines.push(box_line(
            inner_width,
            format!("[ {} ]", section.title).as_str(),
        ));
        if section.items.is_empty() {
            lines.push(box_line(inner_width, " • none"));
            continue;
        }
        for item in section.items {
            let wrapped = wrap_text_by_display_width(item.as_str(), inner_width.saturating_sub(4));
            for (idx, line) in wrapped.iter().enumerate() {
                let prefix = if idx == 0 { " • " } else { "   " };
                lines.push(box_line(inner_width, format!("{prefix}{line}").as_str()));
            }
        }
    }
    lines.push(bottom_border(inner_width));
    style_help_panel_lines(lines, colorful)
}

#[derive(Debug, Default)]
struct ParsedHelp {
    title: String,
    intro: Vec<String>,
    sections: Vec<HelpSection>,
}

#[derive(Debug, Default)]
struct HelpSection {
    title: String,
    items: Vec<String>,
}

fn parse_prettified_help(markdown: &str) -> ParsedHelp {
    let mut parsed = ParsedHelp {
        title: "MachineClaw".to_string(),
        intro: Vec::new(),
        sections: Vec::new(),
    };
    let mut current_section: Option<HelpSection> = None;

    for line in markdown.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("# ") {
            parsed.title = title.trim().to_string();
            continue;
        }
        if let Some(section_title) = trimmed.strip_prefix("## ") {
            if let Some(section) = current_section.take() {
                parsed.sections.push(section);
            }
            current_section = Some(HelpSection {
                title: section_title.trim().to_string(),
                items: Vec::new(),
            });
            continue;
        }
        if trimmed.is_empty() {
            continue;
        }
        let item = trimmed.strip_prefix("- ").unwrap_or(trimmed).to_string();
        if let Some(section) = current_section.as_mut() {
            section.items.push(item);
        } else {
            parsed.intro.push(item);
        }
    }
    if let Some(section) = current_section.take() {
        parsed.sections.push(section);
    }
    parsed
}

fn style_help_panel_lines(lines: Vec<String>, colorful: bool) -> String {
    if !colorful {
        return lines.join("\n");
    }
    let mut styled = Vec::<String>::with_capacity(lines.len());
    for (idx, line) in lines.iter().enumerate() {
        if line.starts_with('┏') || line.starts_with('┣') || line.starts_with('┗') {
            styled.push(line.bright_blue().bold().to_string());
            continue;
        }
        let Some((left, body, right)) = split_box_line(line.as_str()) else {
            styled.push(line.to_string());
            continue;
        };
        if idx == 1 {
            styled.push(format!(
                "{}{}{}",
                left.bright_blue().bold(),
                body.bright_cyan().bold(),
                right.bright_blue().bold()
            ));
            continue;
        }
        if body.contains("[ ") && body.contains(" ]") {
            styled.push(format!(
                "{}{}{}",
                left.bright_blue().bold(),
                body.bright_magenta().bold(),
                right.bright_blue().bold()
            ));
            continue;
        }
        if body.contains("•") {
            let mut body_styled = body.bright_white().to_string();
            body_styled = body_styled.replace("•", &"•".bright_green().bold().to_string());
            body_styled = colorize_inline_code(body_styled.as_str());
            body_styled = colorize_urls(body_styled.as_str());
            styled.push(format!(
                "{}{}{}",
                left.bright_blue().bold(),
                body_styled,
                right.bright_blue().bold()
            ));
            continue;
        }
        styled.push(format!(
            "{}{}{}",
            left.bright_blue().bold(),
            body.bright_white(),
            right.bright_blue().bold()
        ));
    }
    styled.join("\n")
}

fn colorize_inline_code(text: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    let mut buff = String::new();
    for ch in text.chars() {
        if ch == '`' {
            if in_code {
                out.push('`');
                out.push_str(&buff.bright_cyan().bold().to_string());
                out.push('`');
                buff.clear();
                in_code = false;
            } else {
                in_code = true;
            }
            continue;
        }
        if in_code {
            buff.push(ch);
        } else {
            out.push(ch);
        }
    }
    if in_code {
        out.push('`');
        out.push_str(&buff);
    }
    out
}

fn colorize_urls(line: &str) -> String {
    let mut out = String::new();
    let mut remain = line;
    loop {
        let http = remain.find("http://");
        let https = remain.find("https://");
        let start = match (http, https) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => {
                out.push_str(remain);
                break;
            }
        };
        out.push_str(&remain[..start]);
        let after = &remain[start..];
        let end = after.find(' ').unwrap_or(after.len());
        let url = &after[..end];
        out.push_str(&url.bright_blue().underline().to_string());
        remain = &after[end..];
    }
    out
}

fn split_box_line(line: &str) -> Option<(&str, &str, &str)> {
    let body = line.strip_prefix('┃')?.strip_suffix('┃')?;
    Some(("┃", body, "┃"))
}

fn terminal_render_width() -> usize {
    match crossterm::terminal::size() {
        Ok((width, _)) => {
            let safe = (width as usize).max(72);
            safe.min(180)
        }
        Err(_) => 120,
    }
}

fn top_border(inner_width: usize) -> String {
    format!("┏{}┓", "━".repeat(inner_width))
}

fn mid_border(inner_width: usize) -> String {
    format!("┣{}┫", "━".repeat(inner_width))
}

fn bottom_border(inner_width: usize) -> String {
    format!("┗{}┛", "━".repeat(inner_width))
}

fn box_line(inner_width: usize, content: &str) -> String {
    let line = truncate_to_display_width(content, inner_width);
    let padded = pad_to_display_width(line.as_str(), inner_width);
    format!("┃{padded}┃")
}

fn pad_to_display_width(text: &str, target_width: usize) -> String {
    let current = display_width(text);
    if current >= target_width {
        return text.to_string();
    }
    format!("{text}{}", " ".repeat(target_width - current))
}

fn truncate_to_display_width(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut current = 0usize;
    for grapheme in text.graphemes(true) {
        let width = display_width(grapheme);
        if current + width > max_width {
            break;
        }
        out.push_str(grapheme);
        current += width;
    }
    out
}

fn wrap_text_by_display_width(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![String::new()];
    }
    let mut lines = Vec::<String>::new();
    for raw in text.lines() {
        if raw.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current = String::new();
        let mut current_width = 0usize;
        for grapheme in raw.graphemes(true) {
            let width = display_width(grapheme);
            if current_width + width > max_width {
                lines.push(current);
                current = String::new();
                current_width = 0;
            }
            current.push_str(grapheme);
            current_width += width;
        }
        lines.push(current);
    }
    lines
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn help_en(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\nA cross-platform CLI for machine preflight checks, inspection, and interactive diagnosis.\n\nUsage:\n  MachineClaw [OPTIONS] <COMMAND>\n\nCommands:\n  prepare      Run preflight environment checks\n  inspect      Inspect machine state by target\n  test         Validate runtime config and integrity\n  chat         Start interactive chat mode\n  snapshot     Build self-contained executable with config snapshot\n  upgrade      Check latest release and upgrade local binary\n  show-config  Show effective config snapshot (masked)\n  config       Get or set config values\n  help         Print this help or subcommand help\n\nOptions:\n  -c, --conf <path>        Config file path (supports --conf=... and --conf ...)\n  --show-config-template   Print full config template\n  -h, --help               Print help\n  -V, --version            Print version\n\nExamples:\n  MachineClaw --show-config-template\n  MachineClaw show-config --conf ./sample/claw-sample.toml\n  MachineClaw snapshot --conf ./sample/claw-sample.toml -o ./MachineClaw-prod\n  MachineClaw upgrade --check-only\n  MachineClaw test config --conf ./sample/claw-sample.toml\n  MachineClaw config get ai.retry.max-retries --conf ./sample/claw-sample.toml\n  MachineClaw config set ai.retry.max-retries 5 --conf ./sample/claw-sample.toml\n".to_string(),
        HelpTopic::Prepare => "Usage:\n  MachineClaw prepare [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "Usage:\n  MachineClaw inspect <target> [OPTIONS]\n\nTargets:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "Usage:\n  MachineClaw test <target> [OPTIONS]\n\nTargets:\n  config\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "Usage:\n  MachineClaw chat [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "Usage:\n  MachineClaw snapshot [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -o, --output <path>  Output executable path\n  -h, --help\n".to_string(),
        HelpTopic::Upgrade => "Usage:\n  MachineClaw upgrade [OPTIONS]\n\nOptions:\n  --check-only           Only check release without downloading\n  -o, --output <path>    Save downloaded asset to custom path\n  --allow-prerelease     Allow prerelease as candidate\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "Usage:\n  MachineClaw show-config [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "Usage:\n  MachineClaw config <COMMAND> [ARGS] [OPTIONS]\n\nCommands:\n  get <key>          Get effective value\n  set <key> <value>  Set value into config file\n\nExamples:\n  MachineClaw config get ai.retry.max-retries --conf ./sample/claw-sample.toml\n  MachineClaw config set ai.retry.max-retries 5 --conf ./sample/claw-sample.toml\n".to_string(),
    }
}

fn help_zh_cn(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\n跨平台命令行工具，用于机器预检、状态巡检与交互式诊断。\n\n用法:\n  MachineClaw [选项] <命令>\n\n命令:\n  prepare      执行运行前环境检查\n  inspect      按目标检查机器状态\n  test         校验配置与运行参数完整性\n  chat         进入交互式对话模式\n  snapshot     打包内置配置快照的可执行文件\n  upgrade      查询最新 Release 并升级本地程序\n  show-config  展示当前生效配置快照（已脱敏）\n  config       获取或设置配置项\n  help         显示帮助信息\n\n选项:\n  -c, --conf <path>       配置文件路径（支持 --conf=... 与 --conf ...）\n  --show-config-template  展示完整配置模板\n  -h, --help              显示帮助\n  -V, --version           显示版本\n\n示例:\n  MachineClaw --show-config-template\n  MachineClaw show-config --conf ./sample/claw-sample.toml\n  MachineClaw snapshot --conf ./sample/claw-sample.toml -o ./MachineClaw-prod\n  MachineClaw upgrade --check-only\n  MachineClaw test config --conf ./sample/claw-sample.toml\n  MachineClaw config get ai.retry.max-retries --conf ./sample/claw-sample.toml\n  MachineClaw config set ai.retry.max-retries 5 --conf ./sample/claw-sample.toml\n".to_string(),
        HelpTopic::Prepare => "用法:\n  MachineClaw prepare [选项]\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "用法:\n  MachineClaw inspect <target> [选项]\n\ntarget 可选值:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "用法:\n  MachineClaw test <target> [选项]\n\ntarget 可选值:\n  config\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "用法:\n  MachineClaw chat [选项]\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "用法:\n  MachineClaw snapshot [选项]\n\n选项:\n  -c, --conf <path>\n  -o, --output <path>  输出可执行文件路径\n  -h, --help\n".to_string(),
        HelpTopic::Upgrade => "用法:\n  MachineClaw upgrade [选项]\n\n选项:\n  --check-only          仅检查版本，不下载\n  -o, --output <path>   指定下载输出路径\n  --allow-prerelease    允许预发布版本参与比较\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "用法:\n  MachineClaw show-config [选项]\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "用法:\n  MachineClaw config <命令> [参数] [选项]\n\n命令:\n  get <key>          获取配置字段当前生效值\n  set <key> <value>  设置配置字段值并写回配置文件\n\n示例:\n  MachineClaw config get ai.retry.max-retries --conf ./sample/claw-sample.toml\n  MachineClaw config set ai.retry.max-retries 5 --conf ./sample/claw-sample.toml\n".to_string(),
    }
}

fn help_zh_tw(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\n跨平台命令列工具，用於機器預檢、狀態巡檢與互動式診斷。\n\n用法:\n  MachineClaw [選項] <命令>\n\n命令:\n  prepare      執行啟動前環境檢查\n  inspect      依目標檢查機器狀態\n  test         驗證配置與執行參數完整性\n  chat         進入互動式對話模式\n  snapshot     打包內嵌配置快照的可執行檔\n  upgrade      查詢最新 Release 並升級本機程式\n  show-config  顯示目前生效配置快照（已脫敏）\n  config       取得或設定配置項\n  help         顯示說明\n\n選項:\n  -c, --conf <path>       設定檔路徑（支援 --conf=... 與 --conf ...）\n  --show-config-template  顯示完整配置模板\n  -h, --help              顯示說明\n  -V, --version           顯示版本\n\n示例:\n  MachineClaw --show-config-template\n  MachineClaw show-config --conf ./sample/claw-sample.toml\n  MachineClaw snapshot --conf ./sample/claw-sample.toml -o ./MachineClaw-prod\n  MachineClaw upgrade --check-only\n  MachineClaw test config --conf ./sample/claw-sample.toml\n".to_string(),
        HelpTopic::Prepare => "用法:\n  MachineClaw prepare [選項]\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "用法:\n  MachineClaw inspect <target> [選項]\n\ntarget 可選值:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "用法:\n  MachineClaw test <target> [選項]\n\ntarget 可選值:\n  config\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "用法:\n  MachineClaw chat [選項]\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "用法:\n  MachineClaw snapshot [選項]\n\n選項:\n  -c, --conf <path>\n  -o, --output <path>  輸出可執行檔路徑\n  -h, --help\n".to_string(),
        HelpTopic::Upgrade => "用法:\n  MachineClaw upgrade [選項]\n\n選項:\n  --check-only          僅檢查版本，不下載\n  -o, --output <path>   指定下載輸出路徑\n  --allow-prerelease    允許預發布版本參與比較\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "用法:\n  MachineClaw show-config [選項]\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "用法:\n  MachineClaw config <命令> [參數] [選項]\n\n命令:\n  get <key>          取得配置欄位目前生效值\n  set <key> <value>  設定配置欄位值並寫回設定檔\n".to_string(),
    }
}

fn help_fr(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\nUtilitaire CLI multiplateforme pour la pré-vérification, l'inspection et le diagnostic interactif.\n\nUtilisation:\n  MachineClaw [OPTIONS] <COMMANDE>\n\nCommandes:\n  prepare      Vérifier l'environnement avant exécution\n  inspect      Inspecter l'état de la machine par cible\n  test         Valider la configuration et l'intégrité\n  chat         Démarrer le mode conversation interactif\n  snapshot     Générer un exécutable autonome avec snapshot config\n  upgrade      Vérifier la release et mettre à jour le binaire local\n  show-config  Afficher la configuration effective (masquée)\n  config       Lire ou modifier la configuration\n  help         Afficher cette aide\n\nOptions:\n  -c, --conf <path>        Chemin du fichier de configuration (supporte --conf=... et --conf ...)\n  --show-config-template   Afficher le modèle complet de configuration\n  -h, --help               Afficher l'aide\n  -V, --version            Afficher la version\n".to_string(),
        HelpTopic::Prepare => "Utilisation:\n  MachineClaw prepare [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "Utilisation:\n  MachineClaw inspect <target> [OPTIONS]\n\nValeurs target:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "Utilisation:\n  MachineClaw test <target> [OPTIONS]\n\nValeurs target:\n  config\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "Utilisation:\n  MachineClaw chat [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "Utilisation:\n  MachineClaw snapshot [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -o, --output <path>  Chemin de sortie exécutable\n  -h, --help\n".to_string(),
        HelpTopic::Upgrade => "Utilisation:\n  MachineClaw upgrade [OPTIONS]\n\nOptions:\n  --check-only           Vérifier seulement, sans téléchargement\n  -o, --output <path>    Chemin personnalisé du fichier téléchargé\n  --allow-prerelease     Autoriser les versions prerelease\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "Utilisation:\n  MachineClaw show-config [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "Utilisation:\n  MachineClaw config <COMMANDE> [ARGS] [OPTIONS]\n\nCommandes:\n  get <key>          Lire la valeur effective\n  set <key> <value>  Écrire la valeur dans le fichier de config\n".to_string(),
    }
}

fn help_de(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\nPlattformübergreifendes CLI für Preflight-Checks, Inspektion und interaktive Diagnose.\n\nVerwendung:\n  MachineClaw [OPTIONEN] <BEFEHL>\n\nBefehle:\n  prepare      Umgebungsprüfung vor der Ausführung\n  inspect      Maschinenstatus nach Ziel prüfen\n  test         Konfiguration und Integrität prüfen\n  chat         Interaktiven Chat-Modus starten\n  snapshot     Selbstenthaltende Binärdatei mit Konfig-Snapshot bauen\n  upgrade      Neueste Release prüfen und lokal aktualisieren\n  show-config  Effektive Konfiguration (maskiert) anzeigen\n  config       Konfiguration lesen oder setzen\n  help         Hilfe anzeigen\n\nOptionen:\n  -c, --conf <path>        Pfad zur Konfigurationsdatei (unterstützt --conf=... und --conf ...)\n  --show-config-template   Vollständige Konfigurationsvorlage anzeigen\n  -h, --help               Hilfe anzeigen\n  -V, --version            Version anzeigen\n".to_string(),
        HelpTopic::Prepare => "Verwendung:\n  MachineClaw prepare [OPTIONEN]\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "Verwendung:\n  MachineClaw inspect <target> [OPTIONEN]\n\nTarget-Werte:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "Verwendung:\n  MachineClaw test <target> [OPTIONEN]\n\nTarget-Werte:\n  config\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "Verwendung:\n  MachineClaw chat [OPTIONEN]\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "Verwendung:\n  MachineClaw snapshot [OPTIONEN]\n\nOptionen:\n  -c, --conf <path>\n  -o, --output <path>  Ausgabe-Binärpfad\n  -h, --help\n".to_string(),
        HelpTopic::Upgrade => "Verwendung:\n  MachineClaw upgrade [OPTIONEN]\n\nOptionen:\n  --check-only           Nur prüfen, nichts herunterladen\n  -o, --output <path>    Zielpfad für Download\n  --allow-prerelease     Vorabversionen zulassen\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "Verwendung:\n  MachineClaw show-config [OPTIONEN]\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "Verwendung:\n  MachineClaw config <BEFEHL> [ARGS] [OPTIONEN]\n\nBefehle:\n  get <key>          Effektiven Wert lesen\n  set <key> <value>  Wert in Konfigurationsdatei schreiben\n".to_string(),
    }
}

fn help_ja(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\nマシンの事前チェック・状態確認・対話診断のためのクロスプラットフォームCLIです。\n\n使い方:\n  MachineClaw [オプション] <コマンド>\n\nコマンド:\n  prepare      実行前の環境チェックを実施\n  inspect      対象ごとにマシン状態を確認\n  test         設定と整合性を検証\n  chat         対話チャットモードを開始\n  snapshot     設定スナップショット内蔵バイナリを生成\n  upgrade      最新 Release を確認してローカル更新\n  show-config  有効設定のスナップショットを表示（マスク済み）\n  config       設定値を取得/更新\n  help         ヘルプを表示\n\nオプション:\n  -c, --conf <path>       設定ファイルのパス（--conf=... と --conf ... をサポート）\n  --show-config-template  完全な設定テンプレートを表示\n  -h, --help              ヘルプを表示\n  -V, --version           バージョンを表示\n".to_string(),
        HelpTopic::Prepare => "使い方:\n  MachineClaw prepare [オプション]\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "使い方:\n  MachineClaw inspect <target> [オプション]\n\ntarget の値:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "使い方:\n  MachineClaw test <target> [オプション]\n\ntarget の値:\n  config\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "使い方:\n  MachineClaw chat [オプション]\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "使い方:\n  MachineClaw snapshot [オプション]\n\nオプション:\n  -c, --conf <path>\n  -o, --output <path>  出力バイナリのパス\n  -h, --help\n".to_string(),
        HelpTopic::Upgrade => "使い方:\n  MachineClaw upgrade [オプション]\n\nオプション:\n  --check-only           バージョン確認のみ（ダウンロードなし）\n  -o, --output <path>    ダウンロード先のパス\n  --allow-prerelease     プレリリースを許可\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "使い方:\n  MachineClaw show-config [オプション]\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "使い方:\n  MachineClaw config <コマンド> [引数] [オプション]\n\nコマンド:\n  get <key>          実効値を取得\n  set <key> <value>  設定ファイルへ値を書き込み\n".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{prettify_help_markdown, split_help_columns};

    #[test]
    fn prettify_help_markdown_promotes_title_and_sections() {
        let raw = "MachineClaw\n\nUsage:\n  MachineClaw [OPTIONS] <COMMAND>\n\nCommands:\n  prepare      Run preflight checks\n";
        let rendered = prettify_help_markdown(raw);
        assert!(rendered.contains("# MachineClaw"));
        assert!(rendered.contains("## Usage"));
        assert!(rendered.contains("## Commands"));
        assert!(rendered.contains("- `prepare` Run preflight checks"));
    }

    #[test]
    fn split_help_columns_splits_on_multi_space_gap() {
        let parsed = split_help_columns("-c, --conf <path>    Config file path");
        assert_eq!(
            parsed,
            Some((
                "-c, --conf <path>".to_string(),
                "Config file path".to_string()
            ))
        );
    }

    #[test]
    fn prettify_subcommand_help_injects_default_title() {
        let raw = "Usage:\n  MachineClaw upgrade [OPTIONS]\n\nOptions:\n  -h, --help\n";
        let rendered = prettify_help_markdown(raw);
        assert!(rendered.starts_with("# MachineClaw"));
        assert!(rendered.contains("## Usage"));
        assert!(rendered.contains("- `-h, --help`"));
    }
}
