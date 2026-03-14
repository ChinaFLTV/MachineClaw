mod actions;
mod ai;
mod cli;
mod config;
mod config_action;
mod context;
mod error;
mod i18n;
mod logging;
mod mask;
mod mcp;
mod platform;
mod render;
mod shell;
mod skills;
mod snapshot;
mod test_action;
mod tls;

use std::{
    io::{self, IsTerminal, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use clap::Parser;
use clap::error::ErrorKind;

use crate::{
    actions::ActionServices,
    ai::AiClient,
    cli::{Cli, Commands},
    config::{
        config_template_example, expand_tilde, normalize_mcp_availability_check_mode,
        read_console_colorful_hint, read_language_hint, resolve_config_path, validate_config,
    },
    config_action::run_config_command,
    context::SessionStore,
    error::{AppError, ExitCode},
    i18n::{language_code, localize_error, parse_language, resolve_language, set_language},
    mcp::{mcp_summary, validate_mcp_config},
    platform::{current_os, os_name, require_elevated_permissions},
    shell::ShellExecutor,
    skills::detect_skills,
    snapshot::{
        build_snapshot_binary, load_effective_config, render_show_config, render_snapshot_result,
    },
    test_action::run_test_command,
};

fn main() {
    let raw_args: Vec<String> = std::env::args().collect();
    let startup_colorful = resolve_startup_colorful(&raw_args);
    let exit_code = match run() {
        Ok(code) => code,
        Err(err) => {
            let code = err.exit_code();
            let message = mask::mask_sensitive(&localize_error(&err));
            eprintln!("{}", render::render_error_line(&message, startup_colorful));
            logging::error(&format!("program failed: {message}"));
            code
        }
    };
    std::process::exit(exit_code as i32);
}

fn run() -> Result<ExitCode, AppError> {
    set_language(resolve_language(None));
    let raw_args: Vec<String> = std::env::args().collect();
    let startup_colorful = resolve_startup_colorful(&raw_args);
    if let Some(conf_path) = cli::extract_conf_path_from_args(&raw_args)
        && let Ok(resolved_path) = resolve_config_path(Some(conf_path))
        && let Some(language_hint) = read_language_hint(&resolved_path)
    {
        set_language(resolve_language(Some(&language_hint)));
    }
    if let Some(topic) = cli::detect_help_topic(&raw_args) {
        println!(
            "{}",
            render::render_markdown_for_terminal(&cli::localized_help(topic), startup_colorful)
        );
        return Ok(ExitCode::Success);
    }

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => match err.kind() {
            ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                print!("{err}");
                return Ok(ExitCode::Success);
            }
            _ => return Err(AppError::Runtime(err.to_string())),
        },
    };
    if cli.show_config_template {
        println!(
            "{}",
            render::render_markdown_for_terminal(config_template_example(), startup_colorful,)
        );
        return Ok(ExitCode::Success);
    }
    let Some(command) = cli.command else {
        println!(
            "{}",
            render::render_markdown_for_terminal(
                &cli::localized_help(cli::HelpTopic::Global),
                startup_colorful,
            )
        );
        return Ok(ExitCode::Success);
    };
    let config_path = resolve_config_path(cli.conf.clone())?;
    if let Some(language_hint) = read_language_hint(&config_path) {
        set_language(resolve_language(Some(&language_hint)));
    }
    if let Commands::Config { command } = &command {
        let outcome = run_config_command(&config_path, command)?;
        println!(
            "{}",
            render::render_markdown_for_terminal(&outcome.rendered, startup_colorful)
        );
        return Ok(outcome.exit_code);
    }
    if let Commands::Test { target } = &command {
        let assets_setup = render::locate_or_init_assets_dir()?;
        for notice in assets_setup.notices {
            println!("{}", render::render_info_line(&notice, startup_colorful));
        }
        let outcome = run_test_command(&config_path, *target, &assets_setup.path)?;
        println!(
            "{}",
            render::render_markdown_for_terminal(&outcome.rendered, startup_colorful)
        );
        return Ok(outcome.exit_code);
    }

    let effective = load_effective_config(cli.conf.clone())?;
    let mut cfg = effective.cfg;
    let config_source_desc = effective.source.describe();
    let selected_language = resolve_language(cfg.app.language.as_deref());
    set_language(selected_language);
    cfg.console.colorful = render::resolve_colorful_enabled(cfg.console.colorful);
    validate_config(&cfg)?;
    validate_mcp_config(&cfg.mcp)?;
    let language_warning = cfg.app.language.as_deref().and_then(|raw| {
        if parse_language(raw).is_none() {
            Some(i18n::unsupported_language_notice(raw))
        } else {
            None
        }
    });
    if matches!(&command, Commands::ShowConfig) {
        let rendered = render_show_config(&cfg, &config_source_desc)?;
        println!(
            "{}",
            render::render_markdown_for_terminal(&rendered, cfg.console.colorful)
        );
        return Ok(ExitCode::Success);
    }
    if let Commands::Snapshot { output } = &command {
        let result = build_snapshot_binary(&cfg, config_source_desc.clone(), output.clone())?;
        let rendered = render_snapshot_result(&result);
        println!(
            "{}",
            render::render_markdown_for_terminal(&rendered, cfg.console.colorful)
        );
        return Ok(ExitCode::Success);
    }

    let run_dir = std::env::current_dir()
        .map_err(|err| AppError::Runtime(format!("failed to resolve runtime directory: {err}")))?;
    let session_path = SessionStore::session_file(Path::new(&run_dir));
    let mut session = SessionStore::load_or_new(
        session_path,
        cfg.session.recent_messages,
        cfg.session.max_messages,
        cfg.ai.chat.compression.max_history_messages,
        cfg.ai.chat.compression.max_chars_count,
    )?;
    if matches!(&command, Commands::Chat) {
        session.start_new_session_with_new_file()?;
    }
    let executable_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.to_path_buf()))
        .unwrap_or_else(|| run_dir.clone());
    let log_file = logging::init(&cfg.log, &executable_dir, session.session_id())?;
    logging::info(&format!(
        "MachineClaw start, config={}, log_file={}",
        config_source_desc,
        log_file.display()
    ));
    logging::info(&format!(
        "selected language={}",
        language_code(selected_language)
    ));
    if let Some(notice) = language_warning {
        eprintln!(
            "{}",
            render::render_warn_line(&notice, cfg.console.colorful)
        );
        logging::warn(&notice);
    }

    ShellExecutor::install_interrupt_handler()?;

    let assets_setup = render::locate_or_init_assets_dir()?;
    let assets_dir = assets_setup.path;
    logging::info(&format!("assets directory={}", assets_dir.display()));
    for notice in assets_setup.notices {
        println!(
            "{}",
            render::render_info_line(&notice, cfg.console.colorful)
        );
        logging::info(&notice);
    }

    let ai_client = AiClient::new(
        &cfg.ai,
        run_dir.join(".machineclaw").join("model-prices.json"),
        cfg.console.colorful,
    )?;
    let run_ai_connectivity_check =
        !matches!(&command, Commands::Chat) || cfg.ai.connectivity_check;
    let require_elevated = !matches!(&command, Commands::Chat);
    run_preflight_checks(
        &cfg,
        &ai_client,
        run_ai_connectivity_check,
        require_elevated,
    )?;

    let skills_dir = expand_tilde(&cfg.skills.dir);
    let skill_list = if cfg.skills.enabled {
        detect_skills(&skills_dir)?
    } else {
        Vec::new()
    };
    logging::info(&format!(
        "skills enabled={}, directory={}, skills_count={}",
        cfg.skills.enabled,
        skills_dir.display(),
        skill_list.len()
    ));

    let shell = ShellExecutor::new(&cfg.cmd);
    let use_async_mcp_availability_check = matches!(&command, Commands::Chat)
        && cfg.mcp.enabled
        && normalize_mcp_availability_check_mode(cfg.mcp.mcp_availability_check_mode.as_str())
            == "async";
    let mut mcp_manager = if use_async_mcp_availability_check {
        let pending_summary = format!("{}, availability=checking(async)", mcp_summary(&cfg.mcp));
        mcp::McpManager::pending_with_config(&cfg.mcp, pending_summary)
    } else {
        mcp::McpManager::connect(&cfg.mcp)?
    };

    let os_type = current_os();
    let os_name = os_name();
    let mcp_desc = if cfg.mcp.enabled {
        mcp_manager.summary()
    } else {
        mcp_summary(&cfg.mcp)
    };

    let mut services = ActionServices {
        cfg: &cfg,
        assets_dir: &assets_dir,
        shell: &shell,
        ai: &ai_client,
        session: &mut session,
        os_type,
        os_name,
        skills: &skill_list,
        mcp_summary: mcp_desc,
        mcp: &mut mcp_manager,
    };

    let outcome = match command {
        Commands::Prepare => actions::run_prepare(&mut services)?,
        Commands::Inspect { target } => actions::run_inspect(&mut services, target)?,
        Commands::Test { .. } => {
            return Err(AppError::Runtime(
                "test command should be handled before runtime actions".to_string(),
            ));
        }
        Commands::Chat => actions::run_chat(&mut services)?,
        Commands::Config { .. } => {
            return Err(AppError::Runtime(
                "config command should be handled before runtime actions".to_string(),
            ));
        }
        Commands::Snapshot { .. } => {
            return Err(AppError::Runtime(
                "snapshot command should be handled before runtime actions".to_string(),
            ));
        }
        Commands::ShowConfig => {
            return Err(AppError::Runtime(
                "show-config command should be handled before runtime actions".to_string(),
            ));
        }
    };

    println!("{}", outcome.rendered);
    Ok(outcome.exit_code)
}

fn run_preflight_checks(
    cfg: &config::AppConfig,
    ai_client: &AiClient,
    run_ai_connectivity_check: bool,
    require_elevated: bool,
) -> Result<(), AppError> {
    logging::info("preflight start");
    let started = Instant::now();
    println!(
        "{}",
        render::render_info_line(i18n::preflight_notice_start(), cfg.console.colorful)
    );
    println!(
        "{}",
        render::render_info_line(i18n::preflight_notice_config_check(), cfg.console.colorful)
    );
    validate_config(cfg)?;
    println!(
        "{}",
        render::render_info_line(
            i18n::preflight_notice_permission_check(),
            cfg.console.colorful
        )
    );
    if require_elevated {
        require_elevated_permissions()?;
    } else {
        println!(
            "{}",
            render::render_info_line(
                i18n::preflight_notice_permission_check_skipped(),
                cfg.console.colorful
            )
        );
    }
    if run_ai_connectivity_check {
        let mut ai_spinner = PreflightSpinner::start(
            i18n::preflight_notice_ai_check().to_string(),
            cfg.console.colorful,
        );
        let ai_result = ai_client.validate_connectivity();
        ai_spinner.stop();
        ai_result?;
    } else {
        println!(
            "{}",
            render::render_info_line(
                i18n::preflight_notice_ai_check_skipped(),
                cfg.console.colorful
            )
        );
    }
    println!(
        "{}",
        render::render_info_line(
            &i18n::preflight_notice_done(&i18n::human_duration_ms(started.elapsed().as_millis())),
            cfg.console.colorful,
        )
    );
    logging::info("preflight finished");
    Ok(())
}

struct PreflightSpinner {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PreflightSpinner {
    fn start(message: String, colorful: bool) -> Self {
        if !io::stdout().is_terminal() {
            println!("{}", render::render_info_line(&message, colorful));
            return Self {
                stop: Arc::new(AtomicBool::new(true)),
                handle: None,
            };
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stop_cloned = stop.clone();
        let handle = thread::spawn(move || {
            let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut idx = 0usize;
            while !stop_cloned.load(Ordering::SeqCst) {
                let text = format!("{message} {}", frames[idx % frames.len()]);
                print!("\r{}", render::render_chat_notice(&text, colorful));
                let _ = io::stdout().flush();
                idx = idx.wrapping_add(1);
                thread::sleep(Duration::from_millis(100));
            }
            clear_preflight_line();
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PreflightSpinner {
    fn drop(&mut self) {
        self.stop();
    }
}

fn clear_preflight_line() {
    if !io::stdout().is_terminal() {
        return;
    }
    print!("\r{: <180}\r", "");
    let _ = io::stdout().flush();
}

fn resolve_startup_colorful(raw_args: &[String]) -> bool {
    let mut colorful = true;
    let config_path = if let Some(conf_path) = cli::extract_conf_path_from_args(raw_args) {
        resolve_config_path(Some(conf_path)).ok()
    } else {
        resolve_config_path(None).ok()
    };
    if let Some(path) = config_path
        && let Some(colorful_hint) = read_console_colorful_hint(&path)
    {
        colorful = colorful_hint;
    }
    render::resolve_colorful_enabled(colorful)
}
