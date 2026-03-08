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

use std::path::Path;

use clap::Parser;

use crate::{
    actions::ActionServices,
    ai::AiClient,
    cli::{Cli, Commands},
    config::{
        config_template_example, expand_tilde, load_config, read_language_hint,
        resolve_config_path, validate_config,
    },
    config_action::run_config_command,
    context::SessionStore,
    error::{AppError, ExitCode},
    i18n::{language_code, localize_error, parse_language, resolve_language, set_language},
    mcp::{mcp_summary, validate_mcp_config},
    platform::{current_os, os_name, require_elevated_permissions},
    shell::ShellExecutor,
    skills::detect_skills,
};

fn main() {
    let exit_code = match run() {
        Ok(code) => code,
        Err(err) => {
            let code = err.exit_code();
            let message = mask::mask_sensitive(&localize_error(&err));
            eprintln!("{}: {message}", i18n::prefix_error());
            logging::error(&format!("program failed: {message}"));
            code
        }
    };
    std::process::exit(exit_code as i32);
}

fn run() -> Result<ExitCode, AppError> {
    set_language(resolve_language(None));
    let raw_args: Vec<String> = std::env::args().collect();
    if let Some(conf_path) = cli::extract_conf_path_from_args(&raw_args)
        && let Ok(resolved_path) = resolve_config_path(Some(conf_path))
        && let Some(language_hint) = read_language_hint(&resolved_path)
    {
        set_language(resolve_language(Some(&language_hint)));
    }
    if let Some(topic) = cli::detect_help_topic(&raw_args) {
        println!("{}", cli::localized_help(topic));
        return Ok(ExitCode::Success);
    }

    let cli = Cli::try_parse().map_err(|err| AppError::Runtime(err.to_string()))?;
    if cli.show_config_template {
        println!(
            "{}",
            render::render_markdown_for_terminal(
                config_template_example(),
                render::resolve_colorful_enabled(true),
            )
        );
        return Ok(ExitCode::Success);
    }
    let Some(command) = cli.command else {
        println!("{}", cli::localized_help(cli::HelpTopic::Global));
        return Ok(ExitCode::Success);
    };
    let config_path = resolve_config_path(cli.conf)?;
    if let Some(language_hint) = read_language_hint(&config_path) {
        set_language(resolve_language(Some(&language_hint)));
    }
    if let Commands::Config { command } = &command {
        let outcome = run_config_command(&config_path, command)?;
        println!("{}", outcome.rendered);
        return Ok(outcome.exit_code);
    }

    let mut cfg = load_config(&config_path)?;
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

    let run_dir = std::env::current_dir()
        .map_err(|err| AppError::Runtime(format!("failed to resolve runtime directory: {err}")))?;
    let log_file = logging::init(&run_dir.join("logs"))?;
    logging::info(&format!(
        "MachineClaw start, config={}, log_file={}",
        config_path.display(),
        log_file.display()
    ));
    logging::info(&format!(
        "selected language={}",
        language_code(selected_language)
    ));
    if let Some(notice) = language_warning {
        eprintln!("{}: {notice}", i18n::prefix_warn());
        logging::warn(&notice);
    }

    ShellExecutor::install_interrupt_handler()?;

    let assets_setup = render::locate_or_init_assets_dir()?;
    let assets_dir = assets_setup.path;
    logging::info(&format!("assets directory={}", assets_dir.display()));
    for notice in assets_setup.notices {
        println!("{}: {notice}", i18n::prefix_info());
        logging::info(&notice);
    }

    let ai_client = AiClient::new(&cfg.ai)?;
    run_preflight_checks(&cfg, &ai_client)?;

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
    let session_path = SessionStore::session_file(Path::new(&run_dir));
    let mut session = SessionStore::load_or_new(
        session_path,
        cfg.session.recent_messages,
        cfg.session.max_messages,
    )?;
    let mut mcp_manager = mcp::McpManager::connect(&cfg.mcp)?;

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
        Commands::Chat => actions::run_chat(&mut services)?,
        Commands::Config { .. } => {
            return Err(AppError::Runtime(
                "config command should be handled before runtime actions".to_string(),
            ));
        }
    };

    println!("{}", outcome.rendered);
    Ok(outcome.exit_code)
}

fn run_preflight_checks(cfg: &config::AppConfig, ai_client: &AiClient) -> Result<(), AppError> {
    logging::info("preflight start");
    validate_config(cfg)?;
    require_elevated_permissions()?;
    ai_client.validate_connectivity()?;
    logging::info("preflight finished");
    Ok(())
}
