use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

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

fn help_en(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\nA cross-platform CLI for machine preflight checks, inspection, and interactive diagnosis.\n\nUsage:\n  MachineClaw [OPTIONS] <COMMAND>\n\nCommands:\n  prepare      Run preflight environment checks\n  inspect      Inspect machine state by target\n  test         Validate runtime config and integrity\n  chat         Start interactive chat mode\n  snapshot     Build self-contained executable with config snapshot\n  show-config  Show effective config snapshot (masked)\n  config       Get or set config values\n  help         Print this help or subcommand help\n\nOptions:\n  -c, --conf <path>        Config file path (supports --conf=... and --conf ...)\n  --show-config-template   Print full config template\n  -h, --help               Print help\n  -V, --version            Print version\n\nExamples:\n  MachineClaw --show-config-template\n  MachineClaw show-config --conf ./docs/claw.toml\n  MachineClaw snapshot --conf ./docs/claw.toml -o ./MachineClaw-prod\n  MachineClaw test config --conf ./docs/claw.toml\n  MachineClaw config get ai.retry.max-retries --conf ./docs/claw.toml\n  MachineClaw config set ai.retry.max-retries 5 --conf ./docs/claw.toml\n".to_string(),
        HelpTopic::Prepare => "Usage:\n  MachineClaw prepare [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "Usage:\n  MachineClaw inspect <target> [OPTIONS]\n\nTargets:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "Usage:\n  MachineClaw test <target> [OPTIONS]\n\nTargets:\n  config\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "Usage:\n  MachineClaw chat [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "Usage:\n  MachineClaw snapshot [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -o, --output <path>  Output executable path\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "Usage:\n  MachineClaw show-config [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "Usage:\n  MachineClaw config <COMMAND> [ARGS] [OPTIONS]\n\nCommands:\n  get <key>          Get effective value\n  set <key> <value>  Set value into config file\n\nExamples:\n  MachineClaw config get ai.retry.max-retries --conf ./docs/claw.toml\n  MachineClaw config set ai.retry.max-retries 5 --conf ./docs/claw.toml\n".to_string(),
    }
}

fn help_zh_cn(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\n跨平台命令行工具，用于机器预检、状态巡检与交互式诊断。\n\n用法:\n  MachineClaw [选项] <命令>\n\n命令:\n  prepare      执行运行前环境检查\n  inspect      按目标检查机器状态\n  test         校验配置与运行参数完整性\n  chat         进入交互式对话模式\n  snapshot     打包内置配置快照的可执行文件\n  show-config  展示当前生效配置快照（已脱敏）\n  config       获取或设置配置项\n  help         显示帮助信息\n\n选项:\n  -c, --conf <path>       配置文件路径（支持 --conf=... 与 --conf ...）\n  --show-config-template  展示完整配置模板\n  -h, --help              显示帮助\n  -V, --version           显示版本\n\n示例:\n  MachineClaw --show-config-template\n  MachineClaw show-config --conf ./docs/claw.toml\n  MachineClaw snapshot --conf ./docs/claw.toml -o ./MachineClaw-prod\n  MachineClaw test config --conf ./docs/claw.toml\n  MachineClaw config get ai.retry.max-retries --conf ./docs/claw.toml\n  MachineClaw config set ai.retry.max-retries 5 --conf ./docs/claw.toml\n".to_string(),
        HelpTopic::Prepare => "用法:\n  MachineClaw prepare [选项]\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "用法:\n  MachineClaw inspect <target> [选项]\n\ntarget 可选值:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "用法:\n  MachineClaw test <target> [选项]\n\ntarget 可选值:\n  config\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "用法:\n  MachineClaw chat [选项]\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "用法:\n  MachineClaw snapshot [选项]\n\n选项:\n  -c, --conf <path>\n  -o, --output <path>  输出可执行文件路径\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "用法:\n  MachineClaw show-config [选项]\n\n选项:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "用法:\n  MachineClaw config <命令> [参数] [选项]\n\n命令:\n  get <key>          获取配置字段当前生效值\n  set <key> <value>  设置配置字段值并写回配置文件\n\n示例:\n  MachineClaw config get ai.retry.max-retries --conf ./docs/claw.toml\n  MachineClaw config set ai.retry.max-retries 5 --conf ./docs/claw.toml\n".to_string(),
    }
}

fn help_zh_tw(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\n跨平台命令列工具，用於機器預檢、狀態巡檢與互動式診斷。\n\n用法:\n  MachineClaw [選項] <命令>\n\n命令:\n  prepare      執行啟動前環境檢查\n  inspect      依目標檢查機器狀態\n  test         驗證配置與執行參數完整性\n  chat         進入互動式對話模式\n  snapshot     打包內嵌配置快照的可執行檔\n  show-config  顯示目前生效配置快照（已脫敏）\n  config       取得或設定配置項\n  help         顯示說明\n\n選項:\n  -c, --conf <path>       設定檔路徑（支援 --conf=... 與 --conf ...）\n  --show-config-template  顯示完整配置模板\n  -h, --help              顯示說明\n  -V, --version           顯示版本\n\n示例:\n  MachineClaw --show-config-template\n  MachineClaw show-config --conf ./docs/claw.toml\n  MachineClaw snapshot --conf ./docs/claw.toml -o ./MachineClaw-prod\n  MachineClaw test config --conf ./docs/claw.toml\n".to_string(),
        HelpTopic::Prepare => "用法:\n  MachineClaw prepare [選項]\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "用法:\n  MachineClaw inspect <target> [選項]\n\ntarget 可選值:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "用法:\n  MachineClaw test <target> [選項]\n\ntarget 可選值:\n  config\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "用法:\n  MachineClaw chat [選項]\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "用法:\n  MachineClaw snapshot [選項]\n\n選項:\n  -c, --conf <path>\n  -o, --output <path>  輸出可執行檔路徑\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "用法:\n  MachineClaw show-config [選項]\n\n選項:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "用法:\n  MachineClaw config <命令> [參數] [選項]\n\n命令:\n  get <key>          取得配置欄位目前生效值\n  set <key> <value>  設定配置欄位值並寫回設定檔\n".to_string(),
    }
}

fn help_fr(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\nUtilitaire CLI multiplateforme pour la pré-vérification, l'inspection et le diagnostic interactif.\n\nUtilisation:\n  MachineClaw [OPTIONS] <COMMANDE>\n\nCommandes:\n  prepare      Vérifier l'environnement avant exécution\n  inspect      Inspecter l'état de la machine par cible\n  test         Valider la configuration et l'intégrité\n  chat         Démarrer le mode conversation interactif\n  snapshot     Générer un exécutable autonome avec snapshot config\n  show-config  Afficher la configuration effective (masquée)\n  config       Lire ou modifier la configuration\n  help         Afficher cette aide\n\nOptions:\n  -c, --conf <path>        Chemin du fichier de configuration (supporte --conf=... et --conf ...)\n  --show-config-template   Afficher le modèle complet de configuration\n  -h, --help               Afficher l'aide\n  -V, --version            Afficher la version\n".to_string(),
        HelpTopic::Prepare => "Utilisation:\n  MachineClaw prepare [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "Utilisation:\n  MachineClaw inspect <target> [OPTIONS]\n\nValeurs target:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "Utilisation:\n  MachineClaw test <target> [OPTIONS]\n\nValeurs target:\n  config\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "Utilisation:\n  MachineClaw chat [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "Utilisation:\n  MachineClaw snapshot [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -o, --output <path>  Chemin de sortie exécutable\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "Utilisation:\n  MachineClaw show-config [OPTIONS]\n\nOptions:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "Utilisation:\n  MachineClaw config <COMMANDE> [ARGS] [OPTIONS]\n\nCommandes:\n  get <key>          Lire la valeur effective\n  set <key> <value>  Écrire la valeur dans le fichier de config\n".to_string(),
    }
}

fn help_de(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\nPlattformübergreifendes CLI für Preflight-Checks, Inspektion und interaktive Diagnose.\n\nVerwendung:\n  MachineClaw [OPTIONEN] <BEFEHL>\n\nBefehle:\n  prepare      Umgebungsprüfung vor der Ausführung\n  inspect      Maschinenstatus nach Ziel prüfen\n  test         Konfiguration und Integrität prüfen\n  chat         Interaktiven Chat-Modus starten\n  snapshot     Selbstenthaltende Binärdatei mit Konfig-Snapshot bauen\n  show-config  Effektive Konfiguration (maskiert) anzeigen\n  config       Konfiguration lesen oder setzen\n  help         Hilfe anzeigen\n\nOptionen:\n  -c, --conf <path>        Pfad zur Konfigurationsdatei (unterstützt --conf=... und --conf ...)\n  --show-config-template   Vollständige Konfigurationsvorlage anzeigen\n  -h, --help               Hilfe anzeigen\n  -V, --version            Version anzeigen\n".to_string(),
        HelpTopic::Prepare => "Verwendung:\n  MachineClaw prepare [OPTIONEN]\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "Verwendung:\n  MachineClaw inspect <target> [OPTIONEN]\n\nTarget-Werte:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "Verwendung:\n  MachineClaw test <target> [OPTIONEN]\n\nTarget-Werte:\n  config\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "Verwendung:\n  MachineClaw chat [OPTIONEN]\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "Verwendung:\n  MachineClaw snapshot [OPTIONEN]\n\nOptionen:\n  -c, --conf <path>\n  -o, --output <path>  Ausgabe-Binärpfad\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "Verwendung:\n  MachineClaw show-config [OPTIONEN]\n\nOptionen:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "Verwendung:\n  MachineClaw config <BEFEHL> [ARGS] [OPTIONEN]\n\nBefehle:\n  get <key>          Effektiven Wert lesen\n  set <key> <value>  Wert in Konfigurationsdatei schreiben\n".to_string(),
    }
}

fn help_ja(topic: HelpTopic) -> String {
    match topic {
        HelpTopic::Global => "MachineClaw\n\nマシンの事前チェック・状態確認・対話診断のためのクロスプラットフォームCLIです。\n\n使い方:\n  MachineClaw [オプション] <コマンド>\n\nコマンド:\n  prepare      実行前の環境チェックを実施\n  inspect      対象ごとにマシン状態を確認\n  test         設定と整合性を検証\n  chat         対話チャットモードを開始\n  snapshot     設定スナップショット内蔵バイナリを生成\n  show-config  有効設定のスナップショットを表示（マスク済み）\n  config       設定値を取得/更新\n  help         ヘルプを表示\n\nオプション:\n  -c, --conf <path>       設定ファイルのパス（--conf=... と --conf ... をサポート）\n  --show-config-template  完全な設定テンプレートを表示\n  -h, --help              ヘルプを表示\n  -V, --version           バージョンを表示\n".to_string(),
        HelpTopic::Prepare => "使い方:\n  MachineClaw prepare [オプション]\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Inspect => "使い方:\n  MachineClaw inspect <target> [オプション]\n\ntarget の値:\n  cpu, memory, disk, os, process, filesystem, hardware, logs, network, all\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Test => "使い方:\n  MachineClaw test <target> [オプション]\n\ntarget の値:\n  config\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Chat => "使い方:\n  MachineClaw chat [オプション]\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Snapshot => "使い方:\n  MachineClaw snapshot [オプション]\n\nオプション:\n  -c, --conf <path>\n  -o, --output <path>  出力バイナリのパス\n  -h, --help\n".to_string(),
        HelpTopic::ShowConfig => "使い方:\n  MachineClaw show-config [オプション]\n\nオプション:\n  -c, --conf <path>\n  -h, --help\n".to_string(),
        HelpTopic::Config => "使い方:\n  MachineClaw config <コマンド> [引数] [オプション]\n\nコマンド:\n  get <key>          実効値を取得\n  set <key> <value>  設定ファイルへ値を書き込み\n".to_string(),
    }
}
