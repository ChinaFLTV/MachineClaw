use std::{
    cmp::Ordering,
    fs,
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use colored::Colorize;
use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, HeaderMap, HeaderValue, USER_AGENT};
use serde::Deserialize;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::{
    actions::ActionOutcome,
    error::{AppError, ExitCode},
    i18n::{self, Language},
    platform::{OsType, current_os},
    render,
    tls::ensure_rustls_crypto_provider,
};

const RELEASES_API_URL: &str =
    "https://api.github.com/repos/ChinaFLTV/MachineClaw/releases?per_page=20";
const RELEASES_WEB_URL: &str = "https://github.com/ChinaFLTV/MachineClaw/releases";
const UPGRADE_USER_AGENT: &str =
    "MachineClaw-upgrade/1.0 (+https://github.com/ChinaFLTV/MachineClaw)";
const HTTP_CONNECT_TIMEOUT_SECS: u64 = 8;
const HTTP_REQUEST_TIMEOUT_SECS: u64 = 45;
const HTTP_DOWNLOAD_TIMEOUT_SECS: u64 = 300;

#[derive(Debug, Clone, Copy)]
struct UpgradeLocale {
    language: Language,
}

impl UpgradeLocale {
    fn new() -> Self {
        Self {
            language: i18n::current_language(),
        }
    }

    fn title(self) -> &'static str {
        match self.language {
            Language::ZhCn => "MachineClaw 升级中心",
            Language::ZhTw => "MachineClaw 升級中心",
            Language::Fr => "Centre de mise a niveau MachineClaw",
            Language::De => "MachineClaw Upgrade-Zentrale",
            Language::Ja => "MachineClaw アップグレードセンター",
            Language::En => "MachineClaw Upgrade Center",
        }
    }

    fn action_label(self) -> &'static str {
        match self.language {
            Language::ZhCn => "动作",
            Language::ZhTw => "動作",
            Language::Fr => "Action",
            Language::De => "Aktion",
            Language::Ja => "操作",
            Language::En => "Action",
        }
    }

    fn status_label(self) -> &'static str {
        match self.language {
            Language::ZhCn => "状态",
            Language::ZhTw => "狀態",
            Language::Fr => "Statut",
            Language::De => "Status",
            Language::Ja => "状態",
            Language::En => "Status",
        }
    }

    fn elapsed_label(self) -> &'static str {
        match self.language {
            Language::ZhCn => "耗时",
            Language::ZhTw => "耗時",
            Language::Fr => "Duree",
            Language::De => "Dauer",
            Language::Ja => "経過時間",
            Language::En => "Elapsed",
        }
    }

    fn section_version_radar(self) -> &'static str {
        match self.language {
            Language::ZhCn => "版本雷达",
            Language::ZhTw => "版本雷達",
            Language::Fr => "Radar des versions",
            Language::De => "Versionsradar",
            Language::Ja => "バージョンレーダー",
            Language::En => "Version Radar",
        }
    }

    fn section_risk(self) -> &'static str {
        match self.language {
            Language::ZhCn => "风险防护",
            Language::ZhTw => "風險防護",
            Language::Fr => "Boucliers de risque",
            Language::De => "Risikoschilde",
            Language::Ja => "リスクシールド",
            Language::En => "Risk Shields",
        }
    }

    fn section_decision(self) -> &'static str {
        match self.language {
            Language::ZhCn => "升级决策",
            Language::ZhTw => "升級決策",
            Language::Fr => "Decision de mise a niveau",
            Language::De => "Upgrade-Entscheidung",
            Language::Ja => "アップグレード判断",
            Language::En => "Upgrade Decision",
        }
    }

    fn section_pipeline(self) -> &'static str {
        match self.language {
            Language::ZhCn => "下载管线",
            Language::ZhTw => "下載管線",
            Language::Fr => "Pipeline de telechargement",
            Language::De => "Download-Pipeline",
            Language::Ja => "ダウンロードパイプライン",
            Language::En => "Download Pipeline",
        }
    }

    fn item_none(self) -> &'static str {
        match self.language {
            Language::ZhCn => "无",
            Language::ZhTw => "無",
            Language::Fr => "aucun",
            Language::De => "keine",
            Language::Ja => "なし",
            Language::En => "none",
        }
    }

    fn metric_local_version(self) -> &'static str {
        match self.language {
            Language::ZhCn => "本地版本",
            Language::ZhTw => "本機版本",
            Language::Fr => "version locale",
            Language::De => "lokale Version",
            Language::Ja => "ローカルバージョン",
            Language::En => "local version",
        }
    }

    fn metric_repo(self) -> &'static str {
        match self.language {
            Language::ZhCn => "仓库地址",
            Language::ZhTw => "倉庫位址",
            Language::Fr => "depot",
            Language::De => "Repository",
            Language::Ja => "リポジトリ",
            Language::En => "repository",
        }
    }

    fn metric_latest_release(self) -> &'static str {
        match self.language {
            Language::ZhCn => "最新发布",
            Language::ZhTw => "最新發佈",
            Language::Fr => "derniere release",
            Language::De => "neueste Release",
            Language::Ja => "最新リリース",
            Language::En => "latest release",
        }
    }

    fn metric_release_title(self) -> &'static str {
        match self.language {
            Language::ZhCn => "发布标题",
            Language::ZhTw => "發佈標題",
            Language::Fr => "titre de release",
            Language::De => "Release-Titel",
            Language::Ja => "リリースタイトル",
            Language::En => "release title",
        }
    }

    fn metric_release_page(self) -> &'static str {
        match self.language {
            Language::ZhCn => "发布页面",
            Language::ZhTw => "發佈頁面",
            Language::Fr => "page de release",
            Language::De => "Release-Seite",
            Language::Ja => "リリースページ",
            Language::En => "release page",
        }
    }

    fn metric_published_at(self) -> &'static str {
        match self.language {
            Language::ZhCn => "发布时间",
            Language::ZhTw => "發佈時間",
            Language::Fr => "publie a",
            Language::De => "veroeffentlicht am",
            Language::Ja => "公開時刻",
            Language::En => "published at",
        }
    }

    fn metric_platform(self) -> &'static str {
        match self.language {
            Language::ZhCn => "平台",
            Language::ZhTw => "平台",
            Language::Fr => "plateforme",
            Language::De => "Plattform",
            Language::Ja => "プラットフォーム",
            Language::En => "platform",
        }
    }

    fn metric_selected_asset(self) -> &'static str {
        match self.language {
            Language::ZhCn => "选中资源",
            Language::ZhTw => "選中資源",
            Language::Fr => "asset selectionne",
            Language::De => "gewaehltes Asset",
            Language::Ja => "選択アセット",
            Language::En => "selected asset",
        }
    }

    fn metric_declared_asset_size(self) -> &'static str {
        match self.language {
            Language::ZhCn => "声明资源大小",
            Language::ZhTw => "宣告資源大小",
            Language::Fr => "taille declaree",
            Language::De => "angegebene Asset-Groesse",
            Language::Ja => "申告サイズ",
            Language::En => "declared asset size",
        }
    }

    fn metric_download_target(self) -> &'static str {
        match self.language {
            Language::ZhCn => "下载目标",
            Language::ZhTw => "下載目標",
            Language::Fr => "cible de telechargement",
            Language::De => "Download-Ziel",
            Language::Ja => "ダウンロード先",
            Language::En => "download target",
        }
    }

    fn metric_downloaded_size(self) -> &'static str {
        match self.language {
            Language::ZhCn => "已下载大小",
            Language::ZhTw => "已下載大小",
            Language::Fr => "taille telechargee",
            Language::De => "heruntergeladene Groesse",
            Language::Ja => "ダウンロード済みサイズ",
            Language::En => "downloaded size",
        }
    }

    fn risk_local_version_invalid(self, version: &str) -> String {
        match self.language {
            Language::ZhCn => format!("本地版本 `{version}` 格式无效，无法安全比较"),
            Language::ZhTw => format!("本機版本 `{version}` 格式無效，無法安全比較"),
            Language::Fr => {
                format!("la version locale `{version}` est invalide, comparaison impossible")
            }
            Language::De => {
                format!("lokale Version `{version}` ist ungueltig, Vergleich nicht sicher moeglich")
            }
            Language::Ja => format!("ローカル版 `{version}` の形式が不正で安全に比較できません"),
            Language::En => format!("local version `{version}` is invalid, cannot compare safely"),
        }
    }

    fn risk_fetch_failed(self, err: &str) -> String {
        match self.language {
            Language::ZhCn => format!("拉取 release 列表失败: {err}"),
            Language::ZhTw => format!("拉取 release 清單失敗: {err}"),
            Language::Fr => format!("echec de lecture de la liste release: {err}"),
            Language::De => format!("Abruf der Release-Liste fehlgeschlagen: {err}"),
            Language::Ja => format!("release 一覧の取得に失敗しました: {err}"),
            Language::En => format!("failed to fetch release list: {err}"),
        }
    }

    fn decision_keep_current(self) -> &'static str {
        match self.language {
            Language::ZhCn => "无法判断升级可用性，保持当前版本",
            Language::ZhTw => "無法判斷升級可用性，保持當前版本",
            Language::Fr => {
                "impossible de verifier la mise a niveau, conservation de la version actuelle"
            }
            Language::De => "Upgrade-Verfuegbarkeit unklar, aktuelle Version wird beibehalten",
            Language::Ja => "アップグレード可否を判定できないため現行版を維持します",
            Language::En => "unable to check upgrade availability; keep current version",
        }
    }

    fn decision_no_eligible_release(self) -> &'static str {
        match self.language {
            Language::ZhCn => "未找到可用 Release（可能为空、仅草稿或仅预发布）",
            Language::ZhTw => "未找到可用 Release（可能為空、僅草稿或僅預發布）",
            Language::Fr => {
                "aucune release eligible trouvee (vide, brouillon seulement, ou prerelease seulement)"
            }
            Language::De => {
                "keine geeignete Release gefunden (leer, nur Entwurf oder nur Vorabversion)"
            }
            Language::Ja => {
                "対象となる release がありません（空、ドラフトのみ、またはプレリリースのみ）"
            }
            Language::En => {
                "no eligible release found (possibly empty, draft-only, or prerelease-only)"
            }
        }
    }

    fn risk_prerelease_excluded(self) -> &'static str {
        match self.language {
            Language::ZhCn => "默认排除预发布版本，可用 `--allow-prerelease` 重新检查",
            Language::ZhTw => "預設排除預發布版本，可用 `--allow-prerelease` 重新檢查",
            Language::Fr => {
                "les prereleases sont exclues par defaut, relancez avec `--allow-prerelease`"
            }
            Language::De => {
                "Vorabversionen sind standardmaessig ausgeschlossen, erneut mit `--allow-prerelease` pruefen"
            }
            Language::Ja => {
                "プレリリースは既定で除外されています。`--allow-prerelease` で再確認してください"
            }
            Language::En => {
                "prerelease is excluded by default; rerun with `--allow-prerelease` if needed"
            }
        }
    }

    fn na_value(self) -> &'static str {
        "N/A"
    }

    fn risk_latest_version_invalid(self, version: &str) -> String {
        match self.language {
            Language::ZhCn => format!("远端 release 版本号 `{version}` 格式无效"),
            Language::ZhTw => format!("遠端 release 版本號 `{version}` 格式無效"),
            Language::Fr => format!("le tag release distant `{version}` est invalide"),
            Language::De => format!("Remote-Release-Tag `{version}` ist ungueltig"),
            Language::Ja => format!("リモート release タグ `{version}` の形式が不正です"),
            Language::En => format!("latest release tag `{version}` has invalid version format"),
        }
    }

    fn decision_skip_auto_upgrade(self) -> &'static str {
        match self.language {
            Language::ZhCn => "自动升级流程已跳过",
            Language::ZhTw => "自動升級流程已跳過",
            Language::Fr => "le flux de mise a niveau automatique est ignore",
            Language::De => "automatischer Upgrade-Ablauf wurde uebersprungen",
            Language::Ja => "自動アップグレード処理をスキップしました",
            Language::En => "automatic upgrade flow is skipped",
        }
    }

    fn decision_already_up_to_date(self, local: &str, remote: &str) -> String {
        match self.language {
            Language::ZhCn => format!("已是最新版本：本地 `{local}` >= 远端 `{remote}`"),
            Language::ZhTw => format!("已是最新版本：本機 `{local}` >= 遠端 `{remote}`"),
            Language::Fr => format!("deja a jour: local `{local}` >= distant `{remote}`"),
            Language::De => format!("bereits aktuell: lokal `{local}` >= remote `{remote}`"),
            Language::Ja => format!("最新です: ローカル `{local}` >= リモート `{remote}`"),
            Language::En => format!("already up-to-date: local `{local}` >= remote `{remote}`"),
        }
    }

    fn decision_upgrade_available(self, local: &str, remote: &str) -> String {
        match self.language {
            Language::ZhCn => format!("检测到可升级版本：`{local}` -> `{remote}`"),
            Language::ZhTw => format!("偵測到可升級版本：`{local}` -> `{remote}`"),
            Language::Fr => format!("mise a niveau disponible: `{local}` -> `{remote}`"),
            Language::De => format!("Upgrade verfuegbar: `{local}` -> `{remote}`"),
            Language::Ja => format!("アップグレード可能です: `{local}` -> `{remote}`"),
            Language::En => format!("upgrade available: `{local}` -> `{remote}`"),
        }
    }

    fn risk_no_compatible_asset(self) -> &'static str {
        match self.language {
            Language::ZhCn => "未找到与当前平台匹配的可下载资源",
            Language::ZhTw => "未找到與當前平台匹配的可下載資源",
            Language::Fr => "aucun asset telechargeable compatible avec la plateforme actuelle",
            Language::De => "kein kompatibles Download-Asset fuer die aktuelle Plattform gefunden",
            Language::Ja => "現在のプラットフォームに一致するダウンロード可能アセットがありません",
            Language::En => "no compatible downloadable asset found for current platform",
        }
    }

    fn decision_manual_download_fallback(self) -> &'static str {
        match self.language {
            Language::ZhCn => "请打开 release 页面手动下载",
            Language::ZhTw => "請開啟 release 頁面手動下載",
            Language::Fr => "ouvrez la page release pour telecharger manuellement",
            Language::De => "oeffnen Sie die Release-Seite fuer einen manuellen Download",
            Language::Ja => "release ページを開いて手動ダウンロードしてください",
            Language::En => "fallback: open release page and download manually",
        }
    }

    fn decision_check_only(self) -> &'static str {
        match self.language {
            Language::ZhCn => "已启用 `--check-only`，仅检查不下载",
            Language::ZhTw => "已啟用 `--check-only`，僅檢查不下載",
            Language::Fr => "`--check-only` active, verification uniquement",
            Language::De => "`--check-only` aktiv, nur Pruefung ohne Download",
            Language::Ja => "`--check-only` が有効のため、確認のみ実行しました",
            Language::En => "`--check-only` enabled, skip download/apply",
        }
    }

    fn risk_download_failed(self, err: &str) -> String {
        match self.language {
            Language::ZhCn => format!("下载失败: {err}"),
            Language::ZhTw => format!("下載失敗: {err}"),
            Language::Fr => format!("echec du telechargement: {err}"),
            Language::De => format!("Download fehlgeschlagen: {err}"),
            Language::Ja => format!("ダウンロード失敗: {err}"),
            Language::En => format!("download failed: {err}"),
        }
    }

    fn decision_keep_binary_unchanged(self) -> &'static str {
        match self.language {
            Language::ZhCn => "保持当前可执行文件不变",
            Language::ZhTw => "保持當前可執行檔不變",
            Language::Fr => "conserver le binaire actuel",
            Language::De => "aktuelle Binardatei unveraendert beibehalten",
            Language::Ja => "現在の実行ファイルを変更せず維持します",
            Language::En => "keep current executable unchanged",
        }
    }

    fn risk_archive_skip_auto_apply(self) -> &'static str {
        match self.language {
            Language::ZhCn => "下载结果是压缩包，为避免误替换已跳过自动覆盖",
            Language::ZhTw => "下載結果是壓縮包，為避免誤替換已跳過自動覆蓋",
            Language::Fr => "asset archive detecte, remplacement automatique ignore pour securite",
            Language::De => {
                "Archiv erkannt, automatisches Ersetzen wird aus Sicherheitsgruenden uebersprungen"
            }
            Language::Ja => "アーカイブ資産のため、誤置換防止として自動適用をスキップしました",
            Language::En => "downloaded asset is an archive, auto-replace is skipped",
        }
    }

    fn decision_manual_extract_replace(self, archive: &Path, exe: &Path) -> String {
        match self.language {
            Language::ZhCn => format!(
                "需要手动处理：先解压 `{}`，再替换 `{}`",
                archive.display(),
                exe.display()
            ),
            Language::ZhTw => format!(
                "需要手動處理：先解壓 `{}`，再替換 `{}`",
                archive.display(),
                exe.display()
            ),
            Language::Fr => format!(
                "etape manuelle requise: extraire `{}` puis remplacer `{}`",
                archive.display(),
                exe.display()
            ),
            Language::De => format!(
                "manueller Schritt erforderlich: `{}` entpacken und `{}` ersetzen",
                archive.display(),
                exe.display()
            ),
            Language::Ja => format!(
                "手動対応が必要です: `{}` を展開して `{}` を置換してください",
                archive.display(),
                exe.display()
            ),
            Language::En => format!(
                "manual step required: extract `{}` then replace `{}`",
                archive.display(),
                exe.display()
            ),
        }
    }

    fn decision_custom_output_skip_auto_apply(self) -> &'static str {
        match self.language {
            Language::ZhCn => "检测到自定义 `--output`，已仅下载不自动替换",
            Language::ZhTw => "檢測到自訂 `--output`，已僅下載不自動替換",
            Language::Fr => "`--output` personnalise detecte, telechargement uniquement",
            Language::De => "benutzerdefiniertes `--output` erkannt, nur Download ohne Auto-Apply",
            Language::Ja => "カスタム `--output` 指定のため、ダウンロードのみ実行しました",
            Language::En => {
                "custom `--output` path provided, download completed without auto-apply"
            }
        }
    }

    fn decision_manual_replace(self, downloaded: &Path) -> String {
        match self.language {
            Language::ZhCn => format!(
                "如需立即切换，请手动将 `{}` 替换到当前程序路径",
                downloaded.display()
            ),
            Language::ZhTw => format!(
                "如需立即切換，請手動將 `{}` 替換到當前程式路徑",
                downloaded.display()
            ),
            Language::Fr => format!(
                "pour basculer maintenant, remplacez manuellement avec `{}`",
                downloaded.display()
            ),
            Language::De => format!(
                "fuer sofortige Umstellung bitte manuell mit `{}` ersetzen",
                downloaded.display()
            ),
            Language::Ja => format!(
                "今すぐ切り替える場合は `{}` を手動で配置してください",
                downloaded.display()
            ),
            Language::En => format!(
                "if you want to switch now: replace current executable with `{}` manually",
                downloaded.display()
            ),
        }
    }

    fn decision_upgrade_applied(self, version: &str) -> String {
        match self.language {
            Language::ZhCn => format!("升级已完成，请重新启动 MachineClaw（新版本 `{version}`）"),
            Language::ZhTw => format!("升級已完成，請重新啟動 MachineClaw（新版本 `{version}`）"),
            Language::Fr => {
                format!("mise a niveau appliquee, redemarrez MachineClaw (version `{version}`)")
            }
            Language::De => {
                format!("Upgrade abgeschlossen, MachineClaw neu starten (neue Version `{version}`)")
            }
            Language::Ja => format!(
                "アップグレード適用済みです。MachineClaw を再起動してください（新バージョン `{version}`）"
            ),
            Language::En => format!(
                "upgrade applied successfully; restart MachineClaw (new version `{version}`)"
            ),
        }
    }

    fn risk_auto_apply_failed(self, err: &str) -> String {
        match self.language {
            Language::ZhCn => format!("自动替换失败: {err}"),
            Language::ZhTw => format!("自動替換失敗: {err}"),
            Language::Fr => format!("echec du remplacement automatique: {err}"),
            Language::De => format!("automatisches Ersetzen fehlgeschlagen: {err}"),
            Language::Ja => format!("自動適用に失敗しました: {err}"),
            Language::En => format!("auto-apply failed: {err}"),
        }
    }

    fn command_release_candidates(self) -> &'static str {
        match self.language {
            Language::ZhCn => "release_candidates(含草稿与预发布)",
            Language::ZhTw => "release_candidates(含草稿與預發布)",
            Language::Fr => "release_candidates(y compris brouillon/prerelease)",
            Language::De => "release_candidates(inkl. Entwurf/Vorab)",
            Language::Ja => "release_candidates(ドラフト/プレリリース含む)",
            Language::En => "release_candidates(draft+prerelease included)",
        }
    }

    fn command_available_assets(self) -> &'static str {
        match self.language {
            Language::ZhCn => "available_assets",
            Language::ZhTw => "available_assets",
            Language::Fr => "assets_disponibles",
            Language::De => "verfuegbare_assets",
            Language::Ja => "available_assets",
            Language::En => "available_assets",
        }
    }

    fn command_download(self) -> &'static str {
        match self.language {
            Language::ZhCn => "下载",
            Language::ZhTw => "下載",
            Language::Fr => "telechargement",
            Language::De => "download",
            Language::Ja => "ダウンロード",
            Language::En => "download",
        }
    }

    fn command_downloaded_bytes(self) -> &'static str {
        match self.language {
            Language::ZhCn => "downloaded_bytes",
            Language::ZhTw => "downloaded_bytes",
            Language::Fr => "octets_telecharges",
            Language::De => "downloaded_bytes",
            Language::Ja => "downloaded_bytes",
            Language::En => "downloaded_bytes",
        }
    }

    fn command_apply_upgrade(self) -> &'static str {
        match self.language {
            Language::ZhCn => "apply_upgrade",
            Language::ZhTw => "apply_upgrade",
            Language::Fr => "appliquer_upgrade",
            Language::De => "apply_upgrade",
            Language::Ja => "apply_upgrade",
            Language::En => "apply_upgrade",
        }
    }

    fn banner_slogan(self) -> &'static str {
        match self.language {
            Language::ZhCn => "版本探测 · 资源路由 · 安全替换 · 失败回退",
            Language::ZhTw => "版本探測 · 資源路由 · 安全替換 · 失敗回退",
            Language::Fr => {
                "Detection version · Routage assets · Remplacement sur · Repli securise"
            }
            Language::De => "Versionspruefung · Asset-Routing · Sicheres Ersetzen · Rollback",
            Language::Ja => "バージョン検知・資産ルーティング・安全置換・ロールバック",
            Language::En => "Version Probe · Asset Routing · Safe Replace · Rollback",
        }
    }

    fn progress_checking_release(self) -> &'static str {
        match self.language {
            Language::ZhCn => "正在查询 Release",
            Language::ZhTw => "正在查詢 Release",
            Language::Fr => "verification des releases",
            Language::De => "Release-Abfrage laeuft",
            Language::Ja => "release を確認中",
            Language::En => "checking releases",
        }
    }

    fn progress_downloading(self) -> &'static str {
        match self.language {
            Language::ZhCn => "正在下载升级资源",
            Language::ZhTw => "正在下載升級資源",
            Language::Fr => "telechargement de l'asset de mise a niveau",
            Language::De => "Upgrade-Asset wird heruntergeladen",
            Language::Ja => "アップグレード資産をダウンロード中",
            Language::En => "downloading upgrade asset",
        }
    }
}

#[derive(Debug, Clone)]
pub struct UpgradeCommandOptions {
    pub check_only: bool,
    pub output: Option<PathBuf>,
    pub allow_prerelease: bool,
}

#[derive(Debug, Deserialize)]
struct GithubRelease {
    #[serde(default)]
    tag_name: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    html_url: String,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    assets: Vec<GithubReleaseAsset>,
}

#[derive(Debug, Deserialize)]
struct GithubReleaseAsset {
    #[serde(default)]
    name: String,
    #[serde(default)]
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

#[derive(Debug, Clone)]
struct NormalizedVersion {
    raw: String,
    numeric_parts: Vec<u64>,
    prerelease: bool,
}

#[derive(Debug, Clone, Copy)]
struct PlatformProfile {
    os: OsType,
    os_tokens: &'static [&'static str],
    incompatible_os_tokens: &'static [&'static str],
    arch_tokens: &'static [&'static str],
    incompatible_arch_tokens: &'static [&'static str],
}

struct UpgradeSpinner {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl UpgradeSpinner {
    fn start(message: &str, colorful: bool) -> Self {
        if !io::stdout().is_terminal() {
            return Self {
                stop: Arc::new(AtomicBool::new(true)),
                handle: None,
            };
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stop_cloned = stop.clone();
        let message = message.to_string();
        let use_color = render::resolve_colorful_enabled(colorful);
        let handle = thread::spawn(move || {
            let frames = ["◐", "◓", "◑", "◒"];
            let mut idx = 0usize;
            while !stop_cloned.load(AtomicOrdering::SeqCst) {
                let frame = frames[idx % frames.len()];
                let line = if use_color {
                    format!(
                        "{} {} {}",
                        "✦".bright_cyan().bold(),
                        message.bright_white().bold(),
                        frame.bright_magenta().bold()
                    )
                } else {
                    format!("* {message} {frame}")
                };
                print!("\r{line}");
                let _ = io::stdout().flush();
                idx = idx.wrapping_add(1);
                thread::sleep(Duration::from_millis(80));
            }
            clear_spinner_line();
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, AtomicOrdering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for UpgradeSpinner {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn run_upgrade_command(
    assets_dir: &Path,
    colorful: bool,
    options: UpgradeCommandOptions,
) -> Result<ActionOutcome, AppError> {
    let locale = UpgradeLocale::new();
    let started = Instant::now();
    let mut key_metrics = Vec::<String>::new();
    let mut risks = Vec::<String>::new();
    let mut decision = Vec::<String>::new();
    let mut commands = Vec::<String>::new();

    let mut exit_code = ExitCode::Success;
    let mut status = i18n::status_success().to_string();
    let local_version_raw = env!("CARGO_PKG_VERSION").to_string();
    key_metrics.push(metric_entry(
        locale.metric_local_version(),
        format!("`{}`", local_version_raw).as_str(),
    ));
    key_metrics.push(metric_entry(
        locale.metric_repo(),
        format!("`{}`", RELEASES_WEB_URL).as_str(),
    ));
    commands.push(format!("GET {}", RELEASES_API_URL));

    let local_version = normalize_version(local_version_raw.as_str());
    if local_version.is_none() {
        risks.push(bullet(
            locale
                .risk_local_version_invalid(local_version_raw.as_str())
                .as_str(),
        ));
    }

    let client = build_http_client(HTTP_REQUEST_TIMEOUT_SECS)?;
    let mut release_check_spinner =
        UpgradeSpinner::start(locale.progress_checking_release(), colorful);
    let releases = match fetch_releases(&client) {
        Ok(list) => list,
        Err(err) => {
            release_check_spinner.stop();
            status = i18n::status_failed().to_string();
            exit_code = ExitCode::CommandFailure;
            risks.push(bullet(locale.risk_fetch_failed(err.as_str()).as_str()));
            decision.push(bullet(locale.decision_keep_current()));
            return render_upgrade_outcome(
                assets_dir,
                colorful,
                status,
                locale,
                key_metrics,
                risks,
                decision,
                commands,
                started.elapsed().as_millis(),
                exit_code,
            );
        }
    };
    release_check_spinner.stop();
    commands.push(format!(
        "{}={}",
        locale.command_release_candidates(),
        releases.len()
    ));

    let latest = pick_latest_release(releases.as_slice(), options.allow_prerelease);
    let Some(release) = latest else {
        decision.push(bullet(locale.decision_no_eligible_release()));
        key_metrics.push(metric_entry(
            locale.metric_latest_release(),
            format!("`{}`", locale.na_value()).as_str(),
        ));
        if !options.allow_prerelease {
            risks.push(bullet(locale.risk_prerelease_excluded()));
        }
        return render_upgrade_outcome(
            assets_dir,
            colorful,
            status,
            locale,
            key_metrics,
            risks,
            decision,
            commands,
            started.elapsed().as_millis(),
            exit_code,
        );
    };

    let release_title = if release.name.trim().is_empty() {
        release.tag_name.clone()
    } else {
        release.name.clone()
    };
    let release_link = if release.html_url.trim().is_empty() {
        RELEASES_WEB_URL.to_string()
    } else {
        release.html_url.clone()
    };
    key_metrics.push(metric_entry(
        locale.metric_latest_release(),
        format!("`{}`", release.tag_name).as_str(),
    ));
    key_metrics.push(metric_entry(
        locale.metric_release_title(),
        format!("`{}`", release_title).as_str(),
    ));
    key_metrics.push(metric_entry(
        locale.metric_release_page(),
        format!("`{}`", release_link).as_str(),
    ));
    if let Some(published) = release.published_at.as_deref() {
        key_metrics.push(metric_entry(locale.metric_published_at(), published));
    }

    let Some(latest_version) = normalize_version(release.tag_name.as_str()) else {
        status = i18n::status_failed().to_string();
        exit_code = ExitCode::CommandFailure;
        risks.push(bullet(
            locale
                .risk_latest_version_invalid(release.tag_name.as_str())
                .as_str(),
        ));
        decision.push(bullet(locale.decision_skip_auto_upgrade()));
        return render_upgrade_outcome(
            assets_dir,
            colorful,
            status,
            locale,
            key_metrics,
            risks,
            decision,
            commands,
            started.elapsed().as_millis(),
            exit_code,
        );
    };

    let Some(local_version) = local_version else {
        status = i18n::status_failed().to_string();
        exit_code = ExitCode::CommandFailure;
        decision.push(bullet(locale.decision_skip_auto_upgrade()));
        return render_upgrade_outcome(
            assets_dir,
            colorful,
            status,
            locale,
            key_metrics,
            risks,
            decision,
            commands,
            started.elapsed().as_millis(),
            exit_code,
        );
    };

    match compare_version(&latest_version, &local_version) {
        Ordering::Less | Ordering::Equal => {
            decision.push(bullet(
                locale
                    .decision_already_up_to_date(
                        local_version.raw.as_str(),
                        latest_version.raw.as_str(),
                    )
                    .as_str(),
            ));
            return render_upgrade_outcome(
                assets_dir,
                colorful,
                status,
                locale,
                key_metrics,
                risks,
                decision,
                commands,
                started.elapsed().as_millis(),
                exit_code,
            );
        }
        Ordering::Greater => {
            decision.push(bullet(
                locale
                    .decision_upgrade_available(
                        local_version.raw.as_str(),
                        latest_version.raw.as_str(),
                    )
                    .as_str(),
            ));
        }
    }

    let platform = platform_profile(current_os(), std::env::consts::ARCH);
    key_metrics.push(metric_entry(
        locale.metric_platform(),
        format!("`{}-{}`", os_label(platform.os), std::env::consts::ARCH).as_str(),
    ));

    let Some(asset) = pick_best_asset(release.assets.as_slice(), platform) else {
        status = i18n::status_failed().to_string();
        exit_code = ExitCode::CommandFailure;
        risks.push(bullet(locale.risk_no_compatible_asset()));
        if !release.assets.is_empty() {
            commands.push(format!(
                "{}={}",
                locale.command_available_assets(),
                release
                    .assets
                    .iter()
                    .map(|item| item.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        decision.push(bullet(locale.decision_manual_download_fallback()));
        return render_upgrade_outcome(
            assets_dir,
            colorful,
            status,
            locale,
            key_metrics,
            risks,
            decision,
            commands,
            started.elapsed().as_millis(),
            exit_code,
        );
    };

    key_metrics.push(metric_entry(
        locale.metric_selected_asset(),
        format!("`{}`", asset.name).as_str(),
    ));
    key_metrics.push(metric_entry(
        locale.metric_declared_asset_size(),
        format!("`{}`", i18n::human_bytes(asset.size as u128)).as_str(),
    ));

    if options.check_only {
        decision.push(bullet(locale.decision_check_only()));
        return render_upgrade_outcome(
            assets_dir,
            colorful,
            status,
            locale,
            key_metrics,
            risks,
            decision,
            commands,
            started.elapsed().as_millis(),
            exit_code,
        );
    }

    let current_exe = std::env::current_exe()
        .map_err(|err| AppError::Runtime(format!("failed to locate current executable: {err}")))?;
    let download_target =
        resolve_download_path(options.output.as_ref(), &current_exe, asset.name.as_str())?;
    key_metrics.push(metric_entry(
        locale.metric_download_target(),
        format!("`{}`", download_target.display()).as_str(),
    ));
    commands.push(format!(
        "{}: {}",
        locale.command_download(),
        asset.browser_download_url
    ));

    let download_client = build_http_client(HTTP_DOWNLOAD_TIMEOUT_SECS)?;
    let mut download_spinner = UpgradeSpinner::start(locale.progress_downloading(), colorful);
    let downloaded_bytes = match download_asset(&download_client, asset, download_target.as_path())
    {
        Ok(bytes) => bytes,
        Err(err) => {
            download_spinner.stop();
            status = i18n::status_failed().to_string();
            exit_code = ExitCode::CommandFailure;
            risks.push(bullet(locale.risk_download_failed(err.as_str()).as_str()));
            decision.push(bullet(locale.decision_keep_binary_unchanged()));
            return render_upgrade_outcome(
                assets_dir,
                colorful,
                status,
                locale,
                key_metrics,
                risks,
                decision,
                commands,
                started.elapsed().as_millis(),
                exit_code,
            );
        }
    };
    download_spinner.stop();
    commands.push(format!(
        "{}={}",
        locale.command_downloaded_bytes(),
        i18n::human_count_u128(downloaded_bytes)
    ));

    key_metrics.push(metric_entry(
        locale.metric_downloaded_size(),
        format!("`{}`", i18n::human_bytes(downloaded_bytes)).as_str(),
    ));

    if asset_is_archive(asset.name.as_str()) {
        risks.push(bullet(locale.risk_archive_skip_auto_apply()));
        decision.push(bullet(
            locale
                .decision_manual_extract_replace(download_target.as_path(), current_exe.as_path())
                .as_str(),
        ));
        return render_upgrade_outcome(
            assets_dir,
            colorful,
            status,
            locale,
            key_metrics,
            risks,
            decision,
            commands,
            started.elapsed().as_millis(),
            exit_code,
        );
    }

    if options.output.is_some() {
        decision.push(bullet(locale.decision_custom_output_skip_auto_apply()));
        decision.push(bullet(
            locale
                .decision_manual_replace(download_target.as_path())
                .as_str(),
        ));
        return render_upgrade_outcome(
            assets_dir,
            colorful,
            status,
            locale,
            key_metrics,
            risks,
            decision,
            commands,
            started.elapsed().as_millis(),
            exit_code,
        );
    }

    ensure_executable_permission(download_target.as_path())?;
    match apply_upgrade(download_target.as_path(), current_exe.as_path()) {
        Ok(()) => {
            commands.push(format!("{}=success", locale.command_apply_upgrade()));
            decision.push(bullet(
                locale
                    .decision_upgrade_applied(latest_version.raw.as_str())
                    .as_str(),
            ));
        }
        Err(err) => {
            status = i18n::status_failed().to_string();
            exit_code = ExitCode::CommandFailure;
            risks.push(bullet(locale.risk_auto_apply_failed(err.as_str()).as_str()));
            decision.push(bullet(
                locale
                    .decision_manual_replace(download_target.as_path())
                    .as_str(),
            ));
        }
    }

    render_upgrade_outcome(
        assets_dir,
        colorful,
        status,
        locale,
        key_metrics,
        risks,
        decision,
        commands,
        started.elapsed().as_millis(),
        exit_code,
    )
}

fn render_upgrade_outcome(
    _assets_dir: &Path,
    colorful: bool,
    status: String,
    locale: UpgradeLocale,
    key_metrics: Vec<String>,
    risks: Vec<String>,
    decision: Vec<String>,
    commands: Vec<String>,
    elapsed_ms: u128,
    exit_code: ExitCode,
) -> Result<ActionOutcome, AppError> {
    let rendered = render_upgrade_dashboard(
        colorful,
        locale,
        status.as_str(),
        i18n::human_duration_ms(elapsed_ms).as_str(),
        key_metrics.as_slice(),
        risks.as_slice(),
        decision.as_slice(),
        commands.as_slice(),
    );
    Ok(ActionOutcome {
        rendered,
        exit_code,
    })
}

fn render_upgrade_dashboard(
    colorful: bool,
    locale: UpgradeLocale,
    status: &str,
    elapsed: &str,
    key_metrics: &[String],
    risks: &[String],
    decisions: &[String],
    commands: &[String],
) -> String {
    let use_color = render::resolve_colorful_enabled(colorful);
    let width = terminal_render_width();
    let inner_width = width.saturating_sub(2);
    let mut out = Vec::<String>::new();
    let pulse = status_pulse_icon();
    let banner = format!("{}  {}", pulse, locale.title());

    out.push(top_border(inner_width));
    out.push(box_line(inner_width, banner.as_str()));
    out.push(box_line(inner_width, locale.banner_slogan()));
    out.push(mid_border(inner_width));
    out.push(box_line(
        inner_width,
        format!(
            "{} : upgrade   {} : {status}",
            locale.action_label(),
            locale.status_label()
        )
        .as_str(),
    ));
    out.push(box_line(
        inner_width,
        format!("{}: {elapsed}", locale.elapsed_label()).as_str(),
    ));
    out.push(mid_border(inner_width));
    append_section(
        &mut out,
        inner_width,
        locale.section_version_radar(),
        key_metrics,
        locale,
    );
    out.push(mid_border(inner_width));
    append_section(&mut out, inner_width, locale.section_risk(), risks, locale);
    out.push(mid_border(inner_width));
    append_section(
        &mut out,
        inner_width,
        locale.section_decision(),
        decisions,
        locale,
    );
    out.push(mid_border(inner_width));
    append_section(
        &mut out,
        inner_width,
        locale.section_pipeline(),
        commands,
        locale,
    );
    out.push(bottom_border(inner_width));
    style_dashboard_lines(out, use_color, status, locale).join("\n")
}

fn append_section(
    out: &mut Vec<String>,
    inner_width: usize,
    title: &str,
    items: &[String],
    locale: UpgradeLocale,
) {
    out.push(box_line(inner_width, format!("[ {title} ]").as_str()));
    if items.is_empty() {
        out.push(box_line(
            inner_width,
            format!("  • {}", locale.item_none()).as_str(),
        ));
        return;
    }
    for item in items {
        let normalized = normalize_item(item);
        let wrapped =
            wrap_text_by_display_width(normalized.as_str(), inner_width.saturating_sub(4));
        for (idx, line) in wrapped.iter().enumerate() {
            let prefix = if idx == 0 { "  • " } else { "    " };
            out.push(box_line(inner_width, format!("{prefix}{line}").as_str()));
        }
    }
}

fn normalize_item(raw: &str) -> String {
    raw.trim().trim_start_matches("- ").to_string()
}

fn metric_entry(label: &str, value: &str) -> String {
    bullet(format!("{label}: {value}").as_str())
}

fn bullet(text: &str) -> String {
    format!("- {text}")
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

fn terminal_render_width() -> usize {
    match crossterm::terminal::size() {
        Ok((width, _)) => {
            let safe = (width as usize).max(72);
            safe.min(180)
        }
        Err(_) => 120,
    }
}

fn status_pulse_icon() -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let idx = (timestamp_millis() % FRAMES.len() as u128) as usize;
    FRAMES[idx]
}

fn style_dashboard_lines(
    lines: Vec<String>,
    use_color: bool,
    status: &str,
    locale: UpgradeLocale,
) -> Vec<String> {
    if !use_color {
        return lines;
    }
    let mut styled = Vec::with_capacity(lines.len());
    for (idx, line) in lines.iter().enumerate() {
        if line.starts_with('┏') || line.starts_with('┣') || line.starts_with('┗') {
            styled.push(line.bright_blue().bold().to_string());
            continue;
        }
        let Some((left, body, right)) = split_box_line(line.as_str()) else {
            styled.push(line.to_string());
            continue;
        };
        if line.contains("[ ") && line.contains(" ]") {
            styled.push(format!(
                "{}{}{}",
                left.bright_blue().bold(),
                body.bright_magenta().bold(),
                right.bright_blue().bold()
            ));
            continue;
        }
        if idx == 1 {
            styled.push(format!(
                "{}{}{}",
                left.bright_blue().bold(),
                body.bright_cyan().bold(),
                right.bright_blue().bold()
            ));
            continue;
        }
        if idx == 2 {
            styled.push(format!(
                "{}{}{}",
                left.bright_blue().bold(),
                body.bright_yellow().bold(),
                right.bright_blue().bold()
            ));
            continue;
        }
        if line.contains("•") {
            let mut body_styled = body.bright_white().to_string();
            body_styled = body_styled.replace("•", &"•".bright_green().bold().to_string());
            if body.contains("http://") || body.contains("https://") {
                body_styled = colorize_urls(body_styled.as_str());
            }
            styled.push(format!(
                "{}{}{}",
                left.bright_blue().bold(),
                body_styled,
                right.bright_blue().bold()
            ));
            continue;
        }
        let status_key = format!("{} :", locale.status_label());
        if line.contains(status_key.as_str()) {
            let mut body_styled = body.bright_white().to_string();
            if status == i18n::status_success() {
                body_styled =
                    body_styled.replace(status, &status.bright_green().bold().to_string());
            } else if status == i18n::status_failed() {
                body_styled = body_styled.replace(status, &status.bright_red().bold().to_string());
            }
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
    styled
}

fn split_box_line(line: &str) -> Option<(&str, &str, &str)> {
    let body = line.strip_prefix('┃')?.strip_suffix('┃')?;
    Some(("┃", body, "┃"))
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

fn clear_spinner_line() {
    if !io::stdout().is_terminal() {
        return;
    }
    print!("\r\x1b[2K");
    let _ = io::stdout().flush();
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

fn build_http_client(timeout_secs: u64) -> Result<Client, AppError> {
    ensure_rustls_crypto_provider();
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static(UPGRADE_USER_AGENT));
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("application/vnd.github+json"),
    );
    Client::builder()
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(HTTP_CONNECT_TIMEOUT_SECS))
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|err| AppError::Command(format!("failed to build upgrade http client: {err}")))
}

fn fetch_releases(client: &Client) -> Result<Vec<GithubRelease>, String> {
    let response = client
        .get(RELEASES_API_URL)
        .send()
        .map_err(|err| format!("request failed: {err}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_else(|_| String::new());
        return Err(format!(
            "http status={} body={} ",
            status.as_u16(),
            shorten(body.as_str(), 280)
        ));
    }
    response
        .json::<Vec<GithubRelease>>()
        .map_err(|err| format!("invalid release json: {err}"))
}

fn pick_latest_release<'a>(
    releases: &'a [GithubRelease],
    allow_prerelease: bool,
) -> Option<&'a GithubRelease> {
    releases
        .iter()
        .find(|release| !release.draft && (allow_prerelease || !release.prerelease))
}

fn pick_best_asset<'a>(
    assets: &'a [GithubReleaseAsset],
    platform: PlatformProfile,
) -> Option<&'a GithubReleaseAsset> {
    let mut best: Option<(&GithubReleaseAsset, i32)> = None;
    for asset in assets {
        if asset.name.trim().is_empty() || asset.browser_download_url.trim().is_empty() {
            continue;
        }
        if looks_like_checksum_asset(asset.name.as_str()) {
            continue;
        }
        let score = score_asset(asset.name.as_str(), platform);
        if let Some((_, best_score)) = best {
            if score > best_score {
                best = Some((asset, score));
            }
            continue;
        }
        best = Some((asset, score));
    }
    best.map(|(asset, _)| asset)
}

fn score_asset(name: &str, platform: PlatformProfile) -> i32 {
    let normalized = name.to_ascii_lowercase();
    let mut score: i32 = 0;

    if normalized.contains("machineclaw") {
        score += 6;
    }
    if platform
        .os_tokens
        .iter()
        .any(|token| normalized.contains(token))
    {
        score += 30;
    }
    if platform
        .incompatible_os_tokens
        .iter()
        .any(|token| normalized.contains(token))
    {
        score -= 50;
    }
    if platform
        .arch_tokens
        .iter()
        .any(|token| normalized.contains(token))
    {
        score += 30;
    }
    if platform
        .incompatible_arch_tokens
        .iter()
        .any(|token| normalized.contains(token))
    {
        score -= 35;
    }
    if normalized.ends_with(".exe") {
        score += if matches!(platform.os, OsType::Windows) {
            8
        } else {
            -16
        };
    }
    if asset_is_archive(name) {
        score += 3;
    } else {
        score += 5;
    }
    score
}

fn resolve_download_path(
    output: Option<&PathBuf>,
    current_exe: &Path,
    asset_name: &str,
) -> Result<PathBuf, AppError> {
    if let Some(path) = output {
        if path.exists() && path.is_dir() {
            return Ok(path.join(asset_name));
        }
        return Ok(path.clone());
    }

    let parent = current_exe.parent().ok_or_else(|| {
        AppError::Runtime("failed to resolve current executable directory".to_string())
    })?;
    Ok(parent.join(asset_name))
}

fn download_asset(
    client: &Client,
    asset: &GithubReleaseAsset,
    destination: &Path,
) -> Result<u128, String> {
    let parent = destination
        .parent()
        .ok_or_else(|| "download destination has no parent directory".to_string())?;
    fs::create_dir_all(parent).map_err(|err| {
        format!(
            "failed to create download directory {}: {err}",
            parent.display()
        )
    })?;

    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("machineclaw-upgrade");
    let temp_path = parent.join(format!(
        ".{}.{}.{}.tmp",
        file_name,
        std::process::id(),
        timestamp_millis()
    ));

    let mut response = client
        .get(asset.browser_download_url.as_str())
        .send()
        .map_err(|err| format!("request failed: {err}"))?;
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().unwrap_or_else(|_| String::new());
        return Err(format!(
            "http status={} body={}",
            status,
            shorten(body.as_str(), 280)
        ));
    }

    let mut output = fs::File::create(&temp_path)
        .map_err(|err| format!("failed to create temp file {}: {err}", temp_path.display()))?;
    let copied = io::copy(&mut response, &mut output)
        .map_err(|err| format!("failed to write downloaded bytes: {err}"))?
        as u128;
    output
        .sync_all()
        .map_err(|err| format!("failed to flush temp file {}: {err}", temp_path.display()))?;

    if copied == 0 {
        let _ = fs::remove_file(&temp_path);
        return Err("downloaded file is empty".to_string());
    }

    move_file_with_fallback(temp_path.as_path(), destination)
        .map_err(|err| format!("failed to move downloaded file into place: {err}"))?;

    Ok(copied)
}

fn apply_upgrade(downloaded: &Path, current_exe: &Path) -> Result<(), String> {
    let current_name = current_exe
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("MachineClaw");
    let backup = current_exe.with_file_name(format!(
        ".{}.upgrade-backup.{}",
        current_name,
        timestamp_millis()
    ));

    move_file_with_fallback(current_exe, &backup)
        .map_err(|err| format!("failed to backup current executable: {err}"))?;
    if let Err(err) = move_file_with_fallback(downloaded, current_exe) {
        let _ = move_file_with_fallback(&backup, current_exe);
        return Err(format!("failed to replace executable: {err}"));
    }
    let _ = fs::remove_file(backup);
    Ok(())
}

fn move_file_with_fallback(source: &Path, target: &Path) -> io::Result<()> {
    if source == target {
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Err(rename_err) = fs::rename(source, target) {
        match fs::copy(source, target) {
            Ok(_) => {
                let _ = fs::remove_file(source);
                return Ok(());
            }
            Err(copy_err) => {
                return Err(io::Error::other(format!(
                    "rename failed: {rename_err}; copy failed: {copy_err}"
                )));
            }
        }
    }
    Ok(())
}

fn ensure_executable_permission(path: &Path) -> Result<(), AppError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let metadata = fs::metadata(path).map_err(|err| {
            AppError::Command(format!("failed to read {}: {err}", path.display()))
        })?;
        let mut permissions = metadata.permissions();
        let mode = permissions.mode();
        if mode & 0o111 == 0 {
            permissions.set_mode(mode | 0o755);
            fs::set_permissions(path, permissions).map_err(|err| {
                AppError::Command(format!(
                    "failed to set executable permission for {}: {err}",
                    path.display()
                ))
            })?;
        }
    }
    Ok(())
}

fn looks_like_checksum_asset(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.ends_with(".sha256")
        || normalized.ends_with(".sha512")
        || normalized.ends_with(".sig")
        || normalized.contains("checksum")
}

fn asset_is_archive(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.ends_with(".zip")
        || normalized.ends_with(".tar")
        || normalized.ends_with(".tar.gz")
        || normalized.ends_with(".tgz")
        || normalized.ends_with(".tar.xz")
        || normalized.ends_with(".txz")
}

fn normalize_version(raw: &str) -> Option<NormalizedVersion> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }
    let first_digit = text.chars().position(|ch| ch.is_ascii_digit())?;
    let trimmed = &text[first_digit..];
    let mut numeric = String::new();
    let mut rest = String::new();
    for (idx, ch) in trimmed.char_indices() {
        if ch.is_ascii_digit() || ch == '.' {
            numeric.push(ch);
            continue;
        }
        rest = trimmed[idx..].to_string();
        break;
    }
    if numeric.is_empty() {
        return None;
    }
    let parts = numeric
        .split('.')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.parse::<u64>().ok())
        .collect::<Option<Vec<u64>>>()?;
    if parts.is_empty() {
        return None;
    }
    let prerelease = rest.trim_start().starts_with('-');
    Some(NormalizedVersion {
        raw: text.to_string(),
        numeric_parts: parts,
        prerelease,
    })
}

fn compare_version(left: &NormalizedVersion, right: &NormalizedVersion) -> Ordering {
    let max_len = left.numeric_parts.len().max(right.numeric_parts.len());
    for index in 0..max_len {
        let lv = *left.numeric_parts.get(index).unwrap_or(&0);
        let rv = *right.numeric_parts.get(index).unwrap_or(&0);
        match lv.cmp(&rv) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
    }
    match (left.prerelease, right.prerelease) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

fn platform_profile(os: OsType, arch: &str) -> PlatformProfile {
    let normalized_arch = arch.to_ascii_lowercase();
    let (arch_tokens, incompatible_arch_tokens): (&[&str], &[&str]) = match normalized_arch.as_str()
    {
        "x86_64" | "amd64" => (&["x86_64", "amd64", "x64"], &["arm64", "aarch64", "armv7"]),
        "aarch64" | "arm64" => (&["aarch64", "arm64"], &["x86_64", "amd64", "x64", "armv7"]),
        "arm" | "armv7" => (
            &["armv7", "arm"],
            &["x86_64", "amd64", "x64", "arm64", "aarch64"],
        ),
        _ => (&["unknown"], &[]),
    };
    match os {
        OsType::Windows => PlatformProfile {
            os,
            os_tokens: &["windows", "win"],
            incompatible_os_tokens: &["linux", "darwin", "macos", "osx"],
            arch_tokens,
            incompatible_arch_tokens,
        },
        OsType::MacOS => PlatformProfile {
            os,
            os_tokens: &["darwin", "macos", "osx"],
            incompatible_os_tokens: &["windows", "linux"],
            arch_tokens,
            incompatible_arch_tokens,
        },
        OsType::Linux | OsType::Other => PlatformProfile {
            os,
            os_tokens: &["linux", "gnu", "musl"],
            incompatible_os_tokens: &["windows", "darwin", "macos", "osx"],
            arch_tokens,
            incompatible_arch_tokens,
        },
    }
}

fn os_label(os: OsType) -> &'static str {
    match os {
        OsType::Windows => "windows",
        OsType::MacOS => "macos",
        OsType::Linux => "linux",
        OsType::Other => "other",
    }
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn shorten(raw: &str, max_chars: usize) -> String {
    let mut it = raw.chars();
    let shortened: String = it.by_ref().take(max_chars).collect();
    if it.next().is_some() {
        format!("{shortened}...")
    } else {
        shortened
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NormalizedVersion, OsType, compare_version, normalize_version, pick_best_asset,
        platform_profile, resolve_download_path,
    };
    use std::cmp::Ordering;
    use std::path::{Path, PathBuf};

    #[test]
    fn normalize_version_supports_prefixed_tag() {
        let parsed = normalize_version("release-v1.2.3-beta.1").expect("should parse version");
        assert_eq!(parsed.numeric_parts, vec![1, 2, 3]);
        assert!(parsed.prerelease);
    }

    #[test]
    fn compare_version_prefers_stable_when_numeric_equal() {
        let stable = NormalizedVersion {
            raw: "1.2.3".to_string(),
            numeric_parts: vec![1, 2, 3],
            prerelease: false,
        };
        let pre = NormalizedVersion {
            raw: "1.2.3-rc1".to_string(),
            numeric_parts: vec![1, 2, 3],
            prerelease: true,
        };
        assert_eq!(compare_version(&stable, &pre), Ordering::Greater);
    }

    #[test]
    fn resolve_download_path_supports_output_directory() {
        let path = resolve_download_path(
            Some(&PathBuf::from(".")),
            Path::new("/usr/local/bin/MachineClaw"),
            "MachineClaw-linux-x86_64",
        )
        .expect("should resolve path");
        assert_eq!(path, PathBuf::from("./MachineClaw-linux-x86_64"));
    }

    #[test]
    fn pick_best_asset_prefers_platform_match() {
        let platform = platform_profile(OsType::Linux, "x86_64");
        let assets = vec![
            super::GithubReleaseAsset {
                name: "MachineClaw-darwin-arm64".to_string(),
                browser_download_url: "https://example.com/darwin".to_string(),
                size: 100,
            },
            super::GithubReleaseAsset {
                name: "MachineClaw-linux-x86_64".to_string(),
                browser_download_url: "https://example.com/linux".to_string(),
                size: 100,
            },
        ];
        let selected = pick_best_asset(assets.as_slice(), platform).expect("selected asset");
        assert_eq!(selected.name, "MachineClaw-linux-x86_64");
    }

    #[test]
    fn keep_numeric_order_compare() {
        let latest = normalize_version("v1.10.0").expect("latest");
        let local = normalize_version("v1.9.9").expect("local");
        assert_eq!(compare_version(&latest, &local), Ordering::Greater);
    }
}
