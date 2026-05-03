from pathlib import Path
import shutil, logging
from git import Repo, InvalidGitRepositoryError

log = logging.getLogger(__name__)

SKIP_NAMES = {
    "readme.md",
    "changelog.md",
    "contributing.md",
    "license.md",
    "_sidebar.md",
    "_meta.md",
}

def ensure_repo(repo_path: str, repo_url: str) -> tuple[bool, str]:
    p = Path(repo_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        Repo.clone_from(repo_url, repo_path, depth=1)
        repo = Repo(repo_path)
        head = repo.head.commit.hexsha
        log.info("[KQM] Git cloned %s @ %s", repo_url, head[:7])
        return True, head

    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        shutil.rmtree(repo_path, ignore_errors=True)
        Repo.clone_from(repo_url, repo_path, depth=1)
        repo = Repo(repo_path)
        head = repo.head.commit.hexsha
        log.warning("[KQM] Re-cloned invalid repo %s @ %s", repo_url, head[:7])
        return True, head
        
    origin = repo.remotes.origin
    origin.fetch()
    try:
        default_ref = origin.refs.HEAD.reference
    except Exception:
        if "origin/main" in repo.refs:
            default_ref = repo.refs["origin/main"]
        elif "origin/master" in repo.refs:
            default_ref = repo.refs["origin/master"]
        else:
            raise RuntimeError("Cannot determine remote default branch")

    remote_commit = default_ref.commit.hexsha
    local_commit = repo.head.commit.hexsha

    if local_commit != remote_commit:
        log.info(
            "[KQM] Git repo outdated: local=%s remote=%s → updating",
            local_commit[:7],
            remote_commit[:7],
        )
        repo.git.reset("--hard", remote_commit)
        repo.git.clean("-fdx")
        return True, remote_commit
    
    log.info("[KQM] Git repo up to date (%s)", local_commit[:7])
    return False, local_commit

def iter_markdown_files(repo_path: str):
    root = Path(repo_path)
    for path in root.rglob("*.md"):
        name = path.name.lower()
        if any(part in ("node_modules", ".git") for part in path.parts):
            continue
        if name in SKIP_NAMES:
            continue
        yield path

def load_kqm_tcl_docs(source_cfg: dict):
    REPO_MAP = {
        "kqm_tcl": "https://github.com/KQM-git/TCL.git",
        "kqm_news": "https://github.com/KQM-git/GINews.git",
    }

    name = source_cfg["name"]
    repo_url = REPO_MAP.get(name)
    if not repo_url:
        raise ValueError(f"[KQM] Unknown repo for {name}")
    
    repo_path = source_cfg["path"]
    skip_if_unchanged = bool(source_cfg.get("skip_if_unchanged", True))
    force_rescan = bool(source_cfg.get("force_rescan", False))

    changed, head = ensure_repo(repo_path, repo_url)

    if skip_if_unchanged and not changed and not force_rescan:
        log.info("[KQM] %s unchanged (%s). Skipping scan.", name, head[:7])
        return

    log.info("[KQM] %s scanning markdown (head=%s)", name, head[:7])

    for md_path in iter_markdown_files(repo_path):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            log.exception("[KQM] Failed to read %s, skipping", md_path)
            continue

        title = md_path.stem
        rel = md_path.relative_to(repo_path).as_posix()
        url = f"kqm://{name}/{rel}"
        yield url, title, text, None, None