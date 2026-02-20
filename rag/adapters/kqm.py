from pathlib import Path
import shutil, logging
from git import Repo, InvalidGitRepositoryError, GitCommandError

log = logging.getLogger(__name__)

SKIP_NAMES = {
    "readme.md",
    "changelog.md",
    "contributing.md",
    "license.md",
    "_sidebar.md",
    "_meta.md",
}

def ensure_repo(repo_path: str, repo_url: str):
    p = Path(repo_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        Repo.clone_from(repo_url, repo_path, depth=1)
        return

    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        shutil.rmtree(repo_path, ignore_errors=True)
        Repo.clone_from(repo_url, repo_path, depth=1)
        log.exception(f"Cloning repository for {repo_url}")
        return
        
    origin = repo.remotes.origin
    origin.fetch()
    targets = ["origin/HEAD", "origin/master", "origin/main"]
    for t in targets:
        try:
            repo.git.reset("--hard", t)
            break
        except GitCommandError:
            log.exception("Failed reset git HEAD")
            continue

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
    repo_url = REPO_MAP.get(source_cfg["name"])
    if not repo_url:
        raise ValueError(f"Unknown repo for {name}")
    
    repo_path = source_cfg["path"]
    ensure_repo(repo_path, repo_url)

    for md_path in iter_markdown_files(repo_path):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            log.exception(f"Failed to read text for {md_path}, skipping")
            continue

        title = md_path.stem
        rel = md_path.relative_to(repo_path).as_posix()
        url = f"kqm://{name}/{rel}"
        yield url, title, text