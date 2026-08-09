#!/usr/bin/env bash
# setup.sh
# Non-admin one-shot setup for Genshin RAG pipeline on Linux:
# VS Code user-local tar install, Git via system/Conda, Miniforge Conda,
# Ollama user-local tar install, repo clone/update, conda env create/update,
# Ollama model pulls, optional SPLADE migration and dataset creation.

set -Eeuo pipefail

# -----------------------------
# Config
# -----------------------------
REPO_URL="https://github.com/AFOEK/Genshin-Wiki-RAG-Ollama"
WORK_ROOT="$HOME/Documents"
REPO_PATH="$WORK_ROOT/Genshin-Wiki-RAG-Ollama"
FAISS_CURRENT_DIR="/mnt/ssd/genshin_rag/data/faiss/current"

APP_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/ragstack"
DOWNLOAD_ROOT="${TMPDIR:-/tmp}/rag-stack-downloads"
BIN_DIR="$HOME/.local/bin"

VSCODE_DIR="$APP_ROOT/vscode"
CONDA_DEFAULT_DIR="$HOME/miniforge3"
CONDA_CUSTOM_DIR="$APP_ROOT/miniforge3"
OLLAMA_DIR="$APP_ROOT/ollama"

RUN_SPLADE_MIGRATION=true
RUN_DATASET_CREATION=true
DATASET_LIMIT=8500
DATASET_SEED=40151652

MODELS=(
  "qwen3.5:9b"
  "deepseek-r1:14b"
  "snowflake-arctic-embed2:568m"
  "gemma3:12b"
  "qwen3:8b"
  "llama3.2:3b"
  "all-minilm:latest"
  "qwen3.6:27b"
  "gemma4:12b"
)

mkdir -p "$WORK_ROOT" "$APP_ROOT" "$DOWNLOAD_ROOT" "$BIN_DIR"

# -----------------------------
# Helpers
# -----------------------------
write_step() {
  printf '\n==> %s\n' "$1"
}

warn() {
  printf 'WARNING: %s\n' "$1" >&2
}

die() {
  printf 'ERROR: %s\n' "$1" >&2
  return 1 2>/dev/null || exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_file() {
  local url="$1"
  local out_file="$2"

  printf 'Downloading: %s\n' "$url"

  if have_cmd curl; then
    curl -L --fail --retry 3 --connect-timeout 20 -o "$out_file" "$url"
  elif have_cmd wget; then
    wget -O "$out_file" "$url"
  else
    die "Neither curl nor wget is available. Install one of them first."
  fi
}

add_path_current() {
  local path_to_add="$1"
  [[ -d "$path_to_add" ]] || return 0

  case ":$PATH:" in
    *":$path_to_add:"*) ;;
    *) export PATH="$path_to_add:$PATH" ;;
  esac
}

add_path_bashrc_once() {
  local path_to_add="$1"
  local bashrc="$HOME/.bashrc"
  local marker="# ragstack path: $path_to_add"

  [[ -d "$path_to_add" ]] || return 0
  touch "$bashrc"

  if ! grep -Fq "$marker" "$bashrc"; then
    {
      printf '\n%s\n' "$marker"
      printf 'export PATH="%s:$PATH"\n' "$path_to_add"
    } >> "$bashrc"
    printf 'Added to ~/.bashrc PATH: %s\n' "$path_to_add"
  fi
}

refresh_path() {
  add_path_current "$BIN_DIR"
  add_path_current "$VSCODE_DIR/bin"
  add_path_current "$CONDA_DIR/bin"
  add_path_current "$OLLAMA_DIR/bin"
}

resolve_arch() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'amd64' ;;
    aarch64|arm64) printf 'arm64' ;;
    *) die "Unsupported CPU architecture: $(uname -m)" ;;
  esac
}

resolve_vscode_os() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'linux-x64' ;;
    aarch64|arm64) printf 'linux-arm64' ;;
    *) die "Unsupported CPU architecture for VS Code: $(uname -m)" ;;
  esac
}

resolve_miniforge_arch() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'x86_64' ;;
    aarch64|arm64) printf 'aarch64' ;;
    *) die "Unsupported CPU architecture for Miniforge: $(uname -m)" ;;
  esac
}

# -----------------------------
# Detect Conda directory early
# -----------------------------
if [[ -x "$CONDA_DEFAULT_DIR/bin/conda" ]]; then
  CONDA_DIR="$CONDA_DEFAULT_DIR"
elif [[ -x "$CONDA_CUSTOM_DIR/bin/conda" ]]; then
  CONDA_DIR="$CONDA_CUSTOM_DIR"
else
  CONDA_DIR="$CONDA_CUSTOM_DIR"
fi

refresh_path

# -----------------------------
# Create RAG data / FAISS directory
# -----------------------------
write_step "Creating RAG data directory"

if mkdir -p "$FAISS_CURRENT_DIR"; then
  write_test_file="$FAISS_CURRENT_DIR/.write_test"
  if printf 'ok\n' > "$write_test_file" && rm -f "$write_test_file"; then
    printf 'Created and verified writable directory: %s\n' "$FAISS_CURRENT_DIR"
  else
    die "Directory exists but is not writable: $FAISS_CURRENT_DIR"
  fi
else
  die "Failed to create: $FAISS_CURRENT_DIR. If /mnt/ssd is a protected mount, ask IT/admin to create it or mount it writable."
fi

# -----------------------------
# VS Code user-local install
# -----------------------------
write_step "Installing VS Code user-local if needed"

if have_cmd code; then
  CODE_EXE="$(command -v code)"
  printf 'VS Code already found: %s\n' "$CODE_EXE"
elif [[ -x "$VSCODE_DIR/bin/code" ]]; then
  CODE_EXE="$VSCODE_DIR/bin/code"
  add_path_current "$VSCODE_DIR/bin"
  add_path_bashrc_once "$VSCODE_DIR/bin"
  printf 'VS Code already found: %s\n' "$CODE_EXE"
else
  vscode_os="$(resolve_vscode_os)"
  vscode_archive="$DOWNLOAD_ROOT/vscode.tar.gz"
  vscode_url="https://code.visualstudio.com/sha/download?build=stable&os=$vscode_os"

  rm -rf "$VSCODE_DIR"
  mkdir -p "$VSCODE_DIR"
  download_file "$vscode_url" "$vscode_archive"

  tmp_vscode="$DOWNLOAD_ROOT/vscode-extract"
  rm -rf "$tmp_vscode"
  mkdir -p "$tmp_vscode"
  tar -xzf "$vscode_archive" -C "$tmp_vscode"

  extracted_dir="$(find "$tmp_vscode" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  [[ -n "$extracted_dir" ]] || die "Could not find extracted VS Code directory."

  shopt -s dotglob
  mv "$extracted_dir"/* "$VSCODE_DIR"/
  shopt -u dotglob

  add_path_current "$VSCODE_DIR/bin"
  add_path_bashrc_once "$VSCODE_DIR/bin"
  CODE_EXE="$VSCODE_DIR/bin/code"

  [[ -x "$CODE_EXE" ]] || warn "VS Code installed, but code executable was not found at: $CODE_EXE"
fi

# -----------------------------
# Miniforge / Conda user install
# -----------------------------
write_step "Installing or detecting Miniforge Conda"

if have_cmd conda; then
  CONDA_EXE="$(command -v conda)"
  # Prefer the physical conda root if available.
  CONDA_DIR="$(conda info --base 2>/dev/null || dirname "$(dirname "$CONDA_EXE")")"
  printf 'Conda already found: %s\n' "$CONDA_EXE"
elif [[ -x "$CONDA_DEFAULT_DIR/bin/conda" ]]; then
  CONDA_DIR="$CONDA_DEFAULT_DIR"
  CONDA_EXE="$CONDA_DIR/bin/conda"
  printf 'Conda already found: %s\n' "$CONDA_EXE"
elif [[ -x "$CONDA_CUSTOM_DIR/bin/conda" ]]; then
  CONDA_DIR="$CONDA_CUSTOM_DIR"
  CONDA_EXE="$CONDA_DIR/bin/conda"
  printf 'Conda already found: %s\n' "$CONDA_EXE"
else
  miniforge_arch="$(resolve_miniforge_arch)"
  conda_installer="$DOWNLOAD_ROOT/Miniforge3-Linux-$miniforge_arch.sh"
  conda_url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$miniforge_arch.sh"

  download_file "$conda_url" "$conda_installer"
  bash "$conda_installer" -b -p "$CONDA_CUSTOM_DIR"

  CONDA_DIR="$CONDA_CUSTOM_DIR"
  CONDA_EXE="$CONDA_DIR/bin/conda"

  [[ -x "$CONDA_EXE" ]] || die "Conda installation finished, but conda was not found at: $CONDA_EXE"
fi

add_path_current "$CONDA_DIR/bin"
add_path_bashrc_once "$CONDA_DIR/bin"
refresh_path

printf 'Using Conda directory: %s\n' "$CONDA_DIR"
printf 'Using Conda executable: %s\n' "$CONDA_EXE"

# -----------------------------
# Git install/detect
# -----------------------------
write_step "Installing or detecting Git"

if have_cmd git; then
  GIT_EXE="$(command -v git)"
  printf 'Git already found: %s\n' "$GIT_EXE"
else
  printf 'Git not found system-wide. Installing Git into Conda base environment...\n'
  "$CONDA_EXE" install -n base -c conda-forge git -y
  add_path_current "$CONDA_DIR/bin"

  if have_cmd git; then
    GIT_EXE="$(command -v git)"
  elif [[ -x "$CONDA_DIR/bin/git" ]]; then
    GIT_EXE="$CONDA_DIR/bin/git"
  else
    die "Git installation finished, but git was not found."
  fi

  printf 'Using Git executable: %s\n' "$GIT_EXE"
fi

# -----------------------------
# Ollama user-local install
# -----------------------------
write_step "Installing Ollama user-local if needed"

if have_cmd ollama; then
  OLLAMA_EXE="$(command -v ollama)"
  printf 'Ollama already found: %s\n' "$OLLAMA_EXE"
elif [[ -x "$OLLAMA_DIR/bin/ollama" ]]; then
  OLLAMA_EXE="$OLLAMA_DIR/bin/ollama"
  add_path_current "$OLLAMA_DIR/bin"
  add_path_bashrc_once "$OLLAMA_DIR/bin"
  printf 'Ollama already found: %s\n' "$OLLAMA_EXE"
else
  ollama_arch="$(resolve_arch)"
  ollama_archive="$DOWNLOAD_ROOT/ollama-linux-$ollama_arch.tgz"
  ollama_url="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-$ollama_arch.tgz"

  rm -rf "$OLLAMA_DIR"
  mkdir -p "$OLLAMA_DIR"
  download_file "$ollama_url" "$ollama_archive"
  tar -xzf "$ollama_archive" -C "$OLLAMA_DIR"

  OLLAMA_EXE="$OLLAMA_DIR/bin/ollama"
  [[ -x "$OLLAMA_EXE" ]] || die "Ollama installation finished, but ollama was not found at: $OLLAMA_EXE"

  add_path_current "$OLLAMA_DIR/bin"
  add_path_bashrc_once "$OLLAMA_DIR/bin"
fi

# -----------------------------
# Clone or update repo
# -----------------------------
write_step "Cloning or updating RAG repo"

if [[ -d "$REPO_PATH/.git" ]]; then
  printf 'Repo already exists. Pulling latest changes...\n'
  if ! git -C "$REPO_PATH" pull; then
    warn "git pull failed. Continuing with existing local repo."
  fi
elif [[ -e "$REPO_PATH" ]]; then
  die "Target path already exists but is not a Git repo: $REPO_PATH"
else
  git clone "$REPO_URL" "$REPO_PATH"
fi

# -----------------------------
# Create or update Conda environment
# -----------------------------
write_step "Creating or updating Conda environment"

ENV_FILE=""
if [[ -f "$REPO_PATH/environment.yml" ]]; then
  ENV_FILE="$REPO_PATH/environment.yml"
elif [[ -f "$REPO_PATH/environment.yaml" ]]; then
  ENV_FILE="$REPO_PATH/environment.yaml"
else
  die "No environment.yml or environment.yaml found in: $REPO_PATH"
fi

ENV_NAME="$(grep -E '^[[:space:]]*name[[:space:]]*:' "$ENV_FILE" | head -n 1 | sed -E 's/^[[:space:]]*name[[:space:]]*:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^['\''\"]//; s/['\''\"]$//')"

if [[ -n "$ENV_NAME" ]]; then
  if "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    printf "Conda env '%s' already exists. Updating...\n" "$ENV_NAME"
    "$CONDA_EXE" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
  else
    printf "Creating Conda env '%s'...\n" "$ENV_NAME"
    "$CONDA_EXE" env create -f "$ENV_FILE"
  fi
else
  printf 'No env name found in environment file. Running conda env create directly...\n'
  "$CONDA_EXE" env create -f "$ENV_FILE"
fi

# -----------------------------
# Initialize Conda for Bash
# -----------------------------
write_step "Initializing Conda for Bash"

if "$CONDA_EXE" init bash; then
  printf "Conda Bash initialization done. New terminals can use 'conda activate'.\n"
else
  warn "conda init bash failed. Continuing because this script can still use the Conda hook directly."
fi

# Load Conda activation support for this script/session.
CONDA_HOOK="$CONDA_DIR/etc/profile.d/conda.sh"
if [[ -f "$CONDA_HOOK" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_HOOK"
else
  eval "$("$CONDA_EXE" shell.bash hook)"
fi

# -----------------------------
# Start Ollama server
# -----------------------------
write_step "Starting Ollama server"

test_ollama_server() {
  if have_cmd curl; then
    curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1
  elif have_cmd wget; then
    wget -q -O /dev/null "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1
  else
    return 1
  fi
}

if test_ollama_server; then
  printf 'Ollama server already running.\n'
else
  printf 'Starting Ollama server in background...\n'
  nohup "$OLLAMA_EXE" serve > "$APP_ROOT/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  printf 'Ollama PID: %s\n' "$OLLAMA_PID"
  printf 'Ollama log: %s\n' "$APP_ROOT/ollama.log"

  ready=false
  for _ in $(seq 1 30); do
    sleep 2
    if test_ollama_server; then
      ready=true
      break
    fi
  done

  [[ "$ready" == true ]] || die "Ollama server did not become ready at http://127.0.0.1:11434. Check $APP_ROOT/ollama.log"
fi

# -----------------------------
# Pull Ollama models
# -----------------------------
write_step "Pulling Ollama models"

get_local_ollama_models() {
  "$OLLAMA_EXE" list 2>/dev/null | awk 'NR > 1 {print $1}' || true
}

mapfile -t LOCAL_MODELS < <(get_local_ollama_models)

model_exists_locally() {
  local wanted="$1"
  local existing
  for existing in "${LOCAL_MODELS[@]:-}"; do
    [[ "$existing" == "$wanted" ]] && return 0
  done
  return 1
}

for model in "${MODELS[@]}"; do
  printf '\nChecking model: %s\n' "$model"

  if model_exists_locally "$model"; then
    printf 'Model already exists: %s\n' "$model"
    continue
  fi

  printf 'Pulling model: %s\n' "$model"
  if "$OLLAMA_EXE" pull "$model"; then
    printf 'Pulled successfully: %s\n' "$model"
    mapfile -t LOCAL_MODELS < <(get_local_ollama_models)
  else
    warn "Failed to pull model '$model'. Check whether this tag exists in Ollama."
  fi
done

# -----------------------------
# Final check
# -----------------------------
write_step "Final versions"

if have_cmd code; then code --version || warn "VS Code version check failed."; else warn "VS Code 'code' command may need a new shell."; fi
git --version || warn "Git version check failed."
"$CONDA_EXE" --version || warn "Conda version check failed."
"$OLLAMA_EXE" --version || warn "Ollama version check failed."

printf '\nSetup complete.\n'
printf 'Repo path: %s\n' "$REPO_PATH"
printf '\nUseful next commands:\n'
printf '  cd "%s"\n' "$REPO_PATH"
if [[ -n "${ENV_NAME:-}" ]]; then
  printf '  conda activate %s\n' "$ENV_NAME"
else
  printf '  conda env list\n'
  printf '  conda activate <your-env-name>\n'
fi
printf '  ollama list\n'
printf '\nNote: To keep the Conda env active after the script finishes, run it with:\n'
printf '  source ./setup.sh\n'
printf 'or:\n'
printf '  . ./setup.sh\n'
