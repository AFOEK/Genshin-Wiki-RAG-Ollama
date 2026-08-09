# setup-rag-stack.ps1
# Non-admin one-shot setup for Genshin RAG pipeline:
# VS Code User Installer, Git user install, Miniforge Conda, Ollama official installer,
# repo clone/update, conda env create/update, and Ollama model pulls.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# -----------------------------
# Config
# -----------------------------
$RepoUrl = "https://github.com/AFOEK/Genshin-Wiki-RAG-Ollama"
$WorkRoot = Join-Path $env:USERPROFILE "Documents"
$RepoPath = Join-Path $WorkRoot "Genshin-Wiki-RAG-Ollama"
$FaissCurrentDir = "C:\mnt\ssd\genshin_rag\data\faiss\current"

$AppRoot = Join-Path $env:LOCALAPPDATA "RAGStack"
$DownloadRoot = Join-Path $env:TEMP "rag-stack-downloads"

$GitUserDir = Join-Path $env:LOCALAPPDATA "Programs\Git"
$CondaDefaultDir = Join-Path $env:LOCALAPPDATA "miniforge3"
$CondaCustomDir = Join-Path $AppRoot "Miniforge3"
$CondaDir = if (Test-Path (Join-Path $CondaDefaultDir "Scripts\conda.exe")) { $CondaDefaultDir } else { $CondaCustomDir }

$RunSpladeMigration = $true

$Models = @(
    "qwen3.5:9b",
    "deepseek-r1:14b",
    "snowflake-arctic-embed2:568m",
    "gemma3:12b",
    "qwen3:8b",
    "llama3.2:3b",
    "qwen3.6:27b",
    "gemma4:12b",
    "all-minilm:latest"
)

New-Item -ItemType Directory -Force -Path $WorkRoot, $AppRoot, $DownloadRoot | Out-Null

# -----------------------------
# Helpers
# -----------------------------
function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Download-File($url, $outFile) {
    Write-Host "Downloading: $url"
    Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing
}

function Add-UserPath($pathToAdd) {
    if (!(Test-Path $pathToAdd)) { return }

    $currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($null -eq $currentUserPath) { $currentUserPath = "" }

    $parts = $currentUserPath -split ";" | Where-Object { $_ -and $_.Trim() -ne "" }
    $alreadyThere = $parts | Where-Object { $_.TrimEnd("\") -ieq $pathToAdd.TrimEnd("\") }

    if (!$alreadyThere) {
        $newUserPath = if ($currentUserPath.Trim()) { "$currentUserPath;$pathToAdd" } else { $pathToAdd }
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        Write-Host "Added to user PATH: $pathToAdd"
    }

    if (($env:Path -split ";") -notcontains $pathToAdd) {
        $env:Path = "$pathToAdd;$env:Path"
    }
}

function Refresh-Path {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $env:Path = "$userPath;$machinePath;$env:Path"
}

function Get-LatestGithubAssetUrl($repo, $pattern) {
    $api = "https://api.github.com/repos/$repo/releases/latest"
    $release = Invoke-RestMethod -Uri $api -Headers @{
        "User-Agent" = "PowerShell-RAG-Setup"
        "Accept" = "application/vnd.github+json"
    }

    $asset = $release.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1

    if (!$asset) {
        throw "Could not find GitHub release asset matching pattern '$pattern' in repo '$repo'."
    }

    return $asset.browser_download_url
}

function Resolve-Exe($commandName, $candidatePaths) {
    $cmd = Get-Command $commandName -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    foreach ($path in $candidatePaths) {
        if (Test-Path $path) { return $path }
    }

    return $null
}

function Invoke-Native($exe, $arguments, $friendlyName) {
    Write-Host "Running: $friendlyName"
    & $exe @arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$friendlyName failed with exit code $LASTEXITCODE"
    }
}

# -----------------------------
# Create RAG data / FAISS directory
# -----------------------------
Write-Step "Creating RAG data directory"

try {
    New-Item -ItemType Directory -Force -Path $FaissCurrentDir | Out-Null

    $WriteTestFile = Join-Path $FaissCurrentDir ".write_test"
    "ok" | Set-Content -Path $WriteTestFile -Encoding UTF8
    Remove-Item $WriteTestFile -Force

    Write-Host "Created and verified writable directory: $FaissCurrentDir"
} catch {
    throw "Failed to create or write to: $FaissCurrentDir. This path may need permission from IT, especially because it starts at C:\mnt."
}

# -----------------------------
# VS Code user install
# -----------------------------
Write-Step "Installing VS Code User Installer if needed"

$CodeExeCandidate = Join-Path $env:LOCALAPPDATA "Programs\Microsoft VS Code\Code.exe"
$CodeBin = Join-Path $env:LOCALAPPDATA "Programs\Microsoft VS Code\bin"
$CodeExe = Resolve-Exe "code" @($CodeExeCandidate)

if ($CodeExe) {
    Write-Host "VS Code already found: $CodeExe"
} else {
    $VsCodeInstaller = Join-Path $DownloadRoot "VSCodeUserSetup-x64.exe"
    $VsCodeUrl = "https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user"
    Download-File $VsCodeUrl $VsCodeInstaller

    $VsCodeArgs = "/VERYSILENT /NORESTART /MERGETASKS=addtopath,!runcode"
    Start-Process -FilePath $VsCodeInstaller -ArgumentList $VsCodeArgs -Wait

    Add-UserPath $CodeBin
    Refresh-Path

    $CodeExe = Resolve-Exe "code" @($CodeExeCandidate)
    if (!$CodeExe) {
        Write-Warning "VS Code installed, but 'code' may require a new PowerShell session."
    }
}

# -----------------------------
# Git for Windows user install
# -----------------------------
Write-Step "Installing Git for Windows for current user if needed"

$GitExeCandidate = Join-Path $GitUserDir "cmd\git.exe"
$GitExe = Resolve-Exe "git" @($GitExeCandidate)

if ($GitExe) {
    Write-Host "Git already found: $GitExe"
} else {
    $GitUrl = Get-LatestGithubAssetUrl "git-for-windows/git" "^Git-.*-64-bit\.exe$"
    $GitInstaller = Join-Path $DownloadRoot "GitForWindows-x64.exe"
    Download-File $GitUrl $GitInstaller

    $GitArgs = "/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /CURRENTUSER /DIR=`"$GitUserDir`" /o:PathOption=Cmd /o:EditorOption=VisualStudioCode /o:DefaultBranchOption=main /o:CRLFOption=CRLFCommitAsIs /o:UseCredentialManager=Enabled"
    Start-Process -FilePath $GitInstaller -ArgumentList $GitArgs -Wait

    Add-UserPath (Join-Path $GitUserDir "cmd")
    Add-UserPath (Join-Path $GitUserDir "bin")
    Refresh-Path

    $GitExe = Resolve-Exe "git" @($GitExeCandidate)
    if (!$GitExe) {
        throw "Git installation finished, but git.exe was not found. Try reopening PowerShell, or check: $GitUserDir"
    }
}

# -----------------------------
# Miniforge / Conda user install
# -----------------------------
Write-Step "Installing or detecting Miniforge Conda"

$CondaCandidateDirs = @(
    (Join-Path $env:LOCALAPPDATA "miniforge3"),
    (Join-Path $AppRoot "Miniforge3"),
    (Join-Path $env:USERPROFILE "miniforge3"),
    (Join-Path $env:USERPROFILE "Miniforge3")
)

$CondaCandidateExes = foreach ($dir in $CondaCandidateDirs) {
    Join-Path $dir "Scripts\conda.exe"
}

$CondaExe = Resolve-Exe "conda" $CondaCandidateExes

if ($CondaExe) {
    Write-Host "Conda already found: $CondaExe"
    $CondaDir = Split-Path (Split-Path $CondaExe -Parent) -Parent
} else {
    $CondaInstaller = Join-Path $DownloadRoot "Miniforge3-Windows-x86_64.exe"
    $CondaUrl = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
    Download-File $CondaUrl $CondaInstaller

    # Important: /D must be last. Do not quote it.
    $CondaArgs = @(
        "/S",
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=0",
        "/D=$CondaCustomDir"
    )

    Start-Process -FilePath $CondaInstaller -ArgumentList $CondaArgs -Wait

    $CondaCandidateExes = foreach ($dir in $CondaCandidateDirs) {
        Join-Path $dir "Scripts\conda.exe"
    }

    $CondaExe = Resolve-Exe "conda" $CondaCandidateExes

    if (!$CondaExe) {
        throw "Conda installation finished, but conda.exe was not found in known user paths."
    }

    $CondaDir = Split-Path (Split-Path $CondaExe -Parent) -Parent
}

Add-UserPath (Join-Path $CondaDir "condabin")
Add-UserPath (Join-Path $CondaDir "Scripts")
Add-UserPath $CondaDir
Refresh-Path

Write-Host "Using Conda directory: $CondaDir"
Write-Host "Using Conda executable: $CondaExe"

# -----------------------------
# Ollama official user install
# -----------------------------
Write-Step "Installing Ollama using official PowerShell installer if needed"

$OllamaCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
    (Join-Path $env:LOCALAPPDATA "Ollama\ollama.exe")
)

$OllamaExe = Resolve-Exe "ollama" $OllamaCandidates

if ($OllamaExe) {
    Write-Host "Ollama already found: $OllamaExe"
} else {
    Write-Host "Installing Ollama from official install.ps1..."
    Invoke-Expression (Invoke-RestMethod "https://ollama.com/install.ps1")

    Refresh-Path
    Add-UserPath (Join-Path $env:LOCALAPPDATA "Programs\Ollama")

    $OllamaExe = Resolve-Exe "ollama" $OllamaCandidates
    if (!$OllamaExe) {
        throw "Ollama installation finished, but ollama.exe was not found. Reopen PowerShell and run: ollama --version"
    }
}

# -----------------------------
# Clone or update repo
# -----------------------------
Write-Step "Cloning or updating RAG repo"

if (Test-Path (Join-Path $RepoPath ".git")) {
    Write-Host "Repo already exists. Pulling latest changes..."
    Push-Location $RepoPath

    & $GitExe pull

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "git pull failed. Continuing with existing local repo."
    }

    Pop-Location
} elseif (Test-Path $RepoPath) {
    throw "Target folder already exists but is not a Git repo: $RepoPath"
} else {
    & $GitExe clone $RepoUrl $RepoPath

    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed with exit code $LASTEXITCODE"
    }
}

# -----------------------------
# Create or update Conda environment
# -----------------------------
Write-Step "Creating or updating Conda environment"

$EnvFileYml = Join-Path $RepoPath "environment.yml"
$EnvFileYaml = Join-Path $RepoPath "environment.yaml"

if (Test-Path $EnvFileYml) {
    $EnvFile = $EnvFileYml
} elseif (Test-Path $EnvFileYaml) {
    $EnvFile = $EnvFileYaml
} else {
    throw "No environment.yml or environment.yaml found in: $RepoPath"
}

$EnvNameLine = Get-Content $EnvFile | Where-Object { $_ -match "^\s*name\s*:" } | Select-Object -First 1
$EnvName = ""

if ($EnvNameLine) {
    $EnvName = ($EnvNameLine -replace "^\s*name\s*:\s*", "").Trim().Trim('"').Trim("'")
}

Push-Location $RepoPath

if ($EnvName) {
    $ExistingEnvs = & $CondaExe env list

    if ($ExistingEnvs -match "^\s*$([regex]::Escape($EnvName))\s+") {
        Write-Host "Conda env '$EnvName' already exists. Updating..."
        & $CondaExe env update -n $EnvName -f $EnvFile --prune

        if ($LASTEXITCODE -ne 0) {
            throw "conda env update failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "Creating Conda env '$EnvName'..."
        & $CondaExe env create -f $EnvFile

        if ($LASTEXITCODE -ne 0) {
            throw "conda env create failed with exit code $LASTEXITCODE"
        }
    }
} else {
    Write-Host "No env name found in environment file. Running conda env create directly..."
    & $CondaExe env create -f $EnvFile

    if ($LASTEXITCODE -ne 0) {
        throw "conda env create failed with exit code $LASTEXITCODE"
    }
}

Pop-Location

# Optional but useful for future shells
Write-Step "Initializing Conda for PowerShell"

try {
    & $CondaExe init powershell
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Conda PowerShell initialization done. Open a new PowerShell window before using 'conda activate'."
    } else {
        Write-Warning "conda init powershell returned exit code $LASTEXITCODE"
    }
} catch {
    Write-Warning "Could not run conda init powershell. You can still use conda with full path: $CondaExe"
}

# -----------------------------
# Start Ollama server
# -----------------------------
Write-Step "Starting Ollama server"

function Test-OllamaServer {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (Test-OllamaServer) {
    Write-Host "Ollama server already running."
} else {
    Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    Write-Host "Started Ollama server. Checking readiness..."

    $ready = $false

    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2

        if (Test-OllamaServer) {
            $ready = $true
            break
        }
    }

    if (!$ready) {
        throw "Ollama server did not become ready at http://127.0.0.1:11434"
    }
}

# -----------------------------
# Pull Ollama models
# -----------------------------
Write-Step "Pulling Ollama models"

function Invoke-NativeAllowFailure($exe, $arguments) {
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        & $exe @arguments
        return $LASTEXITCODE
    } catch {
        Write-Warning $_.Exception.Message
        return 1
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
}

function Get-LocalOllamaModels {
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        $lines = & $OllamaExe list 2>$null

        if (!$lines) {
            return @()
        }

        return $lines |
            Select-Object -Skip 1 |
            ForEach-Object {
                $parts = $_ -split "\s+"
                if ($parts.Count -gt 0) {
                    $parts[0]
                }
            } |
            Where-Object { $_ -and $_.Trim() -ne "" }
    } catch {
        return @()
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
}

$LocalModels = @(Get-LocalOllamaModels)

foreach ($model in $Models) {
    Write-Host ""
    Write-Host "Checking model: $model" -ForegroundColor Yellow

    if ($LocalModels -contains $model) {
        Write-Host "Model already exists: $model"
        continue
    }

    Write-Host "Pulling model: $model"
    $exitCode = Invoke-NativeAllowFailure $OllamaExe @("pull", $model)

    if ($exitCode -eq 0) {
        Write-Host "Pulled successfully: $model" -ForegroundColor Green
        $LocalModels = @(Get-LocalOllamaModels)
    } else {
        Write-Warning "Failed to pull model '$model'. Check whether this tag exists in Ollama."
    }
}

# -----------------------------
# Final check
# -----------------------------
Write-Step "Final versions"

try { & $GitExe --version } catch { Write-Warning "Git version check failed." }
try { & $CondaExe --version } catch { Write-Warning "Conda version check failed." }
try { & $OllamaExe --version } catch { Write-Warning "Ollama version check failed." }

$CodeCmd = Get-Command code -ErrorAction SilentlyContinue
if ($CodeCmd) {
    try { code --version } catch { Write-Warning "VS Code version check failed." }
} else {
    Write-Warning "VS Code 'code' command may need a new PowerShell session."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Repo path: $RepoPath"
Write-Host ""