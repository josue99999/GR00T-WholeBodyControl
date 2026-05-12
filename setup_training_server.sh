#!/usr/bin/env bash
# =============================================================================
# setup_training_server.sh
# GR00T-WholeBodyControl — Training environment setup for a headless server.
#
# What this does:
#   1. Checks prerequisites (GPU, CUDA driver, git-lfs)
#   2. Detects filesystem — if NTFS, creates a 30 GB ext4 loop image on the
#      same disk so pip can install packages without POSIX permission errors
#   3. Creates Python 3.10 venv + bootstraps pip
#   4. Installs Isaac Sim 4.5.0.0 + Isaac Lab 2.1.0 (headless pip; 2.3.x not on PyPI)
#   5. Installs gear_sonic[training] dependencies
#   6. Downloads training checkpoint + SMPL data from Hugging Face (~12 GB)
#   7. Runs environment pre-flight check
#   8. Prints the fine-tuning command
#
# Usage (run from repo root):
#   bash setup_training_server.sh
#   bash setup_training_server.sh --train   # setup + start training immediately
#
# After setup, activate the venv with:
#   source activate_training.sh             # auto-generated, works on any layout
#
# Known quirks handled automatically:
#   - NTFS filesystems: pip cannot chmod/rename-atomic on NTFS → ext4 loop device
#   - Isaac Sim 4.x: requires Python 3.10 (cp310 wheels only, NOT 3.11)
#   - Isaac Sim version: 4.5.0.0 on NVIDIA PyPI (NOT 4.5.0.post0)
#   - uv venv does not include pip → bootstrap with ensurepip
#   - PATH may not have the venv pip → always use `python -m pip`
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.10"
ISAACSIM_VERSION="4.5.0.0"
ISAACLAB_VERSION="2.1.0"
NVIDIA_PYPI="https://pypi.nvidia.com"
EXT4_IMG_SIZE_GB=30

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${YELLOW}━━━ $* ━━━${NC}"; }

RUN_TRAINING=false
for arg in "$@"; do [[ "$arg" == "--train" ]] && RUN_TRAINING=true; done

cd "$REPO_ROOT"

# =============================================================================
# STEP 0 — Prerequisites
# =============================================================================
step "0 / 7  Checking prerequisites"

command -v nvidia-smi &>/dev/null || error "nvidia-smi not found. An NVIDIA GPU with drivers is required."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "$DRIVER" | cut -d. -f1)
[[ "$DRIVER_MAJOR" -lt 525 ]] && error "Driver $DRIVER is too old. Isaac Sim 4.x requires >= 525."
info "GPU: $GPU_NAME  (driver $DRIVER)"

AVAIL_GB=$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
[[ "$AVAIL_GB" -lt 50 ]] && warn "Only ${AVAIL_GB} GB free on repo disk. Need ~50 GB total."
info "Disk space at repo: ${AVAIL_GB} GB free"

if ! command -v git-lfs &>/dev/null; then
    warn "git-lfs not found — installing..."
    sudo apt-get install -y git-lfs || error "Install git-lfs manually: sudo apt-get install git-lfs"
fi
info "git-lfs: $(git-lfs version)"
git lfs pull --include="gear_sonic/data/assets/**" 2>/dev/null || warn "git lfs pull skipped (may already be current)"

# =============================================================================
# STEP 1 — Filesystem detection + venv location
# =============================================================================
step "1 / 7  Setting up Python $PYTHON_VERSION environment"

FS_TYPE=$(df -T "$REPO_ROOT" | awk 'NR==2{print $2}')
info "Repo filesystem type: $FS_TYPE"

if [[ "$FS_TYPE" == "fuseblk" || "$FS_TYPE" == "ntfs" || "$FS_TYPE" == "ntfs-3g" ]]; then
    # ── NTFS path ──────────────────────────────────────────────────────────────
    warn "NTFS detected. pip cannot install to NTFS (chmod/atomic-rename restrictions)."
    warn "Creating a ${EXT4_IMG_SIZE_GB} GB ext4 loop image on this disk for the venv."

    DISK_MOUNT=$(df "$REPO_ROOT" | awk 'NR==2{print $6}')
    USER_DIR="$DISK_MOUNT/$(whoami)"
    EXT4_IMG="$USER_DIR/venv_ext4.img"
    EXT4_MOUNT="$USER_DIR/ext4"
    VENV_DIR="$EXT4_MOUNT/.venv_training"
    PIP_TMPDIR="$EXT4_MOUNT/tmp"
    PIP_CACHE="$USER_DIR/.cache/pip"   # cache stays on NTFS (just .whl files — fine)

    mkdir -p "$USER_DIR" "$PIP_CACHE"

    if [[ ! -f "$EXT4_IMG" ]]; then
        info "Creating ${EXT4_IMG_SIZE_GB} GB ext4 image at $EXT4_IMG ..."
        dd if=/dev/zero of="$EXT4_IMG" bs=1G count="$EXT4_IMG_SIZE_GB" status=progress
        mkfs.ext4 -F "$EXT4_IMG"
        info "ext4 image created."
    else
        info "ext4 image already exists: $EXT4_IMG"
    fi

    if ! mountpoint -q "$EXT4_MOUNT" 2>/dev/null; then
        mkdir -p "$EXT4_MOUNT"
        sudo mount -o loop "$EXT4_IMG" "$EXT4_MOUNT"
        sudo chown "$(whoami):$(id -gn)" "$EXT4_MOUNT"
        info "Mounted $EXT4_IMG → $EXT4_MOUNT"
    else
        info "ext4 already mounted at $EXT4_MOUNT"
    fi

    mkdir -p "$PIP_TMPDIR"
    PIP_CACHE_ARG="--cache-dir $PIP_CACHE"
else
    # ── Normal ext4 / other POSIX path ────────────────────────────────────────
    VENV_DIR="$REPO_ROOT/.venv_training"
    PIP_TMPDIR="/tmp"
    PIP_CACHE_ARG=""
fi

# ── Install uv ────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1090
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
info "uv: $(uv --version)"

# ── Python 3.10 venv ──────────────────────────────────────────────────────────
uv python install "$PYTHON_VERSION"
MANAGED_PY="$(uv python find --no-project "$PYTHON_VERSION")"
info "Python binary: $MANAGED_PY"

info "Creating venv at $VENV_DIR ..."
rm -rf "$VENV_DIR"
uv venv "$VENV_DIR" --python "$MANAGED_PY" --prompt gear_sonic_training
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Bootstrap pip (uv venv does not include pip) ──────────────────────────────
info "Bootstrapping pip inside venv..."
TMPDIR="$PIP_TMPDIR" python -m ensurepip --upgrade
TMPDIR="$PIP_TMPDIR" python -m pip install --upgrade pip $PIP_CACHE_ARG --quiet
info "pip $(python -m pip --version)"

# ── Write activation helper (absolute path, server-independent activation) ───
cat > "$REPO_ROOT/activate_training.sh" << ACTIVATE_EOF
#!/usr/bin/env bash
# Auto-generated by setup_training_server.sh — do not edit manually.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ACTIVATE_EOF
info "Activation helper: source activate_training.sh"

# =============================================================================
# STEP 2 — Install Isaac Sim (headless pip, ~8-15 GB)
# =============================================================================
step "2 / 7  Installing Isaac Sim $ISAACSIM_VERSION (headless, ~8-15 GB)"

echo ""
echo "  Headless pip install — no GUI or Omniverse Launcher needed."
echo "  Download size: ~8-15 GB. This will take a while."
echo ""

# shellcheck disable=SC2086
TMPDIR="$PIP_TMPDIR" python -m pip install \
    "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
    --extra-index-url "$NVIDIA_PYPI" \
    $PIP_CACHE_ARG

info "Isaac Sim installed."

# =============================================================================
# STEP 3 — Install Isaac Lab
# =============================================================================
step "3 / 7  Installing Isaac Lab $ISAACLAB_VERSION"

# Auto-detect latest available if pinned version doesn't exist on PyPI
if ! python -m pip index versions isaaclab 2>/dev/null | grep -q "$ISAACLAB_VERSION"; then
    ISAACLAB_LATEST=$(python -m pip index versions isaaclab 2>/dev/null \
        | grep -oP '\d+\.\d+\.\d+' | head -1)
    if [[ -n "$ISAACLAB_LATEST" ]]; then
        warn "isaaclab==$ISAACLAB_VERSION not found on PyPI. Using latest: $ISAACLAB_LATEST"
        ISAACLAB_VERSION="$ISAACLAB_LATEST"
    fi
fi

# shellcheck disable=SC2086
TMPDIR="$PIP_TMPDIR" python -m pip install \
    "isaaclab==$ISAACLAB_VERSION" \
    $PIP_CACHE_ARG

python -c "import isaaclab; print('Isaac Lab', isaaclab.__version__)" \
    || error "Isaac Lab import failed after installation."
info "Isaac Lab OK."

# =============================================================================
# STEP 4 — Install gear_sonic[training]
# =============================================================================
step "4 / 7  Installing gear_sonic[training]"

# shellcheck disable=SC2086
TMPDIR="$PIP_TMPDIR" python -m pip install \
    -e "gear_sonic/[training]" \
    $PIP_CACHE_ARG

info "gear_sonic training deps installed."

# =============================================================================
# STEP 5 — Download checkpoint + SMPL data from Hugging Face (~12 GB)
# =============================================================================
step "5 / 7  Downloading checkpoint + SMPL data from Hugging Face"

# shellcheck disable=SC2086
TMPDIR="$PIP_TMPDIR" python -m pip install huggingface_hub $PIP_CACHE_ARG --quiet

echo ""
echo "  Downloading to repo root:"
echo "    • sonic_release/last.pt      (~1.4 GB  PyTorch checkpoint)"
echo "    • data/smpl_filtered/        (~10 GB   SMPL motion data)"
echo ""
echo "  If the model is gated, log in first:"
echo "    huggingface-cli login"
echo ""

python download_from_hf.py --training
info "Download complete."

# =============================================================================
# STEP 6 — Pre-flight check
# =============================================================================
step "6 / 7  Running environment check"

python check_environment.py --training \
    || warn "Some checks failed — review output above before running training."

# =============================================================================
# STEP 7 — Summary
# =============================================================================
step "7 / 7  Setup complete"

MOTION_FILE="data/motion_lib_custom/robot"
SMPL_FILE="data/smpl_filtered"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the training environment:"
echo ""
echo "    source activate_training.sh"
echo ""
echo "  Fine-tune on danza_caporal (1 GPU, small batch):"
echo ""
echo "    python gear_sonic/train_agent_trl.py \\"
echo "        +exp=manager/universal_token/all_modes/sonic_release \\"
echo "        +checkpoint=sonic_release/last.pt \\"
echo "        num_envs=64 headless=True \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION_FILE \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL_FILE"
echo ""
echo "  Multi-GPU (8 GPUs, recommended for speed):"
echo ""
echo "    accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \\"
echo "        +exp=manager/universal_token/all_modes/sonic_release \\"
echo "        +checkpoint=sonic_release/last.pt \\"
echo "        num_envs=4096 headless=True \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION_FILE \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL_FILE"
echo ""
echo "  With W&B logging:"
echo ""
echo "    python gear_sonic/train_agent_trl.py \\"
echo "        +exp=manager/universal_token/all_modes/sonic_release \\"
echo "        +checkpoint=sonic_release/last.pt \\"
echo "        +opt=wandb \\"
echo "        ++wandb.wandb_project=gr00t-danza-caporal \\"
echo "        ++wandb.wandb_entity=TU_USUARIO_WANDB \\"
echo "        num_envs=64 headless=True \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION_FILE \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL_FILE"
echo ""
echo "  Checkpoints saved to: logs_rl/TRL_G1_Track/<exp>-<timestamp>/"
echo "════════════════════════════════════════════════════════════════════════"

# =============================================================================
# Optional: run fine-tuning directly (--train flag)
# =============================================================================
if [[ "$RUN_TRAINING" == true ]]; then
    step "Running fine-tuning (num_envs=64, headless)"
    python gear_sonic/train_agent_trl.py \
        +exp=manager/universal_token/all_modes/sonic_release \
        +checkpoint=sonic_release/last.pt \
        num_envs=64 headless=True \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION_FILE" \
        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$SMPL_FILE"
fi
