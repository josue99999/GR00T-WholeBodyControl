#!/usr/bin/env bash
# =============================================================================
# setup_training_server.sh
# GR00T-WholeBodyControl — Training environment setup for a headless server.
#
# What this does:
#   1. Checks prerequisites (GPU, CUDA driver, disk space)
#   2. Installs uv + Python 3.11 virtual environment
#   3. Installs Isaac Lab (pip headless — no Isaac Sim GUI needed)
#   4. Installs gear_sonic[training] dependencies
#   5. Downloads training checkpoint + SMPL data from Hugging Face (~30 GB)
#   6. Runs environment pre-flight check
#   7. Prints the fine-tuning command
#
# Usage (run from repo root):
#   bash setup_training_server.sh
#
# After setup, activate the venv and run training:
#   source .venv_training/bin/activate
#   bash setup_training_server.sh --train        # run fine-tuning directly
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv_training"
PYTHON_VERSION="3.11"

# Isaac Lab / Isaac Sim versions (change here if newer versions are available)
# Ref: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
ISAACSIM_VERSION="4.5.0.post0"
ISAACLAB_VERSION="2.3.0"
NVIDIA_PYPI="https://pypi.nvidia.com"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${YELLOW}━━━ $* ━━━${NC}"; }

# ── Handle --train flag ───────────────────────────────────────────────────────
RUN_TRAINING=false
for arg in "$@"; do
    [[ "$arg" == "--train" ]] && RUN_TRAINING=true
done

cd "$REPO_ROOT"

# =============================================================================
# STEP 0 — Prerequisites check
# =============================================================================
step "0 / 7  Checking prerequisites"

# NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. An NVIDIA GPU with drivers is required."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
info "GPU: $GPU_NAME  (driver $DRIVER)"

# CUDA driver >= 525 (required by Isaac Sim 4.x)
DRIVER_MAJOR=$(echo "$DRIVER" | cut -d. -f1)
if [[ "$DRIVER_MAJOR" -lt 525 ]]; then
    error "Driver $DRIVER is too old. Isaac Sim 4.x requires driver >= 525."
fi

# Disk space: need ~100 GB free
AVAIL_GB=$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
if [[ "$AVAIL_GB" -lt 80 ]]; then
    warn "Only ${AVAIL_GB} GB free. Recommend >= 80 GB (Isaac Sim ~15 GB + checkpoint ~30 GB + SMPL ~10 GB + training logs)."
else
    info "Disk space: ${AVAIL_GB} GB free"
fi

# Git LFS (needed for robot mesh assets)
if ! command -v git-lfs &>/dev/null; then
    warn "git-lfs not found — installing..."
    sudo apt-get install -y git-lfs || error "Please install git-lfs manually: sudo apt-get install git-lfs"
fi
info "git-lfs: $(git-lfs version)"
git lfs pull --include="gear_sonic/data/assets/**" 2>/dev/null || warn "git lfs pull failed (may already be pulled)"

# =============================================================================
# STEP 1 — Install uv and Python 3.11
# =============================================================================
step "1 / 7  Setting up uv + Python $PYTHON_VERSION"

if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
info "uv: $(uv --version)"

info "Installing Python $PYTHON_VERSION (uv-managed)..."
uv python install "$PYTHON_VERSION"
MANAGED_PY="$(uv python find --no-project "$PYTHON_VERSION")"
info "Python binary: $MANAGED_PY"

# Create venv
info "Creating $VENV_DIR..."
rm -rf "$VENV_DIR"
uv venv "$VENV_DIR" --python "$MANAGED_PY" --prompt gear_sonic_training
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Virtual environment activated: $VIRTUAL_ENV"

# =============================================================================
# STEP 2 — Install Isaac Sim (headless pip packages)
# =============================================================================
step "2 / 7  Installing Isaac Sim $ISAACSIM_VERSION (headless, ~8-15 GB)"

echo ""
echo "  This installs Isaac Sim as Python pip packages — no GUI / Omniverse needed."
echo "  Download size: ~8-15 GB depending on CUDA version. This will take a while."
echo ""

uv pip install \
    "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
    --extra-index-url "$NVIDIA_PYPI"

info "Isaac Sim installed."

# =============================================================================
# STEP 3 — Install Isaac Lab
# =============================================================================
step "3 / 7  Installing Isaac Lab $ISAACLAB_VERSION"

uv pip install "isaaclab==$ISAACLAB_VERSION"

# Verify
python -c "import isaaclab; print('Isaac Lab', isaaclab.__version__)" \
    || error "Isaac Lab import failed after installation."
info "Isaac Lab OK."

# =============================================================================
# STEP 4 — Install gear_sonic[training]
# =============================================================================
step "4 / 7  Installing gear_sonic[training]"

uv pip install -e "gear_sonic/[training]"
info "gear_sonic training deps installed."

# =============================================================================
# STEP 5 — Download checkpoint + SMPL data from Hugging Face (~30 GB)
# =============================================================================
step "5 / 7  Downloading checkpoint + SMPL data from Hugging Face"

uv pip install huggingface_hub

echo ""
echo "  Downloading:"
echo "    • sonic_release/last.pt          (~1.4 GB PyTorch checkpoint)"
echo "    • data/smpl_filtered/            (~10 GB SMPL motion data, 7 tar parts)"
echo ""
echo "  You will need to be logged in to Hugging Face if the model is gated:"
echo "    huggingface-cli login"
echo ""

python download_from_hf.py --training
info "Download complete."

# =============================================================================
# STEP 6 — Pre-flight environment check
# =============================================================================
step "6 / 7  Running environment check"

python check_environment.py --training || warn "Some checks failed — review output above before training."

# =============================================================================
# STEP 7 — Summary + training command
# =============================================================================
step "7 / 7  Setup complete"

MOTION_FILE="data/motion_lib_custom/robot"
SMPL_FILE="data/smpl_filtered"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the training environment with:"
echo ""
echo "    source .venv_training/bin/activate"
echo ""
echo "  Fine-tune on danza_caporal (1 GPU, small batch — slow but works):"
echo ""
echo "    python gear_sonic/train_agent_trl.py \\"
echo "        +exp=manager/universal_token/all_modes/sonic_release \\"
echo "        +checkpoint=sonic_release/last.pt \\"
echo "        num_envs=64 headless=True \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION_FILE \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL_FILE"
echo ""
echo "  Fine-tune with W&B logging:"
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
echo "  Multi-GPU (8 GPUs, recommended for speed):"
echo ""
echo "    accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \\"
echo "        +exp=manager/universal_token/all_modes/sonic_release \\"
echo "        +checkpoint=sonic_release/last.pt \\"
echo "        num_envs=4096 headless=True \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION_FILE \\"
echo "        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL_FILE"
echo ""
echo "  Checkpoints are saved to: logs_rl/TRL_G1_Track/<exp>-<timestamp>/"
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
