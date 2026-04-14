#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# setup.sh — One-command setup for ASL Fingerspelling Recognition
#
# Handles: Python check → dependencies → model download → landmark
#          extraction → model training
#
# Usage:
#     chmod +x setup.sh
#     ./setup.sh
# ──────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

# ─── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ─── Helpers ──────────────────────────────────────────────────────────
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✔]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
fail()    { echo -e "${RED}[✘]${NC} $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}   ASL Fingerspelling Recognition — Setup          ${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""

# ─── Step 1: Check Python version ────────────────────────────────────
info "Checking Python version..."

if ! command -v python3 &> /dev/null; then
    fail "python3 not found. Please install Python 3.10 or later."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    fail "Python 3.10+ required, found Python $PYTHON_VERSION"
fi

success "Python $PYTHON_VERSION detected"

# ─── Step 2: Create virtual environment (optional) ───────────────────
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
else
    success "Virtual environment already exists"
fi

# Activate virtual environment
info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
success "Virtual environment activated ($(which python3))"

# ─── Step 3: Install dependencies ────────────────────────────────────
info "Installing Python dependencies..."

pip install --upgrade pip --quiet

if pip install torch torchvision scikit-learn opencv-python mediapipe numpy matplotlib --quiet; then
    success "All dependencies installed"
else
    fail "Dependency installation failed. Check your internet connection."
fi

# ─── Step 4: Download hand landmarker model ──────────────────────────
MODEL_DIR="$SCRIPT_DIR/models"
HAND_MODEL="$MODEL_DIR/hand_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

mkdir -p "$MODEL_DIR"

if [ -f "$HAND_MODEL" ]; then
    success "Hand landmarker model already exists ($(du -h "$HAND_MODEL" | cut -f1))"
else
    info "Downloading MediaPipe hand landmarker model..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$HAND_MODEL" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$HAND_MODEL" "$MODEL_URL"
    else
        fail "Neither wget nor curl found. Please install one of them."
    fi

    if [ -f "$HAND_MODEL" ] && [ -s "$HAND_MODEL" ]; then
        success "Hand landmarker model downloaded ($(du -h "$HAND_MODEL" | cut -f1))"
    else
        fail "Download failed or file is empty. Check your internet connection."
    fi
fi

# ─── Step 5: Check dataset ───────────────────────────────────────────
DATASET_DIR="$SCRIPT_DIR/asl_dataset"

if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
    warn "Dataset not found at $DATASET_DIR"
    warn "Please download the ASL dataset and place images in asl_dataset/<class>/ directories"
    warn "Skipping landmark extraction and training."
    echo ""
    echo -e "${YELLOW}After adding the dataset, run:${NC}"
    echo "    python3 extract_landmarks.py"
    echo "    python3 train_model.py"
    echo "    python3 live_demo.py"
    exit 0
fi

CLASS_COUNT=$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
success "Dataset found: $CLASS_COUNT classes in $DATASET_DIR"

# ─── Step 6: Extract landmarks ───────────────────────────────────────
LANDMARKS_FILE="$SCRIPT_DIR/data/landmarks.npy"

if [ -f "$LANDMARKS_FILE" ]; then
    warn "Landmarks file already exists. Re-extracting..."
fi

info "Extracting and normalizing hand landmarks from dataset..."
echo ""

if python3 extract_landmarks.py; then
    echo ""
    success "Landmark extraction complete"
else
    fail "Landmark extraction failed. Check the error above."
fi

# ─── Step 7: Train model ─────────────────────────────────────────────
info "Training LSTM model..."
echo ""

if python3 train_model.py; then
    echo ""
    success "Model training complete"
else
    fail "Model training failed. Check the error above."
fi

# ─── Done ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}   Setup Complete!                                 ${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  To start the live demo, run:"
echo ""
echo -e "    ${BOLD}source venv/bin/activate${NC}"
echo -e "    ${BOLD}python3 live_demo.py${NC}"
echo ""
