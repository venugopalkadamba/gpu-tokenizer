#!/bin/bash
# GPU Tokenizer - Automated Setup Script

set -e  # Exit on error

echo "=========================================="
echo "  GPU Tokenizer - Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Load CUDA module
echo -e "${YELLOW}[1/5] Loading CUDA module...${NC}"
module load cuda-11.4 || {
    echo -e "${RED}Failed to load CUDA module. Try manually: module load cuda-11.4${NC}"
    exit 1
}
echo -e "${GREEN}✓ CUDA 11.4 loaded${NC}"
nvcc --version | head -n 1
echo ""

# Step 2: Create virtual environment
echo -e "${YELLOW}[2/5] Creating Python virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Step 3: Activate virtual environment
echo -e "${YELLOW}[3/5] Activating virtual environment...${NC}"
source venv/bin/activate
pip install --upgrade pip -q
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 4: Install Python packages
echo -e "${YELLOW}[4/5] Installing Python dependencies...${NC}"
echo "This may take 5-10 minutes..."
echo ""

echo "  → Installing PyTorch with CUDA 11.x support..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "  → Installing CuPy..."
pip install -q cupy-cuda11x

echo "  → Installing Numba..."
pip install -q numba

echo "  → Installing HuggingFace libraries..."
pip install -q transformers tokenizers datasets

echo "  → Installing utilities..."
pip install -q numpy pandas matplotlib tqdm jupyter ipykernel

echo -e "${GREEN}✓ All Python packages installed${NC}"
echo ""

# Step 5: Verify installation
echo -e "${YELLOW}[5/5] Verifying installation...${NC}"
python3 test.py
echo ""

# Step 6: Create project structure
echo -e "${YELLOW}Creating project structure...${NC}"
mkdir -p src
mkdir -p data
mkdir -p benchmarks
mkdir -p notebooks
mkdir -p tests
mkdir -p results

echo -e "${GREEN}✓ Project directories created${NC}"
echo ""

# Success message
echo "=========================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Load CUDA: module load cuda-11.4"
echo "  3. Read SETUP_GUIDE.md for detailed roadmap"
echo "  4. Start with matrix multiplication exercise"
echo ""
echo "Quick test:"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""