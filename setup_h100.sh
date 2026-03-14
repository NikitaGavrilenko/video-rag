#!/bin/bash
# Setup script for H100 server — run once after provisioning.
# Usage: bash setup_h100.sh
set -e

export PATH=$PATH:/home/ubuntu/.local/bin

echo "=== [1/5] Creating directory structure ==="
sudo mkdir -p /kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge
sudo mkdir -p /kaggle/working
sudo chown -R ubuntu:ubuntu /kaggle

echo "=== [2/5] Installing kaggle CLI ==="
pip install --break-system-packages kaggle

echo "=== [3/5] Setting up Kaggle credentials ==="
mkdir -p ~/.kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo '{"username":"ark2016","key":""}' > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
fi

echo "=== [4/5] Downloading competition data (~29GB) ==="
kaggle competitions download -c multi-lingual-video-fragment-retrieval-challenge \
    -p /kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge

echo "=== [4b/5] Extracting data ==="
cd /kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge
unzip -q *.zip -d video-rag
rm -f *.zip
echo "Data extracted. Contents:"
ls video-rag/

echo "=== [5/5] Installing Python dependencies ==="
pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --break-system-packages faiss-gpu-cu12 faster-whisper vllm FlagEmbedding rank-bm25 \
    transformers Pillow pandas tqdm

echo "=== [6/6] Cloning repo ==="
cd /kaggle/working
git clone https://github.com/NikitaGavrilenko/video-rag.git || echo "Repo already cloned"

echo ""
echo "=== DONE ==="
echo "Data:  /kaggle/input/competitions/multi-lingual-video-fragment-retrieval-challenge/video-rag/"
echo "Code:  /kaggle/working/video-rag/"
echo ""
echo "Run pipeline:"
echo "  cd /kaggle/working/video-rag"
echo "  python -m kaggle.pipeline.run_pipeline"
echo ""
echo "Or with streaming (step2+3 merged):"
echo "  python -m kaggle.pipeline.run_pipeline --stream"
