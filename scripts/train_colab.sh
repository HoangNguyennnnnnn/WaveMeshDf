#!/bin/bash
# Optimized training script for Google Colab
# Designed to work within Colab's memory constraints

echo "🚀 Starting WaveMesh-Diff training on Google Colab"
echo "=================================================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected - training will be very slow!"
fi

# Memory-optimized settings for Colab (12GB RAM)
echo ""
echo "📊 Memory-optimized settings:"
echo "  - Resolution: 16³ voxels (instead of 32³)"
echo "  - Batch size: 4"
echo "  - Num workers: 0 (in-process loading)"
echo "  - U-Net channels: [8, 16, 32] (reduced from [16, 32, 64])"
echo ""

# Quick debug training (5 minutes)
echo "Starting quick debug training (20 samples, 5 epochs)..."
python train.py \
    --data_root data/ModelNet40 \
    --dataset modelnet40 \
    --resolution 16 \
    --batch_size 4 \
    --epochs 5 \
    --max_samples 20 \
    --num_workers 0 \
    --unet_channels 8 16 32 \
    --diffusion_steps 100 \
    --output_dir outputs/colab_debug \
    --save_freq 5 \
    --log_freq 10

echo ""
echo "✅ Training complete!"
echo "Results saved to: outputs/colab_debug/"
echo ""
echo "To view results:"
echo "  !ls -lh outputs/colab_debug/"
echo "  !cat outputs/colab_debug/*/train.log"
