#!/bin/bash
# Script to download OLYMPUS models using various methods

echo "🚀 OLYMPUS Model Download Script"
echo "================================="

# Set paths
MODELS_DIR="/content/AutomataNexus_Olympus_AGI2/arc_models_v4"
OUTPUT_DIR="/content/downloads"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p $OUTPUT_DIR

# Function to check file size
check_size() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo "  Size: $SIZE"
    fi
}

# Method 1: Create ZIP archive
echo -e "\n📦 Creating ZIP archive..."
ZIP_FILE="$OUTPUT_DIR/olympus_models_$TIMESTAMP.zip"
if [ -d "$MODELS_DIR" ]; then
    cd $MODELS_DIR
    zip -r $ZIP_FILE *.pt *.json 2>/dev/null
    echo "✅ Archive created: $ZIP_FILE"
    check_size $ZIP_FILE
else
    echo "❌ Models directory not found!"
fi

# Method 2: Create TAR.GZ archive (better compression)
echo -e "\n📦 Creating TAR.GZ archive..."
TAR_FILE="$OUTPUT_DIR/olympus_models_$TIMESTAMP.tar.gz"
if [ -d "$MODELS_DIR" ]; then
    tar -czf $TAR_FILE -C $(dirname $MODELS_DIR) $(basename $MODELS_DIR)
    echo "✅ Archive created: $TAR_FILE"
    check_size $TAR_FILE
fi

# Method 3: List individual files
echo -e "\n📄 Individual model files:"
if [ -d "$MODELS_DIR" ]; then
    for model in minerva atlas iris chronos prometheus; do
        BEST_MODEL="$MODELS_DIR/${model}_best.pt"
        CHECKPOINT="$MODELS_DIR/${model}_checkpoint.pt"
        
        if [ -f "$BEST_MODEL" ]; then
            echo "  ✅ ${model}_best.pt"
            check_size "$BEST_MODEL"
        fi
        
        if [ -f "$CHECKPOINT" ]; then
            echo "  ✅ ${model}_checkpoint.pt"
            check_size "$CHECKPOINT"
        fi
    done
fi

# Method 4: Google Drive copy commands
echo -e "\n☁️ Google Drive Commands:"
echo "  # Mount drive first:"
echo "  from google.colab import drive"
echo "  drive.mount('/content/drive')"
echo ""
echo "  # Then copy:"
echo "  !cp -r $MODELS_DIR /content/drive/MyDrive/OLYMPUS_Models_$TIMESTAMP"

# Method 5: wget commands for external download
echo -e "\n🌐 External Download Commands:"
echo "  # Start a simple HTTP server in Colab:"
echo "  !cd $MODELS_DIR && python -m http.server 8080 &"
echo ""
echo "  # Then use ngrok to expose it (requires setup):"
echo "  # !ngrok http 8080"

# Method 6: Split large files
echo -e "\n✂️ Split Large Files (if needed):"
echo "  # Split into 50MB chunks:"
echo "  split -b 50M $ZIP_FILE ${ZIP_FILE}.part_"
echo ""
echo "  # To recombine:"
echo "  cat ${ZIP_FILE}.part_* > ${ZIP_FILE}"

echo -e "\n✅ Download preparation complete!"
echo "================================="
echo -e "\n💡 Tips:"
echo "  - Google Drive method is most reliable for large files"
echo "  - Browser downloads work best under 100MB"
echo "  - Use TAR.GZ for better compression"
echo "  - Split files if download keeps failing"