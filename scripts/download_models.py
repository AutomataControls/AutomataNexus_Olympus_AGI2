"""
Script to prepare model files for download
"""

import os
import zipfile
import shutil
from datetime import datetime

def create_model_archive():
    """Create a zip archive of all model files for easy download"""
    
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    output_dir = '/content/downloads'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f'olympus_models_{timestamp}.zip'
    archive_path = os.path.join(output_dir, archive_name)
    
    # Get all model files
    model_files = []
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pt') or file.endswith('.json'):
                model_files.append(os.path.join(models_dir, file))
    
    if not model_files:
        print("❌ No model files found!")
        return None
    
    print(f"📦 Creating archive with {len(model_files)} files...")
    
    # Create zip archive
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in model_files:
            arcname = os.path.basename(file_path)
            zipf.write(file_path, arcname)
            print(f"  Added: {arcname}")
    
    # Get file size
    size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"\n✅ Archive created: {archive_path}")
    print(f"📊 Size: {size_mb:.1f} MB")
    
    return archive_path

def create_individual_downloads():
    """Create symlinks to individual model files in download directory"""
    
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    output_dir = '/content/downloads/individual_models'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy best models and checkpoints
    important_files = [
        'minerva_best.pt', 'minerva_checkpoint.pt',
        'atlas_best.pt', 'atlas_checkpoint.pt',
        'iris_best.pt', 'iris_checkpoint.pt',
        'chronos_best.pt', 'chronos_checkpoint.pt',
        'prometheus_best.pt', 'prometheus_checkpoint.pt'
    ]
    
    copied_files = []
    for filename in important_files:
        src = os.path.join(models_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            copied_files.append(f"{filename} ({size_mb:.1f} MB)")
            print(f"📄 Copied: {filename} ({size_mb:.1f} MB)")
    
    return output_dir, copied_files

def mount_google_drive():
    """Mount Google Drive for easy file transfer"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except ImportError:
        return False

if __name__ == "__main__":
    print("🚀 OLYMPUS Model Download Helper")
    print("="*50)
    
    # Option 1: Create zip archive
    print("\n📦 Option 1: Creating zip archive of all models...")
    archive_path = create_model_archive()
    
    # Option 2: Create individual downloads
    print("\n📄 Option 2: Preparing individual model files...")
    download_dir, files = create_individual_downloads()
    
    # Option 3: Google Drive
    print("\n☁️ Option 3: Google Drive mount...")
    if mount_google_drive():
        print("✅ Google Drive mounted at /content/drive")
        print("   You can copy files using:")
        print("   !cp -r /content/AutomataNexus_Olympus_AGI2/arc_models_v4 /content/drive/MyDrive/")
    else:
        print("⚠️ Not running in Colab - Google Drive mount not available")
    
    print("\n" + "="*50)
    print("📥 DOWNLOAD INSTRUCTIONS:")
    print("="*50)
    
    if archive_path:
        print(f"\n1️⃣ ZIP Archive: {archive_path}")
        print("   In Colab: Files → downloads → olympus_models_*.zip")
        print("   Or run: from google.colab import files; files.download('{archive_path}')")
    
    if files:
        print(f"\n2️⃣ Individual Files: {download_dir}")
        print("   Files available:")
        for f in files:
            print(f"   - {f}")
    
    print("\n3️⃣ Alternative methods:")
    print("   - Use Google Drive (if mounted)")
    print("   - Use wget/curl from another machine")
    print("   - Use gdown for Google Drive links")
    print("   - Use Colab's file browser (left sidebar)")
    
    print("\n💡 If download fails, try:")
    print("   - Refresh the Colab page")
    print("   - Use a different browser")
    print("   - Download smaller files individually")
    print("   - Use Google Drive for large files")