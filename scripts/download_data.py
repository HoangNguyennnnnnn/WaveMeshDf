"""
Download và setup ModelNet40 dataset
Quick start cho testing
"""
import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_modelnet40(data_dir='./data'):
    """
    Download ModelNet40 dataset (~500MB)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    zip_path = data_dir / "ModelNet40.zip"
    extract_path = data_dir / "ModelNet40"
    
    if extract_path.exists():
        print(f"✅ ModelNet40 đã tồn tại tại {extract_path}")
        return extract_path
    
    # Download
    print(f"📥 Downloading ModelNet40 từ {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='ModelNet40') as t:
        urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    
    print(f"✅ Downloaded to {zip_path}")
    
    # Extract
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print(f"✅ Extracted to {extract_path}")
    
    # Clean up
    zip_path.unlink()
    print("🗑️  Cleaned up zip file")
    
    # Show stats
    # ModelNet40 structure: categories directly, each has train/ and test/ subfolders
    categories = [d.name for d in extract_path.iterdir() if d.is_dir()]
    
    train_files = []
    test_files = []
    for cat in categories:
        cat_path = extract_path / cat
        if (cat_path / "train").exists():
            train_files.extend(list((cat_path / "train").rglob("*.off")))
        if (cat_path / "test").exists():
            test_files.extend(list((cat_path / "test").rglob("*.off")))
    
    print("\n" + "="*60)
    print("📊 ModelNet40 Statistics:")
    print("="*60)
    print(f"Training samples: {len(train_files)}")
    print(f"Test samples: {len(test_files)}")
    print(f"Total: {len(train_files) + len(test_files)}")
    print(f"Categories ({len(categories)}): {', '.join(sorted(categories)[:10])}...")
    print("="*60)
    
    return extract_path

def download_shapenet_instructions():
    """
    ShapeNet cần đăng ký manual, in hướng dẫn
    """
    print("\n" + "="*60)
    print("📋 HƯỚNG DẪN DOWNLOAD SHAPENET")
    print("="*60)
    print("""
ShapeNet là dataset lớn hơn và chất lượng cao hơn ModelNet40.

BƯỚC 1: Đăng ký tài khoản
    1. Truy cập: https://shapenet.org/
    2. Click "Sign Up" và tạo tài khoản
    3. Đợi email xác nhận (thường 1-2 ngày)

BƯỚC 2: Download
    1. Login vào https://shapenet.org/
    2. Vào Downloads → ShapeNetCore.v2
    3. Download file (chọn categories bạn cần):
       - Full dataset: ~50GB
       - Single category (e.g., chairs): ~2-5GB
    
BƯỚC 3: Giải nén
    unzip ShapeNetCore.v2.zip -d ./data/
    
BƯỚC 4: Cấu trúc thư mục
    data/
    └── ShapeNetCore.v2/
        ├── 02691156/  # airplane
        ├── 02958343/  # car
        ├── 03001627/  # chair
        └── ...

CATEGORIES PHỔ BIẾN:
    - 03001627: Chair (~7K models)
    - 02958343: Car (~8K models)
    - 02691156: Airplane (~4K models)
    - 04379243: Table (~9K models)
    - 02828884: Bench (~2K models)

TIP: Bắt đầu với 1 category để test nhanh!
    """)
    print("="*60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['modelnet40', 'shapenet'], 
                       default='modelnet40', help='Dataset to download')
    parser.add_argument('--data_dir', default='./data', help='Data directory')
    args = parser.parse_args()
    
    if args.dataset == 'modelnet40':
        download_modelnet40(args.data_dir)
    else:
        download_shapenet_instructions()
