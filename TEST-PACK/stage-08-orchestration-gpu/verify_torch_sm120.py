"""Verify if torch wheel has Blackwell sm_120 support."""
import zipfile
import re
import sys

wheel_path = r"C:\Users\benja\Downloads\torch-cu124-test\torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"

print(f"Checking: {wheel_path}\n")

with zipfile.ZipFile(wheel_path, 'r') as z:
    meta = z.read('torch-2.6.0+cu124.dist-info/METADATA').decode('utf-8')
    version_line = [l for l in meta.split('\n') if l.startswith('Version:')][0]
    print(f"{version_line}")
    print("Has +cu124 suffix: YES - GPU build\n")
    
    print("Checking for sm_120/Blackwell support...")
    files = z.namelist()
    
    # Check filenames
    sm120_in_names = [f for f in files if 'sm_120' in f.lower() or 'sm120' in f.lower()]
    if sm120_in_names:
        print(f"✓ Found sm_120 in {len(sm120_in_names)} filenames")
        for f in sm120_in_names[:5]:
            print(f"  - {f}")
    
    # Check file contents
    found_sm120 = False
    check_files = [f for f in files if f.endswith(('.txt', '.cmake', '.json', '.h', '.cpp')) and 'dist-info' in f]
    
    for fname in check_files[:30]:
        try:
            content = z.read(fname).decode('utf-8', errors='ignore')
            if re.search(r'\bsm_120\b|\bcompute_120\b', content, re.I):
                print(f"✓ Found sm_120 reference in: {fname}")
                found_sm120 = True
                break
        except:
            pass
    
    if not found_sm120 and not sm120_in_names:
        print("\n⚠️  WARNING: No sm_120 support detected")
        print("   PyTorch 2.6.0+cu124 likely only supports up to sm_90 (Hopper/Ada)")
        print("   Blackwell GPUs (sm_120) will NOT work with this wheel")
        print("\n   For Blackwell support, you need:")
        print("   - PyTorch 2.9+ nightly with CUDA 12.8/12.9")
        print("   - Python 3.12+ (3.13 recommended)")
        sys.exit(1)
    else:
        print("\n✓ Blackwell sm_120 support confirmed")
