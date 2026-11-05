"""
Check if torch wheel has Blackwell sm_120 support.
PyTorch 2.7+ with CUDA 12.6+ should include sm_120 (Blackwell) compute capability.
"""
import sys
import zipfile
from pathlib import Path

wheel_path = Path(r"C:\Users\benja\Downloads\bk-main\bk-main\wheels\torch-2.7.0+cu126-cp311-cp311-win_amd64.whl")

if not wheel_path.exists():
    print(f"ERROR: Wheel not found at {wheel_path}")
    sys.exit(1)

print(f"Checking: {wheel_path.name}")
print(f"Size: {wheel_path.stat().st_size / (1024**3):.2f} GB\n")

with zipfile.ZipFile(wheel_path, 'r') as z:
    # Check METADATA
    try:
        metadata = z.read('torch-2.7.0+cu126.dist-info/METADATA').decode('utf-8')
        for line in metadata.split('\n'):
            if any(k in line.lower() for k in ['version:', 'cuda', 'compute']):
                print(line)
    except KeyError:
        print("METADATA not found in expected location")
    
    print("\n" + "="*60)
    print("Checking for CUDA kernel images (compute capability binaries):")
    print("="*60)
    
    # Look for cubin files which contain the compiled kernels
    cubin_files = [f for f in z.namelist() if f.endswith('.cubin') or 'sm_' in f.lower()]
    
    if cubin_files:
        print(f"\nFound {len(cubin_files)} cubin/sm files:")
        for f in cubin_files[:20]:  # Show first 20
            print(f"  {f}")
        if len(cubin_files) > 20:
            print(f"  ... and {len(cubin_files) - 20} more")
    
    # Check for sm_120 references in any text files
    print("\n" + "="*60)
    print("Searching for 'sm_120' or 'sm_90' references in wheel:")
    print("="*60)
    
    sm120_found = False
    sm90_found = False
    
    for fname in z.namelist():
        if fname.endswith(('.txt', '.json', '.py', '.cmake', '.h', '.cpp')):
            try:
                content = z.read(fname).decode('utf-8', errors='ignore')
                if 'sm_120' in content.lower():
                    print(f"✓ sm_120 found in: {fname}")
                    sm120_found = True
                if 'sm_90' in content.lower() and not sm90_found:
                    print(f"  sm_90 found in: {fname}")
                    sm90_found = True
            except:
                pass
    
    if not sm120_found:
        print("\n⚠️  WARNING: No 'sm_120' references found in wheel!")
        print("   This wheel may NOT support Blackwell (sm_120) GPUs.")
    else:
        print("\n✓ sm_120 support detected - Blackwell compatible")
    
    # Check CUDA version in library files
    print("\n" + "="*60)
    print("CUDA library files in wheel:")
    print("="*60)
    
    cuda_libs = [f for f in z.namelist() if 'cuda' in f.lower() and f.endswith('.dll')]
    for lib in cuda_libs[:10]:
        print(f"  {lib}")
    if len(cuda_libs) > 10:
        print(f"  ... and {len(cuda_libs) - 10} more")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Wheel: torch-2.7.0+cu126")
print(f"Expected CUDA: 12.6")
print(f"Blackwell (sm_120) support: {'YES ✓' if sm120_found else 'NO ⚠️'}")
print("\nPyTorch 2.7.0 with CUDA 12.6 SHOULD include sm_120 support.")
print("If not detected, the wheel may be from an older build.")
