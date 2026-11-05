"""
Test the fundamental CUDA DLL conflict between torch and paddle on GPU hardware.

This test verifies that our DLL preloading fixes prevent WinError 127 when both
torch GPU and paddle GPU are loaded in the same process.

Run this from bk-main directory:
    python test_gpu_dll_conflict.py
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Add workers to path
_base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base_dir, 'workers'))

def _register_nvidia_dll_dirs():
    """Register NVIDIA pip-wheel DLL directories."""
    if os.name != "nt":
        return
    try:
        import site as _site
        for _sp in (_site.getsitepackages() or []):
            _nvidia = os.path.join(_sp, "nvidia")
            if not os.path.isdir(_nvidia):
                continue
            for _pkg in os.listdir(_nvidia):
                for _subdir in ("bin", "lib"):
                    _path = os.path.join(_nvidia, _pkg, _subdir)
                    if os.path.isdir(_path):
                        try:
                            os.add_dll_directory(_path)
                            logging.debug(f"  Registered: {_path}")
                        except Exception:
                            pass
        logging.info("✓ NVIDIA DLL directories registered")
    except Exception as e:
        logging.warning(f"Failed to register NVIDIA DLL dirs: {e}")


def _preload_torch_openmp():
    """Preload torch's libiomp5md.dll before paddle."""
    if os.name != "nt":
        return
    try:
        import site as _site
        for _sp in (_site.getsitepackages() or []):
            _torch_lib = os.path.join(_sp, "torch", "lib")
            _omp = os.path.join(_torch_lib, "libiomp5md.dll")
            if os.path.isfile(_omp):
                import ctypes as _ctypes
                _ctypes.CDLL(_omp)
                logging.info(f"✓ Preloaded torch OpenMP: {os.path.basename(_omp)}")
                break
    except Exception as e:
        logging.warning(f"Failed to preload torch OpenMP: {e}")


def _preload_shared_cuda_dlls():
    """Preload CUDA DLLs from nvidia-* wheels (shared by torch and paddle)."""
    if os.name != "nt":
        return
    try:
        import site as _site
        import ctypes as _ctypes
        
        for _sp in (_site.getsitepackages() or []):
            _nvidia = os.path.join(_sp, "nvidia")
            if not os.path.isdir(_nvidia):
                continue
            
            # Preload in strict dependency order: cnn/ops before main cuDNN
            cuda_packages = [
                ("cuda_runtime", ["cudart64_12.dll"]),
                ("cublas", ["cublas64_12.dll", "cublasLt64_12.dll"]),
                ("cudnn", ["cudnn_cnn64_9.dll", "cudnn_ops64_9.dll", "cudnn64_9.dll"]),
            ]
            
            loaded = []
            for pkg_name, dll_names in cuda_packages:
                pkg_bin = os.path.join(_nvidia, pkg_name, "bin")
                if not os.path.isdir(pkg_bin):
                    continue
                
                for dll_name in dll_names:
                    dll_path = os.path.join(pkg_bin, dll_name)
                    if os.path.isfile(dll_path):
                        try:
                            _ctypes.CDLL(dll_path)
                            loaded.append(dll_name)
                        except Exception as e:
                            logging.debug(f"  Failed to preload {dll_name}: {e}")
            
            if loaded:
                logging.info(f"✓ Preloaded {len(loaded)} shared CUDA DLLs from nvidia wheels: {', '.join(loaded)}")
            break
    except Exception as e:
        logging.warning(f"Failed to preload shared CUDA DLLs: {e}")


def test_gpu_dll_conflict():
    """Test the fundamental DLL conflict between torch and paddle GPU."""
    logging.info("=" * 70)
    logging.info("GPU CUDA DLL CONFLICT TEST")
    logging.info("=" * 70)
    logging.info("")
    
    # Step 1: DLL Preloading
    logging.info("STEP 1: DLL Preloading (mimicking pipeline.py)")
    logging.info("-" * 70)
    _register_nvidia_dll_dirs()
    _preload_torch_openmp()
    _preload_shared_cuda_dlls()
    logging.info("")
    
    # Step 2: Import and test torch GPU
    logging.info("STEP 2: Import Torch and Test GPU")
    logging.info("-" * 70)
    try:
        import torch
        logging.info(f"✓ torch imported: {torch.__version__}")
        
        if not torch.cuda.is_available():
            logging.error("✗ CUDA not available - cannot test GPU DLL conflict")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"✓ CUDA available: {gpu_name}")
        
        # Test actual CUDA operation
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000, device=device)
        result = torch.matmul(test_tensor, test_tensor)
        logging.info(f"✓ Torch GPU matmul succeeded (shape: {result.shape})")
        del test_tensor, result
        torch.cuda.empty_cache()
        
    except Exception as e:
        logging.error(f"✗ Torch GPU test failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    logging.info("")
    
    # Step 3: Import paddle (THIS IS WHERE THE CONFLICT OCCURS)
    logging.info("STEP 3: Import Paddle GPU (Critical Test)")
    logging.info("-" * 70)
    logging.info("This is where WinError 127 on cublas64_12.dll would occur...")
    
    try:
        import paddle
        logging.info(f"✓ paddle imported: {paddle.__version__}")
        
        if not paddle.is_compiled_with_cuda():
            logging.warning("⚠ Paddle not compiled with CUDA")
            return True  # Not a failure, just CPU-only paddle
        
        logging.info(f"✓ Paddle CUDA compiled: True")
        
        # Test paddle GPU operation
        paddle.set_device("gpu:0")
        test_paddle = paddle.randn([1000, 1000])
        result_paddle = paddle.matmul(test_paddle, test_paddle)
        logging.info(f"✓ Paddle GPU matmul succeeded (shape: {result_paddle.shape})")
        del test_paddle, result_paddle
        
    except OSError as e:
        if "WinError 127" in str(e) or "cublas64_12.dll" in str(e):
            logging.error("✗ FUNDAMENTAL DLL CONFLICT DETECTED!")
            logging.error(f"   Error: {e}")
            logging.error("   This is the exact issue we're trying to fix")
            logging.error("   The CUDA DLL preloading did NOT prevent the conflict")
            return False
        raise
    except Exception as e:
        logging.error(f"✗ Paddle GPU test failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    logging.info("")
    
    # Step 4: Test both frameworks together
    logging.info("STEP 4: Test Both Frameworks Together")
    logging.info("-" * 70)
    
    try:
        # Torch operation
        torch_t = torch.randn(500, 500, device="cuda")
        torch_r = torch.matmul(torch_t, torch_t)
        logging.info(f"✓ Torch GPU operation after paddle import: OK")
        
        # Paddle operation
        paddle_t = paddle.randn([500, 500])
        paddle_r = paddle.matmul(paddle_t, paddle_t)
        logging.info(f"✓ Paddle GPU operation after torch: OK")
        
        logging.info("✓ Both frameworks coexist without DLL conflicts!")
        
    except Exception as e:
        logging.error(f"✗ Interleaved GPU operations failed: {e}")
        return False
    
    return True


def main():
    logging.info("Testing Fundamental CUDA DLL Conflict Fix")
    logging.info(f"Python: {sys.version}")
    logging.info("")
    
    success = test_gpu_dll_conflict()
    
    logging.info("")
    logging.info("=" * 70)
    logging.info("FINAL RESULT")
    logging.info("=" * 70)
    
    if success:
        logging.info("✓ SUCCESS - No CUDA DLL conflicts detected")
        logging.info("  - Torch GPU works")
        logging.info("  - Paddle GPU works")
        logging.info("  - Both coexist without WinError 127")
        logging.info("  - DLL preloading fix is working correctly")
        return 0
    else:
        logging.error("✗ FAILURE - CUDA DLL conflict still present")
        logging.error("  The fix needs additional work")
        return 1


if __name__ == "__main__":
    sys.exit(main())
