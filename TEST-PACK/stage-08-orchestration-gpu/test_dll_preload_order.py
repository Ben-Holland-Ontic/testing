"""
Test script to mimic the packaged app's DLL preloading order locally.

This simulates the exact startup sequence from pipeline.py to verify:
1. NVIDIA DLL directories are registered
2. Torch OpenMP is preloaded before paddle
3. Torch CUDA DLLs are preloaded before paddle
4. No WinError 127 or MKL-DNN API errors occur

Run this from bk-main directory:
    python test_dll_preload_order.py
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Add workers to path (same as pipeline.py does)
_base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base_dir, 'workers'))

def _register_nvidia_dll_dirs() -> None:
    """Register NVIDIA pip-wheel DLL directories (from pipeline.py)."""
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
                            logging.info(f"  Registered DLL dir: {_path}")
                        except Exception as e:
                            logging.debug(f"  Failed to register {_path}: {e}")
    except Exception as e:
        logging.warning(f"Failed to register NVIDIA DLL dirs: {e}")


def _preload_torch_openmp() -> None:
    """Preload torch's libiomp5md.dll before paddle (from pipeline.py)."""
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
                logging.info(f"  Preloaded torch OpenMP: {_omp}")
                break
    except Exception as e:
        logging.warning(f"Failed to preload torch OpenMP: {e}")


def _preload_torch_cuda_dlls() -> None:
    """Preload torch's CUDA DLLs before paddle (from pipeline.py)."""
    if os.name != "nt":
        return
    try:
        import site as _site
        for _sp in (_site.getsitepackages() or []):
            _torch_lib = os.path.join(_sp, "torch", "lib")
            if not os.path.isdir(_torch_lib):
                continue
            
            cuda_dlls = [
                "cudart64_12.dll",
                "cublas64_12.dll",
                "cublasLt64_12.dll",
                "cudnn64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_cnn64_9.dll",
            ]
            
            import ctypes as _ctypes
            loaded_count = 0
            for dll_name in cuda_dlls:
                dll_path = os.path.join(_torch_lib, dll_name)
                if os.path.isfile(dll_path):
                    try:
                        _ctypes.CDLL(dll_path)
                        logging.info(f"  Preloaded CUDA DLL: {dll_name}")
                        loaded_count += 1
                    except Exception as e:
                        logging.debug(f"  Failed to preload {dll_name}: {e}")
            
            if loaded_count > 0:
                logging.info(f"  Total CUDA DLLs preloaded: {loaded_count}")
            else:
                logging.info("  No CUDA DLLs found (CPU-only torch wheel)")
            break
    except Exception as e:
        logging.warning(f"Failed to preload torch CUDA DLLs: {e}")


def test_imports():
    """Test imports in the same order as the packaged app."""
    logging.info("=" * 60)
    logging.info("STEP 1: DLL Preloading (mimicking pipeline.py startup)")
    logging.info("=" * 60)
    
    _register_nvidia_dll_dirs()
    _preload_torch_openmp()
    _preload_torch_cuda_dlls()
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("STEP 2: Import Torch (DocTR backend)")
    logging.info("=" * 60)
    
    try:
        import torch
        logging.info(f"✓ torch imported: {torch.__version__}")
        logging.info(f"  CUDA available: {torch.cuda.is_available()}")
        
        # Test OneDNN/MKL-DNN API (this was the original crash)
        try:
            if hasattr(torch.backends, 'onednn'):
                logging.info(f"  OneDNN available: {torch.backends.onednn.is_available()}")
                if hasattr(torch.backends.onednn, 'is_enabled'):
                    logging.info(f"  OneDNN enabled: {torch.backends.onednn.is_enabled()}")
            elif hasattr(torch.backends, 'mkldnn'):
                logging.info(f"  MKL-DNN available: {torch.backends.mkldnn.is_available()}")
                if hasattr(torch.backends.mkldnn, 'is_enabled'):
                    logging.info(f"  MKL-DNN enabled: {torch.backends.mkldnn.is_enabled()}")
                else:
                    logging.warning("  MKL-DNN API incomplete (old torch version - would crash without fix)")
        except AttributeError as e:
            logging.error(f"  ✗ MKL-DNN/OneDNN API error (this is the bug we fixed): {e}")
            raise
        
    except Exception as e:
        logging.error(f"✗ Torch import failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("STEP 3: Import Paddle (PaddleOCR backend)")
    logging.info("=" * 60)
    
    try:
        import paddle
        logging.info(f"✓ paddle imported: {paddle.__version__}")
        logging.info(f"  CUDA compiled: {paddle.is_compiled_with_cuda()}")
    except Exception as e:
        logging.error(f"✗ Paddle import failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("STEP 4: Import Worker Modules (region_ocr, cluster_ocr)")
    logging.info("=" * 60)
    
    try:
        # This will trigger the OneDNN/MKL-DNN code paths
        from region_ocr import prewarm_region_doctr
        logging.info("✓ region_ocr imported (DocTR worker)")
        
        from cluster_ocr import prewarm_cluster_ocr
        logging.info("✓ cluster_ocr imported (PaddleOCR worker)")
        
    except Exception as e:
        logging.error(f"✗ Worker import failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("STEP 5: Test Worker Initialization (prewarm)")
    logging.info("=" * 60)
    
    try:
        logging.info("Prewarming DocTR (region_ocr)...")
        prewarm_region_doctr()
        logging.info("✓ DocTR prewarmed successfully")
    except Exception as e:
        logging.error(f"✗ DocTR prewarm failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    try:
        logging.info("Prewarming PaddleOCR (cluster_ocr)...")
        prewarm_cluster_ocr()
        logging.info("✓ PaddleOCR prewarmed successfully")
    except Exception as e:
        logging.error(f"✗ PaddleOCR prewarm failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    return True


def main():
    logging.info("Testing DLL Preload Order (Packaged App Simulation)")
    logging.info(f"Python: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info("")
    
    success = test_imports()
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("RESULT")
    logging.info("=" * 60)
    
    if success:
        logging.info("✓ ALL IMPORTS SUCCESSFUL")
        logging.info("  - No WinError 127 (CUDA DLL conflicts)")
        logging.info("  - No MKL-DNN/OneDNN API errors")
        logging.info("  - Workers initialized cleanly")
        return 0
    else:
        logging.error("✗ IMPORT FAILED - See errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
