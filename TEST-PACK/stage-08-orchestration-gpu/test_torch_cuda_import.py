"""
Test script to verify Torch CUDA stack imports cleanly in the packaged environment.

This script tests for the WinError 127 cudnn64_9.dll issue that can occur when:
1. Torch CUDA DLLs are not properly loaded before other frameworks
2. cuDNN version mismatches between torch and nvidia-cudnn-cu12 wheel
3. Sentence transformers or other torch-based libraries conflict with paddle

Run this in the packaged runtime environment to verify GPU imports work.
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def test_torch_cuda_import():
    """Test basic torch CUDA import and initialization."""
    logging.info("=" * 60)
    logging.info("TEST 1: Basic Torch CUDA Import")
    logging.info("=" * 60)
    
    try:
        import torch
        logging.info("✓ torch imported successfully")
        logging.info(f"  Torch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        logging.info(f"  CUDA available: {cuda_available}")
        
        if cuda_available:
            logging.info(f"  CUDA version: {torch.version.cuda}")
            logging.info(f"  cuDNN version: {torch.backends.cudnn.version()}")
            logging.info(f"  GPU count: {torch.cuda.device_count()}")
            logging.info(f"  GPU 0 name: {torch.cuda.get_device_name(0)}")
            
            # Test actual CUDA operation
            device = torch.device("cuda:0")
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.matmul(test_tensor, test_tensor)
            logging.info(f"  ✓ CUDA matmul test passed (result shape: {result.shape})")
            del test_tensor, result
            torch.cuda.empty_cache()
        else:
            logging.warning("  CUDA not available - GPU tests skipped")
        
        return True
    except Exception as e:
        logging.error(f"✗ Torch CUDA import failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def test_sentence_transformers_import():
    """Test sentence-transformers import (uses torch backend)."""
    logging.info("=" * 60)
    logging.info("TEST 2: Sentence Transformers Import")
    logging.info("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        logging.info("✓ sentence_transformers imported successfully")
        
        # Don't actually load a model (too slow), just verify import works
        logging.info("  Import successful - model loading skipped for speed")
        return True
    except Exception as e:
        logging.error(f"✗ Sentence transformers import failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def test_paddle_after_torch():
    """Test paddle import after torch (checks for DLL conflicts)."""
    logging.info("=" * 60)
    logging.info("TEST 3: Paddle Import After Torch")
    logging.info("=" * 60)
    
    try:
        import paddle
        logging.info("✓ paddle imported successfully after torch")
        logging.info(f"  Paddle version: {paddle.__version__}")
        logging.info(f"  Paddle CUDA compiled: {paddle.is_compiled_with_cuda()}")
        
        if paddle.is_compiled_with_cuda():
            device_count = paddle.device.cuda.device_count()
            logging.info(f"  Paddle GPU count: {device_count}")
        
        return True
    except Exception as e:
        logging.error(f"✗ Paddle import after torch failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def test_cudnn_dll_loading():
    """Verify cuDNN DLL is loaded correctly."""
    logging.info("=" * 60)
    logging.info("TEST 4: cuDNN DLL Loading Verification")
    logging.info("=" * 60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            logging.warning("  CUDA not available - cuDNN test skipped")
            return True
        
        # Try to use cuDNN explicitly
        cudnn_enabled = torch.backends.cudnn.enabled
        cudnn_version = torch.backends.cudnn.version()
        
        logging.info(f"  cuDNN enabled: {cudnn_enabled}")
        logging.info(f"  cuDNN version: {cudnn_version}")
        
        # Test cuDNN operation (convolution uses cuDNN)
        import torch.nn as nn
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
        test_input = torch.randn(1, 3, 224, 224).cuda()
        output = conv(test_input)
        logging.info(f"  ✓ cuDNN conv2d test passed (output shape: {output.shape})")
        
        del conv, test_input, output
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logging.error(f"✗ cuDNN DLL loading failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def main():
    logging.info("Torch CUDA Stack Import Test")
    logging.info(f"Python: {sys.version}")
    logging.info(f"Platform: {sys.platform}")
    logging.info("")
    
    results = {}
    results["torch_cuda"] = test_torch_cuda_import()
    results["sentence_transformers"] = test_sentence_transformers_import()
    results["paddle_after_torch"] = test_paddle_after_torch()
    results["cudnn_dll"] = test_cudnn_dll_loading()
    
    logging.info("=" * 60)
    logging.info("TEST SUMMARY")
    logging.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logging.info(f"{test_name:25s} {status}")
        if not passed:
            all_passed = False
    
    logging.info("=" * 60)
    if all_passed:
        logging.info("✓ ALL TESTS PASSED - Torch CUDA stack is consistent")
        return 0
    else:
        logging.error("✗ SOME TESTS FAILED - Torch CUDA stack has issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
