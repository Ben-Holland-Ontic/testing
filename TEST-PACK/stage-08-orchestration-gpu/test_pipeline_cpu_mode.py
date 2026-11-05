"""
Test that pipeline.py correctly enforces PADDLEOCR_USE_GPU=0 for CPU mode
and that both PaddleOCR and DocTR workers can be imported without crashes.
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def test_pipeline_cpu_mode():
    """Test that pipeline enforces CPU mode for PaddleOCR."""
    logging.info("=" * 70)
    logging.info("PIPELINE CPU MODE TEST")
    logging.info("=" * 70)
    logging.info("")
    
    # Step 1: Import pipeline (this calls _apply_runtime_config)
    logging.info("STEP 1: Import pipeline.py")
    logging.info("-" * 70)
    
    try:
        import pipeline
        logging.info("✓ pipeline imported successfully")
    except Exception as e:
        logging.error(f"✗ Pipeline import failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    logging.info("")
    
    # Step 2: Verify PADDLEOCR_USE_GPU is set to 0
    logging.info("STEP 2: Verify PADDLEOCR_USE_GPU=0 is enforced")
    logging.info("-" * 70)
    
    paddle_gpu_setting = os.environ.get("PADDLEOCR_USE_GPU", "NOT_SET")
    logging.info(f"  PADDLEOCR_USE_GPU = '{paddle_gpu_setting}'")
    
    if paddle_gpu_setting == "0":
        logging.info("✓ PaddleOCR CPU mode is enforced")
    else:
        logging.error(f"✗ PADDLEOCR_USE_GPU should be '0' but is '{paddle_gpu_setting}'")
        return False
    
    logging.info("")
    
    # Step 3: Verify worker imports work
    logging.info("STEP 3: Verify worker imports (cluster_ocr, region_ocr)")
    logging.info("-" * 70)
    
    try:
        # These are imported by pipeline.py at module level
        from cluster_ocr import process_rectangles, prewarm_cluster_ocr
        from region_ocr import data_extraction_in_memory, prewarm_region_doctr
        logging.info("✓ cluster_ocr imported successfully")
        logging.info("✓ region_ocr imported successfully")
    except Exception as e:
        logging.error(f"✗ Worker import failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False
    
    logging.info("")
    
    # Step 4: Verify PaddleOCR will use CPU
    logging.info("STEP 4: Verify PaddleOCR GPU detection")
    logging.info("-" * 70)
    
    try:
        from cluster_ocr import _gpu_available
        gpu_available = _gpu_available()
        if gpu_available:
            logging.error("✗ PaddleOCR GPU is available (should be disabled)")
            return False
        else:
            logging.info("✓ PaddleOCR GPU is disabled (CPU mode active)")
    except Exception as e:
        logging.error(f"✗ GPU check failed: {e}")
        return False
    
    logging.info("")
    
    # Step 5: Verify torch can still use GPU
    logging.info("STEP 5: Verify torch (DocTR) can still use GPU")
    logging.info("-" * 70)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        logging.info(f"  torch.cuda.is_available() = {cuda_available}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"✓ Torch GPU available: {gpu_name}")
            logging.info("  DocTR can use GPU for region extraction")
        else:
            logging.info("⚠ Torch GPU not available (CPU-only system or no CUDA)")
            logging.info("  DocTR will use CPU (expected on non-GPU systems)")
    except Exception as e:
        logging.error(f"✗ Torch GPU check failed: {e}")
        return False
    
    logging.info("")
    logging.info("=" * 70)
    logging.info("✓ ALL TESTS PASSED")
    logging.info("=" * 70)
    logging.info("Summary:")
    logging.info("  - PaddleOCR forced to CPU mode (PADDLEOCR_USE_GPU=0)")
    logging.info("  - Worker imports successful (no MKL-DNN crashes)")
    logging.info("  - Torch GPU available for DocTR (if GPU present)")
    logging.info("  - No CUDA DLL conflicts")
    logging.info("=" * 70)
    return True


if __name__ == "__main__":
    success = test_pipeline_cpu_mode()
    sys.exit(0 if success else 1)
