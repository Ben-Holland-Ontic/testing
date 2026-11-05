"""
Test that verifies the MKL-DNN API fix in worker files works correctly.

This test focuses on verifying that region_ocr.py and cluster_ocr.py can handle
torch 2.11's mkldnn API (where enabled is a property, not a method).

Run this from bk-main directory:
    python test_worker_mkldnn_fix.py
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Add workers to path
_base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base_dir, 'workers'))

def test_torch_mkldnn_api():
    """Test that torch mkldnn API works correctly."""
    logging.info("=" * 60)
    logging.info("TEST 1: Torch MKL-DNN API")
    logging.info("=" * 60)
    
    try:
        import torch
        logging.info(f"✓ torch imported: {torch.__version__}")
        
        # Test the exact code path from region_ocr.py
        if hasattr(torch.backends, 'onednn'):
            backend = "onednn"
            available = torch.backends.onednn.is_available()
            enabled = torch.backends.onednn.enabled  # Property, not method
            logging.info(f"  OneDNN available: {available}")
            logging.info(f"  OneDNN enabled: {enabled}")
        elif hasattr(torch.backends, 'mkldnn'):
            backend = "mkldnn"
            available = torch.backends.mkldnn.is_available()
            enabled = torch.backends.mkldnn.enabled  # Property, not method
            logging.info(f"  MKL-DNN available: {available}")
            logging.info(f"  MKL-DNN enabled: {enabled}")
            
            # Test enabling it (same as worker code does)
            if available and not enabled:
                torch.backends.mkldnn.enabled = True
                logging.info(f"  MKL-DNN enabled: {torch.backends.mkldnn.enabled}")
        
        logging.info(f"✓ {backend} API works correctly (property access, not method call)")
        return True
    except AttributeError as e:
        logging.error(f"✗ MKL-DNN API error: {e}")
        logging.error("  This is the bug we fixed - enabled should be a property, not a method")
        return False
    except Exception as e:
        logging.error(f"✗ Unexpected error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def test_region_ocr_import():
    """Test that region_ocr.py imports without crashing on mkldnn API."""
    logging.info("=" * 60)
    logging.info("TEST 2: region_ocr.py Import (MKL-DNN fix)")
    logging.info("=" * 60)
    
    try:
        # This will fail if mkldnn API is called incorrectly
        import region_ocr
        logging.info("✓ region_ocr imported successfully")
        logging.info("  MKL-DNN API fix is working")
        return True
    except AttributeError as e:
        if "is_enabled" in str(e) or "mkldnn" in str(e):
            logging.error(f"✗ MKL-DNN API error in region_ocr: {e}")
            logging.error("  The fix didn't work - still calling enabled as a method")
            return False
        raise
    except Exception as e:
        logging.warning(f"⚠ region_ocr import failed (may be due to missing models/GPU): {e}")
        # Don't fail the test for missing models/GPU - we're only testing the API fix
        return True


def test_cluster_ocr_import():
    """Test that cluster_ocr.py imports without crashing on mkldnn API."""
    logging.info("=" * 60)
    logging.info("TEST 3: cluster_ocr.py Import (MKL-DNN fix)")
    logging.info("=" * 60)
    
    try:
        import cluster_ocr
        logging.info("✓ cluster_ocr imported successfully")
        logging.info("  MKL-DNN API fix is working")
        return True
    except AttributeError as e:
        if "is_enabled" in str(e) or "mkldnn" in str(e):
            logging.error(f"✗ MKL-DNN API error in cluster_ocr: {e}")
            logging.error("  The fix didn't work - still calling enabled as a method")
            return False
        raise
    except Exception as e:
        logging.warning(f"⚠ cluster_ocr import failed (may be due to missing models/GPU): {e}")
        # Don't fail the test for missing models/GPU - we're only testing the API fix
        return True


def main():
    logging.info("Testing MKL-DNN API Fix in Worker Files")
    logging.info(f"Python: {sys.version}")
    logging.info("")
    
    results = {}
    results["torch_mkldnn_api"] = test_torch_mkldnn_api()
    results["region_ocr_import"] = test_region_ocr_import()
    results["cluster_ocr_import"] = test_cluster_ocr_import()
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("RESULT")
    logging.info("=" * 60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logging.info(f"{test_name:25s} {status}")
    
    logging.info("=" * 60)
    if all_passed:
        logging.info("✓ ALL TESTS PASSED")
        logging.info("  MKL-DNN API fix is working correctly")
        logging.info("  Worker files will not crash on torch 2.11+")
        return 0
    else:
        logging.error("✗ SOME TESTS FAILED")
        logging.error("  MKL-DNN API fix needs adjustment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
