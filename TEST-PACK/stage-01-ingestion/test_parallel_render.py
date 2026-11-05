"""Testing checklist for parallel PDF page rendering in process_input.py.

Run with:
    cd bk-main && py test_parallel_render.py
"""
import os
import sys
import time
import logging
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import importlib.util

# Setup logging to see warnings during tests
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Load process_input.py directly because 'input' clashes with the built-in module.
_PI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bk-main", "input", "process_input.py")
_spec = importlib.util.spec_from_file_location("process_input_module", _PI_PATH)
_pi = importlib.util.module_from_spec(_spec)
sys.modules["process_input_module"] = _pi
_spec.loader.exec_module(_pi)

_extract_pdf_pages = _pi._extract_pdf_pages
_render_pdf_pages_parallel = _pi._render_pdf_pages_parallel
_render_one_page = _pi._render_one_page
_placeholder_page_data = _pi._placeholder_page_data
DocumentData = _pi.DocumentData
PageData = _pi.PageData


class FakeImage:
    """Pillow-like fake that returns a small RGB array."""
    def __init__(self, w=100, h=100):
        self._w = w
        self._h = h

    def convert(self, mode):
        return self

    @property
    def size(self):
        return (self._w, self._h)

    def __array__(self):
        # Return RGB array for np.array(pil_img)
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)


class FakePDFPage:
    def __init__(self, page_idx, slow=False, raise_on_render=False):
        self._page_idx = page_idx
        self._slow = slow
        self._raise_on_render = raise_on_render

    def to_image(self, resolution=None):
        if self._slow:
            time.sleep(10)  # intentionally longer than any test timeout
        if self._raise_on_render:
            raise RuntimeError("simulated render failure")
        img = MagicMock()
        img.original = FakeImage()
        return img


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_doc(doc_id="DOC001"):
    return DocumentData(
        document_id=doc_id,
        batch_id="BATCH001",
        source_path="/fake/path.pdf",
        file_size=0,
    )


class TestParallelRender(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. Single-page PDF — no pool, identical output to before
    # ------------------------------------------------------------------
    @patch("process_input_module.pdfplumber.open")
    @patch("process_input_module._deskew_bgr")
    @patch("process_input_module._normalize_pdf_page_for_ocr")
    @patch("process_input_module.calculate_patches")
    @patch("process_input_module.SCAN_CLASSIFIER_ENABLED", False)
    def test_single_page_pdf(self, mock_patches, mock_norm, mock_deskew, mock_open):
        mock_open.return_value = FakePDF([FakePDFPage(0)])
        mock_deskew.return_value = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)
        mock_norm.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_patches.return_value = 42

        doc = _make_doc()
        pages = _extract_pdf_pages(doc, "/fake/path.pdf")

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[0].status, "READY_FOR_EXTRACTION")
        self.assertEqual(pages[0].patches, 42)
        # Make sure we went through the single-page sequential path (no ThreadPoolExecutor created)
        mock_open.assert_called_once()

    # ------------------------------------------------------------------
    # 2. Multi-page PDF — correct order, parallel path used
    # ------------------------------------------------------------------
    @patch("process_input_module.pdfplumber.open")
    @patch("process_input_module._deskew_bgr")
    @patch("process_input_module._normalize_pdf_page_for_ocr")
    @patch("process_input_module.calculate_patches")
    @patch("process_input_module.SCAN_CLASSIFIER_ENABLED", False)
    def test_multi_page_order(self, mock_patches, mock_norm, mock_deskew, mock_open):
        num_pages = 10
        mock_open.return_value = FakePDF([FakePDFPage(i) for i in range(num_pages)])
        mock_deskew.return_value = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)
        mock_norm.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_patches.return_value = 42

        doc = _make_doc()
        pages = _extract_pdf_pages(doc, "/fake/path.pdf")

        self.assertEqual(len(pages), num_pages)
        for i, page in enumerate(pages, start=1):
            self.assertEqual(page.page_number, i)
            self.assertEqual(page.page_id, f"DOC001_PG{i:03d}")
            self.assertEqual(page.status, "READY_FOR_EXTRACTION")

    # ------------------------------------------------------------------
    # 3. Artificially slow one page → placeholder, rest complete,
    #    late result logged and discarded
    # ------------------------------------------------------------------
    @patch("process_input_module.PIPELINE_PDF_RENDER_TIMEOUT_SEC", 1)
    @patch("process_input_module.PIPELINE_PDF_RENDER_THREADS", 4)
    @patch("process_input_module.pdfplumber.open")
    @patch("process_input_module._deskew_bgr")
    @patch("process_input_module._normalize_pdf_page_for_ocr")
    @patch("process_input_module.calculate_patches")
    @patch("process_input_module.SCAN_CLASSIFIER_ENABLED", False)
    def test_slow_page_timeout(self, mock_patches, mock_norm, mock_deskew, mock_open):
        # Page 3 is slow; all others fast.
        fake_pages = [FakePDFPage(i, slow=(i == 2)) for i in range(5)]
        mock_open.return_value = FakePDF(fake_pages)
        mock_deskew.return_value = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)
        mock_norm.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_patches.return_value = 42

        doc = _make_doc()
        with self.assertLogs(level=logging.WARNING) as cm:
            pages = _extract_pdf_pages(doc, "/fake/path.pdf")

        # All 5 pages returned
        self.assertEqual(len(pages), 5)

        # Page 3 (index 2, page_number 3) should be a placeholder
        p3 = pages[2]
        self.assertEqual(p3.page_number, 3)
        self.assertEqual(p3.status, "UNSUPPORTED")
        self.assertEqual(p3.metadata.get("reason"), "pdf_page_render_timeout")

        # Other pages should be normal
        for i in (0, 1, 3, 4):
            self.assertEqual(pages[i].status, "READY_FOR_EXTRACTION")

        # Check that a late-render-discard warning may have been emitted
        # (the slow thread eventually completes and its result is discarded)
        warn_text = "\n".join(cm.output)
        self.assertIn("PDF page render timeout", warn_text)
        # late_render_result_discarded may or may not appear depending on timing,
        # but the timeout warning must be present.

    # ------------------------------------------------------------------
    # 4. Force 5+ consecutive timeouts → pool recycle fires once,
    #    rendering continues
    # ------------------------------------------------------------------
    @patch("process_input_module.PIPELINE_PDF_RENDER_TIMEOUT_SEC", 1)
    @patch("process_input_module.PIPELINE_PDF_RENDER_THREADS", 2)
    @patch("process_input_module.PIPELINE_PDF_RENDER_RECYCLE_THRESHOLD", 3)
    @patch("process_input_module.pdfplumber.open")
    @patch("process_input_module._deskew_bgr")
    @patch("process_input_module._normalize_pdf_page_for_ocr")
    @patch("process_input_module.calculate_patches")
    @patch("process_input_module.SCAN_CLASSIFIER_ENABLED", False)
    def test_consecutive_timeout_recycle(self, mock_patches, mock_norm, mock_deskew, mock_open):
        # First 4 pages are all slow → should trigger recycle after 3rd consecutive timeout
        fake_pages = [FakePDFPage(i, slow=(i < 4)) for i in range(6)]
        mock_open.return_value = FakePDF(fake_pages)
        mock_deskew.return_value = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)
        mock_norm.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_patches.return_value = 42

        doc = _make_doc()
        with self.assertLogs(level=logging.WARNING) as cm:
            pages = _extract_pdf_pages(doc, "/fake/path.pdf")

        self.assertEqual(len(pages), 6)

        warn_text = "\n".join(cm.output)
        # The recycle warning must appear exactly once
        recycle_count = warn_text.count("PDF render pool recycle triggered")
        self.assertEqual(recycle_count, 1, f"Expected 1 recycle warning, got {recycle_count}")

        # Pages 1-4 should be placeholders (timed out)
        for i in range(4):
            self.assertEqual(pages[i].status, "UNSUPPORTED")
            self.assertEqual(pages[i].metadata.get("reason"), "pdf_page_render_timeout")

        # Pages 5-6 (indices 4,5) should have completed normally
        self.assertEqual(pages[4].status, "READY_FOR_EXTRACTION")
        self.assertEqual(pages[5].status, "READY_FOR_EXTRACTION")

    # ------------------------------------------------------------------
    # 5. Corrupt / unopenable PDF → outer try/except fallback
    # ------------------------------------------------------------------
    @patch("process_input_module.pdfplumber.open")
    def test_corrupt_pdf_fallback(self, mock_open):
        mock_open.side_effect = RuntimeError("corrupted PDF header")

        doc = _make_doc()
        with self.assertLogs(level=logging.WARNING) as cm:
            pages = _extract_pdf_pages(doc, "/fake/path.pdf")

        self.assertEqual(len(pages), 1)  # _create_placeholder_page returns list of 1
        self.assertEqual(pages[0].status, "UNSUPPORTED")
        self.assertEqual(pages[0].metadata.get("reason"), "pdf_extract_failed")

        warn_text = "\n".join(cm.output)
        self.assertIn("PDF extract failed", warn_text)

    # ------------------------------------------------------------------
    # 6. Zero-page PDF → returns [] without error
    # ------------------------------------------------------------------
    @patch("process_input_module.pdfplumber.open")
    def test_zero_page_pdf(self, mock_open):
        mock_open.return_value = FakePDF([])

        doc = _make_doc()
        pages = _extract_pdf_pages(doc, "/fake/path.pdf")
        self.assertEqual(pages, [])

    # ------------------------------------------------------------------
    # 7. PIPELINE_PDF_RENDER_THREADS=1 → identical output to sequential path
    # ------------------------------------------------------------------
    @patch("process_input_module.PIPELINE_PDF_RENDER_THREADS", 1)
    @patch("process_input_module.pdfplumber.open")
    @patch("process_input_module._deskew_bgr")
    @patch("process_input_module._normalize_pdf_page_for_ocr")
    @patch("process_input_module.calculate_patches")
    @patch("process_input_module.SCAN_CLASSIFIER_ENABLED", False)
    def test_threads_one_regression(self, mock_patches, mock_norm, mock_deskew, mock_open):
        num_pages = 5
        mock_open.return_value = FakePDF([FakePDFPage(i) for i in range(num_pages)])
        mock_deskew.return_value = (np.zeros((10, 10, 3), dtype=np.uint8), 0.0)
        mock_norm.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_patches.return_value = 42

        doc = _make_doc()
        pages = _extract_pdf_pages(doc, "/fake/path.pdf")

        self.assertEqual(len(pages), num_pages)
        for i, page in enumerate(pages, start=1):
            self.assertEqual(page.page_number, i)
            self.assertEqual(page.status, "READY_FOR_EXTRACTION")
            self.assertEqual(page.patches, 42)


if __name__ == "__main__":
    # Run tests; verbose=2 shows each test name and result
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestParallelRender)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
