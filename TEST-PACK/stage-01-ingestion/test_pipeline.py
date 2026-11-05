import sys
import os
import shutil

# Test pipeline: same as main pipeline but runs on test_data. FILE ACCESS ONLY — no internet,
# no registry, no external applications (no .doc input in test_data, so no LibreOffice subprocess).
# Add required paths for imports (align with pipeline.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'input'))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'workers'))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'cleaners'))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'bootstrap'))

import logging
import json
import cv2
import numpy as np
from pathlib import Path

# Configure logging - DEBUG level for more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Debug output directory
DEBUG_OUTPUT_DIR = Path("debug_outputs")


def ensure_debug_dir():
    """Create debug output directory if it doesn't exist."""
    DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)
    return DEBUG_OUTPUT_DIR


def load_image(image_path: str) -> np.ndarray:
    """Load image from path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img


def save_debug_image(img: np.ndarray, name: str):
    """Save debug image to output directory."""
    output_path = DEBUG_OUTPUT_DIR / f"{name}.png"
    cv2.imwrite(str(output_path), img)
    print(f"    [DEBUG] Saved: {output_path}")


def to_json_safe(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    return obj


def draw_rectangles_overlay(img: np.ndarray, rect_data: dict, name: str = "rectangles"):
    """Draw detected rectangles on image."""
    overlay = img.copy()
    rectangles = rect_data.get('rectangles', [])
    
    # Count by source type
    line_based = 0
    contour_based = 0
    null_crops = 0
    
    for i, rect in enumerate(rectangles):
        bbox = rect.get('bbox', rect.get('Bbox', []))
        source = rect.get('source', 'unknown')
        crop = rect.get('crop')
        
        if crop is None:
            null_crops += 1
        
        if len(bbox) >= 4:
            # bbox format is (x, y, w, h) - convert to corners
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Color by source type
            if source == 'line_based':
                color = (0, 255, 0)  # Green for line-based
                line_based += 1
            else:
                color = (255, 0, 255)  # Magenta for contour-based
                contour_based += 1
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
    
    # Add legend
    cv2.putText(overlay, f"Line-based: {line_based}", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(overlay, f"Contour-based: {contour_based}", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(overlay, f"Null crops: {null_crops}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save first few crops for inspection
    crops_dir = DEBUG_OUTPUT_DIR / "crops"
    crops_dir.mkdir(exist_ok=True)
    for i, rect in enumerate(rectangles[:10]):  # First 10 crops
        crop = rect.get('crop')
        if crop is not None:
            cv2.imwrite(str(crops_dir / f"crop_{i:03d}.png"), crop)
    
    save_debug_image(overlay, name)
    return overlay


def draw_cluster_ocr_overlay(img: np.ndarray, cluster_results: dict, name: str = "cluster_ocr"):
    """Draw cluster OCR results on image with rectangles and text."""
    overlay = img.copy()
    cluster_lines = cluster_results.get('Cluster_Line', [])
    
    for line in cluster_lines:
        bbox = line.get('bbox', line.get('Bbox', []))
        text = line.get('text', line.get('Text', ''))
        
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Draw rectangle around the cluster line (green)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text if available
            if text:
                for j, line_text in enumerate(text.splitlines()[:3]):  # Max 3 lines
                    display_text = line_text[:40]
                    px, py = x1 + 2, y1 + 18 + j * 18
                    # Shadow for readability
                    cv2.putText(
                        overlay,
                        display_text,
                        (px + 1, py + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    # Main text (red)
                    cv2.putText(
                        overlay,
                        display_text,
                        (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
    
    save_debug_image(overlay, name)
    return overlay


def draw_region_ocr_overlay(img: np.ndarray, region_results: dict, name: str = "region_ocr"):
    """Draw region OCR results on image: blocks (outlines) and lines with text overlay."""
    overlay = img.copy()
    region_blocks = region_results.get('Region_Block', [])
    region_lines = region_results.get('Region_Line', [])

    # Draw blocks (thicker outline)
    for block in region_blocks:
        bbox = block.get('bbox', block.get('Bbox', []))
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 165, 0), 2)
            block_id = block.get('block_id', block.get('id', ''))
            if block_id:
                cv2.putText(overlay, str(block_id)[:12], (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

    for line in region_lines:
        bbox = line.get('bbox', line.get('Bbox', []))
        text = line.get('text', line.get('Text', ''))[:30]
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(overlay, text, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    save_debug_image(overlay, name)
    return overlay


def draw_classification_overlay(img: np.ndarray, cleaning_inputs, classification: dict,
                                consolidation: dict = None, name: str = "classification", save: bool = True):
    """Draw classification results (TABLE vs INFO_CLUSTER) on image."""
    overlay = img.copy()
    colors = {
        "TABLE": (0, 255, 0),        # Green
        "INFO_CLUSTER": (255, 165, 0),  # Orange
        "ABSTAIN": (128, 128, 128),  # Gray
    }
    clusters = classification.get('clusters', {})
    table_poly_ids = set()
    if consolidation:
        for table in consolidation.get('tables', []):
            for cell in table.get('cells', []):
                table_poly_ids.add(cell.get('poly_id'))
    for rect in cleaning_inputs.rectangles:
        poly_id = rect.poly_id
        bbox = rect.bbox
        cls_info = clusters.get(poly_id, {})
        label = cls_info.get('label', 'UNKNOWN')
        if consolidation and label == "TABLE" and poly_id not in table_poly_ids:
            label = "INFO_CLUSTER"
        color = colors.get(label, (128, 128, 128))
        feat = cleaning_inputs.cluster_features.get(poly_id, {})
        is_grid = feat.get('is_part_of_grid', False)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        thickness = 3 if is_grid else 1
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{label[:3]}" + ("*" if is_grid else "")
        cv2.putText(overlay, label_text, (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    y_offset = 20
    for label, color in colors.items():
        cv2.putText(overlay, f"{label}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += 20
    if save and name:
        save_debug_image(overlay, name)
    return overlay


def draw_region_format_overlay(img: np.ndarray, cleaning_inputs, classification: dict, name: str = "region_format", save: bool = True):
    """Draw region OCR format classification (list/para/string/key_value) on image."""
    overlay = img.copy()
    colors = {
        "KEY_VALUE": (255, 0, 255),   # Magenta
        "LIST": (0, 255, 255),        # Yellow
        "PARAGRAPH": (0, 165, 255),   # Orange
        "STRING": (255, 255, 0),      # Cyan
        "ABSTAIN": (128, 128, 128),   # Gray
    }
    blocks = classification.get("blocks", {})
    standalone = classification.get("standalone_lines", {})
    region_blocks = getattr(cleaning_inputs, "region_blocks", None) or []
    region_lines = getattr(cleaning_inputs, "region_lines", None) or []
    lines_in_blocks = set()
    for block in region_blocks:
        for line_id in getattr(block, "line_ids", []):
            lines_in_blocks.add(line_id)
    for block in region_blocks:
        bbox = getattr(block, "bbox", None)
        if not bbox or len(bbox) < 4:
            continue
        block_id = getattr(block, "id", None)
        cls_info = blocks.get(block_id, {}) if block_id else {}
        label = cls_info.get("label", "ABSTAIN")
        color = colors.get(label, (128, 128, 128))
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        short = label[:4] if label else "?"
        cv2.putText(overlay, short, (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    for line in region_lines:
        if getattr(line, "id", None) in lines_in_blocks:
            continue
        bbox = getattr(line, "bbox", None)
        if not bbox or len(bbox) < 4:
            continue
        line_id = getattr(line, "id", None)
        cls_info = standalone.get(line_id, {}) if line_id else {}
        label = cls_info.get("label", "ABSTAIN")
        color = colors.get(label, (128, 128, 128))
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        short = label[:4] if label else "?"
        cv2.putText(overlay, short, (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    y_off = 20
    for label, color in colors.items():
        cv2.putText(overlay, label, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_off += 20
    if save and name:
        save_debug_image(overlay, name)
    return overlay


def draw_table_consolidation_overlay(img: np.ndarray, consolidation: dict, name: str = "table_consolidation"):
    """Draw consolidated tables on image."""
    overlay = img.copy()
    
    tables = consolidation.get('tables', [])
    
    # Different colors for different tables
    table_colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    for t_idx, table in enumerate(tables):
        color = table_colors[t_idx % len(table_colors)]
        table_id = table.get('table_id', f'T{t_idx}')
        grid_shape = table.get('grid_shape', [0, 0])
        
        # Draw table bounding box
        bbox = table.get('bbox', [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            # Draw table ID and shape
            cv2.putText(overlay, f"{table_id} ({grid_shape[0]}x{grid_shape[1]})", 
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw individual cells
        for cell in table.get('cells', []):
            cell_bbox = cell.get('bbox', [])
            cell_text = cell.get('text', '')[:15]
            
            if len(cell_bbox) >= 4:
                cx1, cy1, cx2, cy2 = int(cell_bbox[0]), int(cell_bbox[1]), int(cell_bbox[2]), int(cell_bbox[3])
                cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), color, 1)
                # Draw cell text
                cv2.putText(overlay, cell_text, (cx1 + 2, cy1 + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Draw non-table clusters in gray
    non_tables = consolidation.get('non_table_clusters', [])
    for cluster in non_tables:
        bbox = cluster.get('bbox', [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), 1)
    
    save_debug_image(overlay, name)
    return overlay


def draw_anchors_overlay(img: np.ndarray, anchors: dict, name: str = "anchors"):
    """Draw query candidate locations: host cell or format highlighted in yellow (BGR 0,255,255)."""
    overlay = img.copy()
    anchor_list = anchors.get('anchors', [])
    yellow_bgr = (0, 255, 255)
    for anchor in anchor_list:
        bbox = anchor.get('bbox', [])
        query = anchor.get('query_name', '')[:10]
        score = anchor.get('score', 0)
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), yellow_bgr, 2)
            cv2.putText(overlay, f"{query}:{score:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, yellow_bgr, 1)
    
    save_debug_image(overlay, name)
    return overlay

def run_debug_pipeline(input_dir: str = None):
    """Run pipeline with debug output at each stage (first document, first page only)."""
    from process_input import process_input_local, get_batch_summary
    from pipeline import (
        process_page_extraction,
        process_page_data_cleaning,
    )
    
    if input_dir is None:
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    
    print(f"Running DEBUG pipeline on: {input_dir}")
    print("=" * 70)
    
    # Setup debug output directory
    ensure_debug_dir()
    print(f"Debug outputs will be saved to: {DEBUG_OUTPUT_DIR.absolute()}")
    
    # Stage 1: Ingest
    batch = process_input_local(input_dir)
    print(f"\n[STAGE 1] Ingestion: {get_batch_summary(batch)}")
    
    # Process first document only for debug
    doc = batch.documents[0]
    page = doc.pages[0]
    
    print(f"\n[DEBUG] Processing: {page.page_id}")
    print(f"  Source: {page.storage_path}")
    
    # Use in-memory page image (works for PDF, image, and converted text/csv/doc)
    img = np.asarray(page.image).copy()

    # Stage 2: Extraction
    print(f"\n[STAGE 2] Extraction...")
    page_result = process_page_extraction(page)

    print(f"\n  Rectangles detected: {len(page_result.rect_data.get('rectangles', [])) if page_result.rect_data else 0}")
    print(f"  Cluster lines: {len(page_result.cluster_results.get('Cluster_Line', [])) if page_result.cluster_results else 0}")
    print(f"  Region lines: {len(page_result.region_results.get('Region_Line', [])) if page_result.region_results else 0}")

    # Debug outputs: exactly 7 visualised stages to debug_outputs
    # 1. Rectangle detection
    draw_rectangles_overlay(img, page_result.rect_data or {"rectangles": []}, "01_rectangle_detection")
    # 2. Cluster OCR (text overlay)
    draw_cluster_ocr_overlay(img, page_result.cluster_results or {}, "02_cluster_ocr")
    cluster_lines = (page_result.cluster_results or {}).get('Cluster_Line', [])
    if cluster_lines:
        print(f"\n  Cluster OCR text (all lines):")
        for idx, line in enumerate(cluster_lines):
            text = line.get('text', line.get('Text', ''))
            print(f"    [{idx:03d}] {text}")
    # 3. Mask structured regions
    if page_result.masked_image is not None:
        save_debug_image(page_result.masked_image, "03_mask_structured_regions")
    else:
        save_debug_image(img.copy(), "03_mask_structured_regions")
    # 4. Region OCR (blocks and text overlay)
    draw_region_ocr_overlay(img, page_result.region_results or {}, "04_region_ocr")
    region_lines = (page_result.region_results or {}).get('Region_Line', [])
    if region_lines:
        print(f"\n  Region OCR text (all lines):")
        for idx, line in enumerate(region_lines):
            text = line.get('text', line.get('Text', ''))
            print(f"    [{idx:03d}] {text}")
    
    # Save raw OCR outputs (cluster vs region) with separated concerns
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    ocr_debug_dir = outputs_dir / "ocr_debug"
    ocr_debug_dir.mkdir(exist_ok=True)

    if page_result.cluster_results:
        raw_cluster = page_result.cluster_results or {}
        raw_lines = raw_cluster.get("Cluster_Line", [])
        raw_blocks = raw_cluster.get("Cluster_Block", [])

        cluster_lines_out = []
        for entry in raw_lines:
            text_raw = (entry.get("text_raw") or entry.get("text") or "").strip()
            if not text_raw:
                continue
            cluster_lines_out.append({
                "line_id": entry.get("line_id"),
                "text_raw": text_raw,
                "text_norm": entry.get("text_norm"),
                "bbox": entry.get("bbox"),
                "poly_id": entry.get("poly_id"),
                "cluster_block_id": entry.get("cluster_block_id"),
            })

        cluster_blocks_out = []
        for block in raw_blocks:
            text_raw = (block.get("text_raw") or block.get("text") or "").strip()
            if not text_raw:
                continue
            cluster_blocks_out.append({
                "block_id": block.get("block_id"),
                "text_raw": text_raw,
                "text_norm": block.get("text_norm"),
                "bbox": block.get("bbox"),
                "poly_id": block.get("poly_id"),
            })

        cluster_path = ocr_debug_dir / f"{page.document_id}_{page.page_id}_cluster_ocr.json"
        cluster_payload = {
            "batch_id": batch.batch_id,
            "document_id": page.document_id,
            "page_id": page.page_id,
            "cluster_lines": to_json_safe(cluster_lines_out),
            "cluster_blocks": to_json_safe(cluster_blocks_out),
        }
        with open(cluster_path, "w", encoding="utf-8") as f:
            json.dump(cluster_payload, f, indent=2, ensure_ascii=False)

    if page_result.region_results:
        region_path = ocr_debug_dir / f"{page.document_id}_{page.page_id}_region_ocr.json"
        region_payload = {
            "batch_id": batch.batch_id,
            "document_id": page.document_id,
            "page_id": page.page_id,
            "region_results": to_json_safe(page_result.region_results),
        }
        with open(region_path, "w", encoding="utf-8") as f:
            json.dump(region_payload, f, indent=2, ensure_ascii=False)
    
    # Stage 3: Data Cleaning
    print(f"\n[STAGE 3] Data Cleaning...")
    cleaned_result = process_page_data_cleaning(page, page_result)
    
    classification = cleaned_result.classification or {}
    consolidation = cleaned_result.consolidation or {}
    anchors = cleaned_result.anchors or {}
    
    print(f"\n  Classification:")
    print(f"    Clusters: {len(classification.get('clusters', {}))}")
    print(f"    Blocks: {len(classification.get('blocks', {}))}")
    print(f"    Standalone lines: {len(classification.get('standalone_lines', {}))}")
    
    # Show cluster labels and grid features
    if classification.get('clusters'):
        # Count TABLE vs INFO_CLUSTER
        table_count = sum(1 for v in classification['clusters'].values() if v.get('label') == 'TABLE')
        info_count = sum(1 for v in classification['clusters'].values() if v.get('label') == 'INFO_CLUSTER')
        abstain_count = sum(1 for v in classification['clusters'].values() if v.get('label') == 'ABSTAIN')
        print(f"\n  Classification summary: TABLE={table_count}, INFO_CLUSTER={info_count}, ABSTAIN={abstain_count}")
        
        # Show TABLE cells not in 9x5 grid
        print(f"\n  TABLE cells NOT in main grid:")
        for poly_id, info in classification['clusters'].items():
            if info.get('label') != 'TABLE':
                continue
            feat = cleaned_result.cleaning_inputs.cluster_features.get(poly_id, {})
            grid_size = f"{feat.get('grid_rows', 0)}x{feat.get('grid_cols', 0)}"
            if grid_size != "9x5":
                is_grid = feat.get('is_part_of_grid', False)
                n_lines = feat.get('n_lines_in_rect', 0)
                scores = info.get('scores', {})
                print(f"    - {poly_id}: grid={is_grid} ({grid_size}) | lines={n_lines}")
                print(f"        Scores: TABLE={scores.get('TABLE', 0):.1f}, INFO={scores.get('INFO_CLUSTER', 0):.1f}")
        
        print(f"\n  Cluster labels (first 20):")
        for i, (poly_id, info) in enumerate(list(classification['clusters'].items())[:20]):
            label = info.get('label', 'N/A')
            scores = info.get('scores', {})
            # Get features for this cluster
            feat = cleaned_result.cleaning_inputs.cluster_features.get(poly_id, {})
            is_grid = feat.get('is_part_of_grid', False)
            is_strong = feat.get('is_strong_grid', False)
            grid_size = f"{feat.get('grid_rows', 0)}x{feat.get('grid_cols', 0)}" if is_grid else "N/A"
            n_lines = feat.get('n_lines_in_rect', 0)
            height = feat.get('poly_height', 0)
            cols = feat.get('cols_hint', 0)
            rows = feat.get('row_count', 0)
            ruling = feat.get('has_ruling_lines', False)
            print(f"    - {poly_id}: {label} | grid={is_grid} strong={is_strong} ({grid_size}) | lines={n_lines} | h={height:.0f}")
            print(f"        Scores: TABLE={scores.get('TABLE', 0):.1f}, INFO={scores.get('INFO_CLUSTER', 0):.1f} | cols={cols} rows={rows} ruling={ruling}")
    
    # 5. Format classification (cell classifications: TABLE/INFO_CLUSTER + region list/para/string/key_value)
    if cleaned_result.cleaning_inputs and classification:
        overlay = draw_classification_overlay(img, cleaned_result.cleaning_inputs, classification, consolidation, None, save=False)
        region_blocks = getattr(cleaned_result.cleaning_inputs, "region_blocks", None) or []
        region_lines = getattr(cleaned_result.cleaning_inputs, "region_lines", None) or []
        if region_blocks or region_lines:
            overlay = draw_region_format_overlay(overlay, cleaned_result.cleaning_inputs, classification, None, save=False)
        save_debug_image(overlay, "05_format_classification")
    else:
        save_debug_image(img.copy(), "05_format_classification")

    print(f"\n  Consolidation:")
    print(f"    Tables: {len(consolidation.get('tables', []))}")
    print(f"    Non-table clusters: {len(consolidation.get('non_table_clusters', []))}")

    # 6. Consolidated tables (visualised table consolidations)
    if consolidation:
        draw_table_consolidation_overlay(img, consolidation, "06_consolidated_tables")
    else:
        save_debug_image(img.copy(), "06_consolidated_tables")

    # Show table info
    if consolidation.get('tables'):
        print(f"\n  Table details (first 5):")
        for table in consolidation['tables'][:5]:
            print(f"    - {table.get('table_id', 'N/A')}: {table.get('grid_shape', 'N/A')} cells={len(table.get('cells', []))}")
            # Show sample cell text
            cells = table.get('cells', [])[:3]
            for cell in cells:
                print(f"      Cell: {cell.get('text', '')[:40]}")
    
    print(f"\n  Anchors collected: {len(anchors.get('anchors', []))}")

    # 7. Query candidate locations (host cell or format highlighted in yellow)
    if anchors:
        draw_anchors_overlay(img, anchors, "07_query_candidate_locations")
    else:
        save_debug_image(img.copy(), "07_query_candidate_locations")
    
    # Show anchors if any
    if anchors.get('anchors'):
        print(f"\n  Anchor details (first 10):")
        for anchor in anchors['anchors'][:10]:
            print(f"    - Query: {anchor.get('query_name', 'N/A')}")
            print(f"      Text: {anchor.get('anchor_text', '')[:50]}")
            print(f"      Score: {anchor.get('score', 0):.3f}")
            print(f"      Block: {anchor.get('block_id', 'N/A')}, Source: {anchor.get('source_type', 'N/A')}")
    else:
        print(f"\n  [!] No anchors found - checking why...")
        print(f"      Query set is matching against extracted text.")
        print(f"      The test images may not contain fields like 'Part Number', 'Quantity', etc.")
        print(f"      Try with documents containing manufacturing/engineering data.")
    
    # Stage 4: Relationship Resolution (with debug)
    print(f"\n[STAGE 4] Relationship Resolution...")
    
    # Debug: manually run resolver to see orientation
    from relationship_resolver_v2_enhanced import (
        RelationshipEngineV2, CandidateAnchor, NeighborEngine, 
        ToleranceEngine, SpatialStatisticsEngine
    )
    
    # Create engine
    engine = RelationshipEngineV2(
        inputs=cleaned_result.cleaning_inputs,
        classification_results=cleaned_result.classification or {},
        consolidation_results=cleaned_result.consolidation,
    )
    
    # Debug first anchor
    if anchors.get('anchors'):
        first_anchor_dict = anchors['anchors'][0]
        first_anchor = CandidateAnchor(
            query_name=first_anchor_dict["query_name"],
            anchor_text=first_anchor_dict["anchor_text"],
            text_norm=first_anchor_dict["text_norm"],
            score=first_anchor_dict["score"],
            alias_matched=first_anchor_dict["alias_matched"],
            source_type=first_anchor_dict["source_type"],
            source_id=first_anchor_dict["source_id"],
            block_id=first_anchor_dict["block_id"],
            bbox=tuple(first_anchor_dict["bbox"]),
        )
        
        # Get lines for this anchor
        lines = engine.geometry.get_lines_for_anchor(first_anchor, engine.consolidation_results)
        anchor_line = engine.geometry.create_anchor_line(first_anchor)
        
        print(f"\n  Debug for anchor: {first_anchor.anchor_text}")
        print(f"    Anchor bbox: {first_anchor.bbox}")
        print(f"    Anchor y_mid: {anchor_line.y_mid}")
        print(f"    Lines retrieved: {len(lines)}")
        
        # Show lines with their positions
        print(f"\n    Lines in table (sorted by y_mid):")
        sorted_lines = sorted(lines, key=lambda x: x.y_mid)
        for ln in sorted_lines[:10]:
            rel_pos = "ABOVE" if ln.y_mid < anchor_line.y_mid else "BELOW" if ln.y_mid > anchor_line.y_mid else "SAME"
            print(f"      [{rel_pos}] y={ln.y_mid:.0f}: {ln.text[:40]}")
        
        # Build neighbor graph and check orientation
        graph = NeighborEngine.build_neighbor_graph(lines + [anchor_line], engine.tolerances)
        info = graph.get(anchor_line.id)
        if info:
            print(f"\n    Neighbor counts:")
            print(f"      UP: {len(info.up)}, DOWN: {len(info.down)}")
            print(f"      LEFT: {len(info.left)}, RIGHT: {len(info.right)}")
            
            if info.up:
                print(f"      First UP neighbor: {info.up[0][0].text[:30]} (dist={info.up[0][1]:.0f})")
            if info.down:
                print(f"      First DOWN neighbor: {info.down[0][0].text[:30]} (dist={info.down[0][1]:.0f})")
        
        print(f"\n    Tolerances: tau_x={engine.tolerances.tau_x:.1f}, tau_y={engine.tolerances.tau_y:.1f}")
    
    from pipeline import process_page_relationship_resolution
    resolved_result = process_page_relationship_resolution(page, cleaned_result)
    
    relationships = resolved_result.relationships or {}
    print(f"\n  Relationships resolved: {len(relationships)}")
    
    # Show relationship details
    resolved_count = 0
    for query_name, rel_result in relationships.items():
        results_list = rel_result if isinstance(rel_result, list) else ([rel_result] if rel_result is not None else [])
        for r in results_list:
            if r is not None:
                resolved_count += 1
                print(f"\n    - Query: {query_name}")
                print(f"      Format: {r.format}")
                print(f"      Values: {r.values[:3] if r.values else 'None'}...")
                print(f"      Score: {r.score:.3f}")
                if r.meta:
                    print(f"      Meta keys: {list(r.meta.keys())[:5]}")
    
    print(f"\n  Total resolved: {resolved_count}/{len(relationships)}")
    
    if resolved_count == 0 and len(anchors.get('anchors', [])) > 0:
        print(f"\n  [!] Anchors found but no values resolved!")
        print(f"      This suggests the resolver couldn't find values near the anchors.")
        print(f"      Check if the geometry/lines are being passed correctly.")
    
    print("\n" + "=" * 70)
    print("Debug complete.")
    
    # Save JSON results to outputs directory (same structure as full pipeline: batch_id/doc_id/)
    from pipeline import (
        aggregate_document_results,
        build_digitised_document_results,
        augment_aggregated_with_validation,
        DocumentExtractionResult,
    )

    # Build a minimal DocumentExtractionResult for aggregation
    doc_result = DocumentExtractionResult(
        document_id=page.document_id,
        batch_id=batch.batch_id,
    )
    doc_result.page_results[page.page_id] = resolved_result
    doc_result.aggregated_results = aggregate_document_results(doc_result)

    # Outputs under outputs/BATCH_ID/DOC_ID/ (same as save_final_results)
    outputs_dir = Path("outputs")
    batch_dir = outputs_dir / batch.batch_id
    doc_dir = batch_dir / page.document_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # extraction_results.json (aggregated + validation if available)
    payload = dict(doc_result.aggregated_results)
    augment_aggregated_with_validation(
        payload, page.document_id,
        getattr(batch, "validation_results", None) or {},
        doc_result=doc_result,
    )
    extraction_path = doc_dir / "extraction_results.json"
    with open(extraction_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[OUTPUT] Aggregated results: {extraction_path}")

    # digitised_document.json (all scraped data, confidence >= 0.45)
    digitised = build_digitised_document_results(doc_result, min_confidence=0.45)
    digitised_path = doc_dir / "digitised_document.json"
    with open(digitised_path, "w", encoding="utf-8") as f:
        json.dump(digitised, f, indent=2, ensure_ascii=False)
    print(f"[OUTPUT] Digitised document: {digitised_path}")


def clear_cache():
    """Clear __pycache__, investigation_out, and experimental cache so the run uses current code."""
    cleared = []
    # investigation_out
    inv_dir = os.path.join(_SCRIPT_DIR, "investigation_out")
    if os.path.isdir(inv_dir):
        for name in os.listdir(inv_dir):
            p = os.path.join(inv_dir, name)
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        cleared.append("investigation_out/")
    # __pycache__ under project subdirs
    for sub in ("cleaners", "workers", "input", "bootstrap"):
        d = os.path.join(_SCRIPT_DIR, sub)
        pycache = os.path.join(d, "__pycache__")
        if os.path.isdir(pycache):
            shutil.rmtree(pycache, ignore_errors=True)
            cleared.append(f"{sub}/__pycache__/")
    # experimental cache (rect/OCR)
    exp_cache = os.path.join(_SCRIPT_DIR, "debug_outputs", "test_data_experimental_cache")
    if os.path.isdir(exp_cache):
        shutil.rmtree(exp_cache, ignore_errors=True)
        cleared.append("debug_outputs/test_data_experimental_cache/")
    if cleared:
        for c in cleared:
            logging.info("Cleared %s", c)
    return cleared


def run_full_batch_no_debug(input_dir: str, output_dir: str = "outputs"):
    """Clear cache and run the full pipeline on all documents in input_dir (no debug outputs)."""
    clear_cache()
    from pipeline import pipeline_main
    logging.info("Running full pipeline (no debug) on: %s -> %s", input_dir, output_dir)
    return pipeline_main(input_dir, output_dir)


if __name__ == "__main__":
    # Default: run full batch on test_data with cache cleared and no debug outputs
    args = [a for a in sys.argv[1:] if a != "--debug"]
    input_dir = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    output_dir = args[1] if len(args) > 1 else "outputs"
    if "--debug" in sys.argv:
        run_debug_pipeline(input_dir)
    else:
        result = run_full_batch_no_debug(input_dir, output_dir)
        print(json.dumps(result, indent=2))
