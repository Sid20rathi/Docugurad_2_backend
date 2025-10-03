import os
import json
import mimetypes
import shutil
import tempfile
import uuid
import threading
import time
import asyncio
from dotenv import load_dotenv

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import google.generativeai as genai
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim
import difflib
import hashlib
import pytesseract 
import re
from PIL import Image, ImageEnhance, ImageFilter
import logging

load_dotenv()

router2 = APIRouter()

tasks = {}

STATIC_FOLDER = 'static/results'
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Updated constants for better content-focused comparison
OCR_CONFIDENCE_THRESHOLD = 0.8
MIN_CONTOUR_AREA = 2200  # Increased significantly to ignore small differences
MIN_TEXT_REGION_AREA = 2100  # Minimum area for text regions
MIN_SIGNATURE_AREA = 2100  # Minimum area for signature regions

# More lenient thresholds for scanner variations
DIFFERENCE_THRESHOLD = 210 # Increased to ignore minor intensity differences
MORPH_KERNEL_SIZE = (9,9)  # Increased to remove more noise

# Standard resolution for comparison
STANDARD_DPI = 300
STANDARD_WIDTH = 2480
STANDARD_HEIGHT = 3508

# Scanner tolerance
SCANNER_NOISE_TOLERANCE = 0.30 # Increased to 5% noise tolerance
TEXT_SIMILARITY_THRESHOLD = 0.85

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    gemini_model = None
    print(f"Could not initialize Gemini: {e}. AI report generation will be disabled.")

def log_progress(task_id, message):
    """Logs a message to the specific task's log list."""
    if task_id in tasks:
        logger.info(message)
        tasks[task_id]["logs"].append(message)

def get_file_hash(file_path):
    """Calculate MD5 hash of file"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def preprocess_for_ocr(image):
    """Enhanced image preprocessing for better OCR accuracy"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Convert to grayscale
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(2.0)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(2.0)
    
    # Apply slight blur to reduce noise
    pil_image = pil_image.filter(ImageFilter.MedianFilter(3))
    
    # Convert back to numpy array
    processed = np.array(pil_image)
    
    # Apply adaptive threshold
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15, 2)
    
    return processed

def convert_pdf_to_tiff(pdf_path, output_dir, task_id, dpi=STANDARD_DPI):
    """Convert PDF to TIFF format at specified DPI"""
    log_progress(task_id, f"Converting PDF to TIFF at {dpi} DPI: {pdf_path}")
    
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        
        if not images:
            raise ValueError("No pages found in PDF")
        
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        tiff_path = os.path.join(output_dir, f"{base_name}.tiff")
        
        if len(images) == 1:
            images[0].save(tiff_path, format='TIFF', dpi=(dpi, dpi))
        else:
            images[0].save(
                tiff_path, 
                format='TIFF', 
                save_all=True, 
                append_images=images[1:],
                dpi=(dpi, dpi)
            )
        
        log_progress(task_id, f"Successfully converted PDF to TIFF at {dpi} DPI: {tiff_path}")
        return tiff_path
        
    except Exception as e:
        log_progress(task_id, f"[ERROR] Failed to convert PDF to TIFF: {e}")
        raise

def read_tiff_file(tiff_path):
    """Read a TIFF file and return a list of images (pages)."""
    images = []
    try:
        tiff_file = cv2.imreadmulti(tiff_path, flags=cv2.IMREAD_COLOR)
        
        if tiff_file[0]:
            images = list(tiff_file[1])
        else:
            img = cv2.imread(tiff_path)
            if img is not None:
                images = [img]
    except Exception as e:
        print(f"Error reading TIFF file: {e}")
    
    return images

def is_pdf_file(file_path):
    """Check if a file is a PDF based on its content."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type == 'application/pdf'

def is_tiff_file(file_path):
    """Check if a file is a TIFF based on its content."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type in ['image/tiff', 'image/tif']

def resize_to_standard(image, width=STANDARD_WIDTH, height=STANDARD_HEIGHT):
    """Resize an image to a standard size while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    scale = min(width/w, height/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    if len(image.shape) == 3:
        canvas = np.zeros((height, width, 3), dtype=np.uint8) + 255
    else:
        canvas = np.zeros((height, width), dtype=np.uint8) + 255
    
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def extract_text_with_tesseract(image):
    """Enhanced text extraction from image using Tesseract OCR"""
    try:
        processed = preprocess_for_ocr(image)
        
        if isinstance(processed, np.ndarray):
            pil_image = Image.fromarray(processed)
        else:
            pil_image = processed
        
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 4',
            r'--oem 3 --psm 3',
        ]
        
        best_text = ""
        best_length = 0
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(pil_image, config=config, lang='eng')
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) > best_length:
                    best_text = text
                    best_length = len(text)
                    
            except Exception as e:
                logger.warning(f"Tesseract config {config} failed: {e}")
                continue
        
        return best_text if best_text else ""
        
    except Exception as e:
        logger.error(f"Tesseract OCR error: {e}")
        return ""

def normalize_scanner_variations(image):
    """Enhanced normalization to handle scanner variations"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Stronger blur to remove scanner noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # CLAHE for contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(blurred)
    
    # Stronger bilateral filter
    filtered = cv2.bilateralFilter(normalized, 15, 80, 80)
    
    return filtered

def detect_content_regions(image):
    """Detect different types of content regions (text, signatures, images)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Use adaptive threshold to handle varying lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_TEXT_REGION_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate region properties for classification
            region = gray[y:y+h, x:x+w]
            aspect_ratio = w / h
            
            # Classify region type based on properties
            if aspect_ratio > 3 and area > 5000:  # Likely text region
                region_type = "text"
            elif 0.5 < aspect_ratio < 2.5 and area > MIN_SIGNATURE_AREA:  # Likely signature/image
                region_type = "signature"
            else:
                region_type = "other"
            
            regions.append({
                "type": region_type,
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": aspect_ratio
            })
    
    return regions

def compare_content_regions(regions1, regions2):
    """Compare content regions between two images"""
    differences = []
    
    # Compare by region type and position
    for region1 in regions1:
        found_match = False
        for region2 in regions2:
            # Check if regions are similar in position and size
            x1, y1, w1, h1 = region1["bbox"]
            x2, y2, w2, h2 = region2["bbox"]
            
            # Calculate overlap and similarity
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            area1 = w1 * h1
            area2 = w2 * h2
            min_area = min(area1, area2)
            
            if overlap_area > 0.7 * min_area and region1["type"] == region2["type"]:
                found_match = True
                break
        
        if not found_match and region1["type"] in ["text", "signature"]:
            differences.append({
                "type": f"missing_{region1['type']}",
                "bbox": region1["bbox"],
                "area": region1["area"]
            })
    
    return differences

class ForgeryDetector:
    def __init__(self, task_id):
        self.task_id = task_id
        log_progress(self.task_id, "Initializing Content-Focused Forgery Detector...")

    def _align_images(self, image1, image2):
        """Enhanced image alignment with better feature matching"""
        standard_img1 = resize_to_standard(image1)
        standard_img2 = resize_to_standard(image2)
        
        gray1 = normalize_scanner_variations(standard_img1)
        gray2 = normalize_scanner_variations(standard_img2)
        
        # Use ORB for faster processing with good enough results
        orb = cv2.ORB_create(1000)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
        
        if descriptors1 is None or descriptors2 is None: 
            return standard_img2
        
        # Use BFMatcher for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        
        if len(matches) < 4: 
            return standard_img2
            
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        if h is None: 
            return standard_img2
            
        height, width = gray1.shape
        aligned = cv2.warpPerspective(standard_img2, h, (width, height))
        
        return aligned

    def _is_scanner_artifact(self, diff_region, original_region, aligned_region):
        """Enhanced scanner artifact detection"""
        region_area = diff_region.size
        changed_pixels = np.sum(diff_region > 0)
        change_percentage = changed_pixels / region_area
        
        # If change is very small and scattered, it's likely scanner noise
        if change_percentage < SCANNER_NOISE_TOLERANCE:
            return True
        
        # Check if the pattern is consistent with scanner lines
        if len(original_region.shape) == 3:
            orig_gray = cv2.cvtColor(original_region, cv2.COLOR_BGR2GRAY)
            align_gray = cv2.cvtColor(aligned_region, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original_region
            align_gray = aligned_region
        
        # Calculate horizontal and vertical projections
        horizontal_proj_orig = np.sum(orig_gray, axis=1)
        horizontal_proj_align = np.sum(align_gray, axis=1)
        vertical_proj_orig = np.sum(orig_gray, axis=0)
        vertical_proj_align = np.sum(align_gray, axis=0)
        
        # Check for consistent patterns (scanner lines)
        h_corr = np.corrcoef(horizontal_proj_orig, horizontal_proj_align)[0,1]
        v_corr = np.corrcoef(vertical_proj_orig, vertical_proj_align)[0,1]
        
        if h_corr > 0.8 or v_corr > 0.8:
            return True
            
        return False

    def _visual_comparison(self, original_img, aligned_img, page_num, output_dir, base_url):
        """Content-focused visual comparison - Always save images for frontend display"""
        standard_original = resize_to_standard(original_img)
        standard_aligned = resize_to_standard(aligned_img)
        
        # Detect content regions in both images
        original_regions = detect_content_regions(standard_original)
        aligned_regions = detect_content_regions(standard_aligned)
        
        # Compare regions to find missing or added content
        region_differences = compare_content_regions(original_regions, aligned_regions)
        
        # Traditional pixel comparison for fine details (but with higher tolerance)
        gray_original = normalize_scanner_variations(standard_original)
        gray_aligned = normalize_scanner_variations(standard_aligned)

        # Compute SSIM
        (score, diff) = ssim(gray_original, gray_aligned, full=True)
        diff = (diff * 255).astype("uint8")

        # Apply threshold - very lenient for scanner variations
        thresh = cv2.threshold(diff, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]

        # Aggressive morphological operations to remove scanner noise
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours from pixel differences
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image = standard_aligned.copy()
        actual_forgeries = 0
        
        # First, mark region-based differences (more reliable)
        for diff_region in region_differences:
            x, y, w, h = diff_region["bbox"]
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            actual_forgeries += 1

        # Then, analyze pixel-based differences with strict filtering
        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                (x, y, w, h) = cv2.boundingRect(contour)
                
                # Extract regions for analysis
                diff_region = thresh[y:y+h, x:x+w]
                orig_region = standard_original[y:y+h, x:x+w]
                align_region = standard_aligned[y:y+h, x:x+w]
                
                # Check if this is scanner artifact - if it is, skip it completely (don't draw any box)
                if not self._is_scanner_artifact(diff_region, orig_region, align_region):
                    # Check if this overlaps with any important content region
                    overlaps_content = False
                    for region in original_regions + aligned_regions:
                        rx, ry, rw, rh = region["bbox"]
                        # Calculate overlap
                        overlap_x = max(0, min(x + w, rx + rw) - max(x, rx))
                        overlap_y = max(0, min(y + h, ry + rh) - max(y, ry))
                        overlap_area = overlap_x * overlap_y
                        
                        if overlap_area > 0.3 * (w * h) and region["type"] in ["text", "signature"]:
                            overlaps_content = True
                            break
                    
                    # Only draw red box if it's not a scanner artifact AND overlaps with important content
                    if overlaps_content:
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        actual_forgeries += 1

        result_data = {
            "similarity_score": score,
            "differences_found": actual_forgeries,
            "content_regions_analyzed": len(original_regions) + len(aligned_regions),
            "region_differences": len(region_differences),
            "diff_image_url": None,
            "original_image_url": None
        }

        # ALWAYS save images for frontend display, regardless of forgery detection
        task_id_folder = os.path.join(output_dir, self.task_id)
        os.makedirs(task_id_folder, exist_ok=True)
        diff_filename = f"page_{page_num}_visual_diff.png"
        original_filename = f"page_{page_num}_original.png"
        diff_path = os.path.join(task_id_folder, diff_filename)
        original_path = os.path.join(task_id_folder, original_filename)
        
        # Save the result image (with or without red boxes)
        cv2.imwrite(diff_path, result_image)
        cv2.imwrite(original_path, standard_original)
        
        # Always set the image URLs so frontend can display them
        result_data["diff_image_url"] = f"{base_url}static/results/{self.task_id}/{diff_filename}"
        result_data["original_image_url"] = f"{base_url}static/results/{self.task_id}/{original_filename}"

       
        return result_data

    def _textual_comparison(self, original_img, aligned_img):
        """Enhanced textual comparison with location awareness"""
        try:
            standard_original = resize_to_standard(original_img)
            standard_aligned = resize_to_standard(aligned_img)
            
            # Extract text with location information
            original_data = pytesseract.image_to_data(standard_original, output_type=pytesseract.Output.DICT)
            duplicate_data = pytesseract.image_to_data(standard_aligned, output_type=pytesseract.Output.DICT)
            
            # Filter confident text detections
            original_text_blocks = []
            duplicate_text_blocks = []
            
            for i in range(len(original_data['text'])):
                if int(original_data['conf'][i]) > 30:  # Confidence threshold
                    text = original_data['text'][i].strip()
                    if len(text) > 1:  # Meaningful text
                        original_text_blocks.append({
                            'text': text,
                            'bbox': (
                                original_data['left'][i],
                                original_data['top'][i],
                                original_data['width'][i],
                                original_data['height'][i]
                            )
                        })
            
            for i in range(len(duplicate_data['text'])):
                if int(duplicate_data['conf'][i]) > 30:
                    text = duplicate_data['text'][i].strip()
                    if len(text) > 1:
                        duplicate_text_blocks.append({
                            'text': text,
                            'bbox': (
                                duplicate_data['left'][i],
                                duplicate_data['top'][i],
                                duplicate_data['width'][i],
                                duplicate_data['height'][i]
                            )
                        })
            
            # Compare text by location and content
            text_differences = []
            matched_indices = set()
            
            for i, orig_block in enumerate(original_text_blocks):
                best_match_idx = -1
                best_similarity = 0
                
                for j, dup_block in enumerate(duplicate_text_blocks):
                    if j in matched_indices:
                        continue
                    
                    # Check spatial proximity
                    ox, oy, ow, oh = orig_block['bbox']
                    dx, dy, dw, dh = dup_block['bbox']
                    
                    distance = np.sqrt((ox - dx)**2 + (oy - dy)**2)
                    max_distance = max(ow, oh, dw, dh) * 2
                    
                    if distance < max_distance:
                        similarity = difflib.SequenceMatcher(
                            None, 
                            orig_block['text'].lower(), 
                            dup_block['text'].lower()
                        ).ratio()
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = j
                
                if best_match_idx != -1 and best_similarity > 0.8:
                    matched_indices.add(best_match_idx)
                    if best_similarity < 0.95:  # Small text variations
                        text_differences.append({
                            'type': 'text_variation',
                            'original': orig_block,
                            'duplicate': duplicate_text_blocks[best_match_idx],
                            'similarity': best_similarity
                        })
                else:
                    # Missing text block
                    text_differences.append({
                        'type': 'missing_text',
                        'original': orig_block,
                        'duplicate': None
                    })
            
            # Find added text blocks
            for j, dup_block in enumerate(duplicate_text_blocks):
                if j not in matched_indices:
                    text_differences.append({
                        'type': 'added_text',
                        'original': None,
                        'duplicate': dup_block
                    })
            
            # Calculate overall similarity
            original_full_text = ' '.join([block['text'] for block in original_text_blocks])
            duplicate_full_text = ' '.join([block['text'] for block in duplicate_text_blocks])
            
            if original_full_text and duplicate_full_text:
                overall_similarity = difflib.SequenceMatcher(
                    None, original_full_text, duplicate_full_text
                ).ratio()
            else:
                overall_similarity = 0
            
            return {
                "status": "success",
                "similarity_ratio": overall_similarity,
                "text_differences": text_differences,
                "original_text_blocks": len(original_text_blocks),
                "duplicate_text_blocks": len(duplicate_text_blocks),
                "original_text_sample": original_full_text[:200] + "..." if len(original_full_text) > 200 else original_full_text,
                "duplicate_text_sample": duplicate_full_text[:200] + "..." if len(duplicate_full_text) > 200 else duplicate_full_text
            }
            
        except Exception as e:
           
            return {
                "status": "error",
                "error": f"Text comparison failed: {str(e)}",
                "similarity_ratio": 0,
                "text_differences": []
            }

    def compare_documents(self, original_path, duplicate_path, output_dir, base_url):
        analysis_result = {"page_by_page_analysis": []}
        
        # First check if files are identical
        original_hash = get_file_hash(original_path)
        duplicate_hash = get_file_hash(duplicate_path)
        
        if original_hash == duplicate_hash:
       
            return {"identical_files": True, "message": "Files are identical"}
        
        # Check file types and convert if necessary
        original_is_pdf = is_pdf_file(original_path)
        duplicate_is_pdf = is_pdf_file(duplicate_path)
        
        # Convert PDFs to TIFF if needed
        if original_is_pdf:
        
            original_path = convert_pdf_to_tiff(original_path, os.path.dirname(original_path), self.task_id, STANDARD_DPI)
        
        if duplicate_is_pdf:
          
            duplicate_path = convert_pdf_to_tiff(duplicate_path, os.path.dirname(duplicate_path), self.task_id, STANDARD_DPI)
        
        # Read TIFF files
        log_progress(self.task_id, "Reading original document...")
        original_images = read_tiff_file(original_path)
        
        log_progress(self.task_id, "Reading suspected document...")
        duplicate_images = read_tiff_file(duplicate_path)
        
        if not original_images:
            error_msg = "Could not read original document"
        
            return {"error": error_msg}
            
        if not duplicate_images:
            error_msg = "Could not read suspected document"
         
            return {"error": error_msg}
        
        if len(original_images) != len(duplicate_images):
            error_msg = f"Documents have different page counts: Original {len(original_images)}, Suspected {len(duplicate_images)}"
  
            return {"error": error_msg}
            
        log_progress(self.task_id, f"Starting page-by-page analysis for {len(original_images)} pages...")
        
        for i in range(len(original_images)):
            page_number = i + 1
            log_progress(self.task_id, f"--- Analyzing Page {page_number}/{len(original_images)} ---")
            page_analysis = {"page": page_number}
            
            # Convert images to the format needed for processing
            original_cv = original_images[i]
            duplicate_cv = duplicate_images[i]
            
            # Ensure both images have the same number of channels
            if len(original_cv.shape) == 2:
                original_cv = cv2.cvtColor(original_cv, cv2.COLOR_GRAY2BGR)
            if len(duplicate_cv.shape) == 2:
                duplicate_cv = cv2.cvtColor(duplicate_cv, cv2.COLOR_GRAY2BGR)
                
            aligned_duplicate = self._align_images(original_cv, duplicate_cv)
            page_analysis["visual_analysis"] = self._visual_comparison(original_cv, aligned_duplicate, page_number, output_dir, base_url)
            page_analysis["textual_analysis"] = self._textual_comparison(original_cv, aligned_duplicate)
            analysis_result["page_by_page_analysis"].append(page_analysis)
            log_progress(self.task_id, f"[Page {page_number}] Analysis complete.")
            
        log_progress(self.task_id, "Finalizing report with Docu Agent...")
        return analysis_result

def create_error_report(error_message="An unknown error occurred."):
    return {
        "verdict": "Inconclusive",
        "summary": {"critical_findings": 1, "moderate_findings": 0, "total_discrepancies": 1},
        "analysis_details": [{"area": "System Error", "discrepancies": [{"page": 0, "description": f"Analysis failed: {error_message}", "severity": "High"}]}]
    }

def generate_gemini_report(analysis_data, task_id):
    if not gemini_model:
        return create_error_report("Gemini AI is not configured on the server.")
    log_progress(task_id, "Generating final analysis report with Docu Agent...")
    
    if analysis_data.get("identical_files"):
        return {
            "verdict": "Authentic",
            "summary": {"critical_findings": 0, "moderate_findings": 0, "total_discrepancies": 0},
            "analysis_details": [],
            "identical_files": True
        }
    
    json_data_string = json.dumps(analysis_data, indent=2)
    prompt = f"""
    You are a helpful assistant. Your task is to summarize a technical JSON report from a document comparison tool into a simple, plain English summary. Focus on clarity and ease of understanding for a non-technical user.

    **IMPORTANT CONTEXT FOR ANALYSIS:**
    - The system now only shows red rectangles for potential forgeries
    - Scanner noise and artifacts are completely filtered out and not shown
    - The system analyzes text content, signatures, and images in their exact locations
    - OCR/text analysis may sometimes fail - this doesn't necessarily indicate forgery

    **Input Data Explanation:**
    - `page`: The page number being analyzed.
    - `visual_analysis.similarity_score`: A visual similarity score from 0.0 to 1.0.
    - `visual_analysis.differences_found`: The number of potential forgeries (red rectangles).
    - `visual_analysis.region_differences`: Differences in content regions (text, signatures, images).
    - `textual_analysis.status`: "success" or "error"
    - `textual_analysis.similarity_ratio`: A ratio from 0.0 to 1.0 indicating text similarity.
    - `textual_analysis.text_differences`: Detailed text differences with locations.

    **Your Task:**
    Analyze the input data and generate a single JSON object with the following exact structure. Describe each finding in a clear, single sentence.

    ```json
    {{
    "verdict": "Authentic" | "Forged" | "Inconclusive",
    "summary": {{
        "critical_findings": <integer>,
        "moderate_findings": <integer>,
        "total_discrepancies": <integer>
    }},
    "analysis_details": [
        {{
        "area": "Signature" | "Text Content" | "Stamp" | "Layout" | "Image" | "Handwritten Notes",
        "discrepancies": [
            {{
            "page": <integer>,
            "description": "A very specific, detailed description of the exact change, quoting evidence.",
            "severity": "Low" | "Medium" | "High"
            }}
        ]
        }}
    ]
    }}
    ```

    **CRITICAL ANALYSIS & REPORTING RULES:**
    1. **ONLY RED BOXES MATTER:** All differences found are potential forgeries (red rectangles only).
    2. **FOCUS ON CONTENT:** Only report differences in text, signatures, stamps, or important visual elements.
    3. **BE LENIENT WITH TEXT:** If OCR failed or shows minor variations, don't count it as forgery unless there are visual forgeries.
    4. **ONLY REPORT ACTUAL FORGERIES:** If no differences found, verdict should be "Authentic".
    5. **ALWAYS MENTION THE PAGE NUMBER:** Every finding must start with "On page X, ...".
    6. **USE PLAIN ENGLISH:** Avoid technical jargon.
    7. **BE CONSERVATIVE:** When in doubt, prefer "Authentic" over "Forged".

    **Input Data for Analysis:**
    ```json
    {json_data_string}
    ```

    Now, generate the simple, plain English JSON output.
    """
    try:
        start_gemini_time = time.time()
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = gemini_model.generate_content(prompt, generation_config=generation_config)
        end_gemini_time = time.time()
        log_progress(task_id, f"Analysis generated in {end_gemini_time - start_gemini_time:.2f} seconds.")
        return json.loads(response.text)
    except Exception as e:
        log_progress(task_id, f"[ERROR] An error occurred while communicating with or parsing Gemini response: {e}")
        raw_text = response.text if 'response' in locals() else "No response from model."
        log_progress(task_id, f"[DEBUG] Raw Gemini Response: {raw_text}")
        return create_error_report(f"Failed to generate or parse AI report. Details: {e}")

def run_analysis_task(task_id, original_path, suspected_path, output_dir, base_url):
    """The function that will run in a separate thread."""
    try:
        detector = ForgeryDetector(task_id=task_id)
        start_analysis_time = time.time()
        analysis_result = detector.compare_documents(original_path, suspected_path, output_dir, base_url)
        end_analysis_time = time.time()
        log_progress(task_id, f"Full document analysis completed in {end_analysis_time - start_analysis_time:.2f} seconds.")
        
        if "error" in analysis_result:
            report = create_error_report(f"An error occurred during document analysis: {analysis_result['error']}")
        else:
            report = generate_gemini_report(analysis_result, task_id)
            report['page_by_page_analysis'] = analysis_result.get('page_by_page_analysis', [])
            
        tasks[task_id]["result"] = report
        tasks[task_id]["status"] = "complete"
        
    except Exception as e:
        error_msg = f"An internal server error occurred: {e}"
       
        tasks[task_id]["result"] = create_error_report(error_msg)
        tasks[task_id]["status"] = "error"
    finally:
        # Clean up temporary files
        temp_dir = os.path.dirname(original_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
           
        log_progress(task_id, "Request processing finished.")

@router2.get('/')
async def status():
    """Provides a status check for the API."""
    return {"status": "DocuGuard Enhanced Forgery Detection API is running"}

@router2.post("/api/verify", status_code=202)
async def verify_documents(
    request: Request,
    originalFile: UploadFile = File(..., description="The original, trusted PDF or TIFF document."),
    suspectedFile: UploadFile = File(..., description="The suspected, potentially forged PDF or TIFF document.")
):
    """
    Starts the asynchronous document verification process.
    - Accepts two PDF or TIFF files: `originalFile` and `suspectedFile`.
    - Returns a `task_id` to track the analysis progress.
    """
    if not originalFile.filename or not suspectedFile.filename:
        raise HTTPException(status_code=400, detail="One or both filenames are empty.")
    
    # Validate file types
    original_filename = originalFile.filename.lower()
    suspected_filename = suspectedFile.filename.lower()
    
    if not (original_filename.endswith(('.pdf', '.tiff', '.tif')) and 
            suspected_filename.endswith(('.pdf', '.tiff', '.tif'))):
        raise HTTPException(status_code=400, detail="Only PDF and TIFF files are supported.")

    # Create a temporary directory to store uploaded files securely
    temp_dir = tempfile.mkdtemp()
    original_path = os.path.join(temp_dir, secure_filename(originalFile.filename))
    suspected_path = os.path.join(temp_dir, secure_filename(suspectedFile.filename))

    # Save uploaded files to the temporary directory
    try:
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(originalFile.file, buffer)
        with open(suspected_path, "wb") as buffer:
            shutil.copyfileobj(suspectedFile.file, buffer)
    finally:
        await originalFile.close()
        await suspectedFile.close()

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "running", "logs": [], "result": None}

    # Get the base URL from the incoming request to construct result URLs
    base_url = str(request.base_url)

    # Start the analysis in a background thread to avoid blocking the API
    thread = threading.Thread(
        target=run_analysis_task,
        args=(task_id, original_path, suspected_path, STATIC_FOLDER, base_url)
    )
    thread.start()

    return {"task_id": task_id}

@router2.get("/api/stream/{task_id}")
async def stream_logs(task_id: str):
    """
    Streams logs and the final result for a given task ID using Server-Sent Events (SSE).
    """
    async def event_generator():
        last_log_index = 0
        while True:
            if task_id not in tasks:
                yield f"data: [ERROR] Task ID not found.\n\n"
                break

            # Stream new log messages
            while last_log_index < len(tasks[task_id]["logs"]):
                log_message = tasks[task_id]["logs"][last_log_index]
                yield f"data: {log_message}\n\n"
                last_log_index += 1

            # Check if the task is finished
            if tasks[task_id]["status"] in ["complete", "error"]:
                final_result = json.dumps(tasks[task_id]["result"])
                yield f"data: {final_result}\n\n"
                yield f"data: [DONE]\n\n"
                break

            # Wait before checking for new logs again
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")