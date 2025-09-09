import cv2
import numpy as np
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim
import easyocr
import difflib
import os
import time
import shutil
import tempfile
import json
import uuid
import threading
import google.generativeai as genai
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(
    app,
    supports_credentials=True,
    resources={r"/api/*": {"origins": "http://localhost:3000"}},
)


tasks = {}

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if origin == "http://localhost:3000":
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With, X-CSRF-Token"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

STATIC_FOLDER = 'static/results'
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# --- Configuration ---
OCR_CONFIDENCE_THRESHOLD = 0.3
MIN_CONTOUR_AREA = 900
ALIGNMENT_MATCH_PERCENT = 0.50

# --- Gemini AI Configuration ---
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {"temperature": 0}
    gemini_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
   
except Exception as e:
    gemini_model = None
    print(f"Could not initialize Backend: {e}. AI report generation will be disabled.")


def log_progress(task_id, message):
    """Logs a message to the specific task's log list."""
    if task_id in tasks:
        print(message)
        tasks[task_id]["logs"].append(message)


class ForgeryDetector:
    def __init__(self, task_id, languages=['en']):
        self.task_id = task_id
        log_progress(self.task_id, "")
        try:
            self.reader = easyocr.Reader(languages)
            log_progress(self.task_id, "")
        except Exception as e:
            log_progress(self.task_id, f"[ERROR] Could not initialize Backend : {e}")
            self.reader = None

    def _align_images(self, image1, image2):
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=5000)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
        if descriptors1 is None or descriptors2 is None: return image2
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descriptors1, descriptors2), key=lambda x: x.distance)
        num_good_matches = int(len(matches) * ALIGNMENT_MATCH_PERCENT)
        matches = matches[:num_good_matches]
        if len(matches) < 4: return image2
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        if h is None: return image2
        height, width, _ = image1.shape
        return cv2.warpPerspective(image2, h, (width, height))

    def _visual_comparison(self, original_img, aligned_img, page_num, output_dir, base_url):
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_aligned = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        (score, diff) = ssim(gray_original, gray_aligned, full=True)
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        diff_count = 0
        result_image = aligned_img.copy()
        result_data = {
            "similarity_score": score,
            "differences_found": diff_count,
            "diff_image_url": None,
            "original_image_url": None
        }
        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                diff_count += 1
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        result_data["differences_found"] = diff_count
        if diff_count > 0:
            diff_filename = f"page_{page_num}_visual_diff.png"
            original_filename = f"page_{page_num}_original.png"
            diff_path = os.path.join(output_dir, diff_filename)
            original_path = os.path.join(output_dir, original_filename)
            cv2.imwrite(diff_path, result_image)
            cv2.imwrite(original_path, original_img)
            result_data["diff_image_url"] = f"{base_url}static/results/{diff_filename}"
            result_data["original_image_url"] = f"{base_url}static/results/{original_filename}"
        return result_data

    def _textual_comparison(self, original_img, aligned_img):
        if not self.reader: return {"error": "OCR Reader not initialized."}
        try:
            original_results = self.reader.readtext(original_img, paragraph=False, detail=0)
            duplicate_results = self.reader.readtext(aligned_img, paragraph=False, detail=0)
            original_text = " ".join(original_results).lower()
            duplicate_text = " ".join(duplicate_results).lower()
            diff = difflib.unified_diff(original_text.split(), duplicate_text.split(), fromfile='original', tofile='suspected', lineterm='')
            text_diffs = [line for line in diff if line.startswith(('+', '-')) and len(line) > 1]
            return {"differences": text_diffs}
        except Exception as e:
            return {"error": f"An error occurred during OCR: {e}"}

    def compare_documents(self, original_pdf_path, duplicate_pdf_path, output_dir, base_url):
        analysis_result = {"page_by_page_analysis": []}
        log_progress(self.task_id, "Converting original PDF to images...")
        original_images = convert_from_path(original_pdf_path)
        log_progress(self.task_id, "Converting suspected PDF to images...")
        duplicate_images = convert_from_path(duplicate_pdf_path)
        if len(original_images) != len(duplicate_images):
            error_msg = f"Documents have different page counts: Original {len(original_images)}, Suspected {len(duplicate_images)}"
            log_progress(self.task_id, f"[ERROR] {error_msg}")
            return {"error": error_msg}
        log_progress(self.task_id, f"Starting page-by-page analysis for {len(original_images)} pages...")
        for i in range(len(original_images)):
            page_number = i + 1
            log_progress(self.task_id, f"--- Analyzing Page {page_number}/{len(original_images)} ---")
            page_analysis = {"page": page_number}
            original_pil_rgb = original_images[i].convert('RGB')
            duplicate_pil_rgb = duplicate_images[i].convert('RGB')
            original_cv = cv2.cvtColor(np.array(original_pil_rgb), cv2.COLOR_RGB2BGR)
            duplicate_cv = cv2.cvtColor(np.array(duplicate_pil_rgb), cv2.COLOR_RGB2BGR)
            aligned_duplicate = self._align_images(original_cv, duplicate_cv)
            page_analysis["visual_analysis"] = self._visual_comparison(original_cv, aligned_duplicate, page_number, output_dir, base_url)
            page_analysis["textual_analysis"] = self._textual_comparison(original_cv, aligned_duplicate)
            analysis_result["page_by_page_analysis"].append(page_analysis)
            log_progress(self.task_id, f"[Page {page_number}] Analysis complete.")
        log_progress(self.task_id, "Finalizing report with AI...")
        return analysis_result

# --- Helper and Report Generation Functions ---
def create_error_report(error_message="An unknown error occurred."):
    return {
        "verdict": "Inconclusive",
        "summary": {"critical_findings": 1, "moderate_findings": 0, "total_discrepancies": 1},
        "analysis_details": [{"area": "System Error", "discrepancies": [{"page": 0, "description": f"Analysis failed: {error_message}", "severity": "High"}]}]
    }

def generate_gemini_report(analysis_data, task_id):
    if not gemini_model:
        return create_error_report("Gemini AI is not configured on the server.")
    log_progress(task_id, "Generating analysis report ...")
    json_data_string = json.dumps(analysis_data, indent=2)
    prompt = f"""
     You are a helpful assistant. Your task is to summarize a technical JSON report from a document comparison tool into a simple, plain English summary. Focus on clarity and ease of understanding for a non-technical user.
     
     **Input Data Explanation:**
     - `page`: The page number being analyzed.
     - `similarity_score`: A visual similarity score from 0.0 to 1.0. A low score means the pages look very different.
     - `differences_found`: The number of visual differences (like changed images, stamps, or signatures).
     - `textual_analysis.differences`: A list of specific words that were added ('+' prefix) or removed ('-' prefix).
     - `diff_image_url`: A URL to the suspected page image with differences highlighted in red boxes.
     - `original_image_url`: A URL to the clean, original page image for comparison.
     
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
     1.  **USE PLAIN ENGLISH:** Do NOT use technical jargon.
     2.  **BE DIRECT AND CLEAR:** Describe the change itself.
     3.  **SUMMARIZE CHANGES:** If you see evidence of a signature being changed, describe it as a "signature change". If a stamp or seal looks different, call it a "stamp or seal alteration", if there is change in text or numbers as part of the documents, call it a 'text change'and mention the change.
     4.  **ALWAYS MENTION THE PAGE NUMBER:** Every finding must start with "On page X, ...".
     5.  **CREATE A FINDING FOR EACH SIGNIFICANT CHANGE.**
     6.  **Do not use numeric values in the description.**
     
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

# --- Background Worker Function ---
def run_analysis_task(task_id, original_path, suspected_path, output_dir, base_url):
    """The function that will run in a separate thread."""
    try:
        detector = ForgeryDetector(task_id=task_id)
        start_analysis_time = time.time()
        analysis_result = detector.compare_documents(original_path, suspected_path, output_dir, base_url)
        end_analysis_time = time.time()
        log_progress(task_id, f"[Full analysis completed in {end_analysis_time - start_analysis_time:.2f} seconds.")
        if "error" in analysis_result:
            report = create_error_report(f"An error occurred during document analysis: {analysis_result['error']}")
        else:
            report = generate_gemini_report(analysis_result, task_id)
            report['page_by_page_analysis'] = analysis_result.get('page_by_page_analysis', [])
        tasks[task_id]["result"] = report
        tasks[task_id]["status"] = "complete"
    except Exception as e:
        error_msg = f"An internal server error occurred: {e}"
        log_progress(task_id, f"[FATAL ERROR] {error_msg}")
        tasks[task_id]["result"] = create_error_report(error_msg)
        tasks[task_id]["status"] = "error"
    finally:
        temp_dir = os.path.dirname(original_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            log_progress(task_id, f"Cleaned up temporary directory: {temp_dir}")
        log_progress(task_id, "Request processing finished.")

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def status():
    return jsonify({"status": "DocuGuard Forgery Detection API is running"})

@app.route("/api/verify", methods=["POST"])
def verify_documents():
    """Starts the verification process and returns a task ID."""
    if 'original' not in request.files or 'suspected' not in request.files:
        return jsonify(create_error_report("Missing 'original' or 'suspected' file.")), 400
    original_file = request.files['original']
    suspected_file = request.files['suspected']
    if original_file.filename == '' or suspected_file.filename == '':
        return jsonify(create_error_report("One or both filenames are empty.")), 400
    temp_dir = tempfile.mkdtemp()
    original_path = os.path.join(temp_dir, secure_filename(original_file.filename))
    suspected_path = os.path.join(temp_dir, secure_filename(suspected_file.filename))
    original_file.save(original_path)
    suspected_file.save(suspected_path)
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "running", "logs": [], "result": None}
    base_url = request.host_url
    thread = threading.Thread(
        target=run_analysis_task,
        args=(task_id, original_path, suspected_path, app.config['STATIC_FOLDER'], base_url)
    )
    thread.start()
    return jsonify({"task_id": task_id}), 202

@app.route("/api/stream/<task_id>")
def stream_logs(task_id):
    """Streams logs for a given task ID."""
    def generate():
        last_log_index = 0
        while True:
            if task_id not in tasks:
                yield f"data: [ERROR] Task ID not found.\n\n"
                break
            while last_log_index < len(tasks[task_id]["logs"]):
                log_message = tasks[task_id]["logs"][last_log_index]
                yield f"data: {log_message}\n\n"
                last_log_index += 1
            if tasks[task_id]["status"] in ["complete", "error"]:
                final_result = json.dumps(tasks[task_id]["result"])
                yield f"data: {final_result}\n\n"
                yield f"data: [DONE]\n\n"
                break
            time.sleep(0.5)
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True, port=8000)