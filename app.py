"""
app.py
------
Main Flask application for the AI Resume Analyzer.

Routes
------
GET  /                    – Serves the frontend (index.html).
POST /analyze             – Accepts a PDF resume + job description;
                            returns ATS score, matched/missing skills,
                            pros, cons, and improvement suggestion.
POST /rank_jobs           – Ranks jobs.csv entries against an uploaded resume.
POST /extract_resume_skills – Returns comma-separated skills from a PDF resume.

Parsing pipeline
----------------
  PDF → model.parser.extract_text_from_pdf   (block-sorted, multi-column aware)
      → model.preprocessing.clean_text        (noise removal, dedup)
      → model.skill_extractor.extract_skills  (250+ vocab, alias normalisation)

Scoring pipeline (updated)
--------------------------
  resume_text + job_text
      → model.embedding_service   (all-MiniLM-L6-v2 embeddings)
      → model.similarity_engine   (semantic cosine + skill matching)
      → model.matcher.compute_match  (blended score + confidence)
"""

import os
import uuid
import hashlib
import time
import logging
from flask import Flask, request, jsonify, send_from_directory, g
from werkzeug.utils import secure_filename

from model.parser import extract_text_from_pdf
from model.skill_extractor import extract_skills, detect_sections
from model.matcher import compute_match
from model.ranker import rank_jobs_for_resume
from model.comparator import compare_resumes

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        from flask import has_request_context
        record.request_id = g.request_id if has_request_context() and hasattr(g, 'request_id') else 'SYSTEM'
        return True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [req:%(request_id)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addFilter(RequestIdFilter())
# Ensure other loggers (like similarity_engine) also get the filter if possible, 
# but setting it on root is safer.
logging.getLogger().addFilter(RequestIdFilter())

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(__file__)
FRONTEND_DIR    = os.path.join(BASE_DIR, "frontend")
UPLOAD_FOLDER   = os.path.join(BASE_DIR, "uploads")
DATA_CSV_PATH   = os.path.join(BASE_DIR, "data", "jobs.csv")
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.before_request
def assign_request_id():
    g.request_id = str(uuid.uuid4())[:8]
    g.start_time = time.time()
    logger.info(f"Started {request.method} {request.path}")

@app.after_request
def log_response(response):
    if hasattr(g, 'start_time'):
        elapsed = (time.time() - g.start_time) * 1000
        logger.info(f"Completed {request.method} {request.path} - Status: {response.status_code} - {elapsed:.2f}ms")
    return response

# ---------------------------------------------------------------------------
# ML Model Warm-up
# ---------------------------------------------------------------------------
try:
    from model.ml_scorer import warm_up
    from model.embedding_service import get_model
    
    logger.info("Warming up ML models...")
    warm_up(os.path.join(BASE_DIR, "model_store"))
    get_model()  
    logger.info("ML models warm-up complete.")
except Exception as e:
    logger.warning(f"Model warm-up failed or partially failed: {e}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# In-memory LRU cache for PDF texts to eliminate duplicate extraction 
import functools
@functools.lru_cache(maxsize=32)
def _extract_from_hash(file_hash: str, save_path: str) -> str:
    start_time = time.time()
    text = extract_text_from_pdf(save_path)
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"PDF parsed successfully. Characters: {len(text)}. Time: {elapsed:.2f}ms")
    return text

def _process_uploaded_pdf(resume_file) -> str:
    content = resume_file.read()
    file_hash = hashlib.md5(content).hexdigest()
    
    # Save the file only once per hash to save disk IO if possible, 
    # but for simplicity we'll just use the hash for the memory cache
    safe_name = f"{file_hash}_{secure_filename(resume_file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    
    if not os.path.exists(save_path):
        with open(save_path, "wb") as f:
            f.write(content)
            
    # Reset file pointer if needed by callers
    resume_file.seek(0)
    
    logger.debug(f"Processing PDF with hash: {file_hash}")
    return _extract_from_hash(file_hash, save_path)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def homepage():
    """
    GET /
    Serves the frontend (index.html).
    """
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    -------------
    Form fields expected:
        resume          (file)   – PDF resume to be analyzed.
        job_description (string) – Plain-text job description.

    Returns JSON:
        {
            "score":          <int 0-100>,    # blended semantic ATS score
            "confidence":     <float 0-1>,    # mean skill-match confidence
            "matched_skills": [<str>, ...],   # semantically matched skills
            "missing_skills": [<str>, ...],   # JD skills absent from resume
            "scoring_method": <str>,          # "semantic" | "tfidf_fallback"
            "pros":           [<str>, ...],
            "cons":           [<str>, ...],
            "suggestion":     <str>
        }

    Error responses follow the shape:
        {"error": "<message>"}
    """
    # ------------------------------------------------------------------
    # 1. Validate incoming file
    # ------------------------------------------------------------------
    if "resume" not in request.files:
        return jsonify({"error": "No resume file included in the request. "
                                 "Use field name 'resume'."}), 400

    resume_file = request.files["resume"]

    if resume_file.filename == "":
        return jsonify({"error": "Resume file field is empty."}), 400

    if not _is_allowed_file(resume_file.filename):
        return jsonify({"error": "Only PDF files are accepted."}), 415

    # ------------------------------------------------------------------
    # 2. Validate job description
    # ------------------------------------------------------------------
    job_description: str = request.form.get("job_description", "").strip()
    if not job_description:
        return jsonify({"error": "job_description field is required and "
                                 "must not be empty."}), 400

    # ------------------------------------------------------------------
    # 4. Parse PDF → extract text
    # ------------------------------------------------------------------
    try:
        resume_text = _process_uploaded_pdf(resume_file)
    except Exception as exc:
        logger.error(f"Failed to parse PDF: {exc}")
        return jsonify({"error": f"Failed to parse PDF: {exc}"}), 422

    # ------------------------------------------------------------------
    # 5. Extract skills (for gap analysis)
    # ------------------------------------------------------------------
    start_time = time.time()
    resume_sections = detect_sections(resume_text)
    
    # We can use the LRU cache directly on extract_skills if we want, 
    # but extracting it directly is fine as we'll just log it.
    resume_skills = extract_skills(resume_text)
    logger.info(f"Extracted {len(resume_skills)} skills from resume in {(time.time() - start_time)*1000:.2f}ms")
    
    import pandas as pd
    try:
        df = pd.read_csv(DATA_CSV_PATH)
        for _, row in df.iterrows():
            title = str(row["title"]).lower()
            if title in job_description.lower():
                job_description += " " + str(row["description"]) + " " + str(row["skills"])
    except Exception as exc:
        logger.warning(f"Failed to augment JD with dataset: {exc}")

    start_time = time.time()
    job_skills = extract_skills(job_description)
    logger.info(f"Extracted {len(job_skills)} skills from job description in {(time.time() - start_time)*1000:.2f}ms")

    # ------------------------------------------------------------------
    # 6. Compute match
    # ------------------------------------------------------------------
    start_time = time.time()
    result = compute_match(
        resume_text=resume_text,
        job_text=job_description,
        resume_skills=resume_skills,
        job_skills=job_skills,
    )
    logger.info(f"Match computed in {(time.time() - start_time)*1000:.2f}ms. Score: {result['score']}%")
    logger.info(f"Semantic info - Method: {result.get('scoring_method')}, Confidence: {result.get('confidence', 0):.2f}")
    logger.info(f"Skill overlap - Matched: {len(result.get('matched_skills', []))}, Missing: {len(result.get('missing_skills', []))}")

    # Attach the full resume skill list as extra diagnostic context
    result["resume_skills_found"] = resume_skills

    # Load ideal resume and compare
    ideal_resume_path = os.path.join(BASE_DIR, "model", "ideal_resume.txt")
    try:
        with open(ideal_resume_path, "r", encoding="utf-8") as f:
            ideal_text = f.read()
        comparison = compare_resumes(
            resume_text, ideal_text, resume_skills, job_skills=job_skills
        )
        result["pros"] = comparison["pros"]
        result["cons"] = comparison["cons"]
        result["suggestion"] = comparison["suggestion"]
    except Exception as e:
        logger.error(f"Failed to compare with ideal resume: {e}")
        result["pros"] = []
        result["cons"] = ["Could not load ideal resume for comparison."]
        result["suggestion"] = ""

    # Add the recommendation_reason missing field for standardized output
    from model.ranking_engine import generate_recommendation_reason
    result["recommendation_reason"] = generate_recommendation_reason(result["score"], result["missing_skills"])

    return jsonify(result), 200


@app.route("/rank_jobs", methods=["POST"])
def rank_jobs():
    """
    POST /rank_jobs
    -------------
    Form fields expected:
        resume (file)   – PDF resume to be analyzed.

    Returns JSON:
        {
            "top_jobs": [
                {
                    "title": "...",
                    "score": 85,
                    "missing_skills": []
                }
            ]
        }
    """
    if "resume" not in request.files:
        return jsonify({"error": "No resume file included in the request. Use field name 'resume'."}), 400

    resume_file = request.files["resume"]
    if resume_file.filename == "":
        return jsonify({"error": "Resume file field is empty."}), 400

    if not _is_allowed_file(resume_file.filename):
        return jsonify({"error": "Only PDF files are accepted."}), 415

    try:
        resume_text = _process_uploaded_pdf(resume_file)
    except Exception as exc:
        return jsonify({"error": f"Failed to parse PDF: {exc}"}), 422

    resume_skills = extract_skills(resume_text)

    try:
        result = rank_jobs_for_resume(resume_text, resume_skills, DATA_CSV_PATH)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result), 200


@app.route("/extract_resume_skills", methods=["POST"])
def extract_resume_skills():
    """
    POST /extract_resume_skills
    Returns comma-separated skills from an uploaded PDF.
    """
    if "resume" not in request.files:
        return jsonify({"error": "No resume file."}), 400

    resume_file = request.files["resume"]
    if resume_file.filename == "" or not _is_allowed_file(resume_file.filename):
        return jsonify({"error": "Invalid file."}), 415

    try:
        resume_text = _process_uploaded_pdf(resume_file)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422

    skills = extract_skills(resume_text)
    joined_skills = ", ".join(skills)
    
    return jsonify({"skills": joined_skills}), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
