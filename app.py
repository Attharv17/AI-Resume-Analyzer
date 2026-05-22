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
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from model.parser import extract_text_from_pdf
from model.skill_extractor import extract_skills, detect_sections
from model.matcher import compute_match
from model.ranker import rank_jobs_for_resume
from model.comparator import compare_resumes

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_allowed_file(filename: str) -> bool:
    """Return True if the uploaded file has a .pdf extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def _process_uploaded_pdf(resume_file) -> str:
    """
    Save the uploaded PDF securely and return cleaned, parsed text.

    Uses the enhanced parser (block-sorted PyMuPDF + pdfplumber fallback)
    followed by the preprocessing noise-removal pipeline.
    """
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(resume_file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    resume_file.save(save_path)
    # extract_text_from_pdf now handles multi-column layout and noise removal
    return extract_text_from_pdf(save_path)


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

    # 4. Parse PDF → extract text
    # ------------------------------------------------------------------
    try:
        resume_text = _process_uploaded_pdf(resume_file)
    except Exception as exc:
        return jsonify({"error": f"Failed to parse PDF: {exc}"}), 422

    # ------------------------------------------------------------------
    # 5. Extract skills (for gap analysis)
    # ------------------------------------------------------------------
    # Use section-aware extraction: skills found in the Skills section
    # are surfaced first; the full-text scan catches the rest.
    resume_sections = detect_sections(resume_text)
    resume_skills = extract_skills(resume_text)
    
    # Optional enhancement: if user typed a job title like "Data Scientist" in the description,
    # augment it with our database so we can accurately output missing skills!
    import pandas as pd
    try:
        df = pd.read_csv(DATA_CSV_PATH)
        for _, row in df.iterrows():
            title = str(row["title"]).lower()
            if title in job_description.lower():
                # Augment the text for tf-idf and extraction exactly like ranker.py
                job_description += " " + str(row["description"]) + " " + str(row["skills"])
    except Exception:
        pass

    job_skills = extract_skills(job_description)

    # ------------------------------------------------------------------
    # 6. Compute match:
    #    • TF-IDF cosine similarity on full texts → score
    #    • Set intersection / difference         → matched / missing skills
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    result = compute_match(
        resume_text=resume_text,
        job_text=job_description,
        resume_skills=resume_skills,
        job_skills=job_skills,
    )

    # Attach the full resume skill list as extra diagnostic context
    result["resume_skills_found"] = resume_skills

    # Load ideal resume and compare
    ideal_resume_path = os.path.join(BASE_DIR, "model", "ideal_resume.txt")
    try:
        with open(ideal_resume_path, "r", encoding="utf-8") as f:
            ideal_text = f.read()
        # Pass job_skills so pros/cons reflect the actual JD, not always
        # the hardcoded Data Science ideal profile.
        comparison = compare_resumes(
            resume_text, ideal_text, resume_skills, job_skills=job_skills
        )
        result["pros"] = comparison["pros"]
        result["cons"] = comparison["cons"]
        result["suggestion"] = comparison["suggestion"]
    except Exception as e:
        result["pros"] = []
        result["cons"] = ["Could not load ideal resume for comparison."]
        result["suggestion"] = ""

    import json
    with open(os.path.join(BASE_DIR, "debug_analyze.json"), "w") as f:
        json.dump({
            "job_description_raw": request.form.get("job_description", "").strip(),
            "job_description_aug": job_description,
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "scoring_method": result.get("scoring_method", "unknown"),
            "confidence": result.get("confidence", None),
            "result": result,
        }, f, indent=2)

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
