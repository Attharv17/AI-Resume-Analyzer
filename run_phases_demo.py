"""
Demo script: run Phases 1–2 locally and print a short ranking sample (Phases 4–7).

Usage:
  python run_phases_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from collect_jobs import collect_jobs
from model.ml_scorer import ensure_default_model
from model.ranking_engine import rank_jobs
from model.resume_structured import parse_resume_to_json_file


def main() -> None:
    root = Path(__file__).resolve().parent
    pdf = root / "sample_resume.pdf"
    if not pdf.is_file():
        raise SystemExit("sample_resume.pdf not found in project root.")

    parse_resume_to_json_file(pdf, root / "resume_data.json")
    collect_jobs(out_path=root / "jobs_data.json", limit=30, prefer_api=False)

    resume = json.loads((root / "resume_data.json").read_text(encoding="utf-8"))
    jobs = json.loads((root / "jobs_data.json").read_text(encoding="utf-8"))
    model = ensure_default_model(root / "model_store")
    top = rank_jobs(resume, jobs, model, top_n=5)

    print("Top 5 jobs:")
    for row in top:
        print(f"- {row['title']}: {row['score']} | missing: {', '.join(row['missing_skills']) or '—'}")


if __name__ == "__main__":
    main()
