"""
collect_jobs.py (Phase 2)
-------------------------
Fetch job listings from a public API (Remote OK) with CSV fallback,
normalize to JSON with:
  - title
  - skills_required (list)
  - experience_required (int)
  - description

Output: jobs_data.json
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import requests

from model.job_requirements import extract_job_requirements

REMOTE_OK_URL = "https://remoteok.com/api"
DEFAULT_OUT = "jobs_data.json"
USER_AGENT = "AIResumeAnalyzer/1.0 (contact: local; education project)"


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"(?s)<script.*?>.*?</script>", " ", raw, flags=re.I)
    text = re.sub(r"(?s)<style.*?>.*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_remote_ok_jobs(limit: int = 25) -> List[Dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(REMOTE_OK_URL, headers=headers, timeout=45)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    jobs: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        if "position" not in row:
            continue
        title = str(row.get("position", "")).strip()
        desc_html = str(row.get("description", "") or "")
        description = _strip_html(desc_html)
        tags = row.get("tags") or []
        skill_tags = [str(t).strip().lower() for t in tags if str(t).strip()]

        req = extract_job_requirements(description + " " + " ".join(skill_tags))
        # Merge tag-like skills with extracted skills (dedupe)
        merged_skills = sorted(set(req["skills_required"]) | set(skill_tags))

        jobs.append(
            {
                "title": title,
                "skills_required": merged_skills,
                "experience_required": req["experience_required"],
                "description": description[:8000],
            }
        )
        if len(jobs) >= limit:
            break
    return jobs


def load_jobs_from_csv(csv_path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = str(row.get("title", "")).strip()
            description = str(row.get("description", "")).strip()
            skills_cell = str(row.get("skills", "")).strip()
            extra = skills_cell.replace(",", " ")
            full_text = f"{description}\n{extra}"
            req = extract_job_requirements(full_text)
            csv_skills = [s.strip().lower() for s in skills_cell.split(",") if s.strip()]
            merged = sorted(set(req["skills_required"]) | set(csv_skills))
            jobs.append(
                {
                    "title": title,
                    "skills_required": merged,
                    "experience_required": req["experience_required"],
                    "description": description,
                }
            )
            if len(jobs) >= limit:
                break
    return jobs


def collect_jobs(
    out_path: str | Path = DEFAULT_OUT,
    limit: int = 25,
    prefer_api: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch jobs and write JSON list to disk.
    Falls back to data/jobs.csv if API fails.
    """
    jobs: List[Dict[str, Any]] = []
    if prefer_api:
        try:
            jobs = fetch_remote_ok_jobs(limit=limit)
        except Exception:
            jobs = []

    if not jobs:
        csv_path = Path(__file__).resolve().parent / "data" / "jobs.csv"
        jobs = load_jobs_from_csv(csv_path, limit=limit)

    Path(out_path).write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    return jobs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Collect job listings into jobs_data.json")
    p.add_argument("-o", "--output", default=DEFAULT_OUT, help="Output JSON path")
    p.add_argument("-n", "--limit", type=int, default=25, help="Max jobs to collect")
    p.add_argument(
        "--no-api",
        action="store_true",
        help="Skip Remote OK API; use local CSV only",
    )
    args = p.parse_args()
    collect_jobs(out_path=args.output, limit=args.limit, prefer_api=not args.no_api)
    print(f"Wrote {args.output}")
