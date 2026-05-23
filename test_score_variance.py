import sys
sys.path.insert(0, r'd:\Coding\AI Resume Analyzer')

# Suppress transformer download noise
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from services.ats_engine import compute_semantic_match

cases = [
    {
        "label": "EXCELLENT (perfect skill match)",
        "resume_skills": ["python", "fastapi", "postgresql", "docker", "redis", "aws", "git", "rest api", "kubernetes"],
        "job_skills":    ["python", "fastapi", "postgresql", "docker", "redis", "aws"],
        "resume_text":   "Senior backend engineer 6 years Python FastAPI PostgreSQL Docker Redis AWS Kubernetes microservices REST API CI/CD",
        "job_text":      "Backend engineer Python FastAPI PostgreSQL Docker Redis AWS cloud deployment microservices",
    },
    {
        "label": "GOOD (60-70% overlap, related)",
        "resume_skills": ["python", "fastapi", "docker", "aws", "git", "linux"],
        "job_skills":    ["python", "fastapi", "postgresql", "docker", "redis", "aws"],
        "resume_text":   "Backend developer Python FastAPI Docker AWS Git Linux 4 years building APIs",
        "job_text":      "Backend engineer Python FastAPI PostgreSQL Docker Redis AWS cloud",
    },
    {
        "label": "AVERAGE (partial overlap, different stack)",
        "resume_skills": ["python", "django", "mysql", "html", "css", "git"],
        "job_skills":    ["python", "fastapi", "postgresql", "docker", "redis", "aws"],
        "resume_text":   "Python developer using Django and MySQL. Frontend with HTML CSS. 2 years experience.",
        "job_text":      "Backend engineer Python FastAPI PostgreSQL Docker Redis AWS Kubernetes",
    },
    {
        "label": "WEAK (wrong domain, designer)",
        "resume_skills": ["photoshop", "figma", "illustrator", "sketch", "ux design"],
        "job_skills":    ["python", "fastapi", "postgresql", "docker", "redis", "aws"],
        "resume_text":   "Graphic designer Figma Photoshop Illustrator UX UI wireframing prototyping logo design branding",
        "job_text":      "Backend engineer Python FastAPI PostgreSQL Docker Redis AWS Kubernetes microservices",
    },
]

print("=" * 75)
print(f"{'Label':<38} {'Score':>6}   cos   str  exact  sem   pen   bst")
print("=" * 75)

for c in cases:
    result = compute_semantic_match(
        resume_text=c["resume_text"],
        job_text=c["job_text"],
        resume_skills=c["resume_skills"],
        job_skills=c["job_skills"],
    )
    bd = result["score_breakdown"]
    print(
        f"{c['label']:<38} {result['score']:>5}%"
        f"  {bd['raw_cosine']:.2f}"
        f"  {bd['stretched_sim']:.2f}"
        f"  {bd['exact_overlap']:.2f}"
        f"  {bd['semantic_coverage']:.2f}"
        f"  -{bd['penalty']:.2f}"
        f"  +{bd['boost']:.2f}"
    )
    if bd["penalty_reasons"]:
        for r in bd["penalty_reasons"]:
            print(f"   {'':38}  PENALTY: {r}")
    if bd["boost_reasons"]:
        for r in bd["boost_reasons"]:
            print(f"   {'':38}  BOOST:   {r}")

print("=" * 75)
print("\nTarget ranges:")
print("  Excellent  -> 85-95%")
print("  Good       -> 70-85%")
print("  Average    -> 45-70%")
print("  Weak       -> below 45%")
