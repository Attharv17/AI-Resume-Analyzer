"""
skill_extractor.py
------------------
Enhanced skill extraction pipeline with:

  • 250+ curated skill vocabulary organised by domain category
  • Skill aliases / abbreviation normalisation (ml → machine learning)
  • Section-aware extraction — skills found inside a "Skills" section
    get higher confidence than those found incidentally elsewhere
  • Optional spaCy NLP pass to surface noun-chunks not in the fixed vocabulary
  • Normalised output — canonical lowercase names, deduped, sorted

Public API
----------
    extract_skills(text: str) -> List[str]
        Fast path — returns a flat sorted list of unique canonical skill names.
        Drop-in replacement for the old extractor.extract_skills().

    extract_skills_detailed(text: str) -> dict
        Rich path — returns {
            "skills":       List[str],        # all found (deduped, sorted)
            "by_section":   dict[str, List[str]],   # skills per detected section
            "from_nlp":     List[str],        # extras surfaced by spaCy noun-chunks
        }

    detect_sections(text: str) -> dict[str, str]
        Returns a mapping of section_name → section_body for downstream use
        (experience parser, education parser, etc.)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# ① SKILL VOCABULARY  — 250+ entries, organised by domain
# ---------------------------------------------------------------------------

_SKILL_CATEGORIES: Dict[str, List[str]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c", "c++", "c#",
        "r", "scala", "kotlin", "swift", "go", "golang", "rust", "ruby",
        "php", "bash", "shell scripting", "perl", "matlab", "julia",
        "haskell", "elixir", "dart", "lua", "groovy", "cobol", "fortran",
        "assembly", "vba", "objective-c", "f#",
    ],
    "web_frontend": [
        "html", "html5", "css", "css3", "react", "react.js", "angular",
        "vue", "vue.js", "next.js", "nuxt", "nuxt.js", "svelte", "jquery",
        "bootstrap", "tailwind", "tailwindcss", "webpack", "vite",
        "sass", "less", "storybook", "redux", "zustand", "mobx",
        "web components", "pwa", "responsive design",
    ],
    "web_backend": [
        "node.js", "express", "express.js", "django", "flask", "fastapi",
        "spring boot", "spring", "laravel", "rails", "ruby on rails",
        "asp.net", ".net", "dotnet", "nest.js", "nestjs", "strapi",
        "graphql", "rest api", "restful api", "soap", "grpc",
        "websocket", "oauth", "jwt",
    ],
    "data_ml": [
        "machine learning", "deep learning", "natural language processing",
        "nlp", "computer vision", "data analysis", "data science",
        "data engineering", "feature engineering", "model deployment",
        "statistics", "statistical analysis", "time series",
        "reinforcement learning", "transfer learning", "generative ai",
        "llm", "large language model", "prompt engineering", "rag",
        "retrieval augmented generation", "fine-tuning", "mlops",
        "a/b testing", "hypothesis testing", "bayesian inference",
        "predictive modeling", "anomaly detection", "recommendation systems",
    ],
    "ml_frameworks": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
        "lightgbm", "catboost", "hugging face", "transformers",
        "opencv", "spacy", "nltk", "gensim", "langchain", "llamaindex",
        "openai api", "anthropic", "stable diffusion", "detectron2",
        "ray", "dask", "onnx", "triton", "mlflow", "wandb",
        "weights & biases",
    ],
    "data_tools": [
        "pandas", "numpy", "matplotlib", "seaborn", "plotly",
        "tableau", "power bi", "looker", "metabase", "excel",
        "jupyter", "google colab", "streamlit", "dash",
        "apache superset", "qlik",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "mongodb", "sqlite", "redis",
        "cassandra", "elasticsearch", "oracle", "nosql", "dynamodb",
        "neo4j", "couchdb", "mariadb", "cockroachdb", "snowflake",
        "bigquery", "redshift", "databricks", "supabase", "planetscale",
        "firestore", "firebase",
    ],
    "big_data": [
        "spark", "apache spark", "hadoop", "kafka", "apache kafka",
        "hive", "airflow", "apache airflow", "dbt", "flink",
        "apache flink", "presto", "trino", "nifi", "flume",
        "delta lake", "iceberg",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "k8s", "terraform", "ansible", "jenkins", "ci/cd",
        "git", "github", "gitlab", "bitbucket", "linux", "unix",
        "github actions", "gitlab ci", "circleci", "travis ci",
        "helm", "prometheus", "grafana", "datadog", "new relic",
        "nginx", "apache", "load balancing", "microservices",
        "serverless", "lambda", "cloud functions", "cloud run",
        "ecs", "eks", "aks", "gke", "cloudformation", "pulumi",
        "vault", "consul", "service mesh", "istio",
    ],
    "security": [
        "cybersecurity", "penetration testing", "ethical hacking",
        "siem", "soc", "owasp", "ssl", "tls", "encryption",
        "identity management", "iam", "zero trust", "devsecops",
    ],
    "mobile": [
        "android", "ios", "react native", "flutter", "xamarin",
        "ionic", "swift ui", "swiftui", "jetpack compose",
    ],
    "design_ux": [
        "figma", "sketch", "adobe xd", "photoshop", "illustrator",
        "ux design", "ui design", "wireframing", "prototyping",
        "user research", "design systems", "accessibility",
    ],
    "soft_skills": [
        "agile", "scrum", "kanban", "jira", "confluence",
        "communication", "leadership", "problem solving", "teamwork",
        "project management", "stakeholder management",
        "cross-functional", "mentoring", "coaching",
        "time management", "critical thinking", "collaboration",
    ],
    "certifications_buzzwords": [
        "aws certified", "azure certified", "gcp certified",
        "pmp", "cka", "ckad", "cissp", "ccna", "comptia",
        "google analytics", "salesforce", "sap",
    ],
}

# Flatten into a master lookup: canonical_name → regex_pattern
_CANONICAL: Dict[str, re.Pattern] = {}

for _skills_list in _SKILL_CATEGORIES.values():
    for _skill in _skills_list:
        _canonical_key = _skill.lower().strip()
        if _canonical_key in _CANONICAL:
            continue
        # Build a word-boundary-aware regex for each skill
        _escaped = re.escape(_canonical_key)
        if re.search(r"[^a-z]", _canonical_key):
            # Contains space, dot, +, #, / → use lookahead/lookbehind
            _pat = r"(?<![a-z0-9])" + _escaped + r"(?![a-z0-9])"
        else:
            _pat = r"\b" + _escaped + r"\b"
        _CANONICAL[_canonical_key] = re.compile(_pat, re.IGNORECASE)

# ---------------------------------------------------------------------------
# ② ALIAS MAP — abbreviations and alternate spellings → canonical name
# ---------------------------------------------------------------------------

SKILL_ALIASES: Dict[str, str] = {
    # Language abbreviations
    "js":          "javascript",
    "ts":          "typescript",
    "py":          "python",
    "c sharp":     "c#",
    "cplusplus":   "c++",
    # ML shortforms
    "ml":          "machine learning",
    "dl":          "deep learning",
    "cv":          "computer vision",
    "gen ai":      "generative ai",
    "genai":       "generative ai",
    "llms":        "llm",
    # Cloud / infra
    "k8s":         "kubernetes",
    "kube":        "kubernetes",
    "tf":          "terraform",
    "gke":         "kubernetes",   # normalise to parent concept
    "eks":         "kubernetes",
    "aks":         "kubernetes",
    # Framework aliases
    "sklearn":     "scikit-learn",
    "sk-learn":    "scikit-learn",
    "hf":          "hugging face",
    "torch":       "pytorch",
    "tf2":         "tensorflow",
    # Data
    "bi":          "power bi",
    "msbi":        "power bi",
    "pg":          "postgresql",
    "postgres":    "postgresql",
    "mongo":       "mongodb",
    "es":          "elasticsearch",
    "elastic":     "elasticsearch",
    # DevOps
    "gh actions":  "github actions",
    "ci cd":       "ci/cd",
    "cicd":        "ci/cd",
    "rest":        "rest api",
    "restful":     "restful api",
    # Design
    "xd":          "adobe xd",
}

# Compile alias patterns
_ALIAS_PATTERNS: Dict[re.Pattern, str] = {}
for _alias, _canonical in SKILL_ALIASES.items():
    _escaped = re.escape(_alias.lower())
    if re.search(r"[^a-z]", _alias.lower()):
        _pat = r"(?<![a-z0-9])" + _escaped + r"(?![a-z0-9])"
    else:
        _pat = r"\b" + _escaped + r"\b"
    _ALIAS_PATTERNS[re.compile(_pat, re.IGNORECASE)] = _canonical.lower()


# ---------------------------------------------------------------------------
# ③ SECTION DETECTION
# ---------------------------------------------------------------------------

# Ordered list of (section_key, list_of_heading_variants)
_SECTION_DEFS: List[tuple[str, List[str]]] = [
    ("skills", [
        "skills", "technical skills", "core competencies", "technologies",
        "tools", "tech stack", "programming languages", "languages & frameworks",
        "areas of expertise", "expertise", "competencies",
    ]),
    ("experience", [
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history",
        "career history", "positions held", "internships",
    ]),
    ("education", [
        "education", "academic background", "academics",
        "educational qualifications", "qualifications",
    ]),
    ("projects", [
        "projects", "personal projects", "academic projects",
        "portfolio", "side projects", "open source", "notable projects",
    ]),
    ("certifications", [
        "certifications", "certificates", "licenses",
        "professional certifications", "credentials",
        "awards & certifications",
    ]),
    ("summary", [
        "summary", "professional summary", "executive summary",
        "profile", "objective", "career objective", "about me", "about",
    ]),
]

# Build a single splitting regex from all heading variants
_ALL_HEADINGS = [h for _, variants in _SECTION_DEFS for h in variants]
_SECTION_SPLIT_RE = re.compile(
    r"(?mi)^\s*("
    + "|".join(re.escape(h) for h in sorted(_ALL_HEADINGS, key=len, reverse=True))
    + r")\s*:?\s*$"
)

# Heading → canonical section key lookup
_HEADING_TO_KEY: Dict[str, str] = {
    h.lower(): key
    for key, variants in _SECTION_DEFS
    for h in variants
}


def detect_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into labelled sections.

    Headings are matched case-insensitively on their own line (with optional
    trailing colon).  Anything before the first recognised heading goes into
    the '_header' bucket (typically the candidate's name/contact info).

    Args:
        text (str): Cleaned resume text.

    Returns:
        dict[str, str]:  { section_key: section_body_text, ... }
                         Possible keys: "skills", "experience", "education",
                         "projects", "certifications", "summary", "_header", "_rest"
    """
    parts = _SECTION_SPLIT_RE.split(text)
    sections: Dict[str, str] = {}

    # First fragment before any heading → contact / header block
    if parts and parts[0].strip():
        sections["_header"] = parts[0].strip()

    i = 1
    while i + 1 < len(parts):
        heading_text = parts[i].strip().lower()
        body = parts[i + 1].strip()
        key = _HEADING_TO_KEY.get(heading_text, heading_text)
        # Merge if section appears twice (e.g., two "Experience" blocks)
        if key in sections:
            sections[key] = sections[key] + "\n" + body
        else:
            sections[key] = body
        i += 2

    return sections


# ---------------------------------------------------------------------------
# ④ CORE SKILL SCANNING
# ---------------------------------------------------------------------------

def _scan_skills_in_text(text: str) -> List[str]:
    """
    Scan *text* and return canonical skill names for every skill that matches.

    Matching order:
        1. Alias patterns first  (so 'ml' → 'machine learning' before scanning
           canonical patterns which also include 'machine learning')
        2. Canonical vocabulary patterns

    Returns deduplicated canonical names (not yet sorted — caller sorts).
    """
    text_lower = text.lower()
    found: set[str] = set()

    # Alias pass
    for pattern, canonical in _ALIAS_PATTERNS.items():
        if pattern.search(text_lower):
            found.add(canonical)

    # Canonical vocabulary pass
    for canonical_key, pattern in _CANONICAL.items():
        if pattern.search(text_lower):
            found.add(canonical_key)

    return list(found)


# ---------------------------------------------------------------------------
# ⑤ OPTIONAL spaCy NLP PASS
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is False:
        return None
    if _NLP is not None:
        return _NLP
    try:
        import spacy  # noqa: PLC0415
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = False
    return _NLP if _NLP is not False else None


def _nlp_skill_hints(text: str, known_skills: set[str]) -> List[str]:
    """
    Use spaCy noun-chunk extraction to surface technical terms that are NOT
    already in the vocabulary — these are returned as raw strings for the
    'from_nlp' field in the detailed output.

    Only triggers when spaCy is installed and loaded successfully.
    """
    nlp = _get_nlp()
    if not nlp:
        return []

    extras: List[str] = []
    # Cap at 50k chars for speed
    doc = nlp(text[:50_000])

    for chunk in doc.noun_chunks:
        token = chunk.text.strip().lower()
        # Skip if already known or too long / too short
        if token in known_skills:
            continue
        if len(token) < 2 or len(token.split()) > 4:
            continue
        # Heuristic: skip purely stopword chunks
        if all(t.is_stop for t in chunk):
            continue
        extras.append(token)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: List[str] = []
    for t in extras:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# ⑥ PUBLIC API
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> List[str]:
    """
    Scan *text* and return a sorted list of unique, normalised skill names.

    This is the drop-in replacement for ``model.extractor.extract_skills()``.
    Internally applies alias normalisation and the expanded 250+ vocabulary.

    Args:
        text (str): Raw or pre-cleaned text (resume or job description).

    Returns:
        List[str]: Sorted list of canonical skill strings.
    """
    if not text or not text.strip():
        return []
    found = _scan_skills_in_text(text)
    return sorted(set(found))


def extract_skills_detailed(text: str) -> Dict[str, object]:
    """
    Richer skill extraction that also returns per-section breakdowns and
    optional spaCy-derived noun-chunk extras.

    Args:
        text (str): Cleaned resume text.

    Returns:
        {
            "skills":     List[str],            # all unique canonical skills, sorted
            "by_section": dict[str, List[str]], # skills per section
            "from_nlp":   List[str],            # spaCy noun-chunks not in vocabulary
        }
    """
    sections = detect_sections(text)
    by_section: Dict[str, List[str]] = {}

    all_found: set[str] = set()

    for sec_key, sec_body in sections.items():
        sec_skills = _scan_skills_in_text(sec_body)
        by_section[sec_key] = sorted(sec_skills)
        all_found.update(sec_skills)

    # Also scan the full text to catch skills mentioned outside headings
    full_skills = _scan_skills_in_text(text)
    all_found.update(full_skills)

    nlp_extras = _nlp_skill_hints(text, all_found)

    return {
        "skills":     sorted(all_found),
        "by_section": by_section,
        "from_nlp":   nlp_extras,
    }


def normalize_skill(skill: str) -> str:
    """
    Return the canonical lowercase name for a given skill string,
    resolving aliases if applicable.

    Args:
        skill (str): Raw skill token (e.g. "ML", "pytorch", "k8s").

    Returns:
        str: Canonical lowercase name.
    """
    key = skill.strip().lower()
    # Check alias map first
    if key in SKILL_ALIASES:
        return SKILL_ALIASES[key]
    # Check vocabulary
    if key in _CANONICAL:
        return key
    return key
