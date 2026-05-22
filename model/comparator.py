"""
comparator.py
-------------
Compares a user's resume against an ideal profile to generate pros, cons,
and improvement suggestions.

Now accepts an optional ``job_skills`` argument so comparisons can be
made against the actual job description rather than the static ideal_resume.txt.
"""

from model.skill_extractor import extract_skills, detect_sections

def get_sections(text: str) -> dict:
    """Detect presence of key sections in the resume text."""
    detected = detect_sections(text)
    known = {"summary", "skills", "projects", "experience", "certifications", "education"}
    return {k: (k in detected) for k in known}

def compare_resumes(
    user_text: str,
    ideal_text: str,
    user_skills: list,
    job_skills: list | None = None,
) -> dict:
    """
    Compare user resume against an ideal profile.

    If ``job_skills`` is provided, pros/cons are generated relative to the
    actual job description skills instead of the static ideal_resume.txt.
    This makes the feedback far more relevant to what the user applied for.

    Returns a dictionary with pros, cons, and a score improvement suggestion.
    """
    # Use job_skills as the reference if provided, else fall back to ideal_resume.txt
    ideal_skills = job_skills if job_skills else extract_skills(ideal_text)
    
    user_sections = get_sections(user_text)
    ideal_sections = get_sections(ideal_text)
    
    missing_skills = [s for s in ideal_skills if s not in user_skills]
    extra_skills = [s for s in user_skills if s not in ideal_skills]
    
    pros = []
    cons = []
    
    # Generate Pros
    matched = [s for s in ideal_skills if s in user_skills]
    if matched:
        if len(matched) > 2:
            pros.append(f"Strong match for key skills: {', '.join(matched[:3])} and more")
        else:
            pros.append(f"Has important key skills: {', '.join(matched)}")
            
    if extra_skills:
        pros.append(f"Bonus skills detected offering a unique advantage: {', '.join(extra_skills[:3])}")
        
    strong_sections = [sec.capitalize() for sec, present in user_sections.items() if present]
    if strong_sections:
        pros.append(f"Good structure! Strong sections present: {', '.join(strong_sections)}")
        
    # Generate Cons
    if missing_skills:
         cons.append(f"Missing important skills from ideal profile: {', '.join(missing_skills[:3])}")
         
    missing_sections = [sec.capitalize() for sec, present in user_sections.items() if not present]
    if missing_sections:
         cons.append(f"Missing important sections: {', '.join(missing_sections)}")
         
    if not user_skills:
         cons.append("Lacks industry keywords and verifiable tech skills")
         
    # Generate Suggestion
    improvement_score = (len(missing_skills[:5]) * 2) + (len(missing_sections) * 5)
    suggestion = ""
    if missing_skills or missing_sections:
        to_add = []
        if missing_skills: 
            to_add.append(f"{missing_skills[0]}")
        if missing_sections: 
            to_add.append(f"a {missing_sections[0].lower()} section")
            
        suggestion = f"Adding {', '.join(to_add)} can improve your score by up to {improvement_score}%!"
    else:
        suggestion = "You have an excellent resume structure! Keep it up to date."
        
    return {
        "pros": pros,
        "cons": cons,
        "suggestion": suggestion
    }
