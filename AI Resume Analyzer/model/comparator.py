"""
comparator.py
-------------
Responsible for comparing a user's resume against an ideal resume to generate pros, cons, and improvement suggestions.
"""

from model.extractor import extract_skills

def get_sections(text: str) -> dict:
    """Detect presence of key sections in the resume text."""
    text_lower = text.lower()
    return {
        "summary": "summary" in text_lower or "profile" in text_lower or "objective" in text_lower,
        "skills": "skills" in text_lower or "technologies" in text_lower,
        "projects": "projects" in text_lower or "portfolio" in text_lower,
        "experience": "experience" in text_lower or "employment" in text_lower or "history" in text_lower or "work" in text_lower
    }

def compare_resumes(user_text: str, ideal_text: str, user_skills: list) -> dict:
    """
    Compare user resume against ideal resume.
    Returns a dictionary with pros, cons, and a score improvement suggestion.
    """
    ideal_skills = extract_skills(ideal_text)
    
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
