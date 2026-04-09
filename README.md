AI Resume Analyzer
The AI Resume Analyzer is a lightweight, modular web application built with a Flask Python backend and a vanilla HTML/JS frontend. It empowers users to instantly evaluate how well their resume matches a specific job description, and goes a step further by comparing the resume against a database of open roles to recommend the top matches.

✨ Features
Intelligent PDF Parsing: Uses PyMuPDF to cleanly extract structured text directly from PDF resumes.
Semantic Matching: Calculates a match score utilizing scikit-learn's TfidfVectorizer and cosine_similarity to measure natural language overlap between your resume and a job requirement.
Skill Gap Analysis: Extracts over 60 exact skill keywords (from basic frontend to deep ML tech) using regex, explicitly listing what skills you currently match and what you need to acquire.
Job Ranking System: Employs Pandas to rank your resume globally against a CSV dataset of open roles (data/jobs.csv), presenting the top 5 positions where your skills are most desired.
Ideal Profile Evaluation: Compares your resume against an "ideal resume" standard, evaluating section completeness and matching skills to give you exact Pros, Cons, and actionable Improvement Suggestions.
Dynamic GUI: A professional Slate/Navy theme built with vanilla HTML/CSS/JS that features animated score rings and asynchronously updates both single-job analysis and top dataset matches.
🚀 Getting Started
1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

2. Installation
Navigate into the project directory and install the required dependencies:

pip install -r requirements.txt
(Dependencies include Flask, PyMuPDF, scikit-learn, and pandas)

3. Running the Server
Launch the Flask development server by running:

python app.py
Wait for the terminal to display * Running on http://127.0.0.1:5000. Keep this terminal window open.

4. Open the Website
Open your web browser and navigate to: http://127.0.0.1:5000/

🖥️ How to Use
Upload Resume: Drag & drop your PDF resume into the upload zone.
Provide Job Description:
You can explicitly type or paste the text of the job description you are targeting into the text area.
Pro tip: Once you upload your resume, the system automatically runs an extraction step and pre-fills the text area with the skills it found. You can either keep this to analyze against your own skill set, or replace it entirely with a real Job description from a job board (like LinkedIn or Indeed) for accurate targeting!
Analyze: Click Analyze Resume. The system will parse your document and asynchronously return:
Your direct match score and skill breakdown for the job description.
Pros, Cons, and a suggestion score based on a comparison with a built-in industry standard ideal resume.
A ranked list of the absolute best alternative jobs you fit into from the local jobs.csv dataset.
