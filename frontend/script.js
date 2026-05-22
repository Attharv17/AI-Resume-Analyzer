/**
 * script.js  — AI Resume Analyzer frontend logic
 * Updated for the new professional HTML structure.
 */

"use strict";

/* ── DOM refs ───────────────────────────────────────────────── */
const form = document.getElementById("analyzeForm");
const submitBtn = document.getElementById("submitBtn");
const btnText = document.getElementById("btnText");
const btnSpinner = document.getElementById("btnSpinner");
const btnIcon = document.getElementById("btnIcon");

// File
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("resumeFile");
const fileDropUI = document.getElementById("fileDropUI");
const fileSelected = document.getElementById("fileSelected");
const fileSelName = document.getElementById("fileSelectedName");
const clearFileBtn = document.getElementById("clearFile");
const fileGroup = document.getElementById("fileGroup");
const fileError = document.getElementById("fileError");

// JD
const jdTextarea = document.getElementById("jobDescription");
const jdGroup = document.getElementById("jdGroup");
const jdError = document.getElementById("jdError");

// Results
const resultsSection = document.getElementById("resultsSection");
const scoreNumber = document.getElementById("scoreNumber");
const scoreRingFill = document.getElementById("scoreRingFill");
const scoreDesc = document.getElementById("scoreDesc");
const scorePill = document.getElementById("scorePill");
const statMatched = document.getElementById("statMatched");
const statMissing = document.getElementById("statMissing");
const matchedCount = document.getElementById("matchedCount");
const missingCount = document.getElementById("missingCount");
const matchedList = document.getElementById("matchedList");
const missingList = document.getElementById("missingList");
const matchedEmpty = document.getElementById("matchedEmpty");
const missingEmpty = document.getElementById("missingEmpty");

const prosConsGrid = document.getElementById("prosConsGrid");
const prosList = document.getElementById("prosList");
const consList = document.getElementById("consList");
const prosEmpty = document.getElementById("prosEmpty");
const consEmpty = document.getElementById("consEmpty");
const suggestionPanel = document.getElementById("suggestionPanel");
const suggestionText = document.getElementById("suggestionText");

const jobsList = document.getElementById("jobsList");
const resetBtn = document.getElementById("resetBtn");

// Toast
const errorBanner = document.getElementById("errorBanner");
const errorMessage = document.getElementById("errorMessage");
const dismissError = document.getElementById("dismissError");

/* ── Constants ──────────────────────────────────────────────── */
const API_ENDPOINT = "/analyze";
const RANK_ENDPOINT = "/rank_jobs";
const CIRCUMFERENCE = 2 * Math.PI * 52;   // ≈ 326.73 px  (r = 52)

/* ══════════════════════════════════════════════════════════════
   FILE HANDLING
   ══════════════════════════════════════════════════════════════ */
function showFileChosen(file) {
  fileDropUI.hidden = true;
  fileSelected.hidden = false;
  fileSelName.textContent = file.name;
  clearFieldError(fileGroup, fileError);

  // Auto-extract skills from backend
  autoExtractSkills(file);
}

async function autoExtractSkills(file) {
  // Optionally show a loading state in the textarea
  const originalPlaceholder = jdTextarea.placeholder;
  jdTextarea.placeholder = "Extracting skills from resume...";

  try {
    const fd = new FormData();
    fd.append("resume", file);

    const res = await fetch("/extract_resume_skills", { method: "POST", body: fd });
    if (res.ok) {
      const data = await res.json();
      if (data.skills && data.skills.trim() !== "") {
        jdTextarea.value = data.skills;
        clearFieldError(jdGroup, jdError);
      }
    }
  } catch (err) {
    // Auto-extraction failed silently
  } finally {
    jdTextarea.placeholder = originalPlaceholder;
  }
}

function clearFileSelection() {
  fileInput.value = "";
  fileDropUI.hidden = false;
  fileSelected.hidden = true;
  fileSelName.textContent = "";
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) showFileChosen(fileInput.files[0]);
});

clearFileBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearFileSelection();
});

// Keyboard: Enter/Space opens file dialog
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

// Drag-and-drop
["dragenter", "dragover"].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  })
);
["dragleave", "dragend", "drop"].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"))
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  const files = e.dataTransfer?.files;
  if (!files?.length) return;
  const file = files[0];
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showFieldError(fileGroup, fileError, "Only PDF files are accepted.");
    return;
  }
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  showFileChosen(file);
});

/* ══════════════════════════════════════════════════════════════
   VALIDATION
   ══════════════════════════════════════════════════════════════ */
function showFieldError(group, el, msg) {
  group.classList.add("form-group--error");
  el.textContent = msg;
  el.hidden = false;
}
function clearFieldError(group, el) {
  group.classList.remove("form-group--error");
  el.hidden = true;
  el.textContent = "";
}

function validateForm() {
  let valid = true;
  clearFieldError(fileGroup, fileError);
  if (!fileInput.files?.length) {
    showFieldError(fileGroup, fileError, "Please upload your resume as a PDF.");
    valid = false;
  } else if (!fileInput.files[0].name.toLowerCase().endsWith(".pdf")) {
    showFieldError(fileGroup, fileError, "Only PDF files are accepted.");
    valid = false;
  }
  clearFieldError(jdGroup, jdError);
  if (!jdTextarea.value.trim()) {
    showFieldError(jdGroup, jdError, "Please paste the job description.");
    valid = false;
  }
  return valid;
}

/* ══════════════════════════════════════════════════════════════
   LOADING STATE
   ══════════════════════════════════════════════════════════════ */
function setLoading(on) {
  submitBtn.disabled = on;
  btnIcon.hidden = on;
  btnSpinner.hidden = !on;
  btnText.textContent = on ? "Analyzing…" : "Analyze Resume";
}

/* ══════════════════════════════════════════════════════════════
   ERROR TOAST
   ══════════════════════════════════════════════════════════════ */
function showError(msg) {
  errorMessage.textContent = msg;
  errorBanner.hidden = false;
}
function hideError() {
  errorBanner.hidden = true;
  errorMessage.textContent = "";
}
dismissError.addEventListener("click", hideError);

/* ══════════════════════════════════════════════════════════════
   SCORE HELPERS
   ══════════════════════════════════════════════════════════════ */
function scoreStroke(score) {
  if (score >= 75) return "#10b981";   // green
  if (score >= 45) return "#f59e0b";   // amber
  return "#ef4444";                    // red
}

function scorePillInfo(score) {
  if (score >= 80) return { text: "Strong Match", cls: "pill--strong" };
  if (score >= 60) return { text: "Good Match", cls: "pill--good" };
  if (score >= 40) return { text: "Fair Match", cls: "pill--fair" };
  return { text: "Weak Match", cls: "pill--weak" };
}

function scoreText(score) {
  if (score >= 80) return "Your resume is highly aligned with this job. You're a strong candidate for this role.";
  if (score >= 60) return "You match most of the required skills. A few additions to your resume could make it stand out.";
  if (score >= 40) return "Moderate alignment. Review the missing skills — bridging these gaps will significantly improve your fit.";
  if (score >= 20) return "Your resume covers some basics but misses many key requirements. Consider up-skilling in the highlighted areas.";
  return "Low match detected. This role requires skills not yet reflected in your resume.";
}

/* ── Score ring + counter animation ── */
function animateScore(target) {
  // Ring arc
  const stroke = scoreStroke(target);
  const offset = CIRCUMFERENCE * (1 - target / 100);
  scoreRingFill.style.stroke = stroke;
  scoreRingFill.style.strokeDashoffset = offset;

  // Counter
  const ms = 1100;
  const start = performance.now();
  (function tick(now) {
    const p = Math.min((now - start) / ms, 1);
    const e = 1 - Math.pow(1 - p, 3);          // ease-out-cubic
    scoreNumber.textContent = Math.round(e * target);
    if (p < 1) requestAnimationFrame(tick);
  })(performance.now());
}

/* ── Skill pill list ── */
function renderSkills(listEl, skills, emptyEl) {
  listEl.innerHTML = "";
  if (!skills.length) { emptyEl.hidden = false; return; }
  emptyEl.hidden = true;
  skills.forEach((skill, i) => {
    const li = document.createElement("li");
    li.textContent = skill;
    li.style.cssText = `opacity:0;transform:translateY(5px);
      transition:opacity .25s ease ${i * 35}ms, transform .25s ease ${i * 35}ms`;
    listEl.appendChild(li);
    requestAnimationFrame(() => {
      li.style.opacity = "1";
      li.style.transform = "translateY(0)";
    });
  });
}

/* ── Pros/Cons list ── */
function renderProsCons(listEl, items, emptyEl) {
  listEl.innerHTML = "";
  if (!items || !items.length) { emptyEl.hidden = false; return; }
  emptyEl.hidden = true;
  items.forEach((item, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span style="opacity:0.7; margin-right:6px;">•</span><span>${item}</span>`;
    li.style.cssText = `opacity:0; transform:translateY(5px); display:flex; font-size:0.9rem; color:#cbd5e1;
      transition:opacity .25s ease ${i * 35}ms, transform .25s ease ${i * 35}ms`;
    listEl.appendChild(li);
    requestAnimationFrame(() => {
      li.style.opacity = "1";
      li.style.transform = "translateY(0)";
    });
  });
}

/* ── Render Top Jobs ── */
function renderJobs(topJobs) {
  jobsList.innerHTML = "";
  if (!topJobs || !topJobs.length) return;

  topJobs.forEach((job, i) => {
    const el = document.createElement("div");
    el.className = "job-item";
    el.style.cssText = `opacity:0;transform:translateY(10px);
      transition:opacity .3s ease ${i * 100}ms, transform .3s ease ${i * 100}ms`;

    const missingSkills = job.missing_skills.length > 0
      ? job.missing_skills.join(", ")
      : "None";

    el.innerHTML = `
      <div class="job-item__header">
        <h4 class="job-title">${job.title}</h4>
        <span class="job-score-text">${job.score}% Match</span>
      </div>
      <div class="job-progress-track">
        <div class="job-progress-fill" style="width: 0%;" data-target="${job.score}%"></div>
      </div>
      <div class="job-missing-wrap">
        <span class="job-missing-label">Missing:</span>
        <span class="job-missing-skills">${missingSkills}</span>
      </div>
    `;
    jobsList.appendChild(el);

    // Trigger animations
    requestAnimationFrame(() => {
      el.style.opacity = "1";
      el.style.transform = "translateY(0)";

      // Animate progress bar slightly after initial reveal
      setTimeout(() => {
        const fill = el.querySelector(".job-progress-fill");
        fill.style.width = fill.getAttribute("data-target");
      }, 100 + i * 100);
    });
  });
}

/* ══════════════════════════════════════════════════════════════
   RENDER RESULTS
   ══════════════════════════════════════════════════════════════ */
function renderResults(data) {
  const { score, matched_skills, missing_skills } = data;

  // Score ring + pill
  const pill = scorePillInfo(score);
  scorePill.textContent = pill.text;
  scorePill.className = `score-label-pill ${pill.cls}`;
  scoreDesc.textContent = scoreText(score);
  animateScore(score);

  // Summary stats
  statMatched.textContent = matched_skills.length;
  statMissing.textContent = missing_skills.length;

  // Panel counts
  matchedCount.textContent = matched_skills.length;
  missingCount.textContent = missing_skills.length;

  // Skill lists
  renderSkills(matchedList, matched_skills, matchedEmpty);
  renderSkills(missingList, missing_skills, missingEmpty);

  // Pros & Cons
  if (data.pros || data.cons) {
    prosConsGrid.hidden = false;
    renderProsCons(prosList, data.pros || [], prosEmpty);
    renderProsCons(consList, data.cons || [], consEmpty);
    
    if (data.suggestion) {
      suggestionPanel.hidden = false;
      suggestionText.textContent = data.suggestion;
    } else {
      suggestionPanel.hidden = true;
    }
  } else {
    prosConsGrid.hidden = true;
    suggestionPanel.hidden = true;
  }

  // Swap views
  resultsSection.hidden = false;

  // Smooth scroll to the results section
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

/* ══════════════════════════════════════════════════════════════
   RESET
   ══════════════════════════════════════════════════════════════ */
function resetUI() {
  resultsSection.hidden = true;

  // Scroll back to the top form area
  document.body.scrollIntoView({ behavior: "smooth", block: "start" });

  // Reset ring
  scoreRingFill.style.stroke = "#3b82f6";
  scoreRingFill.style.strokeDashoffset = CIRCUMFERENCE;
  scoreNumber.textContent = "0";
  
  prosConsGrid.hidden = true;
  suggestionPanel.hidden = true;

  clearFileSelection();
  jdTextarea.value = "";
  clearFieldError(fileGroup, fileError);
  clearFieldError(jdGroup, jdError);
  hideError();
}
resetBtn.addEventListener("click", resetUI);

/* ══════════════════════════════════════════════════════════════
   FORM SUBMIT → fetch
   ══════════════════════════════════════════════════════════════ */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  if (!validateForm()) return;

  setLoading(true);

  try {
    const fdAnalyze = new FormData();
    fdAnalyze.append("resume", fileInput.files[0]);
    fdAnalyze.append("job_description", jdTextarea.value.trim());

    const fdRank = new FormData();
    fdRank.append("resume", fileInput.files[0]);

    // Perform both fetch operations concurrently
    const [analyzeRes, rankRes] = await Promise.all([
      fetch(API_ENDPOINT, { method: "POST", body: fdAnalyze }),
      fetch(RANK_ENDPOINT, { method: "POST", body: fdRank })
    ]);

    let data;
    try { data = await analyzeRes.json(); }
    catch { throw new Error("Server returned an unreadable response."); }

    if (!analyzeRes.ok) throw new Error(data?.error || `Server error (HTTP ${analyzeRes.status}).`);

    let rankData = { top_jobs: [] };
    if (rankRes.ok) {
      try { rankData = await rankRes.json(); } catch (e) { }
    }

    setLoading(false);
    renderResults(data);
    renderJobs(rankData.top_jobs);

  } catch (err) {
    setLoading(false);
    showError(
      err.name === "TypeError"
        ? "Cannot reach the Flask server. Make sure it is running on http://127.0.0.1:5000."
        : err.message
    );
  }
});
