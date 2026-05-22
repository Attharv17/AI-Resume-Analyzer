import { useMemo, useState } from "react";

const API = ""; // Vite proxy -> FastAPI on :8000

export default function App() {
  const [file, setFile] = useState(null);
  const [resume, setResume] = useState(null);
  const [jobs, setJobs] = useState(null);
  const [top, setTop] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const canRun = useMemo(() => !!file, [file]);

  async function uploadResume() {
    setErr("");
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("resume", file);
      const r = await fetch(`${API}/upload-resume`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      setResume(await r.json());
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function loadJobs() {
    setErr("");
    setBusy(true);
    try {
      const r = await fetch(`${API}/jobs`);
      if (!r.ok) throw new Error(await r.text());
      setJobs(await r.json());
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function match() {
    setErr("");
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("resume", file);
      const r = await fetch(`${API}/match?top_n=10`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      setTop(await r.json());
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>AI Resume Analyzer</h1>
        <p>Upload a PDF, preview parsed fields, then run hybrid matching against jobs.</p>
      </header>

      <section className="card">
        <h2>1) Resume PDF</h2>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <div className="row">
          <button disabled={!canRun || busy} onClick={uploadResume}>
            Parse resume
          </button>
          <button disabled={!canRun || busy} onClick={match}>
            Find top matches
          </button>
          <button disabled={busy} onClick={loadJobs}>
            Load jobs JSON
          </button>
        </div>
        {err ? <pre className="err">{err}</pre> : null}
      </section>

      {resume ? (
        <section className="card">
          <h2>Parsed resume</h2>
          <pre className="pre">{JSON.stringify(resume, null, 2)}</pre>
        </section>
      ) : null}

      {jobs ? (
        <section className="card">
          <h2>Jobs ({Array.isArray(jobs) ? jobs.length : 0})</h2>
          <pre className="pre">{JSON.stringify(jobs, null, 2).slice(0, 8000)}</pre>
        </section>
      ) : null}

      {top ? (
        <section className="card">
          <h2>Top matches</h2>
          <pre className="pre">{JSON.stringify(top, null, 2)}</pre>

          <h3>Missing skills (top jobs)</h3>
          <ul className="list">
            {(top.top_jobs || []).map((j) => (
              <li key={j.title}>
                <div className="title">
                  <strong>{j.title}</strong> — score: {j.score}
                </div>
                <div className="sub">Missing: {(j.missing_skills || []).join(", ") || "—"}</div>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <footer className="foot">
        Run API: <code>python api_fastapi.py</code> then <code>npm run dev</code> here.
      </footer>
    </div>
  );
}
