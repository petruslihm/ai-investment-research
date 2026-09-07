import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";

const routes = [
  { path: "/", label: "Dashboard" },
  { path: "/portfolio", label: "Portfolio" },
  { path: "/recommendations", label: "Stock Recommendations" },
  { path: "/btc", label: "BTC Signal" },
  { path: "/models", label: "Models / Learning" },
  { path: "/runtime", label: "Runtime Activity" },
  { path: "/data-health", label: "Data Health" },
  { path: "/settings", label: "Settings" },
];

function useJson<T>(url: string): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    fetch(url)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null));
  }, [url]);
  return data;
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <main>
      <h1>{title}</h1>
      {children}
    </main>
  );
}

function Dashboard() {
  const snap = useJson<Record<string, unknown>>("/api/v1/snapshot");
  const alloc = (snap?.allocation || {}) as Record<string, unknown>;
  return (
    <Panel title="Dashboard">
      <p>No automatic trading. Historical diagnostics are not live performance.</p>
      <p>{String(snap?.survivorship_warning || "")}</p>
      <pre>{JSON.stringify(alloc, null, 2)}</pre>
    </Panel>
  );
}

function Recommendations() {
  const snap = useJson<{
    quant?: unknown[];
    llm_final?: unknown[];
    llm_judge_status?: string;
    quant_status?: string;
  }>("/api/v1/snapshot");
  return (
    <Panel title="Stock Recommendations">
      <p>
        LLM FINAL JUDGE Status: {snap?.llm_judge_status || "…"} — Quant recommendation:{" "}
        {snap?.quant_status || "…"}
      </p>
      <h2>quant_only</h2>
      <pre>{JSON.stringify(snap?.quant, null, 2)}</pre>
      <h2>llm_final</h2>
      <pre>{JSON.stringify(snap?.llm_final, null, 2)}</pre>
    </Panel>
  );
}

function Btc() {
  const snap = useJson<{ quant?: Array<Record<string, unknown>> }>("/api/v1/snapshot");
  const rows = (snap?.quant || []).filter((r) => String(r.instrument_id).includes("btc"));
  return (
    <Panel title="BTC Signal">
      <p>BTC/USD only. Transfer friction and hysteresis apply.</p>
      <pre>{JSON.stringify(rows, null, 2)}</pre>
    </Panel>
  );
}

function Models() {
  const data = useJson<{ events?: unknown[] }>("/api/v1/journal");
  return (
    <Panel title="Models / Learning">
      <p>Walk-forward MAE is stored on promote events. Live scores are separate.</p>
      <pre>{JSON.stringify(data?.events, null, 2)}</pre>
    </Panel>
  );
}

function Runtime() {
  const data = useJson<{ events?: unknown[] }>("/api/v1/runtime");
  return (
    <Panel title="Runtime Activity">
      <pre>{JSON.stringify(data?.events, null, 2)}</pre>
    </Panel>
  );
}

function DataHealth() {
  const data = useJson<Record<string, unknown>>("/api/v1/data-health");
  const snap = useJson<{ capabilities?: Record<string, string> }>("/api/v1/snapshot");
  return (
    <Panel title="Data Health">
      <p>Missing keys = NOT_CONFIGURED. Failed live calls = UNAVAILABLE.</p>
      <pre>{JSON.stringify({ health: data, capabilities: snap?.capabilities }, null, 2)}</pre>
    </Panel>
  );
}

function SettingsHint() {
  return (
    <Panel title="Settings">
      <p>
        Use the FastAPI Settings page at <code>/settings</code> to save API keys. This React view
        reads live JSON from the same backend.
      </p>
    </Panel>
  );
}

function PortfolioHint() {
  return (
    <Panel title="Portfolio">
      <p>Unit-first lots are edited on the FastAPI Portfolio page. AI never mutates holdings.</p>
    </Panel>
  );
}

export function App() {
  return (
    <div className="layout">
      <aside>
        <div className="brand">Investment Assistant</div>
        <nav>
          {routes.map((r) => (
            <NavLink key={r.path} to={r.path} end={r.path === "/"}>
              {r.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/portfolio" element={<PortfolioHint />} />
        <Route path="/recommendations" element={<Recommendations />} />
        <Route path="/btc" element={<Btc />} />
        <Route path="/models" element={<Models />} />
        <Route path="/runtime" element={<Runtime />} />
        <Route path="/data-health" element={<DataHealth />} />
        <Route path="/settings" element={<SettingsHint />} />
      </Routes>
    </div>
  );
}
