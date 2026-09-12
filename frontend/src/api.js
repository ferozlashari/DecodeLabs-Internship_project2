const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const api = {
  getSummary: () => request("/api/summary"),
  getSample: (rows = 10) => request(`/api/sample?rows=${rows}`),
  getOptions: () => request("/api/options"),
  train: (k, testSize) =>
    request("/api/train", {
      method: "POST",
      body: JSON.stringify({ k, test_size: testSize }),
    }),
  predict: (payload) =>
    request("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
