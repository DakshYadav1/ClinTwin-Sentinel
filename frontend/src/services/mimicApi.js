const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

async function mimicFetch(endpoint) {
  const response = await fetch(`${API_BASE}${endpoint}`);

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`MIMIC API error (${response.status}): ${body}`);
  }

  return response.json();
}

export async function fetchMIMICRoster() {
  return mimicFetch('/api/mimic/roster');
}

export async function fetchMIMICPatient(stayId) {
  return mimicFetch(`/api/mimic/patients/${encodeURIComponent(stayId)}`);
}

export async function fetchMIMICTrajectory(stayId) {
  return mimicFetch(
    `/api/mimic/patients/${encodeURIComponent(stayId)}/trajectory`
  );
}
