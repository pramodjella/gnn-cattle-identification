// Centralised API client
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

export const api = {
  // Stats
  getStats: () => request('/api/stats'),

  // Cattle CRUD
  listCattle: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return request(`/api/cattle/${q ? '?' + q : ''}`);
  },
  getCattle: (id) => request(`/api/cattle/${id}`),
  deleteCattle: (id) => request(`/api/cattle/${id}`, { method: 'DELETE' }),

  // Registration
  registerCattle: (formData) =>
    request('/api/cattle/register', { method: 'POST', body: formData }),

  // Identification
  identifyCattle: (formData, params = {}) => {
    const q = new URLSearchParams(params).toString();
    return request(`/api/cattle/identify${q ? '?' + q : ''}`, {
      method: 'POST',
      body: formData,
    });
  },

  // Logs
  getLogs: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return request(`/api/logs${q ? '?' + q : ''}`);
  },

  // Helpers
  photoUrl: (url) => url ? `${BASE_URL}${url}` : null,
};
