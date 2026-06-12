import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';

export default function Gallery() {
  const [cattle, setCattle] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 24;

  const fetchCattle = async () => {
    setLoading(true);
    try {
      const params = { limit: PAGE_SIZE, skip: page * PAGE_SIZE };
      if (search) params.search = search;
      if (statusFilter) params.status = statusFilter;
      const data = await api.listCattle(params);
      setCattle(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchCattle(); }, [search, statusFilter, page]);

  return (
    <div className="page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-header__title">Cattle Gallery</h1>
          <p className="page-header__subtitle">Browse all registered cattle</p>
        </div>
        <Link to="/register" className="btn btn--primary">➕ Register New</Link>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
        <div className="search-bar">
          <span className="search-bar__icon">🔍</span>
          <input className="form-input" id="gallery-search" placeholder="Search by tag, name, breed…"
            value={search} onChange={e => { setSearch(e.target.value); setPage(0); }} />
        </div>
        <select className="form-select" id="status-filter" style={{ width: 160 }}
          value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setPage(0); }}>
          <option value="">All Status</option>
          <option value="active">Active</option>
          <option value="sold">Sold</option>
          <option value="deceased">Deceased</option>
        </select>
        <button className="btn btn--ghost btn--sm" onClick={() => { setSearch(''); setStatusFilter(''); setPage(0); }}>
          Clear
        </button>
      </div>

      {loading ? (
        <div className="loading-wrapper"><div className="spinner" /><span>Loading cattle…</span></div>
      ) : cattle.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state__icon">🐄</div>
          <div className="empty-state__title">No cattle found</div>
          <p>{search ? 'Try a different search term.' : 'Register your first cattle to get started.'}</p>
          <Link to="/register" className="btn btn--primary" style={{ marginTop: 20 }}>Register Cattle</Link>
        </div>
      ) : (
        <>
          <div className="cattle-grid">
            {cattle.map(c => (
              <Link to={`/cattle/${c.id}`} key={c.id} className="cattle-card" style={{ textDecoration: 'none' }}>
                <div className="cattle-card__image">
                  {c.photo_url
                    ? <img src={api.photoUrl(c.photo_url)} alt={c.name || c.tag_id} />
                    : <span>🐄</span>}
                </div>
                <div className="cattle-card__body">
                  <div className="cattle-card__tag">#{c.tag_id}</div>
                  <div className="cattle-card__name">{c.name || 'Unnamed'}</div>
                  <div className="cattle-card__meta">
                    {c.breed || 'Unknown breed'} · {c.sex || '—'}<br />
                    {c.farm_name || 'Unknown farm'}<br />
                    <span style={{ color: 'var(--accent-primary)' }}>{c.embedding_count} embedding{c.embedding_count !== 1 ? 's' : ''}</span>
                  </div>
                  <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className={`badge badge--${c.status === 'active' ? 'success' : c.status === 'sold' ? 'warning' : 'muted'}`}>
                      {c.status}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {new Date(c.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </Link>
            ))}
          </div>

          <div style={{ display: 'flex', justifyContent: 'center', gap: 12, marginTop: 32 }}>
            <button className="btn btn--ghost" onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}>← Previous</button>
            <span style={{ lineHeight: '38px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>Page {page + 1}</span>
            <button className="btn btn--ghost" onClick={() => setPage(p => p + 1)} disabled={cattle.length < PAGE_SIZE}>Next →</button>
          </div>
        </>
      )}
    </div>
  );
}
