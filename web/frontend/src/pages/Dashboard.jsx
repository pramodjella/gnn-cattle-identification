import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [recentCattle, setRecentCattle] = useState([]);
  const [recentLogs, setRecentLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getStats(),
      api.listCattle({ limit: 6 }),
      api.getLogs({ limit: 5 }),
    ])
      .then(([s, c, l]) => { setStats(s); setRecentCattle(c); setRecentLogs(l); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const statItems = stats ? [
    { icon: '🐄', label: 'Active Cattle', value: stats.active_cattle, color: '#3b82f6' },
    { icon: '📁', label: 'Total Registered', value: stats.total_cattle, color: '#8b5cf6' },
    { icon: '🧬', label: 'Embeddings Stored', value: stats.total_embeddings, color: '#10b981' },
    { icon: '🔍', label: 'Identifications', value: stats.total_identifications, color: '#f59e0b' },
    { icon: '✅', label: 'Successful Matches', value: stats.successful_identifications, color: '#06b6d4' },
  ] : [];

  return (
    <div className="page animate-fade-in">
      {/* Hero */}
      <div className="hero-banner">
        <h1 style={{ marginBottom: 8 }}>
          Biometric Cattle Identification
          <span style={{ background: 'var(--grad-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}> Platform</span>
        </h1>
        <p style={{ maxWidth: 560, marginBottom: 24 }}>
          GNN-powered muzzle print recognition. Register cattle, store biometric embeddings in pgvector,
          and identify individuals from a photo in seconds.
        </p>
        <div style={{ display: 'flex', gap: 12 }}>
          <Link to="/register" className="btn btn--primary btn--lg">➕ Register Cattle</Link>
          <Link to="/identify" className="btn btn--ghost btn--lg">🔍 Identify Now</Link>
        </div>
      </div>

      {/* Stats */}
      {loading ? (
        <div className="loading-wrapper"><div className="spinner" /></div>
      ) : (
        <div className="stat-grid">
          {statItems.map(s => (
            <div className="stat-card" key={s.label}>
              <div className="stat-card__icon" style={{ background: `${s.color}22` }}>
                <span style={{ fontSize: 22 }}>{s.icon}</span>
              </div>
              <div>
                <div className="stat-card__value" style={{ color: s.color }}>{s.value.toLocaleString()}</div>
                <div className="stat-card__label">{s.label}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Recent Registrations */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <h3>Recent Registrations</h3>
            <Link to="/gallery" className="btn btn--ghost btn--sm">View All →</Link>
          </div>
          {recentCattle.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state__icon">🐄</div>
              <div className="empty-state__title">No cattle registered yet</div>
              <Link to="/register" className="btn btn--primary" style={{ marginTop: 16 }}>Register First Cattle</Link>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {recentCattle.map(c => (
                <Link to={`/cattle/${c.id}`} key={c.id}
                  style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '10px 12px',
                    borderRadius: 'var(--radius-md)', background: 'var(--bg-base)',
                    border: '1px solid var(--border)', textDecoration: 'none',
                    transition: 'all 0.2s' }}
                  onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border-accent)'}
                  onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                >
                  <div style={{ width: 44, height: 44, borderRadius: 10, background: 'var(--bg-elevated)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    overflow: 'hidden', flexShrink: 0 }}>
                    {c.photo_url
                      ? <img src={api.photoUrl(c.photo_url)} alt={c.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      : <span style={{ fontSize: 20 }}>🐄</span>}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {c.name || 'Unnamed'}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Tag: {c.tag_id} · {c.breed || 'Unknown breed'}
                    </div>
                  </div>
                  <span className={`badge badge--${c.status === 'active' ? 'success' : 'muted'}`}>{c.status}</span>
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* Recent Identification Logs */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <h3>Identification History</h3>
          </div>
          {recentLogs.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state__icon">🔍</div>
              <div className="empty-state__title">No identifications yet</div>
              <Link to="/identify" className="btn btn--primary" style={{ marginTop: 16 }}>Try Identification</Link>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {recentLogs.map(log => (
                <div key={log.id} style={{ display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 14px', background: 'var(--bg-base)',
                  borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
                  <span style={{ fontSize: 20 }}>{log.accepted ? '✅' : '❌'}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                      {log.accepted ? 'Match Found' : 'No Match'}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Similarity: {log.similarity != null ? (log.similarity * 100).toFixed(1) + '%' : '—'}
                      {' · '}{log.extractor || 'unknown'}
                    </div>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {new Date(log.created_at).toLocaleDateString()}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div style={{ marginTop: 24 }}>
        <h3 style={{ marginBottom: 16 }}>Quick Actions</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
          {[
            { icon: '➕', title: 'Register Cattle', desc: 'Add a new cattle with muzzle photo', to: '/register', color: '#3b82f6' },
            { icon: '🔍', title: 'Identify Cattle', desc: 'Upload photo to find a match', to: '/identify', color: '#10b981' },
            { icon: '🗂️', title: 'Browse Gallery', desc: 'View all registered cattle', to: '/gallery', color: '#8b5cf6' },
          ].map(a => (
            <Link to={a.to} key={a.to} className="card" style={{ textDecoration: 'none', cursor: 'pointer' }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>{a.icon}</div>
              <h3 style={{ color: a.color, marginBottom: 6 }}>{a.title}</h3>
              <p style={{ fontSize: '0.85rem' }}>{a.desc}</p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
