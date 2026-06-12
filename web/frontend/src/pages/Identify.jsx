import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Link } from 'react-router-dom';
import { api } from '../api';

function SimilarityBar({ value }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.8 ? '#10b981' : value >= 0.6 ? '#f59e0b' : '#ef4444';
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Similarity</span>
        <span style={{ fontFamily: 'Space Grotesk', fontWeight: 700, fontSize: '1rem', color }}>{pct}%</span>
      </div>
      <div className="similarity-bar">
        <div className="similarity-bar__fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function Identify() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(0.70);

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return;
    setImageFile(accepted[0]);
    setImagePreview(URL.createObjectURL(accepted[0]));
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/*': [] }, maxFiles: 1,
  });

  const handleIdentify = async () => {
    if (!imageFile) return;
    setLoading(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append('muzzle_image', imageFile);
    try {
      const res = await api.identifyCattle(fd, { threshold, top_k: 5 });
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-header__title">Identify Cattle</h1>
          <p className="page-header__subtitle">Upload a muzzle photo to find matching cattle in the database</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
        {/* Left – upload panel */}
        <div>
          <div className="card">
            <h3 style={{ marginBottom: 16 }}>Query Image</h3>
            {!imagePreview ? (
              <div {...getRootProps()} className={`dropzone${isDragActive ? ' active' : ''}`}>
                <input {...getInputProps()} id="identify-upload" />
                <div className="dropzone__icon">🔍</div>
                <div className="dropzone__title">{isDragActive ? 'Drop here…' : 'Drop muzzle photo to identify'}</div>
                <div className="dropzone__subtitle">JPG, PNG, WEBP supported</div>
              </div>
            ) : (
              <div>
                <div className="image-preview" style={{ marginBottom: 16 }}>
                  <img src={imagePreview} alt="Query" />
                </div>
                <button className="btn btn--ghost btn--sm" onClick={() => { setImageFile(null); setImagePreview(null); setResult(null); }}>
                  🔄 Use different image
                </button>
              </div>
            )}

            <div style={{ marginTop: 20 }}>
              <label className="form-label" htmlFor="threshold-slider">
                Similarity Threshold: <strong style={{ color: 'var(--accent-primary)' }}>{(threshold * 100).toFixed(0)}%</strong>
              </label>
              <input id="threshold-slider" type="range" min="0.3" max="0.99" step="0.01"
                value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-primary)', marginTop: 6 }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                <span>Lenient (30%)</span><span>Strict (99%)</span>
              </div>
            </div>

            <button className="btn btn--primary" style={{ width: '100%', marginTop: 20 }}
              onClick={handleIdentify} disabled={!imageFile || loading} id="identify-btn">
              {loading ? '⏳ Searching…' : '🔍 Identify Cattle'}
            </button>
          </div>

          {/* Info box */}
          <div className="card" style={{ marginTop: 16, background: 'rgba(139,92,246,0.06)', borderColor: 'rgba(139,92,246,0.2)' }}>
            <h3 style={{ fontSize: '0.9rem', marginBottom: 8 }}>How it works</h3>
            <ol style={{ paddingLeft: 18, fontSize: '0.83rem', color: 'var(--text-secondary)', lineHeight: 1.9 }}>
              <li>Upload a clear muzzle photo</li>
              <li>SuperPoint extracts keypoints &amp; 256-d descriptors</li>
              <li>CattleGNN builds a graph and produces an embedding</li>
              <li>pgvector HNSW search finds closest matches</li>
              <li>Top-5 results ranked by cosine similarity</li>
            </ol>
          </div>
        </div>

        {/* Right – results */}
        <div>
          {loading && (
            <div className="card">
              <div className="loading-wrapper">
                <div className="spinner" />
                <span>Extracting biometric features…</span>
              </div>
            </div>
          )}

          {error && (
            <div className="card" style={{ borderColor: 'rgba(239,68,68,0.4)', background: 'rgba(239,68,68,0.05)' }}>
              <div style={{ fontSize: 32, marginBottom: 8 }}>❌</div>
              <h3 style={{ color: 'var(--accent-danger)', marginBottom: 4 }}>Identification Failed</h3>
              <p style={{ fontSize: '0.9rem' }}>{error}</p>
            </div>
          )}

          {result && !loading && (
            <div className="animate-fade-in">
              {/* Verdict banner */}
              <div className="card" style={{
                marginBottom: 16,
                borderColor: result.accepted ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.3)',
                background: result.accepted ? 'rgba(16,185,129,0.07)' : 'rgba(239,68,68,0.05)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                  <span style={{ fontSize: 40 }}>{result.accepted ? '✅' : '❌'}</span>
                  <div>
                    <h2 style={{ color: result.accepted ? '#10b981' : '#ef4444', marginBottom: 4 }}>
                      {result.accepted ? 'Match Found!' : 'No Match'}
                    </h2>
                    <p style={{ fontSize: '0.85rem' }}>
                      {result.accepted
                        ? `Identified as ${result.top_match?.name || result.top_match?.tag_id}`
                        : `Best similarity ${((result.top_match?.similarity || 0) * 100).toFixed(1)}% is below threshold ${(result.threshold * 100).toFixed(0)}%`}
                    </p>
                  </div>
                </div>
                <div style={{ marginTop: 16, display: 'flex', gap: 12, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  <span>🔑 {result.num_keypoints} keypoints</span>
                  <span>🤖 {result.extractor}</span>
                  <span>📐 threshold {(result.threshold * 100).toFixed(0)}%</span>
                </div>
              </div>

              {/* Top-k matches */}
              <h3 style={{ marginBottom: 12 }}>Top Matches</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {result.top_k.length === 0 ? (
                  <div className="card empty-state">
                    <div className="empty-state__icon">🐄</div>
                    <div className="empty-state__title">No cattle in database</div>
                    <Link to="/register" className="btn btn--primary" style={{ marginTop: 12 }}>Register Cattle</Link>
                  </div>
                ) : result.top_k.map((m) => (
                  <Link to={`/cattle/${m.cattle_id}`} key={m.cattle_id}
                    style={{ textDecoration: 'none' }}
                    className={`match-card${m.rank === 1 && result.accepted ? ' top-match' : ''}`}>
                    <div className="match-card__rank" style={{ color: m.rank === 1 ? '#10b981' : 'var(--text-muted)' }}>
                      #{m.rank}
                    </div>
                    <div style={{ width: 52, height: 52, borderRadius: 10, overflow: 'hidden',
                      background: 'var(--bg-elevated)', flexShrink: 0 }}>
                      {m.photo_url
                        ? <img src={api.photoUrl(m.photo_url)} alt={m.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        : <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22 }}>🐄</div>}
                    </div>
                    <div className="match-card__info">
                      <div className="match-card__name">{m.name || 'Unnamed'}</div>
                      <div className="match-card__tag">Tag: {m.tag_id} · {m.breed || 'Unknown'}</div>
                      <SimilarityBar value={m.similarity} />
                    </div>
                    {m.accepted
                      ? <span className="badge badge--success">✓ Match</span>
                      : <span className="badge badge--muted">Low</span>}
                  </Link>
                ))}
              </div>
            </div>
          )}

          {!result && !loading && !error && (
            <div className="card empty-state">
              <div className="empty-state__icon">🔬</div>
              <div className="empty-state__title">Upload a photo to begin</div>
              <p style={{ fontSize: '0.85rem', marginTop: 8 }}>
                The biometric identification engine will search across all registered cattle.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
