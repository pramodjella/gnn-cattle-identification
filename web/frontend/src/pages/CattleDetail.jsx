import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import toast from 'react-hot-toast';
import { api } from '../api';

export default function CattleDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [cattle, setCattle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    api.getCattle(id)
      .then(setCattle)
      .catch(e => { toast.error(e.message); navigate('/gallery'); })
      .finally(() => setLoading(false));
  }, [id]);

  const handleDelete = async () => {
    if (!window.confirm('Delete this cattle and all embeddings? This cannot be undone.')) return;
    setDeleting(true);
    try {
      await api.deleteCattle(id);
      toast.success('Cattle deleted.');
      navigate('/gallery');
    } catch (e) {
      toast.error(e.message);
      setDeleting(false);
    }
  };

  if (loading) return <div className="page"><div className="loading-wrapper"><div className="spinner" /></div></div>;
  if (!cattle) return null;

  const fields = [
    ['Ear Tag', cattle.tag_id],
    ['Breed', cattle.breed],
    ['Sex', cattle.sex],
    ['Date of Birth', cattle.date_of_birth],
    ['Weight', cattle.weight_kg ? `${cattle.weight_kg} kg` : null],
    ['Farm', cattle.farm_name],
    ['Location', cattle.farm_location],
    ['Owner', cattle.owner_name],
    ['Contact', cattle.owner_contact],
    ['Registered', new Date(cattle.created_at).toLocaleDateString()],
  ];

  return (
    <div className="page animate-fade-in">
      <div style={{ marginBottom: 24 }}>
        <Link to="/gallery" style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>← Gallery</Link>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 24, alignItems: 'start' }}>
        {/* Photo & actions */}
        <div>
          <div style={{ borderRadius: 'var(--radius-xl)', overflow: 'hidden', marginBottom: 16,
            background: 'var(--bg-card)', border: '1px solid var(--border)', aspectRatio: '1' }}>
            {cattle.photo_url
              ? <img src={api.photoUrl(cattle.photo_url)} alt={cattle.name || cattle.tag_id}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              : <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 80 }}>🐄</div>}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <Link to="/identify" className="btn btn--primary" style={{ justifyContent: 'center' }}>
              🔍 Use in Identification
            </Link>
            <button className="btn btn--danger" onClick={handleDelete} disabled={deleting} id="delete-cattle-btn">
              {deleting ? '⏳ Deleting…' : '🗑️ Delete Cattle'}
            </button>
          </div>
        </div>

        {/* Details */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
            <div>
              <h1 style={{ marginBottom: 4 }}>{cattle.name || 'Unnamed Cattle'}</h1>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <span style={{ fontSize: '0.9rem', color: 'var(--accent-primary)', fontWeight: 600 }}>
                  #{cattle.tag_id}
                </span>
                <span className={`badge badge--${cattle.status === 'active' ? 'success' : 'muted'}`}>
                  {cattle.status}
                </span>
                <span className="badge badge--info">
                  🧬 {cattle.embedding_count} embedding{cattle.embedding_count !== 1 ? 's' : ''}
                </span>
              </div>
            </div>
          </div>

          {/* Info card */}
          <div className="card" style={{ marginBottom: 16 }}>
            <h3 style={{ marginBottom: 20 }}>Animal Information</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 24px' }}>
              {fields.map(([label, val]) => val ? (
                <div key={label}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase',
                    letterSpacing: '0.05em', marginBottom: 2 }}>{label}</div>
                  <div style={{ fontWeight: 500, fontSize: '0.95rem' }}>{val}</div>
                </div>
              ) : null)}
            </div>
          </div>

          {/* Notes */}
          {cattle.notes && (
            <div className="card">
              <h3 style={{ marginBottom: 12 }}>Notes</h3>
              <p style={{ fontSize: '0.9rem', lineHeight: 1.7 }}>{cattle.notes}</p>
            </div>
          )}

          {/* Biometric info */}
          <div className="card" style={{ marginTop: 16, background: 'rgba(59,130,246,0.05)', borderColor: 'rgba(59,130,246,0.2)' }}>
            <h3 style={{ marginBottom: 12, color: 'var(--accent-primary)' }}>🧬 Biometric Profile</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, textAlign: 'center' }}>
              {[
                ['Embeddings', cattle.embedding_count, '🧬'],
                ['Vector Dim', '256-d', '📐'],
                ['Index', 'HNSW', '⚡'],
              ].map(([label, val, icon]) => (
                <div key={label} style={{ padding: '16px 8px', background: 'var(--bg-base)',
                  borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 24, marginBottom: 6 }}>{icon}</div>
                  <div style={{ fontFamily: 'Space Grotesk', fontWeight: 700, fontSize: '1.2rem', color: 'var(--accent-primary)' }}>{val}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
