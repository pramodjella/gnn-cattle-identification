import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { api } from '../api';

const STEPS = ['Upload Photo', 'Cattle Details', 'Review & Submit'];
const BREEDS = ['Angus', 'Hereford', 'Brahman', 'Holstein', 'Simmental', 'Limousin', 'Charolais', 'Shorthorn', 'Murray Grey', 'Other'];

export default function Register() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState({
    tag_id: '', name: '', breed: '', sex: '', date_of_birth: '',
    farm_name: '', farm_location: '', owner_name: '', owner_contact: '', weight_kg: '', notes: '',
  });

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return;
    const file = accepted[0];
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setStep(1);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/*': [] }, maxFiles: 1, maxSize: 10 * 1024 * 1024,
  });

  const handleChange = (e) => setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  const handleSubmit = async () => {
    if (!imageFile) { toast.error('Please upload a muzzle photo.'); return; }
    if (!form.tag_id) { toast.error('Tag ID is required.'); return; }
    setSubmitting(true);
    const fd = new FormData();
    fd.append('muzzle_image', imageFile);
    Object.entries(form).forEach(([k, v]) => { if (v) fd.append(k, v); });
    try {
      const result = await api.registerCattle(fd);
      toast.success(`✅ Cattle registered successfully!`);
      navigate(`/cattle/${result.id}`);
    } catch (err) {
      toast.error(err.message || 'Registration failed.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-header__title">Register Cattle</h1>
          <p className="page-header__subtitle">Add a new cattle to the biometric database</p>
        </div>
      </div>

      <div className="steps">
        {STEPS.map((s, i) => (
          <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className={`step ${i === step ? 'active' : i < step ? 'done' : ''}`}
              onClick={() => i < step && setStep(i)} style={{ cursor: i < step ? 'pointer' : 'default' }}>
              <div className="step__num">{i < step ? '✓' : i + 1}</div>
              <span>{s}</span>
            </div>
            {i < STEPS.length - 1 && <div className="step-divider" />}
          </div>
        ))}
      </div>

      <div style={{ maxWidth: 700, margin: '0 auto' }}>

        {step === 0 && (
          <div className="card animate-fade-in">
            <h3 style={{ marginBottom: 8 }}>Upload Muzzle Photo</h3>
            <p style={{ marginBottom: 24, fontSize: '0.9rem' }}>
              Take a clear, well-lit photo of the cattle's muzzle. The muzzle print is unique to each animal.
            </p>
            <div {...getRootProps()} className={`dropzone${isDragActive ? ' active' : ''}`}>
              <input {...getInputProps()} id="muzzle-upload" />
              <div className="dropzone__icon">📸</div>
              <div className="dropzone__title">{isDragActive ? 'Drop the image here…' : 'Drag & drop or click to upload'}</div>
              <div className="dropzone__subtitle">JPG, PNG, WEBP · Max 10 MB</div>
            </div>
            {imagePreview && (
              <div style={{ marginTop: 20 }}>
                <div className="image-preview">
                  <img src={imagePreview} alt="Muzzle preview" />
                </div>
                <button className="btn btn--primary" style={{ marginTop: 16, width: '100%' }} onClick={() => setStep(1)}>
                  Continue with this photo →
                </button>
              </div>
            )}
          </div>
        )}

        {step === 1 && (
          <div className="card animate-fade-in">
            {imagePreview && (
              <div style={{ display: 'flex', gap: 16, marginBottom: 24, alignItems: 'center' }}>
                <img src={imagePreview} alt="Muzzle"
                  style={{ width: 72, height: 72, objectFit: 'cover', borderRadius: 10, flexShrink: 0 }} />
                <div>
                  <div style={{ fontWeight: 600 }}>Photo uploaded ✅</div>
                  <button className="btn btn--ghost btn--sm" onClick={() => setStep(0)}>Change photo</button>
                </div>
              </div>
            )}
            <h3 style={{ marginBottom: 20 }}>Cattle Details</h3>
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label" htmlFor="tag_id">Ear Tag / ID *</label>
                <input id="tag_id" name="tag_id" className="form-input" required value={form.tag_id} onChange={handleChange} placeholder="e.g. AU-1234-5678" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="name">Name</label>
                <input id="name" name="name" className="form-input" value={form.name} onChange={handleChange} placeholder="Optional name" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="breed">Breed</label>
                <select id="breed" name="breed" className="form-select" value={form.breed} onChange={handleChange}>
                  <option value="">Select breed</option>
                  {BREEDS.map(b => <option key={b} value={b}>{b}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="sex">Sex</label>
                <select id="sex" name="sex" className="form-select" value={form.sex} onChange={handleChange}>
                  <option value="">Select sex</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="date_of_birth">Date of Birth</label>
                <input id="date_of_birth" name="date_of_birth" type="date" className="form-input" value={form.date_of_birth} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="weight_kg">Weight (kg)</label>
                <input id="weight_kg" name="weight_kg" type="number" className="form-input" value={form.weight_kg} onChange={handleChange} placeholder="e.g. 450" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="farm_name">Farm Name</label>
                <input id="farm_name" name="farm_name" className="form-input" value={form.farm_name} onChange={handleChange} placeholder="Green Valley Farm" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="owner_name">Owner Name</label>
                <input id="owner_name" name="owner_name" className="form-input" value={form.owner_name} onChange={handleChange} />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="notes">Notes</label>
              <textarea id="notes" name="notes" className="form-textarea" value={form.notes} onChange={handleChange} placeholder="Any additional notes…" />
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
              <button className="btn btn--ghost" onClick={() => setStep(0)}>← Back</button>
              <button className="btn btn--primary" style={{ flex: 1 }}
                onClick={() => { if (!form.tag_id) { toast.error('Tag ID is required'); return; } setStep(2); }}>
                Review →
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="card animate-fade-in">
            <h3 style={{ marginBottom: 20 }}>Review & Confirm</h3>
            <div style={{ display: 'flex', gap: 20, marginBottom: 24 }}>
              {imagePreview && (
                <img src={imagePreview} alt="Muzzle" style={{ width: 110, height: 110, objectFit: 'cover', borderRadius: 12, flexShrink: 0 }} />
              )}
              <div style={{ flex: 1 }}>
                {[['Tag ID', form.tag_id], ['Name', form.name], ['Breed', form.breed],
                  ['Sex', form.sex], ['Farm', form.farm_name], ['Owner', form.owner_name],
                  ['Weight', form.weight_kg ? `${form.weight_kg} kg` : null]].map(([label, val]) =>
                  val ? (
                    <div key={label} style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', width: 70 }}>{label}</span>
                      <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>{val}</span>
                    </div>
                  ) : null
                )}
              </div>
            </div>
            <div style={{ padding: 16, background: 'rgba(59,130,246,0.08)', borderRadius: 10,
              border: '1px solid rgba(59,130,246,0.2)', marginBottom: 24, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              🧬 The GNN model will extract biometric features from the muzzle photo and store
              a 256-dimensional embedding in the pgvector database for future identification.
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              <button className="btn btn--ghost" onClick={() => setStep(1)}>← Edit</button>
              <button className="btn btn--success" style={{ flex: 1 }} onClick={handleSubmit}
                disabled={submitting} id="submit-register-btn">
                {submitting ? '⏳ Processing…' : '✅ Register Cattle'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
