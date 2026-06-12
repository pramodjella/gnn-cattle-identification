import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Dashboard from './pages/Dashboard';
import Register from './pages/Register';
import Identify from './pages/Identify';
import Gallery from './pages/Gallery';
import CattleDetail from './pages/CattleDetail';
import './index.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar__inner">
        <NavLink to="/" className="navbar__logo">
          <div className="navbar__logo-icon">🐄</div>
          <span>CattleID<span style={{ color: 'var(--accent-primary)' }}>Pro</span></span>
        </NavLink>

        <div className="navbar__nav">
          <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            📊 Dashboard
          </NavLink>
          <NavLink to="/register" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            ➕ Register
          </NavLink>
          <NavLink to="/identify" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            🔍 Identify
          </NavLink>
          <NavLink to="/gallery" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            🗂️ Gallery
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main style={{ flex: 1 }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/register" element={<Register />} />
          <Route path="/identify" element={<Identify />} />
          <Route path="/gallery" element={<Gallery />} />
          <Route path="/cattle/:id" element={<CattleDetail />} />
        </Routes>
      </main>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'var(--bg-card)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border)',
            fontFamily: 'Inter, sans-serif',
          },
        }}
      />
    </BrowserRouter>
  );
}
