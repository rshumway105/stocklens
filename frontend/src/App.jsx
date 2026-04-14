import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Watchlist from './pages/Watchlist'
import TickerDetail from './pages/TickerDetail'
import MacroDashboard from './pages/MacroDashboard'
import ModelPerformance from './pages/ModelPerformance'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        {/* ── Top navigation bar ── */}
        <header className="bg-terminal-surface border-b border-terminal-border sticky top-0 z-50">
          <div className="max-w-[1600px] mx-auto px-6 flex items-center justify-between h-14">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-accent-amber animate-pulse" />
              <span className="font-display font-bold text-lg tracking-tight text-terminal-text">
                Stock<span className="text-accent-amber">Lens</span>
              </span>
              <span className="text-[10px] font-mono text-terminal-muted ml-2 border border-terminal-border px-1.5 py-0.5 rounded">
                v0.1.0
              </span>
            </div>

            <nav className="flex items-center gap-1">
              <NavLink
                to="/"
                end
                className={({ isActive }) =>
                  `nav-link ${isActive ? 'nav-link-active' : ''}`
                }
              >
                Watchlist
              </NavLink>
              <NavLink
                to="/macro"
                className={({ isActive }) =>
                  `nav-link ${isActive ? 'nav-link-active' : ''}`
                }
              >
                Macro
              </NavLink>
              <NavLink
                to="/performance"
                className={({ isActive }) =>
                  `nav-link ${isActive ? 'nav-link-active' : ''}`
                }
              >
                Model
              </NavLink>
            </nav>
          </div>
        </header>

        {/* ── Page content ── */}
        <main className="flex-1 max-w-[1600px] mx-auto w-full px-6 py-6">
          <Routes>
            <Route path="/" element={<Watchlist />} />
            <Route path="/ticker/:ticker" element={<TickerDetail />} />
            <Route path="/macro" element={<MacroDashboard />} />
            <Route path="/performance" element={<ModelPerformance />} />
          </Routes>
        </main>

        {/* ── Footer ── */}
        <footer className="border-t border-terminal-border py-3 px-6">
          <div className="max-w-[1600px] mx-auto flex justify-between text-[11px] font-mono text-terminal-muted">
            <span>StockLens — Educational & research use only. Not financial advice.</span>
            <span>Data: Yahoo Finance • FRED • NewsAPI</span>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  )
}

export default App
