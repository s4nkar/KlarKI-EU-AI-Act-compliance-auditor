// Premium nav bar + main content wrapper.

import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'

interface LayoutProps {
  children: React.ReactNode
}

const PRIMARY_NAV = [
  // { to: '/', label: 'Home', exact: true },
  // { to: '/wizard', label: 'Risk Assessment', exact: false },
  { to: '/upload', label: 'Upload Docs', exact: false },
]

const TOOLS_NAV = [
  { to: '/metrics', label: 'Model Metrics', description: 'Classifier accuracy & drift' },
  { to: '/monitoring', label: 'Monitoring', description: 'Pipeline health & runs' },
]

function isActivePath(pathname: string, to: string, exact: boolean) {
  return exact ? pathname === to : pathname.startsWith(to)
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const toolsActive = TOOLS_NAV.some(item => isActivePath(location.pathname, item.to, false))

  const [toolsOpen, setToolsOpen] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const toolsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setToolsOpen(false)
    setMobileOpen(false)
  }, [location.pathname])

  useEffect(() => {
    if (!toolsOpen) return
    const onClickOutside = (e: MouseEvent) => {
      if (toolsRef.current && !toolsRef.current.contains(e.target as Node)) {
        setToolsOpen(false)
      }
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [toolsOpen])

  return (
    <div className="min-h-screen bg-canvas flex flex-col">
      <header className="bg-canvas/80 backdrop-blur-md border-b border-line sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 shrink-0 group">
            <img
              src="/logo.png"
              alt="KlarKI"
              className="h-8 w-auto object-contain group-hover:brightness-110 transition-all"
            />
            <span className="hidden sm:block text-[10px] text-slate-500 font-medium tracking-wide leading-none border-l border-line pl-3">
              EU AI Act &amp; GDPR<br />Compliance
            </span>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-1">
            {PRIMARY_NAV.map(item => {
              const isActive = isActivePath(location.pathname, item.to, item.exact)
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${isActive
                    ? 'text-white bg-surface-raised'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-surface'
                    }`}
                >
                  {item.label}
                </Link>
              )
            })}

            {/* Tools dropdown */}
            <div className="relative" ref={toolsRef}>
              <button
                onClick={() => setToolsOpen(o => !o)}
                className={`flex items-center gap-1 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${toolsActive
                  ? 'text-white bg-surface-raised'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-surface'
                  }`}
                aria-expanded={toolsOpen}
              >
                Tools
                <svg
                  className={`w-3.5 h-3.5 transition-transform duration-150 ${toolsOpen ? 'rotate-180' : ''}`}
                  fill="none" viewBox="0 0 24 24" stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {toolsOpen && (
                <div className="absolute right-0 mt-2 w-64 rounded-xl border border-line bg-surface-raised shadow-panel p-1.5">
                  {TOOLS_NAV.map(item => {
                    const isActive = isActivePath(location.pathname, item.to, false)
                    return (
                      <Link
                        key={item.to}
                        to={item.to}
                        className={`flex flex-col gap-0.5 px-3 py-2 rounded-lg text-sm transition-colors duration-150 ${isActive
                          ? 'bg-surface-hover text-white'
                          : 'text-slate-300 hover:bg-surface-hover hover:text-white'
                          }`}
                      >
                        <span className="font-medium">{item.label}</span>
                        <span className="text-xs text-slate-500">{item.description}</span>
                      </Link>
                    )
                  })}
                </div>
              )}
            </div>

            <Link to="/wizard" className="btn-primary ml-2 !py-1.5 !px-4 text-sm">
              Start Audit
            </Link>
          </nav>

          {/* Mobile menu toggle */}
          <button
            onClick={() => setMobileOpen(o => !o)}
            className="md:hidden p-2 -mr-2 rounded-lg text-slate-300 hover:bg-surface"
            aria-label="Toggle menu"
            aria-expanded={mobileOpen}
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              {mobileOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile nav */}
        {mobileOpen && (
          <nav className="md:hidden border-t border-line px-4 sm:px-6 py-3 flex flex-col gap-1">
            {[...PRIMARY_NAV, ...TOOLS_NAV.map(t => ({ ...t, exact: false }))].map(item => {
              const isActive = isActivePath(location.pathname, item.to, item.exact)
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ${isActive
                    ? 'text-white bg-surface-raised'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-surface'
                    }`}
                >
                  {item.label}
                </Link>
              )
            })}
            <Link to="/wizard" className="btn-primary justify-center mt-2 text-sm">
              Start Audit
            </Link>
          </nav>
        )}
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex-1 w-full">
        {children}
      </main>

      <footer className="border-t border-line">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-8">
            <div className="col-span-2 sm:col-span-1">
              <Link to="/" className="flex items-center gap-2">
                <img src="/logo.png" alt="KlarKI" className="h-6 w-auto object-contain" />
              </Link>
              <p className="mt-3 text-xs text-slate-500 leading-relaxed max-w-[220px]">
                Local-first EU AI Act &amp; GDPR compliance auditor. Nothing leaves your machine.
              </p>
            </div>

            <div>
              <p className="text-xs font-semibold text-slate-300 uppercase tracking-wide mb-3">Product</p>
              <ul className="space-y-2 text-sm text-slate-500">
                <li><Link to="/wizard" className="hover:text-slate-200 transition-colors">Risk Assessment</Link></li>
                <li><Link to="/upload" className="hover:text-slate-200 transition-colors">Upload Docs</Link></li>
              </ul>
            </div>

            <div>
              <p className="text-xs font-semibold text-slate-300 uppercase tracking-wide mb-3">Tools</p>
              <ul className="space-y-2 text-sm text-slate-500">
                <li><Link to="/metrics" className="hover:text-slate-200 transition-colors">Model Metrics</Link></li>
                <li><Link to="/monitoring" className="hover:text-slate-200 transition-colors">Monitoring</Link></li>
              </ul>
            </div>

            <div>
              <p className="text-xs font-semibold text-slate-300 uppercase tracking-wide mb-3">Project</p>
              <ul className="space-y-2 text-sm text-slate-500">
                <li>
                  <a
                    href="https://github.com/s4nkar/KlarKI-EU-AI-Act-compliance-auditor"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-slate-200 transition-colors"
                  >
                    GitHub
                  </a>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-10 pt-6 border-t border-line flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-600">
            <span>&copy; {new Date().getFullYear()} KlarKI</span>
            <span>Runs entirely offline — Ollama &middot; ChromaDB &middot; no external inference APIs</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
