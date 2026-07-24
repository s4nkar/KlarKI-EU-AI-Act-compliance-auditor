// Premium nav bar + main content wrapper.

import { Link, useLocation } from 'react-router-dom'

interface LayoutProps {
  children: React.ReactNode
}

const NAV_ITEMS = [
  { to: '/', label: 'Risk Assessment', exact: true },
  { to: '/upload', label: 'Upload Docs', exact: false },
  { to: '/metrics', label: 'Model Metrics', exact: false },
  { to: '/monitoring', label: 'Monitoring', exact: false },
]

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-canvas">
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

          {/* Nav */}
          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map(item => {
              const isActive = item.exact
                ? location.pathname === item.to
                : location.pathname.startsWith(item.to) && item.to !== '/'
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
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {children}
      </main>
    </div>
  )
}
