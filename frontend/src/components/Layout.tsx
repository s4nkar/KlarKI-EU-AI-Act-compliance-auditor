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
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-nav">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 shrink-0 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-brand flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5}
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-base font-bold text-gradient-brand">KlarKI</span>
              <span className="hidden sm:block text-[10px] text-slate-400 font-medium tracking-wide mt-0.5">
                EU AI Act &amp; GDPR Compliance
              </span>
            </div>
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
                  className={`relative px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${isActive
                      ? 'text-brand-700 bg-brand-50'
                      : 'text-slate-500 hover:text-slate-900 hover:bg-slate-100'
                    }`}
                >
                  {item.label}
                  {/* {isActive && (
                    <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-brand-600" />
                  )} */}
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
