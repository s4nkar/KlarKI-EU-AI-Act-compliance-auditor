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
    <div className="min-h-screen bg-cream">
      <header className="bg-white border-b-4 border-black sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 shrink-0 group">
            <div className="w-9 h-9 border-2 border-black bg-yellow-500 flex items-center justify-center shadow-sm group-hover:-translate-y-0.5 group-hover:shadow-md transition-transform">
              <svg className="w-4.5 h-4.5 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5}
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-base font-black uppercase tracking-tight text-black">KlarKI</span>
              <span className="hidden sm:block text-[10px] text-slate-500 font-bold uppercase tracking-wide mt-0.5">
                EU AI Act &amp; GDPR Compliance
              </span>
            </div>
          </Link>

          {/* Nav */}
          <nav className="flex items-center gap-2">
            {NAV_ITEMS.map(item => {
              const isActive = item.exact
                ? location.pathname === item.to
                : location.pathname.startsWith(item.to) && item.to !== '/'
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`px-3.5 py-1.5 border-2 text-sm font-bold uppercase tracking-wide transition-all duration-100 ${isActive
                      ? 'text-black bg-yellow-500 border-black shadow-sm'
                      : 'text-slate-600 border-transparent hover:text-black hover:border-black'
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
