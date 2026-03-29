// Nav bar + main content wrapper.

import { Link } from 'react-router-dom'

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <span className="text-xl font-bold text-brand-700">KlarKI</span>
            <span className="hidden sm:inline text-xs text-slate-400 font-medium tracking-wide uppercase">
              EU AI Act &amp; GDPR Compliance
            </span>
          </Link>
          <nav className="flex items-center gap-4 text-sm text-slate-500">
            <Link to="/" className="hover:text-brand-600 transition-colors">
              New Audit
            </Link>
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
        {children}
      </main>
    </div>
  )
}
