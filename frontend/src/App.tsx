// Router setup — Phase 3 adds full page components.
// Phase 1 placeholder shows health status only.

import { BrowserRouter, Routes, Route } from 'react-router-dom'

function HealthPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-brand-600 mb-2">KlarKI</h1>
        <p className="text-slate-500 text-lg">EU AI Act &amp; GDPR Compliance Auditor</p>
        <p className="mt-6 text-sm text-slate-400">Phase 3 frontend — coming soon</p>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/*" element={<HealthPage />} />
      </Routes>
    </BrowserRouter>
  )
}
