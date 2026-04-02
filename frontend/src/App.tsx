// Router setup — Wizard-first flow: Step 1 (/) → Step 2 (/upload) → Dashboard.

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Upload from './pages/Upload'
import Dashboard from './pages/Dashboard'
import ArticleDetail from './pages/ArticleDetail'
import RiskWizard from './pages/RiskWizard'
import ClassifierMetrics from './pages/ClassifierMetrics'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<RiskWizard />} />
        <Route path="/upload" element={<Upload />} />
        {/* Legacy alias so existing /wizard links still work */}
        <Route path="/wizard" element={<Navigate to="/" replace />} />
        <Route path="/audit/:auditId" element={<Dashboard />} />
        <Route path="/audit/:auditId/article/:articleNum" element={<ArticleDetail />} />
        <Route path="/metrics" element={<ClassifierMetrics />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
