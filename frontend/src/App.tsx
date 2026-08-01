// Router setup — Landing page (/) → Wizard (/wizard) → Upload (/upload) → Dashboard.

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Home from './pages/Home'
import Upload from './pages/Upload'
import Dashboard from './pages/Dashboard'
import ArticleDetail from './pages/ArticleDetail'
import RiskWizard from './pages/RiskWizard'
import ClassifierMetrics from './pages/ClassifierMetrics'
import Monitoring from './pages/Monitoring'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/wizard" element={<RiskWizard />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/audit/:auditId" element={<Dashboard />} />
        <Route path="/audit/:auditId/article/:articleNum" element={<ArticleDetail />} />
        <Route path="/metrics" element={<ClassifierMetrics />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
