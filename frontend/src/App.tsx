// Router setup — Phase 3 full page routing.

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Upload from './pages/Upload'
import Dashboard from './pages/Dashboard'
import ArticleDetail from './pages/ArticleDetail'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Upload />} />
        <Route path="/audit/:auditId" element={<Dashboard />} />
        <Route path="/audit/:auditId/article/:articleNum" element={<ArticleDetail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
