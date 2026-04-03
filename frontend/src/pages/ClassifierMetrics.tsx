// Premium classifier metrics dashboard: BERT macro F1 + confusion matrix + spaCy NER metrics.
// Falls back to /static-metrics/bert.json when the backend API is not running.

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Layout from '../components/Layout'
import apiClient from '../api/client'

interface PerClassMetric {
  label: string
  precision: number
  recall: number
  f1: number
  support: number
}

interface ClassifierMetricsData {
  macro_f1: number
  per_class: PerClassMetric[]
  confusion_matrix: number[][]
  labels: string[]
  val_size: number
  train_size: number
  base_model: string
}

interface NerLabelMetric {
  label: string
  precision: number
  recall: number
  f1: number
}

interface NerMetricsData {
  overall_f1: number
  overall_p: number
  overall_r: number
  per_label: NerLabelMetric[]
  labels: string[]
  val_size: number
  train_size: number
  final_loss: number
}

export default function ClassifierMetrics() {
  const [data, setData]       = useState<ClassifierMetricsData | null>(null)
  const [nerData, setNerData] = useState<NerMetricsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)
  const [fromStatic, setFromStatic] = useState(false)

  useEffect(() => {
    Promise.allSettled([
      apiClient.get<{ status: string; data: ClassifierMetricsData }>('/api/v1/metrics/classifier'),
      apiClient.get<{ status: string; data: NerMetricsData }>('/api/v1/metrics/ner'),
    ]).then(async ([bertResult, nerResult]) => {
      if (bertResult.status === 'fulfilled') {
        setData(bertResult.value.data.data)
      } else {
        // Try static fallback (works even when backend is offline)
        try {
          const r = await fetch('/static-metrics/bert.json')
          if (!r.ok) throw new Error('Static fallback not found')
          const json = await r.json()
          setData(json)
          setFromStatic(true)
        } catch {
          const err = (bertResult.reason as any)?.response?.data?.detail
            ?? 'BERT metrics not found. Run ./run.sh setup or python training/train_classifier.py to generate them.'
          setError(err)
        }
      }
      if (nerResult.status === 'fulfilled') setNerData(nerResult.value.data.data)
      // NER metrics are optional — no error if missing
    }).finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <Layout>
        <div className="flex flex-col items-center justify-center h-64 gap-3">
          <svg className="w-8 h-8 animate-spin text-brand-500" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          <p className="text-sm text-slate-400">Loading model metrics…</p>
        </div>
      </Layout>
    )
  }

  if (error || !data) {
    return (
      <Layout>
        <div className="max-w-xl mx-auto py-20">
          <div className="card p-10 text-center">
            <div className="w-14 h-14 rounded-2xl bg-slate-100 flex items-center justify-center mx-auto mb-4">
              <svg className="w-7 h-7 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h2 className="text-lg font-bold text-slate-800 mb-2">Metrics Not Available</h2>
            <p className="text-sm text-slate-500 mb-4">{error ?? 'Metrics not found.'}</p>
            <div className="bg-slate-50 rounded-xl p-4 text-left mb-6">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">To generate metrics, run:</p>
              <code className="block text-xs text-slate-700 font-mono leading-relaxed">
                ./run.sh setup<br/>
                <span className="text-slate-400"># or individually:</span><br/>
                python training/train_classifier.py
              </code>
            </div>
            <Link to="/upload" className="btn-secondary">← Back to audit</Link>
          </div>
        </div>
      </Layout>
    )
  }

  const f1Theme = data.macro_f1 >= 0.85
    ? { text: 'text-emerald-600', bg: 'bg-emerald-50', border: 'border-emerald-200', bar: '#10b981', label: 'Above 85% target' }
    : data.macro_f1 >= 0.70
      ? { text: 'text-amber-600',  bg: 'bg-amber-50',  border: 'border-amber-200',  bar: '#f59e0b', label: 'Below 85% target' }
      : { text: 'text-red-600',    bg: 'bg-red-50',    border: 'border-red-200',    bar: '#ef4444', label: 'Needs retraining' }

  const maxCellValue = Math.max(...data.confusion_matrix.flatMap(r => r))

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-7">
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Model Metrics</h1>
          {fromStatic && (
            <span className="badge bg-amber-100 text-amber-700 border border-amber-200">
              Static snapshot · Start the API for live data
            </span>
          )}
        </div>
        <p className="text-slate-500 text-sm">
          BERT classifier evaluation on the held-out validation set ({data.val_size} examples).
          Base model:{' '}
          <code className="text-xs bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded-md font-mono">
            {data.base_model}
          </code>
        </p>
      </div>

      {/* ── Hero stats ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-7">
        {/* Macro F1 hero */}
        <div className={`sm:col-span-1 card p-6 flex flex-col items-center justify-center text-center border-2 ${f1Theme.border} ${f1Theme.bg}`}>
          <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Macro F1</p>
          <p className={`text-5xl font-extrabold tabular-nums ${f1Theme.text}`}>
            {(data.macro_f1 * 100).toFixed(1)}%
          </p>
          <p className={`text-xs mt-2 font-medium ${f1Theme.text} opacity-75`}>{f1Theme.label}</p>
        </div>

        {/* Other stats */}
        <div className="sm:col-span-3 grid grid-cols-1 sm:grid-cols-3 gap-4">
          <MetaCard
            label="Classes"
            value={data.labels.length}
            sub="EU AI Act article domains"
            icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" /></svg>}
          />
          <MetaCard
            label="Training Examples"
            value={data.train_size.toLocaleString()}
            sub="Synthetic + augmented"
            icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" /></svg>}
          />
          <MetaCard
            label="Validation Examples"
            value={data.val_size.toLocaleString()}
            sub="15% held-out split"
            icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" /></svg>}
          />
        </div>
      </div>

      {/* ── Per-class metrics ────────────────────────────────────────────────── */}
      <h2 className="section-label">Per-Class Performance</h2>
      <div className="card overflow-hidden mb-7">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50/80">
              <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Class</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Precision</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Recall</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">F1</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Support</th>
              <th className="px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide w-44">F1 Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {data.per_class.map((cls, i) => {
              const theme = cls.f1 >= 0.85
                ? { text: 'text-emerald-600', bar: '#10b981' }
                : cls.f1 >= 0.70
                  ? { text: 'text-amber-600', bar: '#f59e0b' }
                  : { text: 'text-red-600', bar: '#ef4444' }
              return (
                <tr key={i} className="hover:bg-slate-50/70 transition-colors">
                  <td className="px-5 py-3.5">
                    <span className="font-semibold text-slate-800 capitalize">
                      {cls.label.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-600 tabular-nums">
                    {(cls.precision * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-600 tabular-nums">
                    {(cls.recall * 100).toFixed(1)}%
                  </td>
                  <td className={`px-4 py-3.5 text-center font-bold tabular-nums ${theme.text}`}>
                    {(cls.f1 * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">{cls.support}</td>
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${cls.f1 * 100}%`, backgroundColor: theme.bar }}
                        />
                      </div>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* ── Confusion matrix ─────────────────────────────────────────────────── */}
      <h2 className="section-label">Confusion Matrix</h2>
      <div className="card p-5 overflow-x-auto mb-7">
        <p className="text-xs text-slate-400 mb-4">
          Rows = actual class · Columns = predicted class ·
          <span className="text-emerald-600 font-medium"> Green = correct</span>
          {' · '}
          <span className="text-red-500 font-medium">Red = misclassified</span>
        </p>
        <table className="text-xs border-separate border-spacing-0.5">
          <thead>
            <tr>
              <th className="w-36 pr-3 text-right text-slate-400 font-normal pb-2 text-[11px]">
                Actual ↓ / Predicted →
              </th>
              {data.labels.map(l => (
                <th key={l} className="w-12 text-center pb-2 font-semibold text-slate-500 text-[10px] leading-tight">
                  {l.replace(/_/g, ' ').split(' ').map((w, i) => <div key={i}>{w}</div>)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.confusion_matrix.map((row, ri) => (
              <tr key={ri}>
                <td className="pr-3 text-right font-semibold text-slate-600 py-0.5 text-[11px] whitespace-nowrap">
                  {data.labels[ri].replace(/_/g, ' ')}
                </td>
                {row.map((val, ci) => {
                  const intensity = maxCellValue > 0 ? val / maxCellValue : 0
                  const isDiag = ri === ci
                  const bg = isDiag
                    ? `rgba(16, 185, 129, ${0.12 + intensity * 0.75})`
                    : intensity > 0
                      ? `rgba(239, 68, 68, ${0.08 + intensity * 0.55})`
                      : 'transparent'
                  return (
                    <td
                      key={ci}
                      className="w-12 h-9 text-center font-bold rounded-lg"
                      style={{
                        backgroundColor: bg,
                        color: isDiag
                          ? intensity > 0.5 ? '#065f46' : '#047857'
                          : intensity > 0.3 ? '#991b1b' : val > 0 ? '#7f1d1d' : '#94a3b8',
                      }}
                    >
                      {val > 0 ? val : <span className="text-slate-200">·</span>}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* NER metrics section */}
      {nerData && <NerMetricsSection data={nerData} />}
    </Layout>
  )
}

// ── Sub-components ──────────────────────────────────────────────────────────

function MetaCard({ label, value, sub, icon }: { label: string; value: string | number; sub: string; icon: React.ReactNode }) {
  return (
    <div className="card p-5">
      <div className="flex items-center gap-2.5 mb-3">
        <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center text-brand-600">
          {icon}
        </div>
      </div>
      <p className="stat-value">{value}</p>
      <p className="text-xs font-semibold text-slate-700 mt-0.5">{label}</p>
      <p className="text-xs text-slate-400 mt-0.5">{sub}</p>
    </div>
  )
}

function NerMetricsSection({ data }: { data: NerMetricsData }) {
  const theme = data.overall_f1 >= 0.80
    ? { text: 'text-emerald-600', bg: 'bg-emerald-50', border: 'border-emerald-200', label: 'Above 80% target' }
    : data.overall_f1 >= 0.60
      ? { text: 'text-amber-600',  bg: 'bg-amber-50',  border: 'border-amber-200',  label: 'Below 80% target' }
      : { text: 'text-red-600',    bg: 'bg-red-50',    border: 'border-red-200',    label: 'Needs retraining' }

  return (
    <>
      <div className="border-t border-slate-200 pt-8 mb-7">
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight mb-1">NER Model Metrics</h1>
        <p className="text-sm text-slate-500">
          spaCy entity recogniser on the held-out dev set ({data.val_size} sentences).
          Train: {data.train_size} · Final loss:{' '}
          <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded-md font-mono text-slate-700">
            {data.final_loss.toFixed(4)}
          </code>
        </p>
      </div>

      {/* NER hero */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-7">
        <div className={`card p-6 flex flex-col items-center justify-center text-center border-2 ${theme.border} ${theme.bg}`}>
          <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Overall F1</p>
          <p className={`text-5xl font-extrabold tabular-nums ${theme.text}`}>
            {(data.overall_f1 * 100).toFixed(1)}%
          </p>
          <p className={`text-xs mt-2 font-medium ${theme.text} opacity-75`}>{theme.label}</p>
        </div>
        <div className="sm:col-span-3 grid grid-cols-3 gap-4">
          <MetaCard label="Precision" value={`${(data.overall_p * 100).toFixed(1)}%`} sub="Overall" icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>} />
          <MetaCard label="Recall" value={`${(data.overall_r * 100).toFixed(1)}%`} sub="Overall" icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>} />
          <MetaCard label="Entity Types" value={data.labels.length} sub="ARTICLE, OBLIGATION…" icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" /></svg>} />
        </div>
      </div>

      <h2 className="section-label">Per-Entity Performance</h2>
      <div className="card overflow-hidden mb-10">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50/80">
              <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Entity Type</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Precision</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Recall</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">F1</th>
              <th className="px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide w-44">F1 Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {data.per_label.map((row, i) => {
              const t = row.f1 >= 0.80
                ? { text: 'text-emerald-600', bar: '#10b981' }
                : row.f1 >= 0.60
                  ? { text: 'text-amber-600', bar: '#f59e0b' }
                  : { text: 'text-red-600', bar: '#ef4444' }
              return (
                <tr key={i} className="hover:bg-slate-50/70 transition-colors">
                  <td className="px-5 py-3.5">
                    <code className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded-md font-mono font-semibold">
                      {row.label}
                    </code>
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-600 tabular-nums">{(row.precision * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3.5 text-center text-slate-600 tabular-nums">{(row.recall * 100).toFixed(1)}%</td>
                  <td className={`px-4 py-3.5 text-center font-bold tabular-nums ${t.text}`}>{(row.f1 * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3.5">
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${row.f1 * 100}%`, backgroundColor: t.bar }}
                      />
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}
