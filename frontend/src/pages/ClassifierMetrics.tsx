// Classifier metrics dashboard: BERT macro F1 + confusion matrix, spaCy NER per-entity F1.

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
  const [data, setData] = useState<ClassifierMetricsData | null>(null)
  const [nerData, setNerData] = useState<NerMetricsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.allSettled([
      apiClient.get<{ status: string; data: ClassifierMetricsData }>('/api/v1/metrics/classifier'),
      apiClient.get<{ status: string; data: NerMetricsData }>('/api/v1/metrics/ner'),
    ]).then(([bertResult, nerResult]) => {
      if (bertResult.status === 'fulfilled') setData(bertResult.value.data.data)
      else {
        const err = (bertResult.reason as any)?.response?.data?.detail ?? 'Failed to load BERT metrics.'
        setError(err)
      }
      if (nerResult.status === 'fulfilled') setNerData(nerResult.value.data.data)
      // NER metrics are optional — no error if missing
    }).finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <svg className="w-8 h-8 animate-spin text-brand-400" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
        </div>
      </Layout>
    )
  }

  if (error || !data) {
    return (
      <Layout>
        <div className="max-w-xl mx-auto text-center py-16">
          <p className="text-slate-500 mb-2">{error ?? 'Metrics not available.'}</p>
          <p className="text-sm text-slate-400 mb-4">
            Run <code className="bg-slate-100 px-1 rounded">./run.sh setup</code> or{' '}
            <code className="bg-slate-100 px-1 rounded">python training/train_classifier.py</code> to generate metrics.
          </p>
          <Link to="/upload" className="text-brand-600 hover:underline text-sm">← Back to audit</Link>
        </div>
      </Layout>
    )
  }

  const f1Color = data.macro_f1 >= 0.85 ? 'text-green-600' : data.macro_f1 >= 0.70 ? 'text-amber-600' : 'text-red-600'
  const maxCellValue = Math.max(...data.confusion_matrix.flatMap(row => row))

  return (
    <Layout>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Classifier Metrics</h1>
        <p className="mt-1 text-slate-500 text-sm">
          BERT classifier performance on the held-out validation set ({data.val_size} examples).
          Base model: <span className="font-mono text-xs bg-slate-100 px-1 rounded">{data.base_model}</span>
          &nbsp;·&nbsp;Train: {data.train_size} · Val: {data.val_size}
        </p>
      </div>

      {/* Macro F1 hero */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 mb-6 flex items-center gap-6">
        <div className="text-center">
          <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">Macro F1</p>
          <p className={`text-5xl font-bold ${f1Color}`}>{(data.macro_f1 * 100).toFixed(1)}%</p>
          <p className="text-xs text-slate-400 mt-1">
            {data.macro_f1 >= 0.85 ? 'Above 85% target' : 'Below 85% target — consider retraining'}
          </p>
        </div>
        <div className="h-16 w-px bg-slate-200" />
        <div className="grid grid-cols-3 gap-4 text-center text-sm flex-1">
          <MetaBox label="Classes" value={data.labels.length} />
          <MetaBox label="Train examples" value={data.train_size} />
          <MetaBox label="Val examples" value={data.val_size} />
        </div>
      </div>

      {/* Per-class metrics */}
      <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
        Per-Class Metrics
      </h2>
      <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50 text-xs text-slate-500 uppercase tracking-wide">
              <th className="text-left px-4 py-3">Class</th>
              <th className="px-4 py-3">Precision</th>
              <th className="px-4 py-3">Recall</th>
              <th className="px-4 py-3">F1</th>
              <th className="px-4 py-3">Support</th>
              <th className="px-4 py-3 w-40">F1 bar</th>
            </tr>
          </thead>
          <tbody>
            {data.per_class.map((cls, i) => (
              <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                <td className="px-4 py-3 font-medium text-slate-700">
                  {cls.label.replace(/_/g, ' ')}
                </td>
                <td className="px-4 py-3 text-center text-slate-600">{(cls.precision * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 text-center text-slate-600">{(cls.recall * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 text-center font-semibold" style={{ color: f1BarColor(cls.f1) }}>
                  {(cls.f1 * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-center text-slate-400">{cls.support}</td>
                <td className="px-4 py-3">
                  <div className="bg-slate-100 rounded-full h-2 w-full">
                    <div
                      className="h-2 rounded-full transition-all"
                      style={{ width: `${cls.f1 * 100}%`, backgroundColor: f1BarColor(cls.f1) }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Confusion matrix */}
      <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
        Confusion Matrix
      </h2>
      <div className="bg-white border border-slate-200 rounded-2xl p-4 overflow-x-auto mb-10">
        <p className="text-xs text-slate-400 mb-3">Rows = actual class · Columns = predicted class · Darker = more predictions</p>
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <th className="w-28 pr-2 text-right text-slate-400 font-normal pb-1">Actual ↓ / Pred →</th>
              {data.labels.map(l => (
                <th key={l} className="w-14 text-center pb-1 font-medium text-slate-500 rotate-0">
                  {l.replace(/_/g, '\u200b_')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.confusion_matrix.map((row, ri) => (
              <tr key={ri}>
                <td className="pr-2 text-right font-medium text-slate-600 py-0.5">
                  {data.labels[ri].replace(/_/g, ' ')}
                </td>
                {row.map((val, ci) => {
                  const intensity = maxCellValue > 0 ? val / maxCellValue : 0
                  const isDiag = ri === ci
                  const bg = isDiag
                    ? `rgba(34,197,94,${0.15 + intensity * 0.7})`
                    : intensity > 0
                      ? `rgba(239,68,68,${intensity * 0.6})`
                      : 'transparent'
                  return (
                    <td
                      key={ci}
                      className="w-14 h-9 text-center font-semibold rounded"
                      style={{ backgroundColor: bg, color: intensity > 0.5 ? '#1e293b' : '#64748b' }}
                    >
                      {val}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* NER metrics */}
      {nerData && <NerMetricsSection data={nerData} />}
    </Layout>
  )
}

function MetaBox({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="text-xs text-slate-400">{label}</p>
      <p className="font-semibold text-slate-700">{value}</p>
    </div>
  )
}

function f1BarColor(f1: number): string {
  if (f1 >= 0.85) return '#16a34a'
  if (f1 >= 0.70) return '#d97706'
  return '#dc2626'
}

function NerMetricsSection({ data }: { data: NerMetricsData }) {
  const overallColor = data.overall_f1 >= 0.80 ? 'text-green-600' : data.overall_f1 >= 0.60 ? 'text-amber-600' : 'text-red-600'
  return (
    <>
      <div className="border-t border-slate-200 pt-8 mb-6">
        <h1 className="text-2xl font-bold text-slate-800">NER Model Metrics</h1>
        <p className="mt-1 text-slate-500 text-sm">
          spaCy entity recogniser performance on the held-out dev set ({data.val_size} sentences).
          Train: {data.train_size} &nbsp;·&nbsp; Final loss: {data.final_loss.toFixed(4)}
        </p>
      </div>

      {/* Overall F1 hero */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 mb-6 flex items-center gap-6">
        <div className="text-center">
          <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">Overall F1</p>
          <p className={`text-5xl font-bold ${overallColor}`}>{(data.overall_f1 * 100).toFixed(1)}%</p>
          <p className="text-xs text-slate-400 mt-1">
            {data.overall_f1 >= 0.80 ? 'Above 80% target' : 'Below 80% target — consider more training data'}
          </p>
        </div>
        <div className="h-16 w-px bg-slate-200" />
        <div className="grid grid-cols-3 gap-4 text-center text-sm flex-1">
          <MetaBox label="Precision" value={`${(data.overall_p * 100).toFixed(1)}%`} />
          <MetaBox label="Recall" value={`${(data.overall_r * 100).toFixed(1)}%`} />
          <MetaBox label="Entity types" value={data.labels.length} />
        </div>
      </div>

      {/* Per-label table */}
      <h2 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
        Per-Entity Metrics
      </h2>
      <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden mb-10">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50 text-xs text-slate-500 uppercase tracking-wide">
              <th className="text-left px-4 py-3">Entity</th>
              <th className="px-4 py-3">Precision</th>
              <th className="px-4 py-3">Recall</th>
              <th className="px-4 py-3">F1</th>
              <th className="px-4 py-3 w-40">F1 bar</th>
            </tr>
          </thead>
          <tbody>
            {data.per_label.map((row, i) => (
              <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                <td className="px-4 py-3 font-medium text-slate-700">{row.label}</td>
                <td className="px-4 py-3 text-center text-slate-600">{(row.precision * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 text-center text-slate-600">{(row.recall * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 text-center font-semibold" style={{ color: f1BarColor(row.f1) }}>
                  {(row.f1 * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3">
                  <div className="bg-slate-100 rounded-full h-2 w-full">
                    <div
                      className="h-2 rounded-full transition-all"
                      style={{ width: `${row.f1 * 100}%`, backgroundColor: f1BarColor(row.f1) }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
