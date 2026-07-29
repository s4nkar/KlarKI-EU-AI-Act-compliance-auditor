// Model metrics dashboard: BERT, NER, specialist classifiers, version registry, eval suite.
// Falls back to /static-metrics/bert.json for BERT when the backend API is not running.

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

interface SpecialistMetric {
  classifier_type: string
  macro_f1: number
  per_class: PerClassMetric[]
  confusion_matrix?: number[][]
  labels?: string[]
  train_size: number
  val_size: number
  base_model: string
}

type SpecialistMetricsData = Record<string, SpecialistMetric>

interface ModelVersionEntry {
  version: string
  created_at: string
  score: number | null
  data_version: string | null
  is_active: boolean
}

interface ModelVersionInfo {
  active: string | null
  metric_key: string
  active_dir: string
  versions: ModelVersionEntry[]
}

interface DataVersionEntry {
  version: string
  created_at: string
  record_count: number | null
  is_active: boolean
}

interface DataVersionInfo {
  active: string | null
  versions: DataVersionEntry[]
}

interface VersionsData {
  models: Record<string, ModelVersionInfo>
  data: Record<string, DataVersionInfo>
}

interface EvalResult {
  eval: string
  status: 'pass' | 'warn' | 'fail' | 'skip'
  reason?: string
  macro_f1?: number
  accuracy?: number
  per_class?: Record<string, { precision: number; recall: number; f1: number; support: number }>
  'recall@1'?: number
  'recall@3'?: number
  'recall@5'?: number
  mrr?: number
  n_queries?: number
  adversarial_accuracy?: number
  n_examples?: number
  n_correct?: number
  bert?: { consistency_rate: number; n_probes: number }
  ollama?: { consistency_rate: number; n_probes: number }
  citation_rate?: number
  n_articles?: number
  checks?: Record<string, boolean>
  checks_passed?: number
  checks_total?: number
  article_scores?: Record<string, number>
  overall_f1?: number
  per_label?: Record<string, { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number }>
  weak_labels?: Record<string, number>
  failing_labels?: Record<string, number>
  n_gold?: number
  // Phase 3 eval fields
  tpr?: number
  tnr?: number
  balanced_accuracy?: number
  by_outcome?: {
    prohibited?: { n_examples: number; n_correct: number; precision: number; recall: number; f1: number }
    high_risk?:  { n_examples: number; n_correct: number; precision: number; recall: number; f1: number }
    minimal?:    { n_examples: number; n_correct: number }
  }
  mode?: string
  n_samples?: number
  n_errors?: number
  threshold_accuracy?: number
  threshold_tpr?: number
  // Specialist ML classifier eval fields (risk + prohibited)
  recall?: number
  precision?: number
  f1?: number
  avg_confidence?: number
  n_high_risk?: number
  n_not_high_risk?: number
  n_prohibited?: number
  n_not_prohibited?: number
  by_type?: Record<string, { recall: number; tp: number; fp: number; tn: number; fn: number }>
}

type EvalResultsMap = Record<string, EvalResult>

export default function ClassifierMetrics() {
  const [data, setData]                   = useState<ClassifierMetricsData | null>(null)
  const [nerData, setNerData]             = useState<NerMetricsData | null>(null)
  const [specialistData, setSpecialist]   = useState<SpecialistMetricsData>({})
  const [versionsData, setVersions]       = useState<VersionsData | null>(null)
  const [evalData, setEvalData]           = useState<EvalResultsMap | null>(null)
  const [loading, setLoading]             = useState(true)
  const [error, setError]                 = useState<string | null>(null)
  const [fromStatic, setFromStatic]       = useState(false)
  const [modelTab, setModelTab]           = useState<'all' | 'bert' | 'ner' | 'actor' | 'risk' | 'prohibited'>('all')

  useEffect(() => {
    Promise.allSettled([
      apiClient.get<{ status: string; data: ClassifierMetricsData }>('/api/v1/metrics/classifier'),
      apiClient.get<{ status: string; data: NerMetricsData }>('/api/v1/metrics/ner'),
      apiClient.get<{ status: string; data: SpecialistMetricsData }>('/api/v1/metrics/specialists'),
      apiClient.get<{ status: string; data: VersionsData }>('/api/v1/metrics/versions'),
      apiClient.get<{ status: string; data: EvalResultsMap }>('/api/v1/metrics/evaluation'),
    ]).then(async ([bertResult, nerResult, specialistResult, versionsResult, evalResult]) => {
      if (bertResult.status === 'fulfilled') {
        setData(bertResult.value.data.data)
      } else {
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
      if (specialistResult.status === 'fulfilled') setSpecialist(specialistResult.value.data.data)
      if (versionsResult.status === 'fulfilled') setVersions(versionsResult.value.data.data)
      if (evalResult.status === 'fulfilled') {
        const map = evalResult.value.data.data
        if (Object.keys(map).length > 0) setEvalData(map)
      }
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
            <div className="w-14 h-14 rounded-2xl bg-surface-raised flex items-center justify-center mx-auto mb-4">
              <svg className="w-7 h-7 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h2 className="text-lg font-bold text-slate-100 mb-2">Metrics Not Available</h2>
            <p className="text-sm text-slate-500 mb-4">{error ?? 'Metrics not found.'}</p>
            <div className="bg-surface rounded-xl p-4 text-left mb-6">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">To generate metrics, run:</p>
              <code className="block text-xs text-slate-300 font-mono leading-relaxed">
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

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Model Metrics</h1>
          {fromStatic && (
            <span className="badge bg-amber-500/10 text-amber-300">
              Static snapshot · Start the API for live data
            </span>
          )}
        </div>
        <p className="text-slate-500 text-sm">
          Held-out validation metrics, per-class evaluation tables, and confusion matrices across all five model classifiers.
        </p>
      </div>

      {/* Model navigation tabs */}
      <div className="flex flex-wrap items-center gap-2 mb-8 border-b border-line pb-4">
        {[
          { id: 'all', label: 'All Models Overview' },
          { id: 'bert', label: 'BERT Domain Classifier' },
          { id: 'ner', label: 'spaCy NER' },
          { id: 'actor', label: 'Actor Classifier (Art. 3)' },
          { id: 'risk', label: 'Risk Classifier (Art. 6)' },
          { id: 'prohibited', label: 'Prohibited Classifier (Art. 5)' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setModelTab(tab.id as any)}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              modelTab === tab.id
                ? 'bg-brand-500 text-white shadow-sm'
                : 'bg-surface-raised text-slate-400 hover:text-slate-200 hover:bg-surface-hover'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* 1. BERT Classifier */}
      {(modelTab === 'all' || modelTab === 'bert') && (
        <ClassificationModelSection
          title="BERT Domain Classifier"
          subtitle={`EU AI Act 8-domain article classifier evaluation on held-out validation set (${data.val_size} examples).`}
          data={data}
          targetF1={0.85}
          trainCmd="python training/train_classifier.py"
        />
      )}

      {/* 2. spaCy NER Model */}
      {(modelTab === 'all' || modelTab === 'ner') && nerData && (
        <NerMetricsSection data={nerData} />
      )}

      {/* 3. Actor Specialist Classifier */}
      {(modelTab === 'all' || modelTab === 'actor') && (
        <ClassificationModelSection
          title="Actor Classifier (Article 3)"
          subtitle="4-class role detection: Provider / Deployer / Importer / Distributor."
          data={specialistData.actor ?? null}
          targetF1={0.90}
          trainCmd="python training/scripts/train_specialist_classifiers.py --type actor"
        />
      )}

      {/* 4. Risk Specialist Classifier */}
      {(modelTab === 'all' || modelTab === 'risk') && (
        <ClassificationModelSection
          title="Risk Classifier (Article 6 + Annex III)"
          subtitle="High-risk vs Not high-risk binary classifier gate."
          data={specialistData.risk ?? null}
          targetF1={0.93}
          trainCmd="python training/scripts/train_specialist_classifiers.py --type risk"
        />
      )}

      {/* 5. Prohibited Specialist Classifier */}
      {(modelTab === 'all' || modelTab === 'prohibited') && (
        <ClassificationModelSection
          title="Prohibited Classifier (Article 5)"
          subtitle="Article 5 prohibited practice binary classifier gate."
          data={specialistData.prohibited ?? null}
          targetF1={0.95}
          trainCmd="python training/scripts/train_specialist_classifiers.py --type prohibited"
        />
      )}

      {/* Model version registry */}
      {versionsData && <VersionRegistrySection data={versionsData} />}

      {/* Evaluation suite results */}
      <EvaluationSection data={evalData} />
    </Layout>
  )
}

// ── Sub-components ──────────────────────────────────────────────────────────

interface ClassificationModelSectionProps {
  title: string
  subtitle: string
  data: {
    macro_f1: number
    per_class: PerClassMetric[]
    confusion_matrix?: number[][]
    labels?: string[]
    val_size: number
    train_size: number
    base_model?: string
  } | null
  targetF1?: number
  trainCmd?: string
}

function ClassificationModelSection({
  title,
  subtitle,
  data,
  targetF1 = 0.85,
  trainCmd,
}: ClassificationModelSectionProps) {
  if (!data) {
    return (
      <div className="border-t border-line pt-8 mb-7">
        <h2 className="text-2xl font-extrabold text-white tracking-tight mb-1">{title}</h2>
        <p className="text-sm text-slate-500 mb-4">{subtitle}</p>
        <div className="card p-6 text-center">
          <p className="text-sm font-semibold text-slate-300 mb-1">Model Not Yet Trained</p>
          <p className="text-xs text-slate-500 mb-3">Run setup or training script to generate model metrics.</p>
          {trainCmd && (
            <code className="text-xs bg-surface-raised text-brand-300 px-3 py-1.5 rounded font-mono inline-block">
              {trainCmd}
            </code>
          )}
        </div>
      </div>
    )
  }

  const f1Theme = data.macro_f1 >= targetF1
    ? { text: 'text-emerald-400', bg: 'bg-emerald-500/[0.06]', border: 'border-emerald-500/20', bar: '#34d399', label: `Above ${(targetF1 * 100).toFixed(0)}% target` }
    : data.macro_f1 >= 0.70
      ? { text: 'text-amber-400',  bg: 'bg-amber-500/[0.06]',  border: 'border-amber-500/20',  bar: '#fbbf24', label: `Below ${(targetF1 * 100).toFixed(0)}% target` }
      : { text: 'text-red-400',    bg: 'bg-red-500/[0.06]',    border: 'border-red-500/20',    bar: '#f87171', label: 'Needs retraining' }

  const labels = data.labels || data.per_class.map(c => c.label)
  const maxCellValue = data.confusion_matrix && data.confusion_matrix.length > 0
    ? Math.max(...data.confusion_matrix.flatMap(r => r), 1)
    : 1

  return (
    <div className="border-t border-line pt-8 mb-7">
      <div className="mb-6">
        <h2 className="text-2xl font-extrabold text-white tracking-tight mb-1">{title}</h2>
        <p className="text-sm text-slate-500">
          {subtitle}
          {data.base_model && (
            <>
              {' · '}Base model:{' '}
              <code className="text-xs bg-surface-raised text-slate-300 px-1.5 py-0.5 rounded-md font-mono">
                {data.base_model}
              </code>
            </>
          )}
        </p>
      </div>

      {/* Hero stats */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-7">
        <div className={`sm:col-span-1 card p-6 flex flex-col items-center justify-center text-center border-2 ${f1Theme.border} ${f1Theme.bg}`}>
          <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Macro F1</p>
          <p className={`text-5xl font-extrabold tabular-nums ${f1Theme.text}`}>
            {(data.macro_f1 * 100).toFixed(1)}%
          </p>
          <p className={`text-xs mt-2 font-medium ${f1Theme.text} opacity-75`}>{f1Theme.label}</p>
        </div>

        <div className="sm:col-span-3 grid grid-cols-1 sm:grid-cols-3 gap-4">
          <MetaCard
            label="Classes"
            value={labels.length}
            sub="Target output categories"
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
            sub="Held-out split"
            icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" /></svg>}
          />
        </div>
      </div>

      {/* Per-class metrics table */}
      <h3 className="section-label">Per-Class Performance</h3>
      <div className="card overflow-hidden mb-7">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-line bg-surface-raised/80">
              <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Class</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Precision</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Recall</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">F1</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Support</th>
              <th className="px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide w-44">F1 Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {data.per_class.map((cls, i) => {
              const theme = cls.f1 >= targetF1
                ? { text: 'text-emerald-400', bar: '#10b981' }
                : cls.f1 >= 0.70
                  ? { text: 'text-amber-400', bar: '#f59e0b' }
                  : { text: 'text-red-400', bar: '#ef4444' }
              return (
                <tr key={i} className="hover:bg-surface-raised/70 transition-colors">
                  <td className="px-5 py-3.5">
                    <span className="font-semibold text-slate-100 capitalize">
                      {cls.label.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">
                    {(cls.precision * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">
                    {(cls.recall * 100).toFixed(1)}%
                  </td>
                  <td className={`px-4 py-3.5 text-center font-bold tabular-nums ${theme.text}`}>
                    {(cls.f1 * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">{cls.support}</td>
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-surface-raised rounded-full overflow-hidden">
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

      {/* Confusion matrix */}
      {data.confusion_matrix && data.confusion_matrix.length > 0 && (
        <>
          <h3 className="section-label">Confusion Matrix</h3>
          <div className="card p-5 overflow-x-auto mb-7">
            <p className="text-xs text-slate-400 mb-4">
              Rows = actual class · Columns = predicted class ·
              <span className="text-emerald-400 font-medium"> Green = correct</span>
              {' · '}
              <span className="text-red-400 font-medium">Red = misclassified</span>
            </p>
            <table className="text-xs border-separate border-spacing-0.5">
              <thead>
                <tr>
                  <th className="w-36 pr-3 text-right text-slate-400 font-normal pb-2 text-[11px]">
                    Actual ↓ / Predicted →
                  </th>
                  {labels.map(l => (
                    <th key={l} className="w-16 text-center pb-2 font-semibold text-slate-500 text-[10px] leading-tight">
                      {l.replace(/_/g, ' ').split(' ').map((w, i) => <div key={i}>{w}</div>)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.confusion_matrix.map((row, ri) => (
                  <tr key={ri}>
                    <td className="pr-3 text-right font-semibold text-slate-400 py-0.5 text-[11px] whitespace-nowrap">
                      {labels[ri]?.replace(/_/g, ' ')}
                    </td>
                    {row.map((val, ci) => {
                      const intensity = maxCellValue > 0 ? val / maxCellValue : 0
                      const isDiag = ri === ci
                      const bg = isDiag
                        ? `rgba(52, 211, 153, ${0.08 + intensity * 0.32})`
                        : intensity > 0
                          ? `rgba(248, 113, 113, ${0.06 + intensity * 0.32})`
                          : 'transparent'
                      return (
                        <td
                          key={ci}
                          className="w-16 h-9 text-center font-bold rounded-lg"
                          style={{
                            backgroundColor: bg,
                            color: isDiag
                              ? '#6ee7b7'
                              : intensity > 0.3 ? '#fca5a5' : val > 0 ? '#fca5a5' : '#64748b',
                          }}
                        >
                          {val > 0 ? val : <span className="text-slate-700">·</span>}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

function MetaCard({ label, value, sub, icon }: { label: string; value: string | number; sub: string; icon: React.ReactNode }) {
  return (
    <div className="card p-5">
      <div className="flex items-center gap-2.5 mb-3">
        <div className="w-8 h-8 rounded-lg bg-brand-500/10 flex items-center justify-center text-brand-400">
          {icon}
        </div>
      </div>
      <p className="stat-value">{value}</p>
      <p className="text-xs font-semibold text-slate-300 mt-0.5">{label}</p>
      <p className="text-xs text-slate-400 mt-0.5">{sub}</p>
    </div>
  )
}

function EvaluationSection({ data }: { data: EvalResultsMap | null }) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})

  function toggle(key: string) {
    setExpanded(prev => ({ ...prev, [key]: !prev[key] }))
  }

  function hasDetails(r: EvalResult): boolean {
    return !!(
      (r.per_class && Object.keys(r.per_class).length > 0) ||
      (r.per_label && Object.keys(r.per_label).length > 0) ||
      r.by_outcome ||
      (r.checks && Object.keys(r.checks).length > 0)
    )
  }

  if (!data) {
    return (
      <div className="border-t border-line pt-8 mb-10">
        <h1 className="text-2xl font-extrabold text-white tracking-tight mb-1">Evaluation Suite</h1>
        <p className="text-sm text-slate-500 mb-4">
          Run <code className="text-xs bg-surface-raised px-1.5 py-0.5 rounded-md font-mono text-slate-300">./run.sh test</code> to execute the evaluation suite and populate this section.
        </p>
        <div className="card p-6 text-center text-slate-400 text-sm">No evaluation results yet.</div>
      </div>
    )
  }

  const EVAL_META: Record<string, { label: string; desc: string }> = {
    classifier:      { label: 'Gold Dataset',        desc: 'BERT on 80 hand-labeled examples' },
    ner:             { label: 'NER Gold Eval',        desc: '8-label entity recognition — 31 gold sentences' },
    rag:             { label: 'RAG Retrieval',        desc: 'Recall@K against ChromaDB' },
    pipeline:        { label: 'End-to-End',           desc: 'Full pipeline on synthetic doc' },
    hallucination:   { label: 'Hallucination',        desc: 'Citation & grounding checks' },
    adversarial:     { label: 'Adversarial',          desc: '30 paraphrase robustness tests' },
    consistency:     { label: 'Consistency',          desc: 'Determinism across repeated runs' },
    actor:           { label: 'Actor Classifier',     desc: 'Pattern accuracy on 31 gold documents' },
    applicability:   { label: 'Applicability Gate',   desc: 'Art.5/6/Annex III decision accuracy on 28 examples' },
    evidence_mapper: { label: 'Evidence Mapper',      desc: 'Synonym hit-rate on 24 gold evidence chunks' },
    risk:            { label: 'Risk Classifier (ML)', desc: 'High-risk vs not-high-risk on 40 policy-doc examples' },
    prohibited:      { label: 'Prohibited Classifier (ML)', desc: 'Art.5 prohibited vs lawful on 40 policy-doc examples' },
  }

  const statusStyle = (s: string) =>
    s === 'pass' ? 'bg-emerald-500/10 text-emerald-400'
    : s === 'warn' ? 'bg-amber-500/10 text-amber-400'
    : s === 'skip' ? 'bg-surface-raised text-slate-500'
    : 'bg-red-500/10 text-red-400'

  const statusDot = (s: string) =>
    s === 'pass' ? '#34d399' : s === 'warn' ? '#fbbf24' : s === 'skip' ? '#64748b' : '#f87171'

  function keyMetric(key: string, r: EvalResult): string {
    if (key === 'classifier')      return r.macro_f1 != null ? `Macro F1 ${(r.macro_f1 * 100).toFixed(1)}%` : '—'
    if (key === 'ner')             return r.overall_f1 != null ? `Overall F1 ${(r.overall_f1 * 100).toFixed(1)}%  ·  ${r.n_gold ?? '—'} gold samples` : '—'
    if (key === 'rag')             return r['recall@3'] != null ? `Recall@3 ${(r['recall@3'] * 100).toFixed(1)}%  ·  MRR ${r.mrr?.toFixed(3) ?? '—'}` : '—'
    if (key === 'adversarial')     return r.adversarial_accuracy != null ? `Accuracy ${(r.adversarial_accuracy * 100).toFixed(1)}%` : '—'
    if (key === 'consistency')     return r.bert?.consistency_rate != null ? `BERT ${(r.bert.consistency_rate * 100).toFixed(0)}%  ·  LLM ${r.ollama?.consistency_rate != null ? (r.ollama.consistency_rate * 100).toFixed(0) + '%' : '—'}` : '—'
    if (key === 'hallucination')   return r.citation_rate != null ? `Citation rate ${(r.citation_rate * 100).toFixed(1)}%` : '—'
    if (key === 'pipeline')        return r.checks_passed != null ? `${r.checks_passed}/${r.checks_total} checks passed` : '—'
    if (key === 'actor')           return r.accuracy != null ? `Accuracy ${(r.accuracy * 100).toFixed(1)}%  ·  F1 ${r.macro_f1 != null ? (r.macro_f1 * 100).toFixed(1) + '%' : '—'}` : '—'
    if (key === 'applicability')   return r.accuracy != null ? `Accuracy ${(r.accuracy * 100).toFixed(1)}%  ·  Prohibited recall ${r.by_outcome?.prohibited?.recall != null ? (r.by_outcome.prohibited.recall * 100).toFixed(0) + '%' : '—'}` : '—'
    if (key === 'evidence_mapper') return r.tpr != null ? `TPR ${(r.tpr * 100).toFixed(1)}%  ·  TNR ${r.tnr != null ? (r.tnr * 100).toFixed(1) + '%' : '—'}` : '—'
    if (key === 'risk')            return r.accuracy != null ? `Accuracy ${(r.accuracy * 100).toFixed(1)}%  ·  Recall ${r.recall != null ? (r.recall * 100).toFixed(1) + '%' : '—'}` : '—'
    if (key === 'prohibited')      return r.accuracy != null ? `Accuracy ${(r.accuracy * 100).toFixed(1)}%  ·  Recall ${r.recall != null ? (r.recall * 100).toFixed(1) + '%' : '—'}` : '—'
    return '—'
  }

  return (
    <div className="border-t border-line pt-8 mb-10">
      <h1 className="text-2xl font-extrabold text-white tracking-tight mb-1">Evaluation Suite</h1>
      <p className="text-sm text-slate-500 mb-6">
        Results from the last <code className="text-xs bg-surface-raised px-1.5 py-0.5 rounded-md font-mono text-slate-300">./run.sh test</code> run.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(EVAL_META).map(([key, meta]) => {
          const r = data[key]
          if (!r) return (
            <div key={key} className="card p-5 opacity-50">
              <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">{meta.label}</p>
              <p className="text-xs text-slate-400">{meta.desc}</p>
              <p className="text-sm text-slate-400 mt-3">Not yet run</p>
            </div>
          )
          return (
            <div key={key} className="card p-5">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">{meta.label}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{meta.desc}</p>
                </div>
                <span className={`badge ${statusStyle(r.status)}`}>
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: statusDot(r.status) }} />
                  {r.status}
                </span>
              </div>
              {r.status === 'skip'
                ? <p className="text-xs text-slate-400 italic">{r.reason ?? 'Skipped'}</p>
                : <div>
                    <p className="text-sm font-semibold text-slate-300 tabular-nums">{keyMetric(key, r)}</p>
                    {key === 'ner' && r.weak_labels && Object.keys(r.weak_labels).length > 0 && (
                      <p className="text-xs text-amber-400 mt-1">
                        Weak: {Object.keys(r.weak_labels).join(', ')}
                      </p>
                    )}
                    {key === 'actor' && r.n_samples != null && (
                      <p className="text-xs text-slate-400 mt-1">
                        {r.n_samples} gold docs · {r.mode === 'with_ml' ? 'pattern + ML' : 'pattern only'}
                      </p>
                    )}
                    {key === 'applicability' && r.by_outcome && (
                      <p className="text-xs text-slate-400 mt-1">
                        High-risk recall{' '}
                        {r.by_outcome.high_risk?.recall != null
                          ? `${(r.by_outcome.high_risk.recall * 100).toFixed(0)}%`
                          : '—'}
                        {' · '}
                        {r.n_samples} examples
                      </p>
                    )}
                    {key === 'evidence_mapper' && r.balanced_accuracy != null && (
                      <p className="text-xs text-slate-400 mt-1">
                        Balanced acc. {(r.balanced_accuracy * 100).toFixed(1)}%
                        {r.n_errors != null && r.n_errors > 0 && (
                          <span className="text-amber-400"> · {r.n_errors} misses</span>
                        )}
                      </p>
                    )}
                    {key === 'risk' && r.tnr != null && (
                      <p className="text-xs text-slate-400 mt-1">
                        Specificity {(r.tnr * 100).toFixed(1)}%
                        {' · '}
                        {r.n_samples ?? '—'} gold docs
                        {r.f1 != null && <span> · F1 {(r.f1 * 100).toFixed(1)}%</span>}
                      </p>
                    )}
                    {key === 'prohibited' && r.tnr != null && (
                      <p className="text-xs text-slate-400 mt-1">
                        Specificity {(r.tnr * 100).toFixed(1)}%
                        {' · '}
                        {r.n_samples ?? '—'} gold docs
                        {r.f1 != null && <span> · F1 {(r.f1 * 100).toFixed(1)}%</span>}
                      </p>
                    )}
                    {hasDetails(r) && (
                      <button
                        onClick={() => toggle(key)}
                        className="mt-3 flex items-center gap-1.5 text-xs text-brand-400 hover:text-brand-300 font-medium"
                      >
                        <svg
                          className={`w-3.5 h-3.5 transition-transform duration-200 ${expanded[key] ? 'rotate-180' : ''}`}
                          fill="none" viewBox="0 0 24 24" stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                        {expanded[key] ? 'Hide' : 'Show'} breakdown
                      </button>
                    )}
                    {expanded[key] && hasDetails(r) && (
                      <div className="mt-3 pt-3 border-t border-line">
                        {r.per_class && <EvalPerClassDetail data={r.per_class} />}
                        {r.per_label && <EvalPerLabelDetail data={r.per_label} />}
                        {r.by_outcome && <EvalByOutcomeDetail data={r.by_outcome} />}
                        {r.checks && <EvalChecksDetail data={r.checks} articleScores={r.article_scores} />}
                      </div>
                    )}
                  </div>
              }
            </div>
          )
        })}
      </div>
    </div>
  )
}




function VersionRegistrySection({ data }: { data: VersionsData }) {
  const MODEL_LABELS: Record<string, string> = {
    bert:       'BERT Domain Classifier',
    ner:        'spaCy NER',
    actor:      'Actor Classifier',
    risk:       'Risk Classifier',
    prohibited: 'Prohibited Classifier',
  }

  const modelEntries = Object.entries(data.models)
  const dataEntries  = Object.entries(data.data)

  if (modelEntries.length === 0) return null

  function fmtDate(iso: string) {
    try { return new Date(iso).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) }
    catch { return iso }
  }

  return (
    <div className="border-t border-line pt-8 mb-7">
      <h1 className="text-2xl font-extrabold text-white tracking-tight mb-1">Model Version Registry</h1>
      <p className="text-sm text-slate-500 mb-6">
        Active model versions, training history, and data-version traceability across all five classifiers.
      </p>

      {/* Models table */}
      <h2 className="section-label">Active Models</h2>
      <div className="card overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-line bg-surface-raised/80">
              <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Model</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Active Ver.</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Score</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Metric</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Data Ver.</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Versions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {modelEntries.map(([key, info]) => {
              const active = info.versions.find(v => v.is_active)
              const scoreVal = active?.score ?? null
              const scoreTheme = scoreVal == null ? 'text-slate-400'
                : scoreVal >= 0.85 ? 'text-emerald-400 font-bold'
                : scoreVal >= 0.70 ? 'text-amber-400 font-bold'
                : 'text-red-400 font-bold'
              return (
                <tr key={key} className="hover:bg-surface-raised/70 transition-colors">
                  <td className="px-5 py-3.5">
                    <span className="font-semibold text-slate-100">{MODEL_LABELS[key] ?? key}</span>
                  </td>
                  <td className="px-4 py-3.5 text-center">
                    {info.active
                      ? <span className="badge bg-brand-500/10 text-brand-300">{info.active}</span>
                      : <span className="text-slate-400 text-xs italic">untrained</span>}
                  </td>
                  <td className={`px-4 py-3.5 text-center tabular-nums ${scoreTheme}`}>
                    {scoreVal != null ? `${(scoreVal * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="px-4 py-3.5 text-center">
                    <code className="text-xs bg-surface-raised text-slate-400 px-1.5 py-0.5 rounded font-mono">
                      {info.metric_key}
                    </code>
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-500 text-xs">
                    {active?.data_version ?? <span className="text-slate-600">—</span>}
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">
                    {info.versions.length}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Version history per model */}
      <h2 className="section-label">Version History</h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4 mb-6">
        {modelEntries.map(([key, info]) => (
          <div key={key} className="card p-4">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">
              {MODEL_LABELS[key] ?? key}
            </p>
            {info.versions.length === 0 ? (
              <p className="text-xs text-slate-400 italic">No versions recorded</p>
            ) : (
              <div className="space-y-2">
                {[...info.versions].reverse().map(v => {
                  const scoreStr = v.score != null ? `${(v.score * 100).toFixed(1)}%` : '—'
                  return (
                    <div key={v.version} className={`flex items-center justify-between rounded-lg px-3 py-2 text-xs ${v.is_active ? 'bg-brand-500/10' : 'bg-surface'}`}>
                      <div className="flex items-center gap-2">
                        <span className={`font-bold ${v.is_active ? 'text-brand-300' : 'text-slate-500'}`}>
                          {v.version}
                        </span>
                        {v.is_active && (
                          <span className="badge bg-brand-500/10 text-brand-300 text-[10px] py-0 px-1.5">active</span>
                        )}
                      </div>
                      <div className="text-right text-slate-500">
                        <div className="tabular-nums font-semibold">{scoreStr}</div>
                        <div className="text-slate-400 text-[10px]">{fmtDate(v.created_at)}</div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Training data versions */}
      {dataEntries.length > 0 && (
        <>
          <h2 className="section-label">Training Data Versions</h2>
          <div className="card overflow-hidden mb-2">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-line bg-surface-raised/80">
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Dataset</th>
                  <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Active Ver.</th>
                  <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Records</th>
                  <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Snapshots</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {dataEntries.map(([key, info]) => {
                  const active = info.versions.find(v => v.is_active)
                  return (
                    <tr key={key} className="hover:bg-surface-raised/70 transition-colors">
                      <td className="px-5 py-3.5">
                        <span className="font-semibold text-slate-100 capitalize">{key.replace(/_/g, ' ')} labels</span>
                      </td>
                      <td className="px-4 py-3.5 text-center">
                        {info.active
                          ? <span className="badge bg-surface-raised text-slate-400">{info.active}</span>
                          : <span className="text-slate-400 text-xs italic">none</span>}
                      </td>
                      <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">
                        {active?.record_count?.toLocaleString() ?? '—'}
                      </td>
                      <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">
                        {info.versions.length}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}


function EvalPerClassDetail({ data }: {
  data: Record<string, { precision: number; recall: number; f1: number; support?: number }>
}) {
  const entries = Object.entries(data)
  const hasSupport = entries.some(([, m]) => m.support !== undefined)
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-[10px] font-semibold text-slate-400 uppercase tracking-wide border-b border-line">
          <th className="pb-1.5">Class</th>
          <th className="pb-1.5 text-center">P</th>
          <th className="pb-1.5 text-center">R</th>
          <th className="pb-1.5 text-center">F1</th>
          {hasSupport && <th className="pb-1.5 text-center">N</th>}
        </tr>
      </thead>
      <tbody>
        {entries.map(([cls, m]) => {
          const color = m.f1 >= 0.85 ? 'text-emerald-400' : m.f1 >= 0.70 ? 'text-amber-400' : 'text-red-400'
          return (
            <tr key={cls} className="border-b border-line last:border-0">
              <td className="py-1 text-slate-400 capitalize font-medium">{cls.replace(/_/g, ' ')}</td>
              <td className="py-1 text-center text-slate-500 tabular-nums">{(m.precision * 100).toFixed(0)}%</td>
              <td className="py-1 text-center text-slate-500 tabular-nums">{(m.recall * 100).toFixed(0)}%</td>
              <td className={`py-1 text-center font-bold tabular-nums ${color}`}>{(m.f1 * 100).toFixed(1)}%</td>
              {hasSupport && <td className="py-1 text-center text-slate-400 tabular-nums">{m.support}</td>}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function EvalPerLabelDetail({ data }: {
  data: Record<string, { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number }>
}) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-[10px] font-semibold text-slate-400 uppercase tracking-wide border-b border-line">
          <th className="pb-1.5">Label</th>
          <th className="pb-1.5 text-center">F1</th>
          <th className="pb-1.5 text-center">TP</th>
          <th className="pb-1.5 text-center">FP</th>
          <th className="pb-1.5 text-center">FN</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(data).map(([lbl, m]) => {
          const color = m.f1 >= 0.80 ? 'text-emerald-400' : m.f1 >= 0.60 ? 'text-amber-400' : 'text-red-400'
          return (
            <tr key={lbl} className="border-b border-line last:border-0">
              <td className="py-1">
                <code className="text-[10px] bg-surface-raised text-slate-400 px-1.5 py-0.5 rounded font-mono">{lbl}</code>
              </td>
              <td className={`py-1 text-center font-bold tabular-nums ${color}`}>{(m.f1 * 100).toFixed(1)}%</td>
              <td className="py-1 text-center text-emerald-400 tabular-nums">{m.tp}</td>
              <td className="py-1 text-center text-red-500 tabular-nums">{m.fp}</td>
              <td className="py-1 text-center text-amber-400 tabular-nums">{m.fn}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function EvalByOutcomeDetail({ data }: { data: NonNullable<EvalResult['by_outcome']> }) {
  return (
    <div className="space-y-1.5">
      {data.prohibited && (
        <div className="flex justify-between items-baseline text-xs">
          <span className="text-slate-400 font-medium">Prohibited</span>
          <span className="text-slate-500 tabular-nums">
            Recall {(data.prohibited.recall * 100).toFixed(0)}%
            {' · '}F1 {(data.prohibited.f1 * 100).toFixed(0)}%
            {' · '}{data.prohibited.n_examples} examples
          </span>
        </div>
      )}
      {data.high_risk && (
        <div className="flex justify-between items-baseline text-xs">
          <span className="text-slate-400 font-medium">High-risk</span>
          <span className="text-slate-500 tabular-nums">
            Recall {(data.high_risk.recall * 100).toFixed(0)}%
            {' · '}F1 {(data.high_risk.f1 * 100).toFixed(0)}%
            {' · '}{data.high_risk.n_examples} examples
          </span>
        </div>
      )}
      {data.minimal && (
        <div className="flex justify-between items-baseline text-xs">
          <span className="text-slate-400 font-medium">Minimal</span>
          <span className="text-slate-500 tabular-nums">
            {data.minimal.n_correct}/{data.minimal.n_examples} correct
          </span>
        </div>
      )}
    </div>
  )
}

function EvalChecksDetail({ data, articleScores }: {
  data: Record<string, boolean>
  articleScores?: Record<string, number>
}) {
  return (
    <div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1">
        {Object.entries(data).map(([check, passed]) => (
          <div key={check} className="flex items-center gap-1 text-xs">
            <span className={passed ? 'text-emerald-500' : 'text-red-500'}>{passed ? '✓' : '✗'}</span>
            <span className={`${passed ? 'text-slate-400' : 'text-red-400'} truncate`}>
              {check.replace(/_/g, ' ')}
            </span>
          </div>
        ))}
      </div>
      {articleScores && Object.keys(articleScores).length > 0 && (
        <div className="mt-2 pt-2 border-t border-line">
          <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide mb-1.5">Article Scores</p>
          <div className="flex gap-3 flex-wrap">
            {Object.entries(articleScores).map(([art, score]) => {
              const color = score >= 75 ? 'text-emerald-400' : score >= 50 ? 'text-amber-400' : 'text-red-400'
              return (
                <div key={art} className="text-center">
                  <div className={`text-sm font-bold tabular-nums ${color}`}>{score}%</div>
                  <div className="text-[10px] text-slate-400">Art. {art}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function NerMetricsSection({ data }: { data: NerMetricsData }) {
  const theme = data.overall_f1 >= 0.80
    ? { text: 'text-emerald-400', bg: 'bg-emerald-500/[0.06]', border: 'border-emerald-500/20', label: 'Above 80% target' }
    : data.overall_f1 >= 0.60
      ? { text: 'text-amber-400',  bg: 'bg-amber-500/[0.06]',  border: 'border-amber-500/20',  label: 'Below 80% target' }
      : { text: 'text-red-400',    bg: 'bg-red-500/[0.06]',    border: 'border-red-500/20',    label: 'Needs retraining' }

  return (
    <>
      <div className="border-t border-line pt-8 mb-7">
        <h1 className="text-2xl font-extrabold text-white tracking-tight mb-1">NER Model Metrics</h1>
        <p className="text-sm text-slate-500">
          spaCy entity recogniser on the held-out dev set ({data.val_size} sentences).
          Train: {data.train_size} · Final loss:{' '}
          <code className="text-xs bg-surface-raised px-1.5 py-0.5 rounded-md font-mono text-slate-300">
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
            <tr className="border-b border-line bg-surface-raised/80">
              <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Entity Type</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Precision</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">Recall</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide text-center">F1</th>
              <th className="px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide w-44">F1 Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {data.per_label.map((row, i) => {
              const t = row.f1 >= 0.80
                ? { text: 'text-emerald-400', bar: '#10b981' }
                : row.f1 >= 0.60
                  ? { text: 'text-amber-400', bar: '#f59e0b' }
                  : { text: 'text-red-400', bar: '#ef4444' }
              return (
                <tr key={i} className="hover:bg-surface-raised/70 transition-colors">
                  <td className="px-5 py-3.5">
                    <code className="text-xs bg-surface-raised text-slate-300 px-2 py-1 rounded-md font-mono font-semibold">
                      {row.label}
                    </code>
                  </td>
                  <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">{(row.precision * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3.5 text-center text-slate-400 tabular-nums">{(row.recall * 100).toFixed(1)}%</td>
                  <td className={`px-4 py-3.5 text-center font-bold tabular-nums ${t.text}`}>{(row.f1 * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3.5">
                    <div className="h-2 bg-surface-raised rounded-full overflow-hidden">
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
