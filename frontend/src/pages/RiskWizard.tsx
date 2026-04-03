// Annex III guided risk classification wizard — 9 yes/no questions.

import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Layout from '../components/Layout'
import apiClient from '../api/client'
import { riskTierLabel } from '../utils/formatters'
import type { RiskTier } from '../types'

interface Question {
  id: string
  text: string
}

type Answers = Record<string, boolean>

const TIER_DESCRIPTIONS: Record<RiskTier, string> = {
  prohibited: 'This system falls under a prohibited use case per Article 5 of the EU AI Act and cannot be legally deployed in the EU.',
  high:       'This system is classified as High-Risk under Annex III. Full compliance with Articles 9–15 is mandatory before deployment.',
  limited:    'This system has limited obligations — primarily transparency requirements under Article 13.',
  minimal:    'This system carries minimal risk. No specific EU AI Act obligations apply beyond general good practice.',
}

const TIER_ICONS: Record<RiskTier, React.ReactNode> = {
  prohibited: (
    <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
    </svg>
  ),
  high: (
    <svg className="w-8 h-8 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  limited: (
    <svg className="w-8 h-8 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  minimal: (
    <svg className="w-8 h-8 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
}

export default function RiskWizard() {
  const [questions, setQuestions] = useState<Question[]>([])
  const [answers, setAnswers]     = useState<Answers>({})
  const [result, setResult]       = useState<RiskTier | null>(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState<string | null>(null)

  useEffect(() => {
    apiClient
      .get<{ status: string; data: { questions: Question[] } }>('/api/v1/wizard/questions')
      .then(r => setQuestions(r.data.data.questions))
      .catch(() => setError('Failed to load questions. Is the API running?'))
  }, [])

  const answered  = Object.keys(answers).length
  const total     = questions.length
  const allAnswered = total > 0 && answered === total
  const progress  = total > 0 ? (answered / total) * 100 : 0

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    try {
      const resp = await apiClient.post<{ status: string; data: { risk_tier: RiskTier } }>(
        '/api/v1/wizard/classify',
        { answers },
      )
      setResult(resp.data.data.risk_tier)
    } catch {
      setError('Classification failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Layout>
      <div className="max-w-2xl mx-auto">
        {/* Page header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-3">
            <span className="badge bg-brand-100 text-brand-700 border border-brand-200">Step 1 of 2</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            Annex III Risk Classification
          </h1>
          <p className="mt-2 text-slate-500 leading-relaxed">
            Answer these questions to determine your AI system's risk tier under the EU AI Act.
            Your result will carry forward into the document audit.
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-5 flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
            <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            {error}
          </div>
        )}

        {result ? (
          <ResultCard tier={result} onReset={() => { setAnswers({}); setResult(null) }} />
        ) : (
          <>
            {/* Progress bar */}
            {total > 0 && (
              <div className="mb-6">
                <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
                  <span className="font-medium">{answered} of {total} answered</span>
                  <span>{Math.round(progress)}% complete</span>
                </div>
                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-600 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Questions */}
            <div className="flex flex-col gap-3">
              {questions.map((q, i) => {
                const answer = answers[q.id]
                const isLast = i === questions.length - 1

                const cardColor =
                  answer === true
                    ? isLast
                      ? 'border-red-200 bg-red-50'
                      : 'border-amber-200 bg-amber-50'
                    : answer === false
                      ? 'border-emerald-200 bg-emerald-50'
                      : 'border-slate-200 bg-white hover:border-slate-300'

                return (
                  <div key={q.id} className={`rounded-xl border p-4 transition-all duration-150 ${cardColor}`}>
                    <div className="flex items-start gap-4">
                      <span className={`flex items-center justify-center w-7 h-7 rounded-lg shrink-0 text-xs font-bold mt-0.5 ${
                        answer === undefined
                          ? 'bg-slate-100 text-slate-500'
                          : answer === true && isLast
                            ? 'bg-red-100 text-red-700'
                            : answer === true
                              ? 'bg-amber-100 text-amber-700'
                              : 'bg-emerald-100 text-emerald-700'
                      }`}>
                        {i + 1}
                      </span>
                      <p className="text-sm text-slate-700 flex-1 leading-relaxed font-medium pt-0.5">
                        {q.text}
                      </p>
                      <div className="flex gap-2 shrink-0">
                        <button
                          onClick={() => setAnswers(a => ({ ...a, [q.id]: true }))}
                          className={`px-3.5 py-1.5 rounded-lg text-sm font-semibold border transition-all duration-150 ${
                            answer === true
                              ? isLast
                                ? 'bg-red-500 text-white border-red-500 shadow-sm'
                                : 'bg-amber-500 text-white border-amber-500 shadow-sm'
                              : 'border-slate-300 bg-white text-slate-600 hover:border-amber-400 hover:text-amber-700 hover:bg-amber-50'
                          }`}
                        >
                          Yes
                        </button>
                        <button
                          onClick={() => setAnswers(a => ({ ...a, [q.id]: false }))}
                          className={`px-3.5 py-1.5 rounded-lg text-sm font-semibold border transition-all duration-150 ${
                            answer === false
                              ? 'bg-emerald-500 text-white border-emerald-500 shadow-sm'
                              : 'border-slate-300 bg-white text-slate-600 hover:border-emerald-400 hover:text-emerald-700 hover:bg-emerald-50'
                          }`}
                        >
                          No
                        </button>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Submit */}
            {total > 0 && (
              <div className="flex items-center justify-end mt-6">
                <button
                  onClick={handleSubmit}
                  disabled={!allAnswered || loading}
                  className="btn-primary"
                >
                  {loading ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                      </svg>
                      Classifying…
                    </>
                  ) : (
                    <>
                      Get Risk Tier
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    </>
                  )}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </Layout>
  )
}

function ResultCard({ tier, onReset }: { tier: RiskTier; onReset: () => void }) {
  const { label, className } = riskTierLabel(tier)
  const description = TIER_DESCRIPTIONS[tier]

  const wrapperColor: Record<RiskTier, string> = {
    prohibited: 'from-red-50 border-red-200',
    high:       'from-amber-50 border-amber-200',
    limited:    'from-blue-50 border-blue-200',
    minimal:    'from-emerald-50 border-emerald-200',
  }

  return (
    <div className={`card bg-gradient-to-b ${wrapperColor[tier]} to-white p-8`}>
      <div className="flex flex-col items-center text-center">
        <div className="mb-4">{TIER_ICONS[tier]}</div>
        <span className={`badge text-sm px-4 py-1.5 ${className}`}>{label}</span>
        <p className="mt-4 text-slate-600 text-sm leading-relaxed max-w-md">
          {description}
        </p>
      </div>

      <div className="mt-8 flex gap-3 justify-center flex-wrap">
        <button onClick={onReset} className="btn-secondary">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Start Over
        </button>
        <Link to={`/upload?tier=${tier}`} className="btn-primary">
          Continue to Upload Docs
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </Link>
      </div>
    </div>
  )
}
