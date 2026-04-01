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

export default function RiskWizard() {
  const [questions, setQuestions] = useState<Question[]>([])
  const [answers, setAnswers] = useState<Answers>({})
  const [result, setResult] = useState<RiskTier | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    apiClient
      .get<{ status: string; data: { questions: Question[] } }>('/api/v1/wizard/questions')
      .then(r => setQuestions(r.data.data.questions))
      .catch(() => setError('Failed to load questions.'))
  }, [])

  const allAnswered = questions.length > 0 && questions.every(q => q.id in answers)

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

  const handleReset = () => {
    setAnswers({})
    setResult(null)
  }

  const TIER_DESCRIPTIONS: Record<RiskTier, string> = {
    prohibited: 'This system falls under a prohibited use case per Article 5 of the EU AI Act and cannot be legally deployed in the EU.',
    high: 'This system is classified as High-Risk under Annex III. Full compliance with Articles 9–15 is mandatory before deployment.',
    limited: 'This system has limited obligations — primarily transparency requirements under Article 13.',
    minimal: 'This system carries minimal risk. No specific EU AI Act obligations apply beyond general good practice.',
  }

  return (
    <Layout>
      <div className="max-w-2xl mx-auto">
        <div className="mb-6">
          <p className="text-xs font-semibold text-brand-600 uppercase tracking-wide mb-1">Step 1 of 2</p>
          <h1 className="text-2xl font-bold text-slate-800">Annex III Risk Classification</h1>
          <p className="mt-1 text-slate-500 text-sm">
            Answer 9 yes/no questions to determine your AI system's risk tier under the EU AI Act.
            Your result will carry forward into the document audit in Step 2.
          </p>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error}
          </div>
        )}

        {result ? (
          /* Result screen */
          <ResultCard tier={result} description={TIER_DESCRIPTIONS[result]} onReset={handleReset} />
        ) : (
          /* Questions */
          <div className="flex flex-col gap-3">
            {questions.map((q, i) => {
              const answer = answers[q.id]
              const isLast = i === questions.length - 1

              return (
                <div
                  key={q.id}
                  className={`bg-white border rounded-xl p-4 transition-colors ${
                    answer === true
                      ? isLast ? 'border-red-300 bg-red-50' : 'border-amber-300 bg-amber-50'
                      : answer === false
                        ? 'border-green-200 bg-green-50'
                        : 'border-slate-200'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-xs font-bold text-slate-400 mt-0.5 shrink-0 w-5">
                      {i + 1}.
                    </span>
                    <p className="text-sm text-slate-700 flex-1 leading-relaxed">{q.text}</p>
                    <div className="flex gap-2 shrink-0">
                      <button
                        onClick={() => setAnswers(a => ({ ...a, [q.id]: true }))}
                        className={`px-3 py-1 rounded-md text-sm font-medium border transition-colors ${
                          answer === true
                            ? isLast
                              ? 'bg-red-500 text-white border-red-500'
                              : 'bg-amber-500 text-white border-amber-500'
                            : 'border-slate-300 text-slate-600 hover:border-amber-400 hover:text-amber-700'
                        }`}
                      >
                        Yes
                      </button>
                      <button
                        onClick={() => setAnswers(a => ({ ...a, [q.id]: false }))}
                        className={`px-3 py-1 rounded-md text-sm font-medium border transition-colors ${
                          answer === false
                            ? 'bg-green-500 text-white border-green-500'
                            : 'border-slate-300 text-slate-600 hover:border-green-400 hover:text-green-700'
                        }`}
                      >
                        No
                      </button>
                    </div>
                  </div>
                </div>
              )
            })}

            <div className="flex items-center justify-between mt-2">
              <p className="text-xs text-slate-400">
                {Object.keys(answers).length} / {questions.length} answered
              </p>
              <button
                onClick={handleSubmit}
                disabled={!allAnswered || loading}
                className="px-6 py-2.5 rounded-lg bg-brand-600 text-white font-semibold text-sm
                  hover:bg-brand-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Classifying…' : 'Get Risk Tier →'}
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  )
}

function ResultCard({
  tier,
  description,
  onReset,
}: {
  tier: RiskTier
  description: string
  onReset: () => void
}) {
  const { label, className } = riskTierLabel(tier)

  const iconMap: Record<RiskTier, string> = {
    prohibited: '⛔',
    high: '⚠',
    limited: 'ℹ',
    minimal: '✓',
  }

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-8 text-center">
      <div className="text-5xl mb-4">{iconMap[tier]}</div>
      <span className={`inline-block px-4 py-1 rounded-full text-sm font-bold ${className}`}>
        {label}
      </span>
      <p className="mt-4 text-slate-600 text-sm leading-relaxed max-w-md mx-auto">
        {description}
      </p>

      <div className="mt-6 flex gap-3 justify-center">
        <button
          onClick={onReset}
          className="px-4 py-2 rounded-lg border border-slate-300 text-slate-600 text-sm
            font-medium hover:border-slate-400 transition-colors"
        >
          Start Over
        </button>
        <Link
          to={`/upload?tier=${tier}`}
          className="px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium
            hover:bg-brand-700 transition-colors"
        >
          Continue to Step 2: Upload Docs →
        </Link>
      </div>
    </div>
  )
}
