// Landing page — product-led modern layout: real UI motifs (score bar, article
// card, evidence highlight) instead of generic icon-in-square feature grids.

import { Link } from 'react-router-dom'
import Layout from '../components/Layout'

const STEPS = [
  {
    n: '1',
    title: 'Classify risk tier',
    description: 'Nine Annex III questions determine prohibited, high-risk, limited, or minimal risk.',
  },
  {
    n: '2',
    title: 'Upload documentation',
    description: 'PDF, DOCX, TXT, or Markdown — parsed and scored individually, multiple files at once.',
  },
  {
    n: '3',
    title: 'Run gap analysis',
    description: 'A legal → technical → synthesis pipeline checks each applicable article for gaps.',
  },
  {
    n: '4',
    title: 'Score and report',
    description: 'Article-level compliance score with confidence, exported as PDF or JSON.',
  },
]

export default function Home() {
  return (
    <Layout>
      {/* Hero */}
      <section className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-12 items-center pt-6 pb-24 sm:pt-12 sm:pb-32">
        <div className="max-w-xl">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-400 mb-6">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400" />
            </span>
            Local-first • zero data leaves your machine
          </div>

          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight text-white leading-[1.05]">
            Compliance,{' '}
            <span className="text-brand-400">audited automatically.</span>
          </h1>

          <p className="mt-6 text-lg text-slate-400 leading-relaxed">
            Klarki analyzes your AI system's documentation
            article by article against the EU AI Act and GDPR - mapping every finding
            with supporting evidence and recommendations, entirely on your own infrastructure.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-x-7 gap-y-3">
            <Link to="/wizard" className="btn-primary">
              Start Risk Assessment
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link to="/upload" className="text-sm font-semibold text-slate-300 hover:text-white transition-colors">
              Upload documentation →
            </Link>
          </div>
        </div>

        {/* Product mockup — Article card + score ring, floating */}
        <div className="hidden lg:block relative w-[340px] h-[320px] shrink-0">
          <div className="absolute -inset-10 bg-brand-500/10 blur-3xl rounded-full" />

          <div className="absolute top-0 right-4 w-44 card p-4 shadow-panel -rotate-2">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-3">Compliance score</p>
            <div className="flex items-center gap-3">
              <svg width="52" height="52" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" fill="none" stroke="rgb(255 255 255 / 0.08)" strokeWidth="12" />
                <circle
                  cx="50" cy="50" r="40" fill="none" stroke="#34d399" strokeWidth="12"
                  strokeLinecap="round" strokeDasharray={2 * Math.PI * 40}
                  strokeDashoffset={2 * Math.PI * 40 * (1 - 0.82)}
                  transform="rotate(-90 50 50)"
                />
              </svg>
              <div>
                <p className="text-2xl font-bold text-white tabular-nums leading-none">82</p>
                <p className="text-xs text-emerald-400 font-medium mt-1">Good</p>
              </div>
            </div>
            <span className="absolute -bottom-3 -right-3 badge bg-red-500/10 text-red-400 shadow-panel rotate-2">
              High-risk tier
            </span>
          </div>

          <div className="absolute bottom-0 left-0 w-64 card p-4 shadow-panel rotate-1">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] font-semibold text-brand-400 uppercase tracking-wide">Art. 13 · Transparency</span>
              <span className="text-lg font-bold text-amber-400 tabular-nums">78</span>
            </div>
            <div className="w-full h-1.5 bg-surface-hover rounded-full overflow-hidden mb-3">
              <div className="h-full rounded-full bg-amber-500" style={{ width: '78%' }} />
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 inline-block" />
              <span className="text-amber-400 font-medium">1 major gap</span>
              <span className="ml-auto text-slate-500">14 chunks</span>
            </div>
          </div>
        </div>
      </section>

      {/* How it works — stepper, not icon cards */}
      <section className="py-16 border-t border-line">
        <span className="section-label">How it works</span>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-x-6 gap-y-10 mt-2">
          {STEPS.map((step, i) => (
            <div key={step.n} className="relative pl-0">
              <div className="flex items-center gap-3 sm:block">
                <span className="flex items-center justify-center w-8 h-8 rounded-full bg-surface-raised border border-line text-sm font-bold text-white shrink-0">
                  {step.n}
                </span>
                {i < STEPS.length - 1 && (
                  <span className="hidden sm:block h-px bg-line w-full mt-4" />
                )}
                <h3 className="text-sm font-semibold text-white sm:mt-4">{step.title}</h3>
              </div>
              <p className="mt-2 text-sm text-slate-400 leading-relaxed sm:pr-2">{step.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Bento features */}
      <section className="py-16 border-t border-line">
        <span className="section-label">Why Klarki</span>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-2">
          {/* Featured: evidence mapping mockup */}
          <div className="card card-hover p-6 lg:col-span-2 lg:row-span-2 flex flex-col justify-between">
            <div>
              <h3 className="text-base font-semibold text-white mb-2">Evidence you can verify</h3>
              <p className="text-sm text-slate-400 leading-relaxed max-w-md">
                Regex + NLI evidence mapping links every finding back to the exact
                passage in your source documents — no LLM guesswork.
              </p>
            </div>
            <div className="mt-6 rounded-lg border border-line bg-surface-raised p-4">
              <p className="text-xs text-slate-500 leading-relaxed font-mono">
                "...the system shall maintain{' '}
                <span className="bg-brand-500/20 text-brand-300 rounded px-1">automatically generated logs</span>{' '}
                covering the period of the system's operation..."
              </p>
              <div className="mt-3 pt-3 border-t border-line flex items-center gap-2 text-xs">
                <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-slate-400">Matched to <span className="text-slate-200 font-medium">Article 12 — Record-keeping</span></span>
              </div>
            </div>
          </div>

          <div className="card card-hover p-6">
            <h3 className="text-base font-semibold text-white mb-2">Runs entirely on your machine</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Ollama, ChromaDB, and every model run locally. No document text or
              metadata is ever sent to an external API.
            </p>
          </div>

          <div className="card card-hover p-6">
            <h3 className="text-base font-semibold text-white mb-2">Deterministic classification</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Every Ollama call runs with temperature 0, a fixed seed, and top-k 1 —
              reproducible results, run to run.
            </p>
          </div>

          <div className="card card-hover p-6 lg:col-span-3 flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-8">
            <div className="flex-1">
              <h3 className="text-base font-semibold text-white mb-2">Built for multi-document, multi-article audits</h3>
              <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
                Upload several files at once — each is parsed and scored individually,
                Articles 9–15 are checked with non-applicable ones excluded, and GDPR
                obligations are considered alongside the AI Act.
              </p>
            </div>
            <Link to="/upload" className="btn-secondary shrink-0">
              Upload Documents
            </Link>
          </div>
        </div>
      </section>

      {/* Closing CTA */}
      <section className="py-20 border-t border-line text-center">
        <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
          Ready to check your AI system?
        </h2>
        <p className="mt-3 text-slate-400 max-w-md mx-auto leading-relaxed">
          Start with the risk assessment, then upload your documentation for a
          full article-level audit.
        </p>
        <div className="mt-7">
          <Link to="/wizard" className="btn-primary">
            Start Risk Assessment
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>
      </section>
    </Layout>
  )
}
