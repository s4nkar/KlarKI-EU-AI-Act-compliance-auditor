// Article 5 emotion recognition alert — shown on dashboard when flag is detected.

import type { EmotionFlag } from '../types'

interface EmotionWarningProps {
  flag: EmotionFlag
}

export default function EmotionWarning({ flag }: EmotionWarningProps) {
  if (!flag.detected) return null

  const isProhibited = flag.is_prohibited

  return (
    <div className={`rounded-xl border-2 p-5 mb-6 ${
      isProhibited
        ? 'bg-red-50 border-red-400'
        : 'bg-amber-50 border-amber-400'
    }`}>
      <div className="flex items-start gap-3">
        <div className={`text-2xl shrink-0 ${isProhibited ? 'text-red-500' : 'text-amber-500'}`}>
          {isProhibited ? '⛔' : '⚠'}
        </div>
        <div>
          <h3 className={`font-bold text-base ${isProhibited ? 'text-red-800' : 'text-amber-800'}`}>
            {isProhibited
              ? 'Article 5 Violation — Prohibited Use Case Detected'
              : 'Article 5 Warning — Emotion Recognition Detected'}
          </h3>
          <p className={`mt-1 text-sm leading-relaxed ${isProhibited ? 'text-red-700' : 'text-amber-700'}`}>
            {flag.explanation}
          </p>
          {flag.context && (
            <span className={`inline-block mt-2 px-2 py-0.5 rounded text-xs font-semibold ${
              isProhibited ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'
            }`}>
              Context: {flag.context}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
