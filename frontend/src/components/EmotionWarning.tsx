// Article 5 emotion recognition alert — shown on dashboard when flag is detected.

import type { EmotionFlag } from '../types'

export default function EmotionWarning({ flag }: { flag: EmotionFlag }) {
  if (!flag.detected) return null

  const isProhibited = flag.is_prohibited

  return (
    <div className={`relative rounded-2xl border-2 p-5 mb-6 overflow-hidden ${
      isProhibited ? 'bg-red-50 border-red-300' : 'bg-amber-50 border-amber-300'
    }`}>
      {/* Decorative stripe */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${
        isProhibited ? 'bg-red-500' : 'bg-amber-500'
      }`} />

      <div className="flex items-start gap-4 mt-1">
        {/* Icon */}
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
          isProhibited ? 'bg-red-100' : 'bg-amber-100'
        }`}>
          {isProhibited ? (
            <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          )}
        </div>

        <div className="flex-1">
          <h3 className={`font-bold text-base mb-1 ${isProhibited ? 'text-red-800' : 'text-amber-800'}`}>
            {isProhibited
              ? 'Article 5 Violation — Prohibited Use Case Detected'
              : 'Article 5 Warning — Emotion Recognition Detected'}
          </h3>
          <p className={`text-sm leading-relaxed ${isProhibited ? 'text-red-700' : 'text-amber-700'}`}>
            {flag.explanation}
          </p>
          {flag.context && (
            <span className={`inline-block mt-2 px-2.5 py-1 rounded-lg text-xs font-semibold ${
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
