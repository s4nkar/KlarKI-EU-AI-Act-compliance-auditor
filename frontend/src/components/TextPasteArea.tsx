// Premium textarea for pasting raw policy text.

interface TextPasteAreaProps {
  value: string
  onChange: (value: string) => void
}

export default function TextPasteArea({ value, onChange }: TextPasteAreaProps) {
  const wordCount = value.trim() ? value.trim().split(/\s+/).length : 0

  return (
    <div>
      <textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="Paste your AI policy, risk management documentation, or any compliance text here…"
        rows={11}
        className="w-full rounded-xl border border-slate-200 px-4 py-3.5 text-sm text-slate-700
          placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400
          focus:border-transparent resize-none leading-relaxed bg-slate-50
          hover:border-slate-300 transition-colors"
      />
      <div className="flex items-center justify-between mt-1.5 px-0.5">
        <span className="text-xs text-slate-400">
          {wordCount > 0 ? `~${wordCount.toLocaleString()} words` : 'Start typing or paste text'}
        </span>
        <span className="text-xs text-slate-400 tabular-nums">
          {value.length.toLocaleString()} chars
        </span>
      </div>
    </div>
  )
}
