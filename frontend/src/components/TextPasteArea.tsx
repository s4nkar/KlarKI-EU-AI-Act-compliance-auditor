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
        className="w-full rounded-xl border border-line px-4 py-3.5 text-sm text-slate-200
          placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500/40
          focus:border-transparent resize-none leading-relaxed bg-surface
          hover:border-line-strong transition-colors"
      />
      <div className="flex items-center justify-between mt-1.5 px-0.5">
        <span className="text-xs text-slate-500">
          {wordCount > 0 ? `~${wordCount.toLocaleString()} words` : 'Start typing or paste text'}
        </span>
        <span className="text-xs text-slate-500 tabular-nums">
          {value.length.toLocaleString()} chars
        </span>
      </div>
    </div>
  )
}
