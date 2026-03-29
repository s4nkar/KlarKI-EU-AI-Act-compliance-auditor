// Textarea for pasting raw policy text.

interface TextPasteAreaProps {
  value: string
  onChange: (value: string) => void
}

export default function TextPasteArea({ value, onChange }: TextPasteAreaProps) {
  return (
    <div>
      <textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="Paste your AI policy, risk management documentation, or any compliance text here…"
        rows={10}
        className="w-full rounded-xl border border-slate-300 px-4 py-3 text-sm text-slate-700
          placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-400
          focus:border-transparent resize-none leading-relaxed"
      />
      <p className="mt-1 text-xs text-slate-400 text-right">
        {value.length.toLocaleString()} characters
      </p>
    </div>
  )
}
