// Circular compliance score gauge 0–100 with gradient stroke.

interface ScoreRadialProps {
  score: number
  size?: number
  label?: string
}

const RADIUS = 36
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

function scoreTheme(score: number) {
  if (score >= 70) return { stroke: '#10b981', text: '#059669', bg: '#d1fae5', label: 'Good' }
  if (score >= 40) return { stroke: '#f59e0b', text: '#d97706', bg: '#fef3c7', label: 'Needs Work' }
  return { stroke: '#ef4444', text: '#dc2626', bg: '#fee2e2', label: 'At Risk' }
}

export default function ScoreRadial({ score, size = 120, label }: ScoreRadialProps) {
  const clamped = Math.max(0, Math.min(100, score))
  const offset = CIRCUMFERENCE * (1 - clamped / 100)
  const theme = scoreTheme(clamped)

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        aria-label={`Compliance score: ${Math.round(clamped)}`}
      >
        {/* Track ring */}
        <circle
          cx="50" cy="50" r={RADIUS}
          fill="none"
          stroke="#f1f5f9"
          strokeWidth="10"
        />
        {/* Progress arc */}
        <circle
          cx="50" cy="50" r={RADIUS}
          fill="none"
          stroke={theme.stroke}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          transform="rotate(-90 50 50)"
          style={{ transition: 'stroke-dashoffset 0.7s cubic-bezier(0.4, 0, 0.2, 1)' }}
        />
        {/* Score number */}
        <text
          x="50" y="44"
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="22"
          fontWeight="800"
          fill={theme.text}
          fontFamily="Inter, system-ui, sans-serif"
        >
          {Math.round(clamped)}
        </text>
        {/* /100 label */}
        <text
          x="50" y="60"
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="9"
          fill="#94a3b8"
          fontFamily="Inter, system-ui, sans-serif"
        >
          / 100
        </text>
      </svg>
      {label && (
        <span className="text-xs font-semibold tracking-wide" style={{ color: theme.text }}>
          {label}
        </span>
      )}
    </div>
  )
}
