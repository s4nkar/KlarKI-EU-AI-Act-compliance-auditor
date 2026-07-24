// Circular compliance score gauge 0–100 with gradient stroke.

interface ScoreRadialProps {
  score: number
  size?: number
  label?: string
}

const RADIUS = 36
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

function scoreTheme(score: number) {
  if (score >= 70) return { stroke: '#34d399', text: '#34d399', bg: 'rgb(52 211 153 / 0.1)', label: 'Good' }
  if (score >= 40) return { stroke: '#fbbf24', text: '#fbbf24', bg: 'rgb(251 191 36 / 0.1)', label: 'Needs Work' }
  return { stroke: '#f87171', text: '#f87171', bg: 'rgb(248 113 113 / 0.1)', label: 'At Risk' }
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
          stroke="rgb(255 255 255 / 0.08)"
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
