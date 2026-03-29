// Circular progress gauge 0–100 with score-based colour coding.

interface ScoreRadialProps {
  score: number
  size?: number
  label?: string
}

const RADIUS = 38
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

function strokeColor(score: number) {
  if (score >= 70) return '#16a34a' // green-600
  if (score >= 40) return '#d97706' // amber-600
  return '#dc2626' // red-600
}

function textColorClass(score: number) {
  if (score >= 70) return 'text-green-600'
  if (score >= 40) return 'text-amber-600'
  return 'text-red-600'
}

export default function ScoreRadial({ score, size = 120, label }: ScoreRadialProps) {
  const clamped = Math.max(0, Math.min(100, score))
  const offset = CIRCUMFERENCE * (1 - clamped / 100)
  const color = strokeColor(clamped)

  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        aria-label={`Compliance score: ${Math.round(clamped)}`}
      >
        {/* Track */}
        <circle
          cx="50" cy="50" r={RADIUS}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="10"
        />
        {/* Progress */}
        <circle
          cx="50" cy="50" r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          transform="rotate(-90 50 50)"
          style={{ transition: 'stroke-dashoffset 0.6s ease' }}
        />
        {/* Score text */}
        <text
          x="50" y="46"
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="20"
          fontWeight="700"
          fill={color}
        >
          {Math.round(clamped)}
        </text>
        <text
          x="50" y="62"
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="9"
          fill="#94a3b8"
        >
          / 100
        </text>
      </svg>
      {label && (
        <span className={`text-sm font-semibold ${textColorClass(clamped)}`}>
          {label}
        </span>
      )}
    </div>
  )
}
