// Observability monitoring dashboard — auto-refreshes every 10 s.
// Sections: service health, pipeline stats + charts, LangGraph node timing,
// ChromaDB sizes, model/data version registry, system resources.

import { useEffect, useState, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import Layout from '../components/Layout'
import apiClient from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface ServiceHealth { chromadb: boolean; ollama: boolean; api: boolean }

interface PipelineStats {
  total: number; successful: number; failed: number
  active: number; success_rate: number; avg_duration_s: number
}

interface StageStat { count: number; avg_s: number; p95_s: number; min_s: number; max_s: number }

interface GraphNodeStat {
  invocations: number; avg_duration_s: number; total_duration_s: number; errors: number
}

interface ModelVersion {
  model_type: string; version: string | null; is_active: boolean
  created_at: string | null; metric_key: string; score: number | null
  data_version: string | null; on_disk: boolean
}

interface DataVersion {
  data_type: string; active_version: string | null; total_versions: number
  current_records: number | null; file_exists: boolean
}

interface SystemResources {
  available?: boolean; cpu_percent?: number
  memory_used_mb?: number; memory_total_mb?: number; memory_percent?: number
}

interface MonitoringData {
  generated_at: string; uptime_s: number
  services: ServiceHealth
  pipeline: PipelineStats
  stages: Record<string, StageStat>
  graph_nodes: Record<string, GraphNodeStat>
  chromadb: { collections: Record<string, number | null> }
  models: ModelVersion[]
  data: DataVersion[]
  system: SystemResources
}

// ── Palette ───────────────────────────────────────────────────────────────────

const COLORS = {
  success: '#34d399', failed: '#f87171', active: '#60a5fa',
  nodes: ['#60a5fa', '#a78bfa', '#fbbf24'],
  stages: '#7c6ef5',
}

// ── Small helpers ─────────────────────────────────────────────────────────────

function formatUptime(s: number) {
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  return `${(s / 3600).toFixed(1)}h`
}

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={`inline-block w-2.5 h-2.5 rounded-full mr-2 ${ok ? 'bg-emerald-500' : 'bg-red-500'}`} />
}

function StatCard({ label, value, sub, accent }: {
  label: string; value: string | number; sub?: string; accent?: string
}) {
  return (
    <div className={`bg-surface border rounded-lg p-4 ${accent ?? 'border-line'}`}>
      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">{label}</div>
      <div className="text-2xl font-bold text-white">{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-3 mt-8 border-l-4 border-brand-500/40 pl-3">
      {children}
    </h2>
  )
}

// Shared dark-mode styling for Recharts primitives (grid, axes, tooltip).
const CHART_GRID_COLOR = 'rgba(255,255,255,0.08)'
const CHART_TICK = { fontSize: 11, fill: '#8b8b9a' }
const CHART_TOOLTIP_STYLE = {
  background: '#17171f',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 8,
  color: '#e2e2ea',
  fontSize: 12,
}
const CHART_LEGEND_STYLE = { fontSize: 12, color: '#8b8b9a' }

// ── Charts ────────────────────────────────────────────────────────────────────

function PipelineDonut({ pipeline }: { pipeline: PipelineStats }) {
  if (pipeline.total === 0) {
    return (
      <div className="flex items-center justify-center h-44 text-slate-500 text-sm">
        No audits yet
      </div>
    )
  }
  const pieData = [
    { name: 'Successful', value: pipeline.successful },
    { name: 'Failed', value: pipeline.failed },
    ...(pipeline.active ? [{ name: 'Active', value: pipeline.active }] : []),
  ].filter(d => d.value > 0)

  const donutColors = [COLORS.success, COLORS.failed, COLORS.active]

  return (
    <ResponsiveContainer width="100%" height={180}>
      <PieChart>
        <Pie
          data={pieData}
          cx="50%"
          cy="50%"
          innerRadius={50}
          outerRadius={75}
          paddingAngle={3}
          dataKey="value"
          stroke="#121218"
        >
          {pieData.map((_, idx) => (
            <Cell key={idx} fill={donutColors[idx % donutColors.length]} />
          ))}
        </Pie>
        <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
        <Legend iconType="circle" iconSize={8} wrapperStyle={CHART_LEGEND_STYLE} />
      </PieChart>
    </ResponsiveContainer>
  )
}

function StageTimingChart({ stages }: { stages: Record<string, StageStat> }) {
  const data = Object.entries(stages).map(([name, s]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    avg: s.avg_s,
    p95: s.p95_s,
  }))

  if (data.length === 0) {
    return <div className="flex items-center justify-center h-44 text-slate-500 text-sm">No stage data yet</div>
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ left: 16, right: 24, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={CHART_GRID_COLOR} />
        <XAxis type="number" unit="s" tick={CHART_TICK} />
        <YAxis type="category" dataKey="name" tick={CHART_TICK} width={80} />
        <Tooltip formatter={(v) => [`${v}s`]} contentStyle={CHART_TOOLTIP_STYLE} />
        <Legend iconType="square" iconSize={10} wrapperStyle={CHART_LEGEND_STYLE} />
        <Bar dataKey="avg" name="Avg" fill={COLORS.stages} radius={[0, 4, 4, 0]} />
        <Bar dataKey="p95" name="p95" fill="#a5b4fc" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

function GraphNodeChart({ nodes }: { nodes: Record<string, GraphNodeStat> }) {
  const data = Object.entries(nodes).map(([name, n], idx) => ({
    name: name.replace('_agent', ''),
    avg_s: n.avg_duration_s,
    invocations: n.invocations,
    errors: n.errors,
    color: COLORS.nodes[idx % COLORS.nodes.length],
  }))

  if (data.length === 0) {
    return <div className="flex items-center justify-center h-44 text-slate-500 text-sm">No LangGraph data yet</div>
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={CHART_GRID_COLOR} />
        <XAxis dataKey="name" tick={{ ...CHART_TICK, fontSize: 12 }} />
        <YAxis unit="s" tick={CHART_TICK} />
        <Tooltip formatter={(v) => [`${v}s`]} contentStyle={CHART_TOOLTIP_STYLE} />
        <Bar dataKey="avg_s" name="Avg latency" radius={[4, 4, 0, 0]}>
          {data.map((d, idx) => <Cell key={idx} fill={d.color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function ChromaBarChart({ collections }: { collections: Record<string, number | null> }) {
  const data = Object.entries(collections).map(([col, count]) => ({
    name: col.replace(/_/g, ' '),
    docs: count ?? 0,
  }))

  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={CHART_GRID_COLOR} />
        <XAxis dataKey="name" tick={CHART_TICK} />
        <YAxis tick={CHART_TICK} />
        <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
        <Bar dataKey="docs" name="Documents" fill="#818cf8" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

function MemoryBar({ used, total }: { used: number; total: number }) {
  const pct = total > 0 ? Math.round((used / total) * 100) : 0
  const color = pct > 85 ? 'bg-red-500' : pct > 65 ? 'bg-amber-500' : 'bg-emerald-500'
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-500 mb-1">
        <span>{used.toLocaleString()} MB used</span>
        <span>{pct}%</span>
      </div>
      <div className="w-full h-3 bg-surface-hover rounded-full overflow-hidden">
        <div className={`h-3 rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="text-xs text-slate-500 mt-1">{total.toLocaleString()} MB total</div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Monitoring() {
  const [data, setData] = useState<MonitoringData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const fetchData = useCallback(async () => {
    setRefreshing(true)
    try {
      const res = await apiClient.get<{ status: string; data: MonitoringData }>('/api/v1/monitoring')
      setData(res.data.data)
      setLastRefresh(new Date())
      setError(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load monitoring data')
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const timer = setInterval(fetchData, 10_000)
    return () => clearInterval(timer)
  }, [fetchData])

  return (
    <Layout>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Monitoring</h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Live observability · auto-refreshes every 10s
            {data && <> · uptime <strong>{formatUptime(data.uptime_s)}</strong></>}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastRefresh && (
            <span className="text-xs text-slate-500">Updated {lastRefresh.toLocaleTimeString()}</span>
          )}
          <button
            onClick={fetchData}
            disabled={refreshing}
            className="text-sm px-4 py-2 rounded-lg bg-brand-500/10 text-brand-300 hover:bg-brand-500/15 font-medium disabled:opacity-50"
          >
            {refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/[0.06] border border-red-500/20 text-red-300 rounded-lg px-4 py-3 text-sm mb-6">
          {error}
        </div>
      )}

      {!data && !error && (
        <div className="text-slate-500 text-sm">Loading…</div>
      )}

      {data && (
        <>
          {/* ── Service Health ─────────────────────────────────────── */}
          <SectionTitle>Service Health</SectionTitle>
          <div className="grid grid-cols-3 gap-3 mb-2">
            {(Object.entries(data.services) as [string, boolean][]).map(([name, ok]) => (
              <div key={name} className={`border rounded-lg px-4 py-3 flex items-center ${
                ok ? 'bg-emerald-500/[0.06] border-emerald-500/20' : 'bg-red-500/[0.06] border-red-500/20'
              }`}>
                <StatusDot ok={ok} />
                <span className="font-medium text-sm capitalize text-slate-200">{name}</span>
                <span className={`ml-auto text-xs font-bold ${ok ? 'text-emerald-400' : 'text-red-400'}`}>
                  {ok ? 'UP' : 'DOWN'}
                </span>
              </div>
            ))}
          </div>

          {/* ── Pipeline Stats + Donut ─────────────────────────────── */}
          <SectionTitle>Audit Pipeline</SectionTitle>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* KPI cards */}
            <div className="lg:col-span-2 grid grid-cols-2 sm:grid-cols-3 gap-3">
              <StatCard label="Total Audits" value={data.pipeline.total} />
              <StatCard
                label="Successful"
                value={data.pipeline.successful}
                accent="border-emerald-500/20"
              />
              <StatCard
                label="Failed"
                value={data.pipeline.failed}
                accent={data.pipeline.failed > 0 ? 'border-red-500/20' : 'border-line'}
              />
              <StatCard label="Active Now" value={data.pipeline.active} sub="in progress" />
              <StatCard
                label="Success Rate"
                value={`${data.pipeline.success_rate}%`}
                accent={data.pipeline.success_rate >= 90 ? 'border-emerald-500/20' : 'border-line'}
              />
              <StatCard
                label="Avg Duration"
                value={data.pipeline.avg_duration_s > 0 ? `${data.pipeline.avg_duration_s}s` : '—'}
                sub="per audit"
              />
            </div>
            {/* Donut */}
            <div className="bg-surface border border-line rounded-lg p-4">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                Audit Outcomes
              </div>
              <PipelineDonut pipeline={data.pipeline} />
            </div>
          </div>

          {/* ── Stage Timing ──────────────────────────────────────── */}
          <SectionTitle>Pipeline Stage Timing</SectionTitle>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-surface border border-line rounded-lg p-4">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
                Avg vs p95 (seconds)
              </div>
              <StageTimingChart stages={data.stages} />
            </div>
            <div className="bg-surface border border-line rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-surface-raised border-b border-line">
                  <tr>
                    {['Stage', 'Runs', 'Avg', 'p95', 'Max'].map(h => (
                      <th key={h} className="text-left px-4 py-2 text-xs font-semibold text-slate-500 uppercase">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(data.stages).length === 0 ? (
                    <tr><td colSpan={5} className="px-4 py-6 text-center text-slate-500 text-sm">No data yet</td></tr>
                  ) : (
                    Object.entries(data.stages).map(([stage, s], i) => (
                      <tr key={stage} className={i % 2 === 0 ? 'bg-surface' : 'bg-surface-raised'}>
                        <td className="px-4 py-2 font-medium text-slate-200 capitalize">{stage}</td>
                        <td className="px-4 py-2 text-slate-400">{s.count}</td>
                        <td className="px-4 py-2 text-slate-400">{s.avg_s}s</td>
                        <td className="px-4 py-2 text-slate-400">{s.p95_s}s</td>
                        <td className="px-4 py-2 text-slate-400">{s.max_s}s</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* ── LangGraph Nodes ───────────────────────────────────── */}
          <SectionTitle>LangGraph Agent Nodes</SectionTitle>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-surface border border-line rounded-lg p-4">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
                Avg Latency per Node (seconds)
              </div>
              <GraphNodeChart nodes={data.graph_nodes} />
            </div>
            <div className="grid grid-cols-1 gap-3">
              {Object.keys(data.graph_nodes).length === 0 ? (
                <div className="bg-surface border border-line rounded-lg p-6 text-center text-slate-500 text-sm">
                  No LangGraph data yet — run an audit first
                </div>
              ) : (
                Object.entries(data.graph_nodes).map(([node, ns], idx) => (
                  <div key={node} className="bg-surface border border-line rounded-lg p-4 flex items-center gap-4">
                    <div
                      className="w-3 h-10 rounded-full flex-shrink-0"
                      style={{ background: COLORS.nodes[idx % COLORS.nodes.length] }}
                    />
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-slate-200 capitalize">
                        {node.replace('_agent', '')} Agent
                      </div>
                      <div className="text-xs text-slate-500 mt-0.5">
                        {ns.invocations} calls · {ns.avg_duration_s}s avg · {ns.total_duration_s}s total
                      </div>
                    </div>
                    <div className={`text-sm font-bold ${ns.errors > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                      {ns.errors > 0 ? `${ns.errors} err` : '✓ clean'}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* ── ChromaDB ──────────────────────────────────────────── */}
          <SectionTitle>ChromaDB Collections</SectionTitle>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-surface border border-line rounded-lg p-4">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
                Document count per collection
              </div>
              <ChromaBarChart collections={data.chromadb.collections} />
            </div>
            <div className="grid grid-cols-3 gap-3 content-start">
              {Object.entries(data.chromadb.collections).map(([col, count]) => (
                <div key={col} className="bg-surface border border-line rounded-lg p-4">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
                    {col.replace(/_/g, ' ')}
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {count !== null ? count.toLocaleString() : '—'}
                  </div>
                  <div className="text-xs text-slate-500">docs</div>
                </div>
              ))}
            </div>
          </div>

          {/* ── Runtime Model Deployment Health ───────────────────────────── */}
          <SectionTitle>Runtime Model Deployment Health</SectionTitle>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-2">
            {data.models.map(m => {
              const modelLabels: Record<string, string> = {
                bert: 'BERT Domain Classifier',
                ner: 'spaCy NER Model',
                actor: 'Actor Classifier (Art. 3)',
                risk: 'Risk Classifier (Art. 6)',
                prohibited: 'Prohibited Classifier (Art. 5)',
              }
              const label = modelLabels[m.model_type] ?? m.model_type
              return (
                <div key={m.model_type} className="bg-surface border border-line rounded-lg p-4 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-slate-100 text-sm">{label}</span>
                      <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                        m.on_disk ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                      }`}>
                        {m.on_disk ? '● Active' : '○ Missing'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-slate-400 mb-3">
                      <span>Active Version:</span>
                      {m.version ? (
                        <span className="px-1.5 py-0.5 rounded bg-brand-500/10 text-brand-300 font-mono font-semibold">
                          {m.version}
                        </span>
                      ) : (
                        <span className="text-slate-500 italic">untrained</span>
                      )}
                    </div>
                  </div>
                  <div className="pt-3 border-t border-line/60 flex items-center justify-between text-xs text-slate-400">
                    <div>
                      Score:{' '}
                      <span className="font-bold text-slate-200 tabular-nums">
                        {m.score !== null ? `${(m.score * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500 font-mono">
                      {m.data_version ? `Data: ${m.data_version}` : ''}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* ── System Resources ──────────────────────────────────── */}
          {data.system.available !== false && (
            <>
              <SectionTitle>System Resources</SectionTitle>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-surface border border-line rounded-lg p-4">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">CPU</div>
                  <div className="text-3xl font-bold text-white mb-2">
                    {data.system.cpu_percent ?? '—'}%
                  </div>
                  <div className="w-full h-2 bg-surface-hover rounded-full overflow-hidden">
                    <div
                      className={`h-2 rounded-full ${
                        (data.system.cpu_percent ?? 0) > 80 ? 'bg-red-500'
                        : (data.system.cpu_percent ?? 0) > 50 ? 'bg-amber-500'
                        : 'bg-emerald-500'
                      }`}
                      style={{ width: `${data.system.cpu_percent ?? 0}%` }}
                    />
                  </div>
                </div>
                <div className="bg-surface border border-line rounded-lg p-4">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Memory</div>
                  {data.system.memory_used_mb != null && data.system.memory_total_mb != null ? (
                    <MemoryBar used={data.system.memory_used_mb} total={data.system.memory_total_mb} />
                  ) : (
                    <div className="text-slate-500 text-sm">—</div>
                  )}
                </div>
                <StatCard label="Uptime" value={formatUptime(data.uptime_s)} />
                <StatCard
                  label="Last Snapshot"
                  value={data.generated_at.replace('T', ' ')}
                />
              </div>
            </>
          )}
        </>
      )}
    </Layout>
  )
}
