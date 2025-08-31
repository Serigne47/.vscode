/**
 * Composant AgentStatus pour afficher l'état détaillé d'un agent
 * Vue complète avec métriques, logs et résultats partiels
 */

'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress, CircularProgress } from '@/components/ui/progress'
import { ConfidenceIndicator } from '@/components/common/ConfidenceIndicator'
import { 
  AgentType,
  AgentProgress,
  Status,
  ConfidenceScore,
  AGENT_LABELS,
  AGENT_COLORS,
} from '@/lib/types'

// ============================================
// Types
// ============================================

export interface AgentStatusProps {
  agent: AgentType
  progress?: AgentProgress
  confidence?: ConfidenceScore
  results?: any
  logs?: LogEntry[]
  onRetry?: () => void
  onViewDetails?: () => void
  expanded?: boolean
  className?: string
}

interface LogEntry {
  timestamp: string
  level: 'info' | 'warning' | 'error' | 'success'
  message: string
}

interface MetricProps {
  label: string
  value: string | number
  trend?: 'up' | 'down' | 'stable'
  suffix?: string
}

// ============================================
// Icons
// ============================================

const Icons = {
  info: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  warning: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  error: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  success: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  processing: (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  ),
  expand: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  ),
  collapse: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
    </svg>
  ),
  retry: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  ),
  view: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  ),
  trendUp: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  ),
  trendDown: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
    </svg>
  ),
  stable: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
    </svg>
  ),
}

// Agent specific icons
const AgentIcons: Record<AgentType, string> = {
  [AgentType.IDENTITY]: '👤',
  [AgentType.EXECUTION]: '⚙️',
  [AgentType.FINANCIAL]: '💰',
  [AgentType.LEGAL]: '⚖️',
  [AgentType.TIMELINE]: '📅',
  [AgentType.VOLUME]: '📊',
}

// ============================================
// Metric Component
// ============================================

const Metric: React.FC<MetricProps> = ({ label, value, trend, suffix }) => {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <span className="text-success">{Icons.trendUp}</span>
      case 'down':
        return <span className="text-destructive">{Icons.trendDown}</span>
      case 'stable':
        return <span className="text-muted-foreground">{Icons.stable}</span>
      default:
        return null
    }
  }

  return (
    <div className="flex flex-col">
      <span className="text-xs text-muted-foreground mb-1">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-lg font-semibold">
          {value}{suffix && <span className="text-sm font-normal text-muted-foreground ml-1">{suffix}</span>}
        </span>
        {getTrendIcon()}
      </div>
    </div>
  )
}

// ============================================
// AgentStatus Component
// ============================================

export const AgentStatus: React.FC<AgentStatusProps> = ({
  agent,
  progress,
  confidence,
  results,
  logs = [],
  onRetry,
  onViewDetails,
  expanded: initialExpanded = false,
  className,
}) => {
  const [isExpanded, setIsExpanded] = React.useState(initialExpanded)
  const label = AGENT_LABELS[agent]
  const color = AGENT_COLORS[agent]
  const icon = AgentIcons[agent]

  // Get status details
  const getStatus = (): Status => {
    return progress?.status || Status.IDLE
  }

  const getStatusBadge = () => {
    const status = getStatus()
    const statusConfig = {
      [Status.IDLE]: { label: 'Idle', color: 'bg-muted text-muted-foreground' },
      [Status.PENDING]: { label: 'Pending', color: 'bg-warning/10 text-warning' },
      [Status.PROCESSING]: { label: 'Processing', color: 'bg-primary/10 text-primary' },
      [Status.COMPLETED]: { label: 'Completed', color: 'bg-success/10 text-success' },
      [Status.FAILED]: { label: 'Failed', color: 'bg-destructive/10 text-destructive' },
      [Status.CANCELLED]: { label: 'Cancelled', color: 'bg-muted text-muted-foreground' },
    }

    const config = statusConfig[status]
    return (
      <span className={cn(
        'inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium',
        config.color
      )}>
        {status === Status.PROCESSING && Icons.processing}
        {status === Status.COMPLETED && Icons.success}
        {status === Status.FAILED && Icons.error}
        <span>{config.label}</span>
      </span>
    )
  }

  // Calculate metrics
  const getMetrics = () => {
    const metrics = []
    
    if (progress) {
      metrics.push({
        label: 'Progress',
        value: `${progress.progress}%`,
      })
      
      if (progress.startedAt) {
        const duration = progress.completedAt 
          ? new Date(progress.completedAt).getTime() - new Date(progress.startedAt).getTime()
          : Date.now() - new Date(progress.startedAt).getTime()
        const seconds = Math.floor(duration / 1000)
        const minutes = Math.floor(seconds / 60)
        const remainingSeconds = seconds % 60
        
        metrics.push({
          label: 'Duration',
          value: `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`,
        })
      }
    }

    if (confidence !== undefined) {
      metrics.push({
        label: 'Confidence',
        value: Math.round(confidence * 100),
        suffix: '%',
      })
    }

    if (results) {
      // Add agent-specific metrics based on results
      if (agent === AgentType.IDENTITY && results.data?.clientName) {
        metrics.push({
          label: 'Client',
          value: results.data.clientName,
        })
      }
      // Add more agent-specific metrics as needed
    }

    return metrics
  }

  // Get recent logs
  const recentLogs = logs.slice(-5)

  // Render log icon
  const getLogIcon = (level: LogEntry['level']) => {
    switch (level) {
      case 'info':
        return <span className="text-info">{Icons.info}</span>
      case 'warning':
        return <span className="text-warning">{Icons.warning}</span>
      case 'error':
        return <span className="text-destructive">{Icons.error}</span>
      case 'success':
        return <span className="text-success">{Icons.success}</span>
    }
  }

  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardHeader 
        className="cursor-pointer select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            <div 
              className="w-10 h-10 rounded-lg flex items-center justify-center text-lg flex-shrink-0"
              style={{ backgroundColor: `${color}20` }}
            >
              {icon}
            </div>
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                {label}
                <button className="text-muted-foreground hover:text-foreground">
                  {isExpanded ? Icons.collapse : Icons.expand}
                </button>
              </CardTitle>
              <CardDescription className="mt-1">
                {progress?.currentStep || `${label} agent for tender analysis`}
              </CardDescription>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {getStatusBadge()}
            {confidence !== undefined && (
              <ConfidenceIndicator
                score={confidence}
                variant="badge"
                size="sm"
                showPercentage={false}
              />
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {/* Progress Bar */}
        {progress && (progress.status === Status.PROCESSING || progress.status === Status.PENDING) && (
          <div className="mb-4">
            <Progress 
              value={progress.progress} 
              showLabel={true}
              animated={progress.status === Status.PROCESSING}
            />
          </div>
        )}

        {/* Metrics Grid */}
        {getMetrics().length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-4">
            {getMetrics().map((metric, index) => (
              <Metric key={index} {...metric} />
            ))}
          </div>
        )}

        {/* Error Message */}
        {progress?.status === Status.FAILED && progress.error && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 mb-4">
            <div className="flex items-start gap-2">
              <span className="text-destructive mt-0.5">{Icons.error}</span>
              <div className="flex-1">
                <p className="text-sm font-medium text-destructive">Analysis Failed</p>
                <p className="text-sm text-muted-foreground mt-1">{progress.error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Expanded Content */}
        {isExpanded && (
          <div className="space-y-4 pt-4 border-t">
            {/* Activity Logs */}
            {recentLogs.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Recent Activity</h4>
                <div className="space-y-2">
                  {recentLogs.map((log, index) => (
                    <div key={index} className="flex items-start gap-2 text-sm">
                      {getLogIcon(log.level)}
                      <div className="flex-1">
                        <span className="text-muted-foreground">{log.timestamp}</span>
                        <span className="ml-2">{log.message}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Results Preview */}
            {results && (
              <div>
                <h4 className="text-sm font-medium mb-2">Results Preview</h4>
                <div className="p-3 rounded-lg bg-muted/50 text-sm">
                  <pre className="whitespace-pre-wrap font-mono text-xs">
                    {JSON.stringify(results, null, 2).slice(0, 500)}...
                  </pre>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center gap-2">
              {progress?.status === Status.FAILED && onRetry && (
                <Button variant="outline" size="sm" onClick={onRetry}>
                  {Icons.retry}
                  Retry Analysis
                </Button>
              )}
              {progress?.status === Status.COMPLETED && onViewDetails && (
                <Button variant="outline" size="sm" onClick={onViewDetails}>
                  {Icons.view}
                  View Full Results
                </Button>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ============================================
// AgentStatusGrid Component
// ============================================

interface AgentStatusGridProps {
  agents: AgentType[]
  progressData: Record<string, AgentProgress>
  confidenceScores?: Record<string, ConfidenceScore>
  results?: Record<string, any>
  onRetry?: (agent: AgentType) => void
  onViewDetails?: (agent: AgentType) => void
  className?: string
}

export const AgentStatusGrid: React.FC<AgentStatusGridProps> = ({
  agents,
  progressData,
  confidenceScores = {},
  results = {},
  onRetry,
  onViewDetails,
  className,
}) => {
  return (
    <div className={cn('grid grid-cols-1 lg:grid-cols-2 gap-4', className)}>
      {agents.map(agent => (
        <AgentStatus
          key={agent}
          agent={agent}
          progress={progressData[agent]}
          confidence={confidenceScores[agent]}
          results={results[agent]}
          onRetry={onRetry ? () => onRetry(agent) : undefined}
          onViewDetails={onViewDetails ? () => onViewDetails(agent) : undefined}
        />
      ))}
    </div>
  )
}

export default AgentStatus