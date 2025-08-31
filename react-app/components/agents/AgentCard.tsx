/**
 * Composant AgentCard pour afficher un résumé d'agent
 * Card réutilisable avec statut, confiance et actions
 */

'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { ConfidenceIndicator } from '@/components/common/ConfidenceIndicator'
import {
  AgentType,
  Status,
  ConfidenceScore,
  AGENT_LABELS,
  AGENT_COLORS,
} from '@/lib/types'

// ============================================
// Types
// ============================================

export interface AgentCardProps {
  agent: AgentType
  status: Status
  confidence?: ConfidenceScore
  progress?: number
  summary?: {
    title: string
    value: string | number
    details?: string[]
  }
  keyFindings?: string[]
  documentsAnalyzed?: number
  processingTime?: number
  onView?: () => void
  onRetry?: () => void
  interactive?: boolean
  className?: string
}

// ============================================
// Icons
// ============================================

const Icons = {
  view: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  ),
  retry: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  ),
  check: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  ),
  clock: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  document: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  ),
  alert: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  processing: (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  ),
}

// Agent specific icons/emojis
const AgentEmojis: Record<AgentType, string> = {
  [AgentType.IDENTITY]: '👤',
  [AgentType.EXECUTION]: '⚙️',
  [AgentType.FINANCIAL]: '💰',
  [AgentType.LEGAL]: '⚖️',
  [AgentType.TIMELINE]: '📅',
  [AgentType.VOLUME]: '📊',
}

// ============================================
// AgentCard Component
// ============================================

export const AgentCard: React.FC<AgentCardProps> = ({
  agent,
  status,
  confidence,
  progress,
  summary,
  keyFindings = [],
  documentsAnalyzed,
  processingTime,
  onView,
  onRetry,
  interactive = true,
  className,
}) => {
  const router = useRouter()
  const label = AGENT_LABELS[agent]
  const color = AGENT_COLORS[agent]
  const emoji = AgentEmojis[agent]

  // Get status badge
  const getStatusBadge = () => {
    const statusConfig = {
      [Status.IDLE]: { icon: Icons.clock, label: 'Waiting', color: 'text-muted-foreground' },
      [Status.PENDING]: { icon: Icons.clock, label: 'Pending', color: 'text-warning' },
      [Status.PROCESSING]: { icon: Icons.processing, label: 'Processing', color: 'text-primary' },
      [Status.COMPLETED]: { icon: Icons.check, label: 'Completed', color: 'text-success' },
      [Status.FAILED]: { icon: Icons.alert, label: 'Failed', color: 'text-destructive' },
      [Status.CANCELLED]: { icon: Icons.alert, label: 'Cancelled', color: 'text-muted-foreground' },
    }

    const config = statusConfig[status]
    return (
      <div className={cn('flex items-center gap-1.5', config.color)}>
        {config.icon}
        <span className="text-xs font-medium">{config.label}</span>
      </div>
    )
  }

  // Format processing time
  const formatTime = (seconds?: number) => {
    if (!seconds) return null
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  // Handle card click
  const handleClick = () => {
    if (interactive && onView && status === Status.COMPLETED) {
      onView()
    }
  }

  return (
    <Card 
      className={cn(
        'relative overflow-hidden transition-all duration-200',
        interactive && status === Status.COMPLETED && 'cursor-pointer hover:shadow-lg hover:-translate-y-0.5',
        className
      )}
      onClick={handleClick}
    >
      {/* Gradient border effect */}
      <div 
        className="absolute inset-x-0 top-0 h-1"
        style={{ backgroundColor: color }}
      />

      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            <div 
              className="w-10 h-10 rounded-lg flex items-center justify-center text-lg flex-shrink-0"
              style={{ backgroundColor: `${color}20` }}
            >
              {emoji}
            </div>
            <div>
              <CardTitle className="text-base">
                {label} Agent
              </CardTitle>
              <CardDescription className="text-xs mt-0.5">
                {getStatusBadge()}
              </CardDescription>
            </div>
          </div>
          
          {confidence !== undefined && status === Status.COMPLETED && (
            <ConfidenceIndicator
              score={confidence}
              variant="badge"
              size="sm"
              showPercentage={true}
            />
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Progress bar for processing */}
        {status === Status.PROCESSING && progress !== undefined && (
          <div>
            <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
              <span>Analyzing...</span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} className="h-1.5" />
          </div>
        )}

        {/* Summary for completed */}
        {status === Status.COMPLETED && summary && (
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-muted/50">
              <p className="text-xs text-muted-foreground mb-1">{summary.title}</p>
              <p className="font-semibold text-lg">{summary.value}</p>
              {summary.details && summary.details.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {summary.details.slice(0, 3).map((detail, index) => (
                    <li key={index} className="text-xs text-muted-foreground flex items-start gap-1">
                      <span className="text-primary mt-0.5">•</span>
                      <span>{detail}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* Key findings */}
            {keyFindings.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-2">Key Findings</p>
                <ul className="space-y-1">
                  {keyFindings.slice(0, 3).map((finding, index) => (
                    <li key={index} className="text-sm flex items-start gap-2">
                      <span className="text-success mt-0.5">{Icons.check}</span>
                      <span className="text-xs">{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Stats */}
            <div className="flex items-center justify-between pt-3 border-t text-xs text-muted-foreground">
              {documentsAnalyzed !== undefined && (
                <div className="flex items-center gap-1">
                  {Icons.document}
                  <span>{documentsAnalyzed} docs</span>
                </div>
              )}
              {processingTime !== undefined && (
                <div className="flex items-center gap-1">
                  {Icons.clock}
                  <span>{formatTime(processingTime)}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error state */}
        {status === Status.FAILED && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
            <p className="text-sm text-destructive font-medium mb-1">Analysis Failed</p>
            <p className="text-xs text-muted-foreground">
              An error occurred during analysis. Please try again.
            </p>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2">
          {status === Status.COMPLETED && onView && (
            <Button 
              size="sm" 
              className="flex-1"
              onClick={(e) => {
                e.stopPropagation()
                onView()
              }}
            >
              {Icons.view}
              View Details
            </Button>
          )}
          
          {status === Status.FAILED && onRetry && (
            <Button 
              size="sm" 
              variant="outline"
              className="flex-1"
              onClick={(e) => {
                e.stopPropagation()
                onRetry()
              }}
            >
              {Icons.retry}
              Retry
            </Button>
          )}
          
          {(status === Status.IDLE || status === Status.PENDING) && (
            <div className="text-center w-full text-xs text-muted-foreground py-2">
              Waiting to start...
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// ============================================
// AgentCardGrid Component
// ============================================

interface AgentCardGridProps {
  agents: Array<{
    type: AgentType
    status: Status
    confidence?: ConfidenceScore
    progress?: number
    summary?: AgentCardProps['summary']
    keyFindings?: string[]
  }>
  onViewAgent?: (agent: AgentType) => void
  onRetryAgent?: (agent: AgentType) => void
  className?: string
}

export const AgentCardGrid: React.FC<AgentCardGridProps> = ({
  agents,
  onViewAgent,
  onRetryAgent,
  className,
}) => {
  return (
    <div className={cn(
      'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4',
      className
    )}>
      {agents.map(agent => (
        <AgentCard
          key={agent.type}
          agent={agent.type}
          status={agent.status}
          confidence={agent.confidence}
          progress={agent.progress}
          summary={agent.summary}
          keyFindings={agent.keyFindings}
          onView={onViewAgent ? () => onViewAgent(agent.type) : undefined}
          onRetry={onRetryAgent ? () => onRetryAgent(agent.type) : undefined}
        />
      ))}
    </div>
  )
}

export default AgentCard