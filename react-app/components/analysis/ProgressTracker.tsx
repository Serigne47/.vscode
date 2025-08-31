/**
 * Composant ProgressTracker pour suivre l'avancement de l'analyse
 * Affichage temps réel avec WebSocket
 */

'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress, CircularProgress, StepProgress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { useAgentProgress } from '@/lib/hooks/useWebSocket'
import { useTenderStore } from '@/lib/store/tenderStore'
import { 
  AgentType, 
  AgentProgress, 
  Status,
  AGENT_LABELS,
  AGENT_COLORS,
  UUID 
} from '@/lib/types'

// ============================================
// Types
// ============================================

export interface ProgressTrackerProps {
  analysisId: UUID | null
  onComplete?: () => void
  onCancel?: () => void
  showDetails?: boolean
  compact?: boolean
  className?: string
}

interface AgentProgressItemProps {
  agent: AgentType
  progress: AgentProgress
  compact?: boolean
}

// ============================================
// Icons
// ============================================

const Icons = {
  check: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  ),
  clock: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  alert: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  cancel: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  processing: (
    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  ),
}

// Agent Icons (reuse from types or define specific ones)
const AgentIcons: Record<AgentType, React.ReactNode> = {
  [AgentType.IDENTITY]: '👤',
  [AgentType.EXECUTION]: '⚙️',
  [AgentType.FINANCIAL]: '💰',
  [AgentType.LEGAL]: '⚖️',
  [AgentType.TIMELINE]: '📅',
  [AgentType.VOLUME]: '📊',
}

// ============================================
// AgentProgressItem Component
// ============================================

const AgentProgressItem: React.FC<AgentProgressItemProps> = ({
  agent,
  progress,
  compact = false,
}) => {
  const label = AGENT_LABELS[agent]
  const color = AGENT_COLORS[agent]
  const icon = AgentIcons[agent]
  
  const getStatusIcon = () => {
    switch (progress.status) {
      case Status.COMPLETED:
        return <span className="text-success">{Icons.check}</span>
      case Status.FAILED:
        return <span className="text-destructive">{Icons.alert}</span>
      case Status.PROCESSING:
        return <span className="text-primary">{Icons.processing}</span>
      default:
        return <span className="text-muted-foreground">{Icons.clock}</span>
    }
  }

  const getElapsedTime = () => {
    if (!progress.startedAt) return null
    const start = new Date(progress.startedAt).getTime()
    const end = progress.completedAt 
      ? new Date(progress.completedAt).getTime() 
      : Date.now()
    const elapsed = Math.floor((end - start) / 1000)
    const minutes = Math.floor(elapsed / 60)
    const seconds = elapsed % 60
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  if (compact) {
    return (
      <div className="flex items-center gap-3 p-2">
        <div 
          className="w-8 h-8 rounded-lg flex items-center justify-center text-sm"
          style={{ backgroundColor: `${color}20` }}
        >
          {icon}
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">{label}</span>
            <span className="text-xs text-muted-foreground">{progress.progress}%</span>
          </div>
          <Progress value={progress.progress} className="h-1 mt-1" />
        </div>
        {getStatusIcon()}
      </div>
    )
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-start gap-4">
          <div 
            className="w-12 h-12 rounded-xl flex items-center justify-center text-lg"
            style={{ backgroundColor: `${color}20` }}
          >
            {icon}
          </div>
          
          <div className="flex-1">
            <div className="flex items-center justify-between mb-2">
              <div>
                <h4 className="font-medium">{label}</h4>
                <p className="text-xs text-muted-foreground">
                  {progress.status === Status.PROCESSING && progress.currentStep && (
                    <span>{progress.currentStep}</span>
                  )}
                  {progress.status === Status.COMPLETED && getElapsedTime() && (
                    <span>Completed in {getElapsedTime()}</span>
                  )}
                  {progress.status === Status.FAILED && progress.error && (
                    <span className="text-destructive">{progress.error}</span>
                  )}
                </p>
              </div>
              {getStatusIcon()}
            </div>
            
            <Progress 
              value={progress.progress} 
              variant={
                progress.status === Status.COMPLETED ? 'success' :
                progress.status === Status.FAILED ? 'destructive' :
                'default'
              }
              showLabel={true}
              animated={progress.status === Status.PROCESSING}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ============================================
// ProgressTracker Component
// ============================================

export const ProgressTracker: React.FC<ProgressTrackerProps> = ({
  analysisId,
  onComplete,
  onCancel,
  showDetails = true,
  compact = false,
  className,
}) => {
  const { agentProgress, overallProgress, isConnected, reset } = useAgentProgress(analysisId)
  const { currentAnalysisId } = useTenderStore()
  
  const [startTime] = React.useState(Date.now())
  const [elapsedTime, setElapsedTime] = React.useState(0)

  // Update elapsed time
  React.useEffect(() => {
    if (!analysisId || overallProgress === 100) return
    
    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)

    return () => clearInterval(interval)
  }, [analysisId, overallProgress, startTime])

  // Handle completion
  React.useEffect(() => {
    if (overallProgress === 100 && onComplete) {
      onComplete()
    }
  }, [overallProgress, onComplete])

  // Format elapsed time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Get active agents count
  const activeAgents = Object.keys(agentProgress).length
  const completedAgents = Object.values(agentProgress).filter(
    a => a.status === Status.COMPLETED
  ).length
  const failedAgents = Object.values(agentProgress).filter(
    a => a.status === Status.FAILED
  ).length

  // Estimate remaining time
  const estimatedTotal = activeAgents > 0 ? (elapsedTime / overallProgress) * 100 : 0
  const estimatedRemaining = Math.max(0, estimatedTotal - elapsedTime)

  if (!analysisId || !currentAnalysisId) {
    return (
      <Card className={className}>
        <CardContent className="p-8 text-center">
          <p className="text-muted-foreground">No analysis in progress</p>
        </CardContent>
      </Card>
    )
  }

  if (compact) {
    return (
      <Card className={className}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Analysis Progress</CardTitle>
            <span className="text-sm font-medium">{overallProgress}%</span>
          </div>
        </CardHeader>
        <CardContent>
          <Progress value={overallProgress} className="mb-3" />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{completedAgents}/{activeAgents} agents</span>
            <span>{formatTime(elapsedTime)}</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className={cn('space-y-6', className)}>
      {/* Main Progress Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Analysis in Progress</CardTitle>
              <CardDescription>
                Processing your tender documents with {activeAgents} AI agents
              </CardDescription>
            </div>
            {onCancel && (
              <Button
                variant="outline"
                size="sm"
                onClick={onCancel}
                className="text-destructive hover:bg-destructive/10"
              >
                {Icons.cancel}
                Cancel
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {/* Overall Progress */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Overall Progress</span>
              <span className="text-sm text-muted-foreground">
                {overallProgress}% Complete
              </span>
            </div>
            <Progress 
              value={overallProgress} 
              size="lg"
              animated
              striped
            />
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <p className="text-2xl font-bold">{activeAgents}</p>
              <p className="text-xs text-muted-foreground">Active Agents</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-success">{completedAgents}</p>
              <p className="text-xs text-muted-foreground">Completed</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold">{formatTime(elapsedTime)}</p>
              <p className="text-xs text-muted-foreground">Elapsed Time</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold">~{formatTime(Math.floor(estimatedRemaining))}</p>
              <p className="text-xs text-muted-foreground">Remaining</p>
            </div>
          </div>

          {/* Connection Status */}
          {!isConnected && (
            <div className="p-3 rounded-lg bg-warning/10 border border-warning/20 flex items-center gap-2 text-sm text-warning">
              {Icons.alert}
              <span>Connection lost. Attempting to reconnect...</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Agent Progress Details */}
      {showDetails && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Agent Progress</h3>
          <div className={cn(
            'grid gap-4',
            compact ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2'
          )}>
            {Object.values(AgentType).map(agent => {
              const progress = agentProgress[agent] || {
                agent,
                status: Status.PENDING,
                progress: 0,
              }
              
              return (
                <AgentProgressItem
                  key={agent}
                  agent={agent}
                  progress={progress}
                  compact={compact}
                />
              )
            })}
          </div>
        </div>
      )}

      {/* Failed Agents Alert */}
      {failedAgents > 0 && (
        <Card className="border-destructive/20 bg-destructive/5">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <span className="text-destructive">{Icons.alert}</span>
              <div>
                <p className="font-medium text-destructive">
                  {failedAgents} agent{failedAgents > 1 ? 's' : ''} failed
                </p>
                <p className="text-sm text-muted-foreground">
                  Some agents encountered errors. You can retry the analysis after completion.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default ProgressTracker