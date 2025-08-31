/**
 * Composant RiskMatrix pour visualiser les risques identifiés
 * Matrice de risques avec impact et probabilité
 */

'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Risk,
  RiskLevel,
  Recommendation,
  AgentType,
  AGENT_LABELS,
} from '@/lib/types'

// ============================================
// Types
// ============================================

export interface RiskMatrixProps {
  risks: Risk[]
  recommendations?: Recommendation[]
  loading?: boolean
  className?: string
  onRiskClick?: (risk: Risk) => void
  viewMode?: 'matrix' | 'list'
}

interface RiskCardProps {
  risk: Risk
  onClick?: () => void
  compact?: boolean
}

interface MatrixCellProps {
  risks: Risk[]
  impact: 'low' | 'medium' | 'high'
  probability: 'low' | 'medium' | 'high'
  onClick?: (risk: Risk) => void
}

// ============================================
// Icons
// ============================================

const Icons = {
  critical: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  high: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  medium: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  low: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  mitigation: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  ),
  recommendation: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  ),
  grid: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
    </svg>
  ),
  list: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  ),
}

// ============================================
// Risk level configurations
// ============================================

const riskLevelConfig = {
  [RiskLevel.CRITICAL]: {
    icon: Icons.critical,
    label: 'Critical',
    color: 'text-destructive',
    bgColor: 'bg-destructive/10',
    borderColor: 'border-destructive/20',
  },
  [RiskLevel.HIGH]: {
    icon: Icons.high,
    label: 'High',
    color: 'text-orange-600',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
  },
  [RiskLevel.MEDIUM]: {
    icon: Icons.medium,
    label: 'Medium',
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/20',
  },
  [RiskLevel.LOW]: {
    icon: Icons.low,
    label: 'Low',
    color: 'text-blue-600',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
  },
}

// ============================================
// Helper functions
// ============================================

const getRiskImpact = (risk: Risk): 'low' | 'medium' | 'high' => {
  // Simple mapping - you could make this more sophisticated
  if (risk.level === RiskLevel.CRITICAL) return 'high'
  if (risk.level === RiskLevel.HIGH) return 'high'
  if (risk.level === RiskLevel.MEDIUM) return 'medium'
  return 'low'
}

const getRiskProbability = (risk: Risk): 'low' | 'medium' | 'high' => {
  // Simple heuristic - adjust based on your actual data
  if (risk.category === 'legal' || risk.category === 'financial') return 'high'
  if (risk.category === 'operational') return 'medium'
  return 'low'
}

// ============================================
// RiskCard Component
// ============================================

const RiskCard: React.FC<RiskCardProps> = ({ risk, onClick, compact = false }) => {
  const config = riskLevelConfig[risk.level]
  const agentLabel = AGENT_LABELS[risk.agent]

  if (compact) {
    return (
      <div
        className={cn(
          'p-2 rounded-lg border cursor-pointer hover:shadow-sm transition-all',
          config.bgColor,
          config.borderColor
        )}
        onClick={onClick}
      >
        <div className="flex items-center gap-2">
          <span className={config.color}>{config.icon}</span>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium truncate">{risk.title}</p>
            <p className="text-xs text-muted-foreground">{agentLabel}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <Card 
      className={cn(
        'cursor-pointer hover:shadow-md transition-all',
        config.borderColor
      )}
      onClick={onClick}
    >
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <div className={cn(
            'w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0',
            config.bgColor
          )}>
            <span className={config.color}>{config.icon}</span>
          </div>
          <div className="flex-1">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-medium text-sm">{risk.title}</p>
                <p className="text-xs text-muted-foreground mt-1">{risk.description}</p>
              </div>
              <span className={cn(
                'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
                config.bgColor,
                config.color
              )}>
                {config.label}
              </span>
            </div>
            
            <div className="flex items-center gap-3 mt-3 text-xs text-muted-foreground">
              <span>Agent: {agentLabel}</span>
              <span>•</span>
              <span>Category: {risk.category}</span>
            </div>
            
            {risk.mitigationSuggestions && risk.mitigationSuggestions.length > 0 && (
              <div className="mt-3 p-2 rounded bg-muted/50">
                <p className="text-xs font-medium flex items-center gap-1 mb-1">
                  {Icons.mitigation}
                  Mitigation Suggestions:
                </p>
                <ul className="space-y-0.5">
                  {risk.mitigationSuggestions.slice(0, 2).map((suggestion, index) => (
                    <li key={index} className="text-xs text-muted-foreground flex items-start gap-1">
                      <span className="text-primary mt-0.5">•</span>
                      {suggestion}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ============================================
// MatrixCell Component
// ============================================

const MatrixCell: React.FC<MatrixCellProps> = ({
  risks,
  impact,
  probability,
  onClick,
}) => {
  // Determine cell color based on risk level
  const getCellColor = () => {
    if (impact === 'high' && probability === 'high') return 'bg-destructive/20 border-destructive/30'
    if ((impact === 'high' && probability === 'medium') || 
        (impact === 'medium' && probability === 'high')) return 'bg-orange-100 border-orange-300'
    if (impact === 'medium' && probability === 'medium') return 'bg-warning/20 border-warning/30'
    if ((impact === 'low' && probability === 'high') || 
        (impact === 'high' && probability === 'low')) return 'bg-blue-100 border-blue-300'
    return 'bg-muted border-border'
  }

  return (
    <div className={cn(
      'p-3 rounded-lg border-2 min-h-[100px] relative',
      getCellColor()
    )}>
      {risks.length > 0 ? (
        <div className="space-y-2">
          {risks.slice(0, 3).map((risk, index) => (
            <div
              key={risk.id}
              className="text-xs p-1.5 rounded bg-background/80 cursor-pointer hover:bg-background transition-colors"
              onClick={() => onClick?.(risk)}
            >
              <p className="font-medium truncate">{risk.title}</p>
            </div>
          ))}
          {risks.length > 3 && (
            <p className="text-xs text-muted-foreground text-center">
              +{risks.length - 3} more
            </p>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-center h-full text-xs text-muted-foreground">
          No risks
        </div>
      )}
      
      {/* Risk count badge */}
      {risks.length > 0 && (
        <div className="absolute top-2 right-2">
          <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-background text-xs font-medium">
            {risks.length}
          </span>
        </div>
      )}
    </div>
  )
}

// ============================================
// RiskMatrix Component
// ============================================

export const RiskMatrix: React.FC<RiskMatrixProps> = ({
  risks,
  recommendations = [],
  loading = false,
  className,
  onRiskClick,
  viewMode: initialViewMode = 'matrix',
}) => {
  const [viewMode, setViewMode] = React.useState(initialViewMode)
  const [selectedRisk, setSelectedRisk] = React.useState<Risk | null>(null)

  // Categorize risks by level
  const risksByLevel = React.useMemo(() => {
    return {
      critical: risks.filter(r => r.level === RiskLevel.CRITICAL),
      high: risks.filter(r => r.level === RiskLevel.HIGH),
      medium: risks.filter(r => r.level === RiskLevel.MEDIUM),
      low: risks.filter(r => r.level === RiskLevel.LOW),
    }
  }, [risks])

  // Group risks by impact and probability for matrix
  const matrixData = React.useMemo(() => {
    const matrix: Record<string, Risk[]> = {}
    
    risks.forEach(risk => {
      const impact = getRiskImpact(risk)
      const probability = getRiskProbability(risk)
      const key = `${impact}-${probability}`
      
      if (!matrix[key]) matrix[key] = []
      matrix[key].push(risk)
    })
    
    return matrix
  }, [risks])

  const handleRiskClick = (risk: Risk) => {
    setSelectedRisk(risk)
    onRiskClick?.(risk)
  }

  if (loading) {
    return (
      <div className={cn('space-y-4', className)}>
        <Card>
          <CardHeader>
            <div className="h-6 w-32 bg-muted animate-pulse rounded" />
          </CardHeader>
          <CardContent>
            <div className="h-64 bg-muted animate-pulse rounded" />
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Header with view toggle */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Risk Assessment</CardTitle>
              <CardDescription>
                {risks.length} risks identified across all agents
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant={viewMode === 'matrix' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('matrix')}
              >
                {Icons.grid}
                Matrix
              </Button>
              <Button
                variant={viewMode === 'list' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('list')}
              >
                {Icons.list}
                List
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          {viewMode === 'matrix' ? (
            /* Matrix View */
            <div>
              <div className="grid grid-cols-4 gap-3">
                {/* Header row */}
                <div className="text-center text-sm font-medium p-2">Impact →</div>
                <div className="text-center text-sm text-muted-foreground p-2">Low</div>
                <div className="text-center text-sm text-muted-foreground p-2">Medium</div>
                <div className="text-center text-sm text-muted-foreground p-2">High</div>
                
                {/* High probability row */}
                <div className="text-sm text-muted-foreground p-2 text-right">High ↑</div>
                <MatrixCell
                  risks={matrixData['low-high'] || []}
                  impact="low"
                  probability="high"
                  onClick={handleRiskClick}
                />
                <MatrixCell
                  risks={matrixData['medium-high'] || []}
                  impact="medium"
                  probability="high"
                  onClick={handleRiskClick}
                />
                <MatrixCell
                  risks={matrixData['high-high'] || []}
                  impact="high"
                  probability="high"
                  onClick={handleRiskClick}
                />
                
                {/* Medium probability row */}
                <div className="text-sm text-muted-foreground p-2 text-right">Medium</div>
                <MatrixCell
                  risks={matrixData['low-medium'] || []}
                  impact="low"
                  probability="medium"
                  onClick={handleRiskClick}
                />
                <MatrixCell
                  risks={matrixData['medium-medium'] || []}
                  impact="medium"
                  probability="medium"
                  onClick={handleRiskClick}
                />
                <MatrixCell
                  risks={matrixData['high-medium'] || []}
                  impact="high"
                  probability="medium"
                  onClick={handleRiskClick}
                />
                
                {/* Low probability row */}
                <div className="text-sm text-muted-foreground p-2 text-right">Low</div>
                <MatrixCell
                  risks={matrixData['low-low'] || []}
                  impact="low"
                  probability="low"
                  onClick={handleRiskClick}
                />
                <MatrixCell
                  risks={matrixData['medium-low'] || []}
                  impact="medium"
                  probability="low"
                  onClick={handleRiskClick}
                />
                <MatrixCell
                  risks={matrixData['high-low'] || []}
                  impact="high"
                  probability="low"
                  onClick={handleRiskClick}
                />
                
                {/* Bottom label */}
                <div className="text-xs text-muted-foreground p-2">Probability</div>
                <div></div>
                <div></div>
                <div></div>
              </div>
            </div>
          ) : (
            /* List View */
            <div className="space-y-4">
              {risksByLevel.critical.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-destructive mb-2">
                    Critical Risks ({risksByLevel.critical.length})
                  </h4>
                  <div className="space-y-2">
                    {risksByLevel.critical.map(risk => (
                      <RiskCard
                        key={risk.id}
                        risk={risk}
                        onClick={() => handleRiskClick(risk)}
                      />
                    ))}
                  </div>
                </div>
              )}
              
              {risksByLevel.high.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-orange-600 mb-2">
                    High Risks ({risksByLevel.high.length})
                  </h4>
                  <div className="space-y-2">
                    {risksByLevel.high.map(risk => (
                      <RiskCard
                        key={risk.id}
                        risk={risk}
                        onClick={() => handleRiskClick(risk)}
                      />
                    ))}
                  </div>
                </div>
              )}
              
              {risksByLevel.medium.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-warning mb-2">
                    Medium Risks ({risksByLevel.medium.length})
                  </h4>
                  <div className="space-y-2">
                    {risksByLevel.medium.map(risk => (
                      <RiskCard
                        key={risk.id}
                        risk={risk}
                        onClick={() => handleRiskClick(risk)}
                      />
                    ))}
                  </div>
                </div>
              )}
              
              {risksByLevel.low.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-blue-600 mb-2">
                    Low Risks ({risksByLevel.low.length})
                  </h4>
                  <div className="space-y-2">
                    {risksByLevel.low.map(risk => (
                      <RiskCard
                        key={risk.id}
                        risk={risk}
                        onClick={() => handleRiskClick(risk)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              {Icons.recommendation}
              Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recommendations.slice(0, 5).map((rec, index) => (
                <div key={rec.id} className="flex items-start gap-3 p-3 rounded-lg bg-muted/50">
                  <span className={cn(
                    'text-xs font-medium px-2 py-0.5 rounded-full',
                    rec.priority === 'high' && 'bg-destructive/10 text-destructive',
                    rec.priority === 'medium' && 'bg-warning/10 text-warning',
                    rec.priority === 'low' && 'bg-primary/10 text-primary'
                  )}>
                    {rec.priority}
                  </span>
                  <div className="flex-1">
                    <p className="text-sm font-medium">{rec.title}</p>
                    <p className="text-xs text-muted-foreground mt-1">{rec.description}</p>
                    {rec.actionItems && rec.actionItems.length > 0 && (
                      <ul className="mt-2 space-y-0.5">
                        {rec.actionItems.map((item, i) => (
                          <li key={i} className="text-xs text-muted-foreground flex items-start gap-1">
                            <span className="text-primary mt-0.5">→</span>
                            {item}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default RiskMatrix