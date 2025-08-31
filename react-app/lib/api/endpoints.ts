/**
 * Définition centralisée de tous les endpoints API
 * Organisation par domaine métier avec types stricts
 */

import apiClient from './client'
import {
  Tender,
  TenderSummary,
  Analysis,
  AnalysisConfig,
  Document,
  ExportOptions,
  ExportResult,
  ApiResponse,
  PaginatedResponse,
  AgentType,
  Status,
  UUID,
  ProgressUpdate,
  AgentProgress,
  IdentityResult,
  ExecutionResult,
  FinancialResult,
  LegalResult,
  TimelineResult,
  VolumeResult,
} from '@/lib/types'

// ============================================
// Configuration des endpoints
// ============================================

const API_VERSION = '/api/v1'

const ENDPOINTS = {
  // Health & Status
  health: '/health',
  status: '/status',
  
  // Tenders
  tenders: {
    base: `${API_VERSION}/tenders`,
    byId: (id: UUID) => `${API_VERSION}/tenders/${id}`,
    documents: (id: UUID) => `${API_VERSION}/tenders/${id}/documents`,
    upload: `${API_VERSION}/tenders/upload`,
  },
  
  // Analysis
  analysis: {
    base: `${API_VERSION}/analysis`,
    byId: (id: UUID) => `${API_VERSION}/analysis/${id}`,
    start: (tenderId: UUID) => `${API_VERSION}/tenders/${tenderId}/analyze`,
    status: (id: UUID) => `${API_VERSION}/analysis/${id}/status`,
    results: (id: UUID) => `${API_VERSION}/analysis/${id}/results`,
    cancel: (id: UUID) => `${API_VERSION}/analysis/${id}/cancel`,
  },
  
  // Agents
  agents: {
    results: (analysisId: UUID, agent: AgentType) => 
      `${API_VERSION}/analysis/${analysisId}/agents/${agent}`,
    progress: (analysisId: UUID) => 
      `${API_VERSION}/analysis/${analysisId}/agents/progress`,
  },
  
  // Export
  export: {
    create: (analysisId: UUID) => `${API_VERSION}/analysis/${analysisId}/export`,
    download: (exportId: UUID) => `${API_VERSION}/exports/${exportId}/download`,
    status: (exportId: UUID) => `${API_VERSION}/exports/${exportId}`,
  },
  
  // Documents
  documents: {
    upload: `${API_VERSION}/documents/upload`,
    byId: (id: UUID) => `${API_VERSION}/documents/${id}`,
    preview: (id: UUID) => `${API_VERSION}/documents/${id}/preview`,
  },
  
  // WebSocket
  ws: {
    analysis: (analysisId: UUID) => `/ws/analysis/${analysisId}`,
  },
} as const

// ============================================
// API Services
// ============================================

/**
 * Service Health Check
 */
export const healthApi = {
  /**
   * Vérifier l'état du serveur
   */
  async check(): Promise<ApiResponse<{ status: string; version: string }>> {
    return apiClient.get(ENDPOINTS.health, { cache: false })
  },

  /**
   * Obtenir le statut détaillé du système
   */
  async getStatus(): Promise<ApiResponse<{
    status: string
    services: Record<string, boolean>
    timestamp: string
  }>> {
    return apiClient.get(ENDPOINTS.status, { cache: false })
  },
}

/**
 * Service Tenders
 */
export const tendersApi = {
  /**
   * Récupérer la liste des tenders
   */
  async getAll(params?: {
    page?: number
    pageSize?: number
    status?: Status
    search?: string
  }): Promise<ApiResponse<PaginatedResponse<TenderSummary>>> {
    return apiClient.get(ENDPOINTS.tenders.base, {
      params,
      cache: true,
      cacheTTL: 30000, // 30 secondes
    })
  },

  /**
   * Récupérer un tender par ID
   */
  async getById(id: UUID): Promise<ApiResponse<Tender>> {
    return apiClient.get(ENDPOINTS.tenders.byId(id), {
      cache: true,
      cacheTTL: 60000, // 1 minute
    })
  },

  /**
   * Créer un nouveau tender avec upload de documents
   */
  async create(data: {
    title: string
    reference: string
    client: string
    description?: string
    files: File[]
  }, onProgress?: (progress: number) => void): Promise<ApiResponse<Tender>> {
    return apiClient.upload(
      ENDPOINTS.tenders.upload,
      data.files,
      {
        title: data.title,
        reference: data.reference,
        client: data.client,
        description: data.description,
      },
      onProgress
    )
  },

  /**
   * Mettre à jour un tender
   */
  async update(id: UUID, data: Partial<Tender>): Promise<ApiResponse<Tender>> {
    return apiClient.patch(ENDPOINTS.tenders.byId(id), data)
  },

  /**
   * Supprimer un tender
   */
  async delete(id: UUID): Promise<ApiResponse<void>> {
    return apiClient.delete(ENDPOINTS.tenders.byId(id))
  },

  /**
   * Ajouter des documents à un tender existant
   */
  async addDocuments(
    tenderId: UUID, 
    files: File[],
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<Document[]>> {
    return apiClient.upload(
      ENDPOINTS.tenders.documents(tenderId),
      files,
      {},
      onProgress
    )
  },
}

/**
 * Service Analysis
 */
export const analysisApi = {
  /**
   * Démarrer une nouvelle analyse
   */
  async start(
    tenderId: UUID, 
    config?: AnalysisConfig
  ): Promise<ApiResponse<Analysis>> {
    return apiClient.post(ENDPOINTS.analysis.start(tenderId), config)
  },

  /**
   * Récupérer le statut d'une analyse
   */
  async getStatus(id: UUID): Promise<ApiResponse<{
    status: Status
    progress: number
    agents: AgentProgress[]
    currentStep?: string
  }>> {
    return apiClient.get(ENDPOINTS.analysis.status(id), {
      cache: false, // Toujours récupérer le statut frais
    })
  },

  /**
   * Récupérer les résultats complets d'une analyse
   */
  async getResults(id: UUID): Promise<ApiResponse<Analysis>> {
    return apiClient.get(ENDPOINTS.analysis.results(id), {
      cache: true,
      cacheTTL: 300000, // 5 minutes
    })
  },

  /**
   * Annuler une analyse en cours
   */
  async cancel(id: UUID): Promise<ApiResponse<void>> {
    return apiClient.post(ENDPOINTS.analysis.cancel(id))
  },

  /**
   * Récupérer la liste des analyses
   */
  async getAll(params?: {
    page?: number
    pageSize?: number
    status?: Status
    tenderId?: UUID
  }): Promise<ApiResponse<PaginatedResponse<Analysis>>> {
    return apiClient.get(ENDPOINTS.analysis.base, {
      params,
      cache: true,
      cacheTTL: 30000,
    })
  },

  /**
   * Récupérer une analyse par ID
   */
  async getById(id: UUID): Promise<ApiResponse<Analysis>> {
    return apiClient.get(ENDPOINTS.analysis.byId(id), {
      cache: true,
      cacheTTL: 60000,
    })
  },
}

/**
 * Service Agents
 */
export const agentsApi = {
  /**
   * Récupérer les résultats d'un agent spécifique
   */
  async getAgentResult<T = any>(
    analysisId: UUID,
    agent: AgentType
  ): Promise<ApiResponse<T>> {
    const agentTypeMap = {
      [AgentType.IDENTITY]: 'identity' as const,
      [AgentType.EXECUTION]: 'execution' as const,
      [AgentType.FINANCIAL]: 'financial' as const,
      [AgentType.LEGAL]: 'legal' as const,
      [AgentType.TIMELINE]: 'timeline' as const,
      [AgentType.VOLUME]: 'volume' as const,
    }

    return apiClient.get(ENDPOINTS.agents.results(analysisId, agent), {
      cache: true,
      cacheTTL: 120000, // 2 minutes
    })
  },

  /**
   * Récupérer le progrès de tous les agents
   */
  async getProgress(analysisId: UUID): Promise<ApiResponse<AgentProgress[]>> {
    return apiClient.get(ENDPOINTS.agents.progress(analysisId), {
      cache: false,
    })
  },

  // Méthodes spécifiques par agent
  async getIdentityResult(analysisId: UUID): Promise<ApiResponse<IdentityResult>> {
    return this.getAgentResult<IdentityResult>(analysisId, AgentType.IDENTITY)
  },

  async getExecutionResult(analysisId: UUID): Promise<ApiResponse<ExecutionResult>> {
    return this.getAgentResult<ExecutionResult>(analysisId, AgentType.EXECUTION)
  },

  async getFinancialResult(analysisId: UUID): Promise<ApiResponse<FinancialResult>> {
    return this.getAgentResult<FinancialResult>(analysisId, AgentType.FINANCIAL)
  },

  async getLegalResult(analysisId: UUID): Promise<ApiResponse<LegalResult>> {
    return this.getAgentResult<LegalResult>(analysisId, AgentType.LEGAL)
  },

  async getTimelineResult(analysisId: UUID): Promise<ApiResponse<TimelineResult>> {
    return this.getAgentResult<TimelineResult>(analysisId, AgentType.TIMELINE)
  },

  async getVolumeResult(analysisId: UUID): Promise<ApiResponse<VolumeResult>> {
    return this.getAgentResult<VolumeResult>(analysisId, AgentType.VOLUME)
  },
}

/**
 * Service Export
 */
export const exportApi = {
  /**
   * Créer un export
   */
  async create(
    analysisId: UUID,
    options: ExportOptions
  ): Promise<ApiResponse<ExportResult>> {
    return apiClient.post(ENDPOINTS.export.create(analysisId), options)
  },

  /**
   * Télécharger un export
   */
  async download(
    exportId: UUID,
    onProgress?: (progress: number) => void
  ): Promise<void> {
    return apiClient.download(
      ENDPOINTS.export.download(exportId),
      undefined,
      onProgress
    )
  },

  /**
   * Vérifier le statut d'un export
   */
  async getStatus(exportId: UUID): Promise<ApiResponse<ExportResult>> {
    return apiClient.get(ENDPOINTS.export.status(exportId), {
      cache: false,
    })
  },
}

/**
 * Service Documents
 */
export const documentsApi = {
  /**
   * Upload de documents
   */
  async upload(
    files: File[],
    metadata?: Record<string, any>,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<Document[]>> {
    return apiClient.upload(
      ENDPOINTS.documents.upload,
      files,
      metadata,
      onProgress
    )
  },

  /**
   * Récupérer un document par ID
   */
  async getById(id: UUID): Promise<ApiResponse<Document>> {
    return apiClient.get(ENDPOINTS.documents.byId(id), {
      cache: true,
      cacheTTL: 300000, // 5 minutes
    })
  },

  /**
   * Obtenir l'aperçu d'un document
   */
  async getPreview(id: UUID): Promise<ApiResponse<{
    url: string
    type: string
    pages?: number
  }>> {
    return apiClient.get(ENDPOINTS.documents.preview(id), {
      cache: true,
      cacheTTL: 600000, // 10 minutes
    })
  },

  /**
   * Supprimer un document
   */
  async delete(id: UUID): Promise<ApiResponse<void>> {
    return apiClient.delete(ENDPOINTS.documents.byId(id))
  },
}

/**
 * WebSocket URLs
 */
export const wsEndpoints = {
  /**
   * Obtenir l'URL WebSocket pour suivre une analyse
   */
  getAnalysisUrl(analysisId: UUID): string {
    const baseUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://127.0.0.1:8000'
    return `${baseUrl}${ENDPOINTS.ws.analysis(analysisId)}`
  },
}

// ============================================
// Helpers pour les appels API
// ============================================

/**
 * Helper pour gérer les erreurs API
 */
export const handleApiError = (error: any, fallbackMessage?: string): string => {
  if (error?.message) {
    return error.message
  }
  return fallbackMessage || 'An unexpected error occurred'
}

/**
 * Helper pour vérifier le succès d'une réponse
 */
export const isApiSuccess = <T>(response: ApiResponse<T>): response is ApiResponse<T> & { data: T } => {
  return response.success === true && response.data !== undefined
}

/**
 * Helper pour extraire les données d'une réponse
 */
export const extractApiData = <T>(response: ApiResponse<T>): T | null => {
  return isApiSuccess(response) ? response.data : null
}

// Export groupé pour faciliter l'import
export const api = {
  health: healthApi,
  tenders: tendersApi,
  analysis: analysisApi,
  agents: agentsApi,
  export: exportApi,
  documents: documentsApi,
  ws: wsEndpoints,
} as const

export default api