/**
 * Hook React pour la gestion des tenders
 * Avec cache, mutations optimistes et gestion d'état
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { toast } from 'react-hot-toast'
import { useTenderStore } from '@/lib/store/tenderStore'
import { api, isApiSuccess } from '@/lib/api/endpoints'
import {
  Tender,
  TenderSummary,
  Analysis,
  AnalysisConfig,
  Status,
  UUID,
  AsyncState,
  AgentType,
} from '@/lib/types'

// ============================================
// Hook principal pour un tender unique
// ============================================

interface UseTenderOptions {
  autoFetch?: boolean
  pollInterval?: number
  onSuccess?: (tender: Tender) => void
  onError?: (error: Error) => void
}

export function useTender(
  tenderId: UUID | null,
  options: UseTenderOptions = {}
) {
  const { autoFetch = true, pollInterval, onSuccess, onError } = options

  // État local
  const [state, setState] = useState<AsyncState<Tender>>({
    data: null,
    loading: false,
    error: null,
  })

  // Store global
  const { 
    tenders, 
    setTender, 
    updateTender,
    selectedTenderId,
    setSelectedTenderId 
  } = useTenderStore()

  // Récupérer le tender depuis le store si disponible
  const cachedTender = useMemo(
    () => tenderId ? tenders[tenderId] : null,
    [tenders, tenderId]
  )

  /**
   * Charger le tender
   */
  const fetchTender = useCallback(async () => {
    if (!tenderId) return

    setState(prev => ({ ...prev, loading: true, error: null }))

    try {
      const response = await api.tenders.getById(tenderId)
      
      if (isApiSuccess(response) && response.data) {
        setState({
          data: response.data,
          loading: false,
          error: null,
        })
        
        // Mettre à jour le store
        setTender(response.data)
        
        onSuccess?.(response.data)
      } else {
        throw new Error('Failed to fetch tender')
      }
    } catch (error) {
      const err = error as Error
      setState({
        data: null,
        loading: false,
        error: err,
      })
      
      onError?.(err)
      toast.error(err.message || 'Failed to load tender')
    }
  }, [tenderId, setTender, onSuccess, onError])

  /**
   * Rafraîchir le tender
   */
  const refresh = useCallback(() => {
    return fetchTender()
  }, [fetchTender])

  /**
   * Mettre à jour le tender
   */
  const update = useCallback(async (data: Partial<Tender>) => {
    if (!tenderId) return

    // Mutation optimiste
    const optimisticTender = { ...cachedTender, ...data } as Tender
    updateTender(tenderId, data)

    try {
      const response = await api.tenders.update(tenderId, data)
      
      if (isApiSuccess(response) && response.data) {
        setTender(response.data)
        toast.success('Tender updated successfully')
        return response.data
      } else {
        throw new Error('Failed to update tender')
      }
    } catch (error) {
      // Rollback en cas d'erreur
      if (cachedTender) {
        setTender(cachedTender)
      }
      
      const err = error as Error
      toast.error(err.message || 'Failed to update tender')
      throw error
    }
  }, [tenderId, cachedTender, updateTender, setTender])

  /**
   * Supprimer le tender
   */
  const remove = useCallback(async () => {
    if (!tenderId) return

    const confirmed = window.confirm('Are you sure you want to delete this tender?')
    if (!confirmed) return

    try {
      await api.tenders.delete(tenderId)
      
      // Retirer du store
      useTenderStore.getState().removeTender(tenderId)
      
      toast.success('Tender deleted successfully')
      
      // Rediriger vers la liste
      if (typeof window !== 'undefined') {
        window.location.href = '/dashboard'
      }
    } catch (error) {
      const err = error as Error
      toast.error(err.message || 'Failed to delete tender')
      throw error
    }
  }, [tenderId])

  /**
   * Démarrer l'analyse
   */
  const startAnalysis = useCallback(async (config?: AnalysisConfig) => {
    if (!tenderId) return

    const toastId = toast.loading('Starting analysis...')

    try {
      const response = await api.analysis.start(tenderId, config)
      
      if (isApiSuccess(response) && response.data) {
        toast.success('Analysis started', { id: toastId })
        
        // Mettre à jour le tender avec l'ID d'analyse
        updateTender(tenderId, { 
          analysisId: response.data.id,
          status: Status.PROCESSING 
        })
        
        return response.data
      } else {
        throw new Error('Failed to start analysis')
      }
    } catch (error) {
      const err = error as Error
      toast.error(err.message || 'Failed to start analysis', { id: toastId })
      throw error
    }
  }, [tenderId, updateTender])

  /**
   * Ajouter des documents
   */
  const addDocuments = useCallback(async (
    files: File[],
    onProgress?: (progress: number) => void
  ) => {
    if (!tenderId) return

    const toastId = toast.loading('Uploading documents...')

    try {
      const response = await api.tenders.addDocuments(tenderId, files, onProgress)
      
      if (isApiSuccess(response) && response.data) {
        toast.success(`${files.length} documents uploaded`, { id: toastId })
        
        // Rafraîchir le tender pour avoir les nouveaux documents
        await refresh()
        
        return response.data
      } else {
        throw new Error('Failed to upload documents')
      }
    } catch (error) {
      const err = error as Error
      toast.error(err.message || 'Failed to upload documents', { id: toastId })
      throw error
    }
  }, [tenderId, refresh])

  // Auto-fetch au montage
  useEffect(() => {
    if (autoFetch && tenderId && !cachedTender) {
      fetchTender()
    } else if (cachedTender && !state.data) {
      setState({
        data: cachedTender,
        loading: false,
        error: null,
      })
    }
  }, [tenderId, autoFetch]) // eslint-disable-line react-hooks/exhaustive-deps

  // Polling optionnel
  useEffect(() => {
    if (!pollInterval || !tenderId) return

    const interval = setInterval(() => {
      fetchTender()
    }, pollInterval)

    return () => clearInterval(interval)
  }, [pollInterval, tenderId, fetchTender])

  // Sélectionner automatiquement le tender
  useEffect(() => {
    if (tenderId && tenderId !== selectedTenderId) {
      setSelectedTenderId(tenderId)
    }
  }, [tenderId, selectedTenderId, setSelectedTenderId])

  return {
    tender: state.data || cachedTender,
    loading: state.loading,
    error: state.error,
    refresh,
    update,
    remove,
    startAnalysis,
    addDocuments,
  }
}

// ============================================
// Hook pour la liste des tenders
// ============================================

interface UseTendersOptions {
  page?: number
  pageSize?: number
  status?: Status
  search?: string
  autoFetch?: boolean
}

export function useTenders(options: UseTendersOptions = {}) {
  const {
    page = 1,
    pageSize = 10,
    status,
    search,
    autoFetch = true,
  } = options

  // État
  const [tenders, setTenders] = useState<TenderSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [total, setTotal] = useState(0)
  const [hasMore, setHasMore] = useState(false)

  /**
   * Charger la liste
   */
  const fetchTenders = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await api.tenders.getAll({
        page,
        pageSize,
        status,
        search,
      })

      if (isApiSuccess(response) && response.data) {
        setTenders(response.data.items)
        setTotal(response.data.total)
        setHasMore(response.data.hasMore)
      } else {
        throw new Error('Failed to fetch tenders')
      }
    } catch (error) {
      const err = error as Error
      setError(err)
      toast.error(err.message || 'Failed to load tenders')
    } finally {
      setLoading(false)
    }
  }, [page, pageSize, status, search])

  /**
   * Créer un nouveau tender
   */
  const createTender = useCallback(async (
    data: {
      title: string
      reference: string
      client: string
      description?: string
      files: File[]
    },
    onProgress?: (progress: number) => void
  ) => {
    const toastId = toast.loading('Creating tender...')

    try {
      const response = await api.tenders.create(data, onProgress)

      if (isApiSuccess(response) && response.data) {
        toast.success('Tender created successfully', { id: toastId })
        
        // Ajouter au store
        useTenderStore.getState().setTender(response.data)
        
        // Rafraîchir la liste
        await fetchTenders()
        
        return response.data
      } else {
        throw new Error('Failed to create tender')
      }
    } catch (error) {
      const err = error as Error
      toast.error(err.message || 'Failed to create tender', { id: toastId })
      throw error
    }
  }, [fetchTenders])

  // Auto-fetch
  useEffect(() => {
    if (autoFetch) {
      fetchTenders()
    }
  }, [autoFetch]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    tenders,
    loading,
    error,
    total,
    hasMore,
    page,
    pageSize,
    refresh: fetchTenders,
    create: createTender,
  }
}

// ============================================
// Hook pour l'analyse d'un tender
// ============================================

interface UseAnalysisOptions {
  autoFetch?: boolean
  pollInterval?: number
}

export function useAnalysis(
  analysisId: UUID | null,
  options: UseAnalysisOptions = {}
) {
  const { autoFetch = true, pollInterval = 5000 } = options

  // État
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  /**
   * Charger l'analyse
   */
  const fetchAnalysis = useCallback(async () => {
    if (!analysisId) return

    setLoading(true)
    setError(null)

    try {
      const response = await api.analysis.getById(analysisId)

      if (isApiSuccess(response) && response.data) {
        setAnalysis(response.data)
        
        // Arrêter le polling si terminé
        if (response.data.status === Status.COMPLETED || 
            response.data.status === Status.FAILED) {
          return true // Signal pour arrêter le polling
        }
      } else {
        throw new Error('Failed to fetch analysis')
      }
    } catch (error) {
      const err = error as Error
      setError(err)
      toast.error(err.message || 'Failed to load analysis')
    } finally {
      setLoading(false)
    }

    return false
  }, [analysisId])

  /**
   * Annuler l'analyse
   */
  const cancelAnalysis = useCallback(async () => {
    if (!analysisId) return

    const confirmed = window.confirm('Are you sure you want to cancel this analysis?')
    if (!confirmed) return

    try {
      await api.analysis.cancel(analysisId)
      toast.success('Analysis cancelled')
      
      // Mettre à jour l'état
      setAnalysis(prev => prev ? {
        ...prev,
        status: Status.CANCELLED
      } : null)
    } catch (error) {
      const err = error as Error
      toast.error(err.message || 'Failed to cancel analysis')
      throw error
    }
  }, [analysisId])

  /**
   * Récupérer les résultats d'un agent
   */
  const getAgentResult = useCallback(async (agent: AgentType) => {
    if (!analysisId) return null

    try {
      const response = await api.agents.getAgentResult(analysisId, agent)
      
      if (isApiSuccess(response)) {
        return response.data
      }
    } catch (error) {
      console.error(`Failed to get ${agent} results:`, error)
    }

    return null
  }, [analysisId])

  // Auto-fetch
  useEffect(() => {
    if (autoFetch && analysisId) {
      fetchAnalysis()
    }
  }, [analysisId, autoFetch]) // eslint-disable-line react-hooks/exhaustive-deps

  // Polling pour les analyses en cours
  useEffect(() => {
    if (!pollInterval || !analysisId) return
    if (analysis?.status === Status.COMPLETED || 
        analysis?.status === Status.FAILED ||
        analysis?.status === Status.CANCELLED) return

    const interval = setInterval(async () => {
      const shouldStop = await fetchAnalysis()
      if (shouldStop) {
        clearInterval(interval)
      }
    }, pollInterval)

    return () => clearInterval(interval)
  }, [pollInterval, analysisId, analysis?.status, fetchAnalysis])

  return {
    analysis,
    loading,
    error,
    refresh: fetchAnalysis,
    cancel: cancelAnalysis,
    getAgentResult,
    isProcessing: analysis?.status === Status.PROCESSING,
    isCompleted: analysis?.status === Status.COMPLETED,
    isFailed: analysis?.status === Status.FAILED,
  }
}

export default useTender