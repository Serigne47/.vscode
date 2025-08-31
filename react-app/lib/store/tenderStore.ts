/**
 * Store global pour la gestion de l'état de l'application
 * Utilise Zustand pour un state management simple et performant
 */

import { create } from 'zustand'
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import {
  Tender,
  Analysis,
  UUID,
  Status,
  AgentType,
  AgentProgress,
  ConfidenceLevel,
  UIState,
  NotificationItem,
  Risk,
  Finding,
  Recommendation,
} from '@/lib/types'

// ============================================
// Types du Store
// ============================================

interface TenderState {
  // Tenders
  tenders: Record<UUID, Tender>
  selectedTenderId: UUID | null
  
  // Analysis
  analyses: Record<UUID, Analysis>
  currentAnalysisId: UUID | null
  agentProgress: Record<string, AgentProgress>
  
  // UI State
  ui: UIState
  
  // Notifications
  notifications: NotificationItem[]
  
  // Filtres et recherche
  searchQuery: string
  filters: {
    status: Status[]
    dateRange: { start: string; end: string } | null
    confidenceThreshold: number
  }
  
  // Résultats mis en cache
  cachedResults: {
    risks: Risk[]
    findings: Finding[]
    recommendations: Recommendation[]
    lastUpdated: string | null
  }
}

interface TenderActions {
  // Tenders
  setTender: (tender: Tender) => void
  setTenders: (tenders: Tender[]) => void
  updateTender: (id: UUID, updates: Partial<Tender>) => void
  removeTender: (id: UUID) => void
  selectTender: (id: UUID | null) => void
  setSelectedTenderId: (id: UUID | null) => void
  
  // Analysis
  setAnalysis: (analysis: Analysis) => void
  updateAnalysis: (id: UUID, updates: Partial<Analysis>) => void
  setCurrentAnalysisId: (id: UUID | null) => void
  updateAgentProgress: (agent: string, progress: AgentProgress) => void
  resetAgentProgress: () => void
  
  // UI State
  setUIState: (updates: Partial<UIState>) => void
  toggleSidebar: () => void
  setActiveAgent: (agent: AgentType | undefined) => void
  setViewMode: (mode: UIState['viewMode']) => void
  setLoading: (loading: boolean) => void
  
  // Notifications
  addNotification: (notification: Omit<NotificationItem, 'id' | 'timestamp' | 'read'>) => void
  markNotificationAsRead: (id: UUID) => void
  removeNotification: (id: UUID) => void
  clearNotifications: () => void
  
  // Filtres
  setSearchQuery: (query: string) => void
  setStatusFilter: (statuses: Status[]) => void
  setDateRangeFilter: (range: { start: string; end: string } | null) => void
  setConfidenceThreshold: (threshold: number) => void
  resetFilters: () => void
  
  // Cache
  setCachedResults: (results: Partial<TenderState['cachedResults']>) => void
  clearCache: () => void
  
  // Utils
  reset: () => void
  getTenderById: (id: UUID) => Tender | undefined
  getAnalysisById: (id: UUID) => Analysis | undefined
  getSelectedTender: () => Tender | undefined
  getCurrentAnalysis: () => Analysis | undefined
}

type TenderStore = TenderState & TenderActions

// ============================================
// État initial
// ============================================

const initialState: TenderState = {
  // Tenders
  tenders: {},
  selectedTenderId: null,
  
  // Analysis
  analyses: {},
  currentAnalysisId: null,
  agentProgress: {},
  
  // UI State
  ui: {
    isLoading: false,
    isSidebarOpen: true,
    activeAgent: undefined,
    selectedTenderId: undefined,
    viewMode: 'grid',
    filters: {
      status: undefined,
      agents: undefined,
      dateRange: undefined,
      confidenceThreshold: undefined,
    },
  },
  
  // Notifications
  notifications: [],
  
  // Filtres
  searchQuery: '',
  filters: {
    status: [],
    dateRange: null,
    confidenceThreshold: 0.5,
  },
  
  // Cache
  cachedResults: {
    risks: [],
    findings: [],
    recommendations: [],
    lastUpdated: null,
  },
}

// ============================================
// Création du Store
// ============================================

export const useTenderStore = create<TenderStore>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get) => ({
          ...initialState,

          // ========================================
          // Actions Tenders
          // ========================================
          
          setTender: (tender) =>
            set((state) => {
              state.tenders[tender.id] = tender
            }),

          setTenders: (tenders) =>
            set((state) => {
              tenders.forEach((tender) => {
                state.tenders[tender.id] = tender
              })
            }),

          updateTender: (id, updates) =>
            set((state) => {
              if (state.tenders[id]) {
                Object.assign(state.tenders[id], updates)
              }
            }),

          removeTender: (id) =>
            set((state) => {
              delete state.tenders[id]
              if (state.selectedTenderId === id) {
                state.selectedTenderId = null
              }
            }),

          selectTender: (id) =>
            set((state) => {
              state.selectedTenderId = id
              if (id) {
                state.ui.selectedTenderId = id
              }
            }),

          setSelectedTenderId: (id) =>
            set((state) => {
              state.selectedTenderId = id
              state.ui.selectedTenderId = id || undefined
            }),

          // ========================================
          // Actions Analysis
          // ========================================
          
          setAnalysis: (analysis) =>
            set((state) => {
              state.analyses[analysis.id] = analysis
            }),

          updateAnalysis: (id, updates) =>
            set((state) => {
              if (state.analyses[id]) {
                Object.assign(state.analyses[id], updates)
              }
            }),

          setCurrentAnalysisId: (id) =>
            set((state) => {
              state.currentAnalysisId = id
            }),

          updateAgentProgress: (agent, progress) =>
            set((state) => {
              state.agentProgress[agent] = progress
            }),

          resetAgentProgress: () =>
            set((state) => {
              state.agentProgress = {}
            }),

          // ========================================
          // Actions UI State
          // ========================================
          
          setUIState: (updates) =>
            set((state) => {
              Object.assign(state.ui, updates)
            }),

          toggleSidebar: () =>
            set((state) => {
              state.ui.isSidebarOpen = !state.ui.isSidebarOpen
            }),

          setActiveAgent: (agent) =>
            set((state) => {
              state.ui.activeAgent = agent
            }),

          setViewMode: (mode) =>
            set((state) => {
              state.ui.viewMode = mode
            }),

          setLoading: (loading) =>
            set((state) => {
              state.ui.isLoading = loading
            }),

          // ========================================
          // Actions Notifications
          // ========================================
          
          addNotification: (notification) =>
            set((state) => {
              const id = crypto.randomUUID()
              state.notifications.unshift({
                ...notification,
                id,
                timestamp: new Date().toISOString(),
                read: false,
              })
              
              // Limiter à 50 notifications
              if (state.notifications.length > 50) {
                state.notifications = state.notifications.slice(0, 50)
              }
            }),

          markNotificationAsRead: (id) =>
            set((state) => {
              const notification = state.notifications.find((n) => n.id === id)
              if (notification) {
                notification.read = true
              }
            }),

          removeNotification: (id) =>
            set((state) => {
              state.notifications = state.notifications.filter((n) => n.id !== id)
            }),

          clearNotifications: () =>
            set((state) => {
              state.notifications = []
            }),

          // ========================================
          // Actions Filtres
          // ========================================
          
          setSearchQuery: (query) =>
            set((state) => {
              state.searchQuery = query
            }),

          setStatusFilter: (statuses) =>
            set((state) => {
              state.filters.status = statuses
              state.ui.filters.status = statuses.length > 0 ? statuses : undefined
            }),

          setDateRangeFilter: (range) =>
            set((state) => {
              state.filters.dateRange = range
              state.ui.filters.dateRange = range || undefined
            }),

          setConfidenceThreshold: (threshold) =>
            set((state) => {
              state.filters.confidenceThreshold = threshold
              state.ui.filters.confidenceThreshold = threshold
            }),

          resetFilters: () =>
            set((state) => {
              state.searchQuery = ''
              state.filters = initialState.filters
              state.ui.filters = {}
            }),

          // ========================================
          // Actions Cache
          // ========================================
          
          setCachedResults: (results) =>
            set((state) => {
              Object.assign(state.cachedResults, results)
              state.cachedResults.lastUpdated = new Date().toISOString()
            }),

          clearCache: () =>
            set((state) => {
              state.cachedResults = initialState.cachedResults
            }),

          // ========================================
          // Actions Utils
          // ========================================
          
          reset: () => set(() => initialState),

          getTenderById: (id) => get().tenders[id],

          getAnalysisById: (id) => get().analyses[id],

          getSelectedTender: () => {
            const state = get()
            return state.selectedTenderId
              ? state.tenders[state.selectedTenderId]
              : undefined
          },

          getCurrentAnalysis: () => {
            const state = get()
            return state.currentAnalysisId
              ? state.analyses[state.currentAnalysisId]
              : undefined
          },
        }))
      ),
      {
        name: 'tender-store',
        // Persister seulement certaines parties du state
        partialize: (state) => ({
          ui: {
            isSidebarOpen: state.ui.isSidebarOpen,
            viewMode: state.ui.viewMode,
          },
          filters: state.filters,
        }),
      }
    ),
    {
      name: 'TenderStore',
    }
  )
)

// ============================================
// Sélecteurs dérivés (pour optimisation)
// ============================================

export const useTenderById = (id: UUID | null) => {
  return useTenderStore((state) => (id ? state.tenders[id] : undefined))
}

export const useAnalysisById = (id: UUID | null) => {
  return useTenderStore((state) => (id ? state.analyses[id] : undefined))
}

export const useSelectedTender = () => {
  return useTenderStore((state) =>
    state.selectedTenderId ? state.tenders[state.selectedTenderId] : undefined
  )
}

export const useCurrentAnalysis = () => {
  return useTenderStore((state) =>
    state.currentAnalysisId ? state.analyses[state.currentAnalysisId] : undefined
  )
}

export const useAgentProgress = (agent: string) => {
  return useTenderStore((state) => state.agentProgress[agent])
}

export const useOverallProgress = () => {
  return useTenderStore((state) => {
    const agents = Object.values(state.agentProgress)
    if (agents.length === 0) return 0
    
    const total = agents.reduce((sum, agent) => sum + agent.progress, 0)
    return Math.round(total / agents.length)
  })
}

export const useUnreadNotifications = () => {
  return useTenderStore((state) => 
    state.notifications.filter((n) => !n.read).length
  )
}

export const useFilteredTenders = () => {
  return useTenderStore((state) => {
    let tenders = Object.values(state.tenders)

    // Filtre par recherche
    if (state.searchQuery) {
      const query = state.searchQuery.toLowerCase()
      tenders = tenders.filter(
        (t) =>
          t.title.toLowerCase().includes(query) ||
          t.reference.toLowerCase().includes(query) ||
          t.client.toLowerCase().includes(query)
      )
    }

    // Filtre par statut
    if (state.filters.status.length > 0) {
      tenders = tenders.filter((t) => state.filters.status.includes(t.status))
    }

    // Filtre par date
    if (state.filters.dateRange) {
      const { start, end } = state.filters.dateRange
      tenders = tenders.filter((t) => {
        const date = new Date(t.createdAt)
        return date >= new Date(start) && date <= new Date(end)
      })
    }

    return tenders
  })
}

// ============================================
// Actions externes (pour composants)
// ============================================

export const tenderActions = {
  selectAndNavigate: (tenderId: UUID) => {
    useTenderStore.getState().selectTender(tenderId)
    if (typeof window !== 'undefined') {
      window.location.href = `/dashboard/${tenderId}`
    }
  },

  startNewAnalysis: async (tenderId: UUID, config?: any) => {
    const store = useTenderStore.getState()
    store.setLoading(true)
    
    try {
      // L'appel API sera fait via le hook useTender
      store.updateTender(tenderId, { status: Status.PROCESSING })
    } finally {
      store.setLoading(false)
    }
  },

  notifySuccess: (title: string, message?: string) => {
    useTenderStore.getState().addNotification({
      type: 'success',
      title,
      message,
    })
  },

  notifyError: (title: string, message?: string) => {
    useTenderStore.getState().addNotification({
      type: 'error',
      title,
      message,
    })
  },

  notifyInfo: (title: string, message?: string) => {
    useTenderStore.getState().addNotification({
      type: 'info',
      title,
      message,
    })
  },
}

export default useTenderStore