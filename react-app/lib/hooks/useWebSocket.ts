/**
 * Hook React pour gérer les connexions WebSocket
 * Avec reconnection automatique et gestion d'état
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { 
  WebSocketMessage, 
  ProgressUpdate, 
  AgentProgress,
  Status,
  UUID 
} from '@/lib/types'

// Configuration WebSocket
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://127.0.0.1:8000'
const RECONNECT_INTERVAL = 3000 // 3 secondes
const MAX_RECONNECT_ATTEMPTS = 5
const HEARTBEAT_INTERVAL = 30000 // 30 secondes
const CONNECTION_TIMEOUT = 10000 // 10 secondes

// Types pour le hook
export interface WebSocketState {
  isConnected: boolean
  isConnecting: boolean
  error: Error | null
  lastMessage: WebSocketMessage | null
  messageHistory: WebSocketMessage[]
  reconnectAttempts: number
}

export interface WebSocketOptions {
  url?: string
  autoConnect?: boolean
  reconnect?: boolean
  maxReconnectAttempts?: number
  onOpen?: (event: Event) => void
  onClose?: (event: CloseEvent) => void
  onError?: (event: Event) => void
  onMessage?: (message: WebSocketMessage) => void
  heartbeat?: boolean
  debug?: boolean
}

export interface WebSocketActions {
  connect: () => void
  disconnect: () => void
  send: (data: any) => void
  reconnect: () => void
  clearHistory: () => void
}

/**
 * Hook principal pour WebSocket
 */
export function useWebSocket(
  endpoint: string,
  options: WebSocketOptions = {}
): [WebSocketState, WebSocketActions] {
  // Configuration avec valeurs par défaut
  const {
    url = WS_BASE_URL,
    autoConnect = true,
    reconnect = true,
    maxReconnectAttempts = MAX_RECONNECT_ATTEMPTS,
    onOpen,
    onClose,
    onError,
    onMessage,
    heartbeat = true,
    debug = process.env.NODE_ENV === 'development',
  } = options

  // État
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastMessage: null,
    messageHistory: [],
    reconnectAttempts: 0,
  })

  // Refs pour éviter les re-renders
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null)
  const heartbeatTimer = useRef<NodeJS.Timeout | null>(null)
  const connectionTimer = useRef<NodeJS.Timeout | null>(null)
  const mounted = useRef(true)

  // Logger conditionnel
  const log = useCallback((message: string, data?: any) => {
    if (debug) {
      console.log(`[WebSocket] ${message}`, data || '')
    }
  }, [debug])

  /**
   * Établir la connexion WebSocket
   */
  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      log('Already connected')
      return
    }

    if (ws.current?.readyState === WebSocket.CONNECTING) {
      log('Connection in progress')
      return
    }

    log('Connecting...', `${url}${endpoint}`)

    setState(prev => ({
      ...prev,
      isConnecting: true,
      error: null,
    }))

    try {
      // Créer la connexion
      ws.current = new WebSocket(`${url}${endpoint}`)

      // Timeout de connexion
      connectionTimer.current = setTimeout(() => {
        if (ws.current?.readyState !== WebSocket.OPEN) {
          log('Connection timeout')
          ws.current?.close()
          setState(prev => ({
            ...prev,
            isConnecting: false,
            error: new Error('Connection timeout'),
          }))
          handleReconnect()
        }
      }, CONNECTION_TIMEOUT)

      // Gestionnaire d'ouverture
      ws.current.onopen = (event) => {
        log('Connected')
        
        if (connectionTimer.current) {
          clearTimeout(connectionTimer.current)
        }

        setState(prev => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          error: null,
          reconnectAttempts: 0,
        }))

        // Démarrer le heartbeat
        if (heartbeat) {
          startHeartbeat()
        }

        onOpen?.(event)
      }

      // Gestionnaire de messages
      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          
          log('Message received', message)

          // Ignorer les pongs du heartbeat
          if (message.type === 'pong') return

          setState(prev => ({
            ...prev,
            lastMessage: message,
            messageHistory: [...prev.messageHistory, message].slice(-100), // Garder les 100 derniers
          }))

          onMessage?.(message)
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error)
        }
      }

      // Gestionnaire d'erreur
      ws.current.onerror = (event) => {
        log('Error', event)
        
        setState(prev => ({
          ...prev,
          error: new Error('WebSocket error'),
        }))

        onError?.(event)
      }

      // Gestionnaire de fermeture
      ws.current.onclose = (event) => {
        log('Disconnected', { code: event.code, reason: event.reason })
        
        if (connectionTimer.current) {
          clearTimeout(connectionTimer.current)
        }

        stopHeartbeat()

        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
        }))

        onClose?.(event)

        // Tentative de reconnexion si nécessaire
        if (mounted.current && reconnect && !event.wasClean) {
          handleReconnect()
        }
      }
    } catch (error) {
      log('Connection error', error)
      
      setState(prev => ({
        ...prev,
        isConnecting: false,
        error: error as Error,
      }))
    }
  }, [url, endpoint, reconnect, heartbeat, onOpen, onClose, onError, onMessage, log])

  /**
   * Fermer la connexion
   */
  const disconnect = useCallback(() => {
    log('Disconnecting...')
    
    mounted.current = false
    
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }

    if (connectionTimer.current) {
      clearTimeout(connectionTimer.current)
      connectionTimer.current = null
    }

    stopHeartbeat()

    if (ws.current) {
      ws.current.close(1000, 'User disconnect')
      ws.current = null
    }

    setState(prev => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
    }))
  }, [log])

  /**
   * Envoyer un message
   */
  const send = useCallback((data: any) => {
    if (ws.current?.readyState !== WebSocket.OPEN) {
      console.error('[WebSocket] Cannot send message: not connected')
      return
    }

    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data)
      ws.current.send(message)
      log('Message sent', data)
    } catch (error) {
      console.error('[WebSocket] Failed to send message:', error)
    }
  }, [log])

  /**
   * Gérer la reconnexion
   */
  const handleReconnect = useCallback(() => {
    if (!mounted.current || !reconnect) return

    setState(prev => {
      if (prev.reconnectAttempts >= maxReconnectAttempts) {
        toast.error('Failed to connect to server. Please refresh the page.')
        return prev
      }

      const attempts = prev.reconnectAttempts + 1
      log(`Reconnecting... (attempt ${attempts}/${maxReconnectAttempts})`)

      reconnectTimer.current = setTimeout(() => {
        if (mounted.current) {
          connect()
        }
      }, RECONNECT_INTERVAL)

      return {
        ...prev,
        reconnectAttempts: attempts,
      }
    })
  }, [reconnect, maxReconnectAttempts, connect, log])

  /**
   * Forcer la reconnexion
   */
  const reconnectManual = useCallback(() => {
    log('Manual reconnect')
    
    setState(prev => ({
      ...prev,
      reconnectAttempts: 0,
    }))

    disconnect()
    
    setTimeout(() => {
      mounted.current = true
      connect()
    }, 100)
  }, [connect, disconnect, log])

  /**
   * Heartbeat pour maintenir la connexion
   */
  const startHeartbeat = useCallback(() => {
    stopHeartbeat()
    
    heartbeatTimer.current = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        send({ type: 'ping', timestamp: new Date().toISOString() })
      }
    }, HEARTBEAT_INTERVAL)
  }, [send])

  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimer.current) {
      clearInterval(heartbeatTimer.current)
      heartbeatTimer.current = null
    }
  }, [])

  /**
   * Nettoyer l'historique des messages
   */
  const clearHistory = useCallback(() => {
    setState(prev => ({
      ...prev,
      messageHistory: [],
      lastMessage: null,
    }))
  }, [])

  // Auto-connexion au montage
  useEffect(() => {
    mounted.current = true

    if (autoConnect) {
      connect()
    }

    return () => {
      mounted.current = false
      disconnect()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Actions exposées
  const actions: WebSocketActions = {
    connect,
    disconnect,
    send,
    reconnect: reconnectManual,
    clearHistory,
  }

  return [state, actions]
}

/**
 * Hook spécialisé pour suivre l'analyse
 */
export function useAnalysisWebSocket(
  analysisId: UUID | null,
  options?: {
    onProgress?: (update: ProgressUpdate) => void
    onComplete?: (results: any) => void
    onError?: (error: any) => void
  }
) {
  const endpoint = analysisId ? `/ws/analysis/${analysisId}` : ''
  
  const [state, actions] = useWebSocket(endpoint, {
    autoConnect: !!analysisId,
    onMessage: (message) => {
      switch (message.type) {
        case 'progress':
          options?.onProgress?.(message.data as ProgressUpdate)
          break
        case 'complete':
          options?.onComplete?.(message.data)
          toast.success('Analysis completed successfully!')
          break
        case 'error':
          options?.onError?.(message.data)
          toast.error('Analysis failed. Please try again.')
          break
      }
    },
  })

  return {
    ...state,
    ...actions,
    isAnalyzing: state.isConnected && analysisId !== null,
  }
}

/**
 * Hook pour suivre le progrès des agents
 */
export function useAgentProgress(analysisId: UUID | null) {
  const [agentProgress, setAgentProgress] = useState<Record<string, AgentProgress>>({})
  const [overallProgress, setOverallProgress] = useState(0)

  const { isConnected } = useAnalysisWebSocket(analysisId, {
    onProgress: (update) => {
      setAgentProgress(prev => ({
        ...prev,
        [update.agent]: {
          agent: update.agent,
          status: Status.PROCESSING,
          progress: update.progress,
          startedAt: prev[update.agent]?.startedAt || new Date().toISOString(),
        },
      }))

      // Calculer le progrès global
      const agents = Object.values(agentProgress)
      if (agents.length > 0) {
        const total = agents.reduce((sum, agent) => sum + agent.progress, 0)
        setOverallProgress(Math.round(total / agents.length))
      }
    },
    onComplete: () => {
      // Marquer tous les agents comme terminés
      setAgentProgress(prev => {
        const updated: Record<string, AgentProgress> = {}
        Object.keys(prev).forEach(key => {
          updated[key] = {
            ...prev[key],
            status: Status.COMPLETED,
            progress: 100,
            completedAt: new Date().toISOString(),
          }
        })
        return updated
      })
      setOverallProgress(100)
    },
  })

  return {
    agentProgress,
    overallProgress,
    isConnected,
    reset: () => {
      setAgentProgress({})
      setOverallProgress(0)
    },
  }
}

export default useWebSocket