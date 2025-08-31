/**
 * Client API principal avec gestion des erreurs et intercepteurs
 * Utilise Axios pour les requêtes HTTP avec retry logic et caching
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios'
import toast from 'react-hot-toast'
import { ApiResponse, ApiError } from '@/lib/types'

// Configuration de base
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
const API_TIMEOUT = 30000 // 30 seconds
const MAX_RETRIES = 3
const RETRY_DELAY = 1000 // 1 second

// Types pour le cache
interface CacheEntry {
  data: any
  timestamp: number
  ttl: number
}

// Cache en mémoire simple
class SimpleCache {
  private cache: Map<string, CacheEntry> = new Map()

  set(key: string, data: any, ttl: number = 60000): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    })
  }

  get(key: string): any | null {
    const entry = this.cache.get(key)
    if (!entry) return null

    const now = Date.now()
    if (now - entry.timestamp > entry.ttl) {
      this.cache.delete(key)
      return null
    }

    return entry.data
  }

  clear(): void {
    this.cache.clear()
  }

  delete(pattern: string): void {
    const keys = Array.from(this.cache.keys())
    keys.forEach(key => {
      if (key.includes(pattern)) {
        this.cache.delete(key)
      }
    })
  }
}

// Instance du cache
const apiCache = new SimpleCache()

// Classe principale du client API
class ApiClient {
  private client: AxiosInstance
  private authToken: string | null = null
  private refreshPromise: Promise<string> | null = null

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      withCredentials: false, // Changez à true si vous utilisez des cookies
    })

    this.setupInterceptors()
  }

  /**
   * Configuration des intercepteurs pour requêtes et réponses
   */
  private setupInterceptors(): void {
    // Intercepteur de requête
    this.client.interceptors.request.use(
      (config) => {
        // Ajouter le token d'authentification si disponible
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`
        }

        // Ajouter un ID de requête unique pour le tracking
        config.headers['X-Request-ID'] = this.generateRequestId()

        // Log en développement
        if (process.env.NODE_ENV === 'development') {
          console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, {
            params: config.params,
            data: config.data,
          })
        }

        return config
      },
      (error) => {
        console.error('[API] Request error:', error)
        return Promise.reject(error)
      }
    )

    // Intercepteur de réponse
    this.client.interceptors.response.use(
      (response) => {
        // Log en développement
        if (process.env.NODE_ENV === 'development') {
          console.log(`[API] Response:`, response.data)
        }

        return response
      },
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: number }

        // Gestion du retry
        if (error.response?.status === 503 || error.code === 'ECONNABORTED') {
          originalRequest._retry = (originalRequest._retry || 0) + 1

          if (originalRequest._retry <= MAX_RETRIES) {
            await this.delay(RETRY_DELAY * originalRequest._retry)
            return this.client(originalRequest)
          }
        }

        // Gestion de l'authentification expirée
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = 1

          try {
            const newToken = await this.refreshToken()
            this.setAuthToken(newToken)
            
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${newToken}`
            }
            
            return this.client(originalRequest)
          } catch (refreshError) {
            this.handleAuthError()
            return Promise.reject(refreshError)
          }
        }

        // Gestion des erreurs
        this.handleApiError(error)
        return Promise.reject(this.formatError(error))
      }
    )
  }

  /**
   * Méthodes HTTP principales avec support du cache
   */
  async get<T = any>(
    url: string, 
    config?: AxiosRequestConfig & { cache?: boolean; cacheTTL?: number }
  ): Promise<ApiResponse<T>> {
    const cacheKey = `GET:${url}:${JSON.stringify(config?.params)}`
    
    // Vérifier le cache
    if (config?.cache !== false) {
      const cached = apiCache.get(cacheKey)
      if (cached) {
        return { success: true, data: cached }
      }
    }

    try {
      const response = await this.client.get<T>(url, config)
      const result = this.formatResponse<T>(response)
      
      // Mettre en cache si succès
      if (result.success && config?.cache !== false) {
        apiCache.set(cacheKey, result.data, config?.cacheTTL)
      }
      
      return result
    } catch (error) {
      throw error
    }
  }

  async post<T = any>(
    url: string, 
    data?: any, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    // Invalider le cache pour cette ressource
    apiCache.delete(url)
    
    const response = await this.client.post<T>(url, data, config)
    return this.formatResponse<T>(response)
  }

  async put<T = any>(
    url: string, 
    data?: any, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    apiCache.delete(url)
    
    const response = await this.client.put<T>(url, data, config)
    return this.formatResponse<T>(response)
  }

  async patch<T = any>(
    url: string, 
    data?: any, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    apiCache.delete(url)
    
    const response = await this.client.patch<T>(url, data, config)
    return this.formatResponse<T>(response)
  }

  async delete<T = any>(
    url: string, 
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    apiCache.delete(url)
    
    const response = await this.client.delete<T>(url, config)
    return this.formatResponse<T>(response)
  }

  /**
   * Upload de fichiers avec progress
   */
  async upload<T = any>(
    url: string,
    files: File[],
    data?: Record<string, any>,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<T>> {
    const formData = new FormData()
    
    // Ajouter les fichiers
    files.forEach((file, index) => {
      formData.append(`files`, file)
    })
    
    // Ajouter les données additionnelles
    if (data) {
      Object.keys(data).forEach(key => {
        formData.append(key, typeof data[key] === 'object' 
          ? JSON.stringify(data[key]) 
          : data[key]
        )
      })
    }

    const response = await this.client.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress?.(progress)
        }
      },
    })

    return this.formatResponse<T>(response)
  }

  /**
   * Téléchargement de fichiers
   */
  async download(
    url: string,
    filename?: string,
    onProgress?: (progress: number) => void
  ): Promise<void> {
    const response = await this.client.get(url, {
      responseType: 'blob',
      onDownloadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress?.(progress)
        }
      },
    })

    // Créer un lien de téléchargement
    const blob = new Blob([response.data])
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename || this.extractFilename(response) || 'download'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  }

  /**
   * Méthodes utilitaires
   */
  private formatResponse<T>(response: AxiosResponse<T>): ApiResponse<T> {
    return {
      success: true,
      data: response.data,
      metadata: {
        timestamp: new Date().toISOString(),
        version: response.headers['x-api-version'] || '1.0.0',
        requestId: response.headers['x-request-id'] || this.generateRequestId(),
      },
    }
  }

  private formatError(error: AxiosError): ApiError {
    if (error.response) {
      const data = error.response.data as any
      return {
        code: data?.code || error.code || 'UNKNOWN_ERROR',
        message: data?.message || error.message || 'An unexpected error occurred',
        details: data?.details || data,
        statusCode: error.response.status,
      }
    }

    return {
      code: error.code || 'NETWORK_ERROR',
      message: error.message || 'Network error occurred',
      details: null,
      statusCode: 0,
    }
  }

  private handleApiError(error: AxiosError): void {
    const status = error.response?.status

    // Ne pas afficher de toast pour certaines erreurs
    const silentErrors = [401, 403, 422]
    if (silentErrors.includes(status || 0)) return

    // Messages d'erreur personnalisés
    const errorMessages: Record<number, string> = {
      400: 'Invalid request. Please check your input.',
      404: 'Resource not found.',
      500: 'Server error. Please try again later.',
      502: 'Server is temporarily unavailable.',
      503: 'Service is currently unavailable.',
    }

    const message = errorMessages[status || 0] || 'An error occurred. Please try again.'
    
    toast.error(message, {
      duration: 5000,
    })
  }

  private handleAuthError(): void {
    // Nettoyer le token
    this.authToken = null
    apiCache.clear()
    
    // Rediriger vers login si nécessaire
    if (typeof window !== 'undefined') {
      // window.location.href = '/login'
    }
  }

  private async refreshToken(): Promise<string> {
    // Éviter les appels multiples simultanés
    if (this.refreshPromise) {
      return this.refreshPromise
    }

    this.refreshPromise = new Promise(async (resolve, reject) => {
      try {
        const response = await this.client.post('/auth/refresh')
        const newToken = response.data.token
        this.refreshPromise = null
        resolve(newToken)
      } catch (error) {
        this.refreshPromise = null
        reject(error)
      }
    })

    return this.refreshPromise
  }

  private generateRequestId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  }

  private extractFilename(response: AxiosResponse): string | null {
    const disposition = response.headers['content-disposition']
    if (!disposition) return null

    const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition)
    if (matches && matches[1]) {
      return matches[1].replace(/['"]/g, '')
    }

    return null
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  /**
   * Méthodes publiques
   */
  setAuthToken(token: string | null): void {
    this.authToken = token
    if (token) {
      localStorage.setItem('auth_token', token)
    } else {
      localStorage.removeItem('auth_token')
    }
  }

  getAuthToken(): string | null {
    if (!this.authToken && typeof window !== 'undefined') {
      this.authToken = localStorage.getItem('auth_token')
    }
    return this.authToken
  }

  clearCache(): void {
    apiCache.clear()
  }

  isAuthenticated(): boolean {
    return !!this.getAuthToken()
  }
}

// Export d'une instance unique
const apiClient = new ApiClient()

// Export pour utilisation dans l'app
export default apiClient
export { apiClient, ApiClient, apiCache }