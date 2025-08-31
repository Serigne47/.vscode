'use client'
import { useState, useEffect } from 'react'

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<string>('Vérification...')
  const [tenders, setTenders] = useState<any[]>([])

  useEffect(() => {
    checkBackend()
    loadTenders()
  }, [])

  const checkBackend = async () => {
    try {
      const response = await fetch('http://localhost:8000/health')
      const data = await response.json()
      setBackendStatus(data.mongodb === 'connected' ? '✅ Connecté' : '❌ Déconnecté')
    } catch {
      setBackendStatus('❌ Backend non accessible')
    }
  }

  const loadTenders = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/tenders')
      const data = await response.json()
      setTenders(Array.isArray(data) ? data : [])
    } catch (error) {
      console.error('Erreur chargement tenders:', error)
    }
  }

  const createTestTender = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/tenders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: 'Tender Test ' + new Date().toLocaleTimeString(),
          description: 'Test depuis React',
          amount: Math.floor(Math.random() * 100000)
        })
      })
      if (response.ok) {
        loadTenders() // Recharger la liste
      }
    } catch (error) {
      console.error('Erreur création:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Système d'Analyse de Tenders</h1>
        
        {/* Status */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-2">État du Système</h2>
          <p>Backend MongoDB: {backendStatus}</p>
        </div>

        {/* Actions */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <button
            onClick={createTestTender}
            className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
          >
            Créer un Tender Test
          </button>
        </div>

        {/* Liste */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">
            Tenders Existants ({tenders.length})
          </h2>
          {tenders.length === 0 ? (
            <p className="text-gray-500">Aucun tender. Créez-en un!</p>
          ) : (
            <div className="space-y-3">
              {tenders.map((tender, index) => (
                <div key={tender.id || index} className="border-l-4 border-blue-500 pl-4">
                  <p className="font-medium">{tender.title}</p>
                  <p className="text-sm text-gray-600">{tender.description}</p>
                  <p className="text-xs text-gray-500">Montant: {tender.amount}€</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}