import { useEffect, useState } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [status, setStatus] = useState('vérification...')

  useEffect(() => {
    axios
      .get(`${API_URL}/health`)
      .then(() => setStatus('connecté ✅'))
      .catch(() => setStatus('backend injoignable ❌'))
  }, [])

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-slate-800">ATS CV Scorer</h1>
        <p className="mt-2 text-slate-500">Statut backend : {status}</p>
      </div>
    </div>
  )
}

export default App
