// Axios instance with base URL from environment variable.
// All API calls in hooks/ use this client.

import axios from 'axios'

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  headers: { 'Content-Type': 'application/json' },
  timeout: 300_000, // 5 min — audit pipeline can take time
})

export default apiClient
