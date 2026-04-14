import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

/**
 * Generic hook for fetching data from the StockLens API.
 *
 * Returns { data, loading, error, refetch }.
 * Automatically fetches on mount and when `url` changes.
 */
export function useApi(url, options = {}) {
  const { autoFetch = true, defaultData = null } = options
  const [data, setData] = useState(defaultData)
  const [loading, setLoading] = useState(autoFetch)
  const [error, setError] = useState(null)

  const fetch = useCallback(async () => {
    if (!url) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.get(url)
      setData(res.data)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Request failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [url])

  useEffect(() => {
    if (autoFetch) fetch()
  }, [fetch, autoFetch])

  return { data, loading, error, refetch: fetch }
}

/**
 * POST request helper.
 */
export async function apiPost(url, body) {
  const res = await api.post(url, body)
  return res.data
}

/**
 * DELETE request helper.
 */
export async function apiDelete(url) {
  const res = await api.delete(url)
  return res.data
}

export default api
