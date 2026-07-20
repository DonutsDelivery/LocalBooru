/**
 * useImageFiltering - Filter state and logic for gallery
 * Extracted from App.jsx Gallery component
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { fetchImages, fetchTags, getLibraryStats, subscribeToLibraryEvents } from '../api'

export function useImageFiltering() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [images, setImages] = useState([])
  const [tags, setTags] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [total, setTotal] = useState(0)
  const [filtersInitialized, setFiltersInitialized] = useState(false)
  const [stats, setStats] = useState(null)
  const statsUpdateTimeout = useRef(null)
  const lightboxIndexRef = useRef(null)

  // Parse filter values from URL params
  const currentTags = searchParams.get('tags') || ''
  const currentRating = searchParams.get('rating') || 'pg,pg13,r,x,xxx'
  const favoritesOnly = searchParams.get('favorites') === 'true'
  const currentSort = searchParams.get('sort') || 'newest'
  const currentDirectoryId = searchParams.get('directory') ? parseInt(searchParams.get('directory')) : null
  const currentMinAge = searchParams.get('min_age') ? parseInt(searchParams.get('min_age')) : null
  const currentMaxAge = searchParams.get('max_age') ? parseInt(searchParams.get('max_age')) : null
  const currentTimeframe = searchParams.get('timeframe') || null
  const currentFilename = searchParams.get('filename') || ''

  // Keep hasMore in sync with actual images count (fixes stale closure bugs)
  useEffect(() => {
    if (total > 0) {
      setHasMore(images.length < total)
    }
  }, [images.length, total])

  // Load saved filters from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('localbooru_filters')
    if (saved && !window.location.search) {
      try {
        const filters = JSON.parse(saved)
        const params = {}
        if (filters.tags) params.tags = filters.tags
        if (filters.rating && filters.rating !== 'pg,pg13,r,x,xxx') params.rating = filters.rating
        if (filters.favorites) params.favorites = 'true'
        if (filters.sort && filters.sort !== 'newest') params.sort = filters.sort
        if (filters.directory) params.directory = filters.directory
        if (filters.min_age !== null && filters.min_age !== undefined) params.min_age = filters.min_age
        if (filters.max_age !== null && filters.max_age !== undefined) params.max_age = filters.max_age
        if (Object.keys(params).length > 0) {
          setSearchParams(params)
        }
      } catch (e) {
        console.error('Failed to load saved filters:', e)
      }
    }
    setFiltersInitialized(true)
  }, [])

  // Save filters to localStorage when they change (only after initial load to avoid overwriting)
  useEffect(() => {
    if (!filtersInitialized) return
    const filters = {
      tags: currentTags || null,
      rating: currentRating,
      favorites: favoritesOnly,
      sort: currentSort,
      directory: currentDirectoryId,
      min_age: currentMinAge,
      max_age: currentMaxAge
    }
    localStorage.setItem('localbooru_filters', JSON.stringify(filters))
  }, [filtersInitialized, currentTags, currentRating, favoritesOnly, currentSort, currentDirectoryId, currentMinAge, currentMaxAge])

  // Load images
  const loadImages = useCallback(async (pageNum = 1, append = false) => {
    setLoading(true)
    try {
      const result = await fetchImages({
        tags: currentTags,
        rating: currentRating,
        favorites_only: favoritesOnly,
        directory_id: currentDirectoryId,
        min_age: currentMinAge,
        max_age: currentMaxAge,
        timeframe: currentTimeframe,
        filename: currentFilename,
        sort: currentSort,
        page: pageNum,
        per_page: 50
      })

      if (append) {
        // Deduplicate when appending to avoid showing same image twice
        setImages(prev => {
          const existingIds = new Set(prev.map(img => img.id))
          const newImages = result.images.filter(img => !existingIds.has(img.id))
          return [...prev, ...newImages]
        })
      } else {
        setImages(result.images)
      }
      setTotal(result.total)
      // Note: hasMore is computed by useEffect based on actual images.length
      setPage(pageNum)
    } catch (error) {
      console.error('Failed to load images:', error)
    }
    setLoading(false)
  }, [currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge, currentTimeframe, currentFilename])

  // Load tags
  const loadTags = useCallback(async () => {
    try {
      const result = await fetchTags({ per_page: 100 })
      setTags(result.tags || [])
    } catch (error) {
      console.error('Failed to load tags:', error)
    }
  }, [])

  // Initial loads
  useEffect(() => {
    if (!filtersInitialized) return
    loadImages(1, false)
  }, [filtersInitialized, currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge, currentTimeframe, loadImages])

  useEffect(() => {
    loadTags()
  }, [loadTags])

  useEffect(() => {
    getLibraryStats().then(setStats).catch(console.error)
  }, [])

  // Subscribe to real-time library events (debounced refresh)
  const triggerDebouncedRefresh = useCallback(() => {
    if (statsUpdateTimeout.current) {
      clearTimeout(statsUpdateTimeout.current)
    }
    statsUpdateTimeout.current = setTimeout(() => {
      // Always update stats
      getLibraryStats().then(setStats).catch(console.error)

      // Only refresh images if sorted by newest, scrolled near top, and not in lightbox
      const isAtTop = window.scrollY < 200
      const isNewest = currentSort === 'newest'
      const isInLightbox = lightboxIndexRef.current !== null

      if (isNewest && isAtTop && !isInLightbox) {
        loadImages(1, false)
      }
    }, 2000)
  }, [loadImages, currentSort])

  // On visibility change, start debounce
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        triggerDebouncedRefresh()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [triggerDebouncedRefresh])

  // SSE events also trigger the same debounce
  useEffect(() => {
    const unsubscribe = subscribeToLibraryEvents((event) => {
      if (event.type === 'image_added') {
        triggerDebouncedRefresh()
      }
    })
    return () => {
      unsubscribe()
      if (statsUpdateTimeout.current) {
        clearTimeout(statsUpdateTimeout.current)
      }
    }
  }, [triggerDebouncedRefresh])

  const handleLoadMore = useCallback(() => {
    if (!loading && hasMore) {
      loadImages(page + 1, true)
    }
  }, [loading, hasMore, page, loadImages])

  const handleTagClick = useCallback((tagName) => {
    const currentTagList = currentTags ? currentTags.split(',').map(t => t.trim()) : []
    let newTagList

    if (currentTagList.includes(tagName)) {
      newTagList = currentTagList.filter(t => t !== tagName)
    } else {
      newTagList = [...currentTagList, tagName]
    }

    const params = {}
    if (newTagList.length > 0) params.tags = newTagList.join(',')
    if (currentRating !== 'pg,pg13,r,x,xxx') params.rating = currentRating
    if (favoritesOnly) params.favorites = 'true'
    if (currentSort !== 'newest') params.sort = currentSort
    if (currentDirectoryId) params.directory = currentDirectoryId
    if (currentMinAge !== null) params.min_age = currentMinAge
    if (currentMaxAge !== null) params.max_age = currentMaxAge
    setSearchParams(params)
  }, [currentTags, currentRating, favoritesOnly, currentSort, currentDirectoryId, currentMinAge, currentMaxAge, setSearchParams])

  const handleSearch = useCallback((tags, rating, sort, favOnly, directoryId, minAge, maxAge, timeframe, filename) => {
    const params = {}
    if (tags) params.tags = tags
    if (rating && rating !== 'pg,pg13,r,x,xxx') params.rating = rating
    if (favOnly) params.favorites = 'true'
    if (sort && sort !== 'newest') params.sort = sort
    if (directoryId) params.directory = directoryId
    if (minAge !== null && minAge !== undefined) params.min_age = minAge
    if (maxAge !== null && maxAge !== undefined) params.max_age = maxAge
    if (timeframe) params.timeframe = timeframe
    if (filename) params.filename = filename
    setSearchParams(params)
  }, [setSearchParams])

  // Update a single image in the images array
  const handleImageUpdate = useCallback((imageId, updates) => {
    setImages(prev => prev.map(img =>
      img.id === imageId ? { ...img, ...updates } : img
    ))
  }, [])

  return {
    // State
    images,
    setImages,
    tags,
    loading,
    page,
    hasMore,
    total,
    filtersInitialized,
    stats,
    lightboxIndexRef,

    // Filter values
    currentTags,
    currentRating,
    favoritesOnly,
    currentSort,
    currentDirectoryId,
    currentMinAge,
    currentMaxAge,
    currentTimeframe,
    currentFilename,

    // Actions
    loadImages,
    handleLoadMore,
    handleTagClick,
    handleSearch,
    handleImageUpdate
  }
}

export default useImageFiltering
