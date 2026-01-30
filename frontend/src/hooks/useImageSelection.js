/**
 * useImageSelection - Selection state and handlers for gallery
 * Extracted from App.jsx Gallery component
 */
import { useState, useCallback } from 'react'
import { batchDeleteImages, batchRetag, batchAgeDetect, batchMoveImages, fetchDirectories } from '../api'

export function useImageSelection({ images, loadImages }) {
  // Selection mode state
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedImages, setSelectedImages] = useState(new Set())
  const [batchActionLoading, setBatchActionLoading] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [deleteWithFiles, setDeleteWithFiles] = useState(false)
  const [showMoveModal, setShowMoveModal] = useState(false)
  const [moveDirectories, setMoveDirectories] = useState([])
  const [selectedMoveDir, setSelectedMoveDir] = useState(null)

  // Selection mode handlers
  const toggleSelectionMode = useCallback(() => {
    setSelectionMode(prev => !prev)
    if (selectionMode) {
      // Exiting selection mode - clear selection
      setSelectedImages(new Set())
    }
  }, [selectionMode])

  const handleSelectImage = useCallback((imageId) => {
    setSelectedImages(prev => {
      const newSet = new Set(prev)
      if (newSet.has(imageId)) {
        newSet.delete(imageId)
      } else {
        newSet.add(imageId)
      }
      return newSet
    })
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedImages(new Set())
  }, [])

  const selectAll = useCallback(() => {
    setSelectedImages(new Set(images.map(img => img.id)))
  }, [images])

  // Batch action handlers
  const handleBatchDelete = useCallback(async () => {
    if (selectedImages.size === 0) return
    setBatchActionLoading(true)
    try {
      const result = await batchDeleteImages(Array.from(selectedImages), deleteWithFiles)
      console.log('Batch delete result:', result)
      // Refresh the gallery
      await loadImages(1, false)
      setSelectedImages(new Set())
      setShowDeleteConfirm(false)
      setDeleteWithFiles(false)
    } catch (error) {
      console.error('Batch delete failed:', error)
    }
    setBatchActionLoading(false)
  }, [selectedImages, deleteWithFiles, loadImages])

  const handleBatchRetag = useCallback(async () => {
    if (selectedImages.size === 0) return
    setBatchActionLoading(true)
    try {
      const result = await batchRetag(Array.from(selectedImages))
      console.log('Batch retag result:', result)
      alert(`Queued ${result.queued} images for retagging`)
      setSelectedImages(new Set())
    } catch (error) {
      console.error('Batch retag failed:', error)
    }
    setBatchActionLoading(false)
  }, [selectedImages])

  const handleBatchAgeDetect = useCallback(async () => {
    if (selectedImages.size === 0) return
    setBatchActionLoading(true)
    try {
      const result = await batchAgeDetect(Array.from(selectedImages))
      console.log('Batch age detect result:', result)
      alert(`Queued ${result.queued} images for age detection`)
      setSelectedImages(new Set())
    } catch (error) {
      console.error('Batch age detect failed:', error)
    }
    setBatchActionLoading(false)
  }, [selectedImages])

  const openMoveModal = useCallback(async () => {
    try {
      const dirs = await fetchDirectories()
      setMoveDirectories(dirs)
      setSelectedMoveDir(null)
      setShowMoveModal(true)
    } catch (error) {
      console.error('Failed to fetch directories:', error)
    }
  }, [])

  const handleBatchMove = useCallback(async () => {
    if (selectedImages.size === 0 || !selectedMoveDir) return
    setBatchActionLoading(true)
    try {
      const result = await batchMoveImages(Array.from(selectedImages), selectedMoveDir)
      console.log('Batch move result:', result)
      alert(`Moved ${result.moved} images`)
      // Refresh the gallery
      await loadImages(1, false)
      setSelectedImages(new Set())
      setShowMoveModal(false)
      setSelectedMoveDir(null)
    } catch (error) {
      console.error('Batch move failed:', error)
    }
    setBatchActionLoading(false)
  }, [selectedImages, selectedMoveDir, loadImages])

  return {
    // State
    selectionMode,
    selectedImages,
    batchActionLoading,
    showDeleteConfirm,
    setShowDeleteConfirm,
    deleteWithFiles,
    setDeleteWithFiles,
    showMoveModal,
    setShowMoveModal,
    moveDirectories,
    selectedMoveDir,
    setSelectedMoveDir,

    // Actions
    toggleSelectionMode,
    handleSelectImage,
    clearSelection,
    selectAll,
    handleBatchDelete,
    handleBatchRetag,
    handleBatchAgeDetect,
    openMoveModal,
    handleBatchMove
  }
}

export default useImageSelection
