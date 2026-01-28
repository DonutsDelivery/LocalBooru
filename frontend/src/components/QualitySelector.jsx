import { useState, useEffect } from 'react'
import './QualitySelector.css'

export default function QualitySelector({ isOpen, onClose, currentQuality, onQualityChange, sourceResolution }) {
  if (!isOpen) return null

  // Define quality options with metadata
  // Bitrates based on relative pixel count to 1080p @ 20 Mbps
  const qualityOptions = [
    { id: 'original', label: 'Original', description: 'No transcoding', maxHeight: Infinity },
    { id: '1440p', label: '1440p (QHD)', description: '30 Mbps', maxHeight: 1440 },
    { id: '1080p_enhanced', label: '1080p Enhanced', description: '20 Mbps', maxHeight: 1080 },
    { id: '1080p', label: '1080p', description: '12 Mbps', maxHeight: 1080 },
    { id: '720p', label: '720p', description: '8 Mbps', maxHeight: 720 },
    { id: '480p', label: '480p', description: '4 Mbps', maxHeight: 480 },
  ]

  // Filter options based on source resolution (prevent upscaling)
  let availableOptions = qualityOptions
  if (sourceResolution && sourceResolution.height) {
    const sourceHeight = sourceResolution.height
    console.log('[QualitySelector] Source resolution:', sourceHeight, 'px')
    // Keep: Original (always), and options that don't upscale (maxHeight <= sourceHeight)
    availableOptions = qualityOptions.filter(opt =>
      opt.id === 'original' || opt.maxHeight <= sourceHeight
    )
    console.log('[QualitySelector] Available options:', availableOptions.map(o => o.id))
  } else {
    console.log('[QualitySelector] No source resolution provided, showing all options')
  }

  const handleQualitySelect = (optionId) => {
    onQualityChange(optionId)
    onClose()
  }

  return (
    <>
      <div className="quality-selector-popup" onClick={(e) => e.stopPropagation()}>
        <div className="quality-selector-header">Quality</div>
        <div className="quality-options">
          {availableOptions.map(option => (
            <button
              key={option.id}
              className={`quality-option ${currentQuality === option.id ? 'active' : ''}`}
              onClick={() => handleQualitySelect(option.id)}
            >
              <div className="quality-option-content">
                <span className="quality-label">{option.label}</span>
                <span className="quality-description">{option.description}</span>
              </div>
              {currentQuality === option.id && (
                <svg className="quality-checkmark" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                </svg>
              )}
            </button>
          ))}
        </div>
      </div>
      <div className="quality-selector-backdrop" onClick={onClose} />
    </>
  )
}
