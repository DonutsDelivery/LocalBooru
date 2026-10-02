import { videoQualityOptions } from '../utils/localVideoResolution'
import './QualitySelector.css'

export default function QualitySelector({ isOpen, onClose, currentQuality, onQualityChange, sourceResolution, rawResize = false }) {
  if (!isOpen) return null

  // Define quality options with metadata
  // Bitrates based on relative pixel count to 1080p @ 20 Mbps
  const qualityOptions = videoQualityOptions(rawResize)

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
        <div className="quality-selector-header">{rawResize ? 'Resolution' : 'Quality'}</div>
        <div className="quality-options">
          {availableOptions.map(option => (
            <button
              key={option.id}
              className={`quality-option ${(rawResize && currentQuality === '1080p_enhanced' ? '1080p' : currentQuality) === option.id ? 'active' : ''}`}
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
