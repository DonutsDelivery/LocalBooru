import { useState } from 'react'

/**
 * Collapsible prompt section with copy button
 * Used to display AI generation prompts (positive/negative)
 */
function PromptSection({ label, text, isNegative }) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)

  const isLong = text.length > 150
  const displayText = expanded || !isLong ? text : text.slice(0, 150) + '...'

  const handleCopy = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`prompt-section ${isNegative ? 'negative' : 'positive'}`}>
      <div className="prompt-header">
        <span className="prompt-label">{label}</span>
        <button className="copy-btn" onClick={handleCopy} title="Copy to clipboard">
          {copied ? (
            <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
              <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
            </svg>
          )}
        </button>
      </div>
      <div className={`prompt-text ${expanded ? 'expanded' : ''}`}>
        {displayText}
      </div>
      {isLong && (
        <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  )
}

export default PromptSection
