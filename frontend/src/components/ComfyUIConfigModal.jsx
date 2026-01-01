import { useState, useEffect } from 'react'
import './ComfyUIConfigModal.css'

function ComfyUIConfigModal({ directoryId, directoryName, onClose, onSave }) {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [nodes, setNodes] = useState([])
  const [selectedPromptNodes, setSelectedPromptNodes] = useState([])
  const [selectedNegativeNodes, setSelectedNegativeNodes] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    loadNodes()
  }, [directoryId])

  const loadNodes = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/directories/${directoryId}/comfyui-nodes`)
      if (!response.ok) throw new Error('Failed to load nodes')
      const data = await response.json()
      setNodes(data.nodes || [])

      // Load current config
      if (data.current_config) {
        setSelectedPromptNodes(data.current_config.comfyui_prompt_node_ids || [])
        setSelectedNegativeNodes(data.current_config.comfyui_negative_node_ids || [])
      }

      if (data.nodes.length === 0) {
        setError('No ComfyUI metadata found in images in this directory. Make sure the directory contains PNG/WebP images generated with ComfyUI.')
      }
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  const togglePromptNode = (nodeId) => {
    setSelectedPromptNodes(prev =>
      prev.includes(nodeId)
        ? prev.filter(id => id !== nodeId)
        : [...prev, nodeId]
    )
    // Remove from negative if selecting as positive
    setSelectedNegativeNodes(prev => prev.filter(id => id !== nodeId))
  }

  const toggleNegativeNode = (nodeId) => {
    setSelectedNegativeNodes(prev =>
      prev.includes(nodeId)
        ? prev.filter(id => id !== nodeId)
        : [...prev, nodeId]
    )
    // Remove from positive if selecting as negative
    setSelectedPromptNodes(prev => prev.filter(id => id !== nodeId))
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      const response = await fetch(`/directories/${directoryId}/comfyui-config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          comfyui_prompt_node_ids: selectedPromptNodes,
          comfyui_negative_node_ids: selectedNegativeNodes,
          metadata_format: selectedPromptNodes.length > 0 || selectedNegativeNodes.length > 0
            ? 'comfyui'
            : 'auto'
        })
      })
      if (!response.ok) throw new Error('Failed to save configuration')

      // Ask if user wants to re-extract metadata for existing images
      if (selectedPromptNodes.length > 0 || selectedNegativeNodes.length > 0) {
        if (confirm('Configuration saved! Do you want to re-extract metadata for all existing images in this directory?')) {
          const reextractResponse = await fetch(`/directories/${directoryId}/reextract-metadata`, {
            method: 'POST'
          })
          if (reextractResponse.ok) {
            const result = await reextractResponse.json()
            alert(`Queued metadata extraction for ${result.queued} images.`)
          }
        }
      }

      onSave()
      onClose()
    } catch (err) {
      setError(err.message)
    }
    setSaving(false)
  }

  // Group nodes by node_type for easier navigation
  const groupedNodes = nodes.reduce((acc, node) => {
    const type = node.node_type || 'Unknown'
    if (!acc[type]) acc[type] = []
    acc[type].push(node)
    return acc
  }, {})

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content comfyui-config-modal" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>

        <h2>Configure ComfyUI Metadata</h2>
        <p className="modal-subtitle">Directory: {directoryName}</p>
        <p className="modal-description">
          Select which nodes contain your prompts. Click on a node to mark it as
          <span className="positive-text"> positive prompt</span> or
          <span className="negative-text"> negative prompt</span>.
        </p>

        {loading && (
          <div className="loading-state">
            <div className="spinner"></div>
            <span>Scanning for ComfyUI nodes...</span>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        {!loading && nodes.length > 0 && (
          <div className="nodes-container">
            {Object.entries(groupedNodes).map(([nodeType, typeNodes]) => (
              <div key={nodeType} className="node-group">
                <h3 className="node-type-header">{nodeType}</h3>
                {typeNodes.map(node => (
                  <div
                    key={node.node_id}
                    className={`node-item ${
                      selectedPromptNodes.includes(node.node_id) ? 'positive' :
                      selectedNegativeNodes.includes(node.node_id) ? 'negative' : ''
                    }`}
                  >
                    <div className="node-header">
                      <span className="node-id">Node #{node.node_id}</span>
                      <span className="node-field">{node.field}</span>
                    </div>
                    <div className="node-sample">{node.sample_text}</div>
                    <div className="node-actions">
                      <button
                        className={`node-btn positive ${selectedPromptNodes.includes(node.node_id) ? 'active' : ''}`}
                        onClick={() => togglePromptNode(node.node_id)}
                      >
                        + Positive
                      </button>
                      <button
                        className={`node-btn negative ${selectedNegativeNodes.includes(node.node_id) ? 'active' : ''}`}
                        onClick={() => toggleNegativeNode(node.node_id)}
                      >
                        - Negative
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        <div className="modal-actions">
          <button className="cancel-btn" onClick={onClose}>Cancel</button>
          <button
            className="save-btn"
            onClick={handleSave}
            disabled={saving || (selectedPromptNodes.length === 0 && selectedNegativeNodes.length === 0)}
          >
            {saving ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>

        {(selectedPromptNodes.length > 0 || selectedNegativeNodes.length > 0) && (
          <div className="selection-summary">
            <strong>Selected:</strong>
            {selectedPromptNodes.length > 0 && (
              <span className="summary-positive">
                {selectedPromptNodes.length} positive node{selectedPromptNodes.length > 1 ? 's' : ''}
              </span>
            )}
            {selectedNegativeNodes.length > 0 && (
              <span className="summary-negative">
                {selectedNegativeNodes.length} negative node{selectedNegativeNodes.length > 1 ? 's' : ''}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ComfyUIConfigModal
