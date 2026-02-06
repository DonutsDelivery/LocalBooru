import { useState } from 'react'
import { getMediaUrl } from '../api'
import './FolderItem.css'

function FolderItem({ folder, onClick }) {
  const [loaded, setLoaded] = useState(false)
  const thumbnailUrl = folder.thumbnail_url ? getMediaUrl(folder.thumbnail_url) : ''

  return (
    <div className={`folder-item ${loaded ? 'loaded' : 'loading'}`} onClick={onClick} data-folder-path={folder.path}>
      {!loaded && <div className="folder-loading-placeholder" />}
      {thumbnailUrl && (
        <img
          src={thumbnailUrl}
          alt={folder.name}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          onError={() => setLoaded(true)}
        />
      )}
      <div className="folder-overlay">
        <div className="folder-info">
          <svg className="folder-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
          </svg>
          <span className="folder-name">{folder.name}</span>
          <span className="folder-count">{folder.count}</span>
        </div>
      </div>
    </div>
  )
}

export default FolderItem
