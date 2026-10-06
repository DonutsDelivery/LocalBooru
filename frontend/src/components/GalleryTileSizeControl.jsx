// SPDX-License-Identifier: GPL-3.0-only
import './GalleryTileSizeControl.css'
const labels = ['XS', 'S', 'M', 'L', 'XL']
export default function GalleryTileSizeControl({ tileSize, onChange }) {
  return <div className="tile-size-control" title="Adjust tile size">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tile-size-icon"><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></svg>
    <div className="tile-size-range"><input type="range" min="1" max="5" value={tileSize} onChange={event => onChange(Number(event.target.value))} className="tile-size-slider" aria-label="Tile size" aria-valuetext={labels[tileSize - 1]} /><div className="tile-size-scale" aria-hidden="true">{labels.map((label, index) => <span key={label} className={tileSize === index + 1 ? 'active' : ''}>{label}</span>)}</div></div>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tile-size-icon"><rect x="4" y="4" width="16" height="16" rx="2" /></svg>
  </div>
}
