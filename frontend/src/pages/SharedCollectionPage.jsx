import { Link, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import OnlinePage from './OnlinePage'
import { useMobileDrawer } from '../hooks/useMobileDrawer'
import { collectionGalleryTarget } from '../utils/collectionState'
import '../components/SharedCollectionsPanel.css'

export default function SharedCollectionPage() {
  const { sourceId } = useParams()
  const [params] = useSearchParams()
  const mediaType = params.get('media_type') === 'video' ? 'video' : 'image'
  const navigate = useNavigate()
  const location = useLocation()
  const galleryReturnTo = collectionGalleryTarget(location, mediaType)
  const drawer = useMobileDrawer()
  return <div className="app"><div className="main-container">
    {drawer.isOpen && <div className="sidebar-backdrop" onClick={drawer.close} />}
    <Sidebar mediaType={mediaType} mobileOpen={drawer.isOpen} onClose={drawer.close} />
    <main className="content with-sidebar shared-collection-page">
      <header><div><button className="mobile-only" onClick={drawer.open}>Menu</button><button onClick={() => navigate(`/collections?media_type=${mediaType}`, { state: { galleryReturnTo } })}>← Back to Collections</button></div><Link to="/settings/fediverse" state={{ returnTo: `${location.pathname}${location.search}` }}>Community settings</Link></header>
      <OnlinePage key={`${sourceId}:${mediaType}`} initialTab="browse" fixedSourceId={sourceId} mediaType={mediaType} />
    </main>
  </div></div>
}
