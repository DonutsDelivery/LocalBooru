// SPDX-License-Identifier: GPL-3.0-only

const MIME_EXTENSIONS = {
  'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'image/gif': 'gif',
  'image/avif': 'avif', 'video/mp4': 'mp4', 'video/webm': 'webm',
  'video/quicktime': 'mov', 'video/x-matroska': 'mkv',
}

export function remoteGalleryIdentity(sourceId, postId) {
  return `remote:${encodeURIComponent(sourceId)}:${encodeURIComponent(postId)}`
}

export function remoteAssetPath(sourceId, postId, kind) {
  return `/api/online/sources/${encodeURIComponent(sourceId)}/items/${encodeURIComponent(postId)}/asset/${encodeURIComponent(kind)}`
}

export function remoteGalleryItem(item, { sourceId = item.source_id, sourceName = '', offlineOnly = false } = {}) {
  if (!sourceId || item.source_id !== sourceId || item.remote_post_id == null) {
    throw new Error('The community returned an item belonging to another source.')
  }
  const media = item.media || []
  const original = media.find(variant => variant.kind === 'original') || media.find(variant => variant.kind === 'sample') || media[0]
  const sample = media.find(variant => variant.kind === 'sample') || original
  const video = media.some(variant => variant.mime_type?.startsWith('video/')) || item.duration != null
  // Video originals must never be used as <img> previews.
  const thumbnail = media.find(variant => variant.kind === 'thumbnail' && !variant.mime_type?.startsWith('video/'))
    || media.find(variant => variant.kind === 'sample' && !variant.mime_type?.startsWith('video/'))
    || (!video ? original : null)
  const path = variant => variant ? remoteAssetPath(sourceId, item.remote_post_id, variant.kind) + (offlineOnly ? '?offline_only=true' : '') : ''
  const extension = MIME_EXTENSIONS[original?.mime_type] || (video ? 'mp4' : 'jpg')
  const dimensions = [original, sample, thumbnail].find(variant => variant?.width > 0 && variant?.height > 0)
  const filename = `remote-${encodeURIComponent(item.remote_post_id)}.${extension}`
  return {
    id: remoteGalleryIdentity(sourceId, item.remote_post_id),
    _remote: true,
    remote_source_id: sourceId,
    remote_post_id: item.remote_post_id,
    remote_source_name: sourceName,
    remote_item: item,
    filename,
    original_filename: filename,
    title: item.title || '',
    url: path(original),
    file_url: path(original),
    sample_url: path(sample),
    thumbnail_url: path(thumbnail),
    width: item.width > 0 && item.height > 0 ? item.width : dimensions?.width || null,
    height: item.width > 0 && item.height > 0 ? item.height : dimensions?.height || null,
    duration: item.duration,
    rating: item.rating,
    tags: (item.tags || []).map(name => ({ name })),
    is_favorite: false,
    file_status: 'available',
  }
}

export function appendRemoteGalleryItems(existing, incoming) {
  const seen = new Set(existing.map(item => item.id))
  return [...existing, ...incoming.filter(item => {
    if (seen.has(item.id)) return false
    seen.add(item.id)
    return true
  })]
}

export function remotePageHasMore(page) {
  if (page.next_cursor) return true
  if (page.total != null) return page.page * page.per_page < page.total
  return (page.items || []).length >= page.per_page
}
