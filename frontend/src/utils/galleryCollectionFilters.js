// SPDX-License-Identifier: GPL-3.0-only
export const REMOTE_SORT_VALUES = ['newest', 'oldest', 'filename_asc', 'filename_desc', 'filesize_largest', 'filesize_smallest', 'resolution_high', 'resolution_low', 'duration_longest', 'duration_shortest', 'random']
const number = value => value != null && value !== '' && value !== 'null' && Number.isFinite(Number(value)) ? Number(value) : null
export function readCollectionFilters(params, remote = false) {
  const [width, height] = (params.get('resolution') || '').split('x').map(number)
  const [min, max] = (params.get('duration') || '').split('-').map(number)
  return {
    tags: remote ? (params.get('q') || '').split(/[,\s]+/).filter(Boolean).join(',') : params.get('tags') || '', rating: params.get('rating') || null,
    sort: params.get('sort') || (remote ? 'newest' : 'collection'), favoritesOnly: params.get('favorites') === 'true',
    directoryId: remote ? null : number(params.get('directory')), libraryId: remote ? null : params.get('library'),
    minAge: remote ? null : number(params.get('min_age')), maxAge: remote ? null : number(params.get('max_age')), timeframe: params.get('timeframe') || null,
    filename: remote ? '' : params.get('filename') || '', resolution: width && height ? { width, height } : null,
    orientation: params.get('orientation') || null, duration: params.get('duration') ? { min, max } : null,
    watchedStatus: remote ? null : params.get('watched') || null,
    creatorId: remote ? params.get('creator_id') || null : null, offlineOnly: remote && params.get('offline_only') === 'true',
  }
}
export function collectionRequestFilters(filters, remote = false) {
  const result = {}
  const put = (key, value) => { if (value !== null && value !== undefined && value !== '' && value !== false) result[key] = value }
  put(remote ? 'q' : 'tags', remote ? filters.tags.split(',').map(tag => tag.trim()).filter(Boolean).join(' ') : filters.tags)
  if (remote) { put('creator_id', filters.creatorId); put('offline_only', filters.offlineOnly) }
  put('rating', filters.rating)
  if (filters.sort !== (remote ? 'newest' : 'collection')) put('sort', filters.sort)
  put('favorites_only', filters.favoritesOnly)
  if (!remote) { put('min_age', filters.minAge); put('max_age', filters.maxAge) }
  put('timeframe', filters.timeframe)
  put('min_width', filters.resolution?.width); put('min_height', filters.resolution?.height); put('orientation', filters.orientation)
  put('min_duration', filters.duration?.min); put('max_duration', filters.duration?.max)
  if (!remote) {
    put('directory_id', filters.directoryId); put('library_id', filters.libraryId); put('filename', filters.filename); put('watched_status', filters.watchedStatus)
    const included = []; const excluded = []
    for (const tag of filters.tags.split(',').map(tag => tag.trim()).filter(Boolean)) { if (tag.startsWith('-')) excluded.push(tag.slice(1)); else included.push(tag) }
    delete result.tags
    put('tags', included.join(',')); put('exclude_tags', excluded.join(','))
  }
  return result
}
export function collectionSidebarProps(filters) {
  return { currentTags: filters.tags, initialTags: filters.tags, initialRating: filters.rating || undefined,
    initialSort: filters.sort, initialFavoritesOnly: filters.favoritesOnly, initialDirectoryId: filters.directoryId,
    initialLibraryId: filters.libraryId, initialMinAge: filters.minAge, initialMaxAge: filters.maxAge,
    initialTimeframe: filters.timeframe, initialFilename: filters.filename, initialResolution: filters.resolution,
    initialOrientation: filters.orientation, initialDuration: filters.duration, initialWatchedStatus: filters.watchedStatus }
}
export function sidebarFilterParams(params, remote, tags, rating, sort, favorites, directory, minAge, maxAge, timeframe, filename, resolution, orientation, duration, library, watched) {
  const next = new URLSearchParams(params)
  for (const key of ['q', 'tags', 'rating', 'sort', 'favorites', 'directory', 'library', 'min_age', 'max_age', 'timeframe', 'filename', 'resolution', 'orientation', 'duration', 'watched']) next.delete(key)
  const put = (key, value) => { if (value !== null && value !== undefined && value !== '' && value !== false) next.set(key, String(value)) }
  put(remote ? 'q' : 'tags', tags)
  if (rating !== 'pg,pg13,r,x,xxx') put('rating', rating)
  if (sort !== (remote ? 'newest' : 'collection')) put('sort', sort)
  put('favorites', favorites)
  if (!remote) { put('min_age', minAge); put('max_age', maxAge) }
  put('timeframe', timeframe)
  if (resolution) put('resolution', `${resolution.width}x${resolution.height}`)
  put('orientation', orientation)
  if (duration) put('duration', `${duration.min ?? 'null'}-${duration.max ?? 'null'}`)
  if (!remote) { put('directory', directory); put('library', library); put('filename', filename); put('watched', watched === undefined ? params.get('watched') : watched) }
  return next
}
export function toggleCollectionTag(params, tag, remote = false) {
  const next = new URLSearchParams(params)
  const key = remote ? 'q' : 'tags'
  const tags = (next.get(key) || '').split(remote ? /[,\s]+/ : ',').map(name => name.trim()).filter(Boolean)
  const name = typeof tag === 'string' ? tag : tag.name
  const updated = tags.includes(name) ? tags.filter(current => current !== name) : [...tags, name]
  if (updated.length) next.set(key, updated.join(',')); else next.delete(key)
  return next
}
