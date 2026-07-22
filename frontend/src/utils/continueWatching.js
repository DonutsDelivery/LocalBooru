export function watchHistoryLocator(item) {
  const imageId = item?.id ?? item?.image_id
  const libraryId = item?.library_id
  const directoryId = item?.directory_id
  if (imageId == null || libraryId == null || directoryId == null) {
    throw new Error('Watch history requires a full image locator')
  }
  return {
    imageId: Number(imageId),
    libraryId: String(libraryId),
    directoryId: Number(directoryId),
  }
}

export function watchHistoryIdentityKey(item) {
  const locator = watchHistoryLocator(item)
  return `${encodeURIComponent(locator.libraryId)}:${locator.directoryId}:${locator.imageId}`
}

export function canonicalWatchItem(history, image) {
  const historyLocator = watchHistoryLocator(history)
  const imageLocator = watchHistoryLocator(image)
  if (
    historyLocator.imageId !== imageLocator.imageId
    || historyLocator.libraryId !== imageLocator.libraryId
    || historyLocator.directoryId !== imageLocator.directoryId
  ) {
    throw new Error('Hydrated image does not match watch history locator')
  }
  return {
    ...image,
    playback_position: history.playback_position,
    duration: history.duration,
    progress: history.progress,
    completed: history.completed,
    last_watched: history.last_watched,
  }
}
