export function directoryLibraryTargetValue(library) {
  return library.is_primary ? 'primary' : library.uuid
}

export function resolveDirectoryLibraryTarget(libraries, selectedTarget) {
  const library = libraries.find(candidate =>
    directoryLibraryTargetValue(candidate) === selectedTarget
  )
  if (!library?.mounted) {
    throw new Error('Select a mounted destination library')
  }
  return directoryLibraryTargetValue(library)
}
