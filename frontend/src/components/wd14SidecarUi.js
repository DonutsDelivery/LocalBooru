export const WD14_OPERATIONS = Object.freeze(['import', 'absorb', 'export'])
export const MAX_WD14_DIRECTORIES = 100

export function wd14DirectoryKey(libraryId, directoryId) {
  return `${libraryId}:${directoryId}`
}

export function buildWd14DirectoryOptions(directories, libraries) {
  const mountedLibraries = new Map(
    libraries
      .filter(library => library.mounted)
      .map(library => [library.uuid, library.name]),
  )

  return directories
    .filter(directory => mountedLibraries.has(directory.library_id))
    .map(directory => ({
      key: wd14DirectoryKey(directory.library_id, directory.id),
      libraryId: directory.library_id,
      directoryId: directory.id,
      libraryName: mountedLibraries.get(directory.library_id),
      name: directory.name,
      path: directory.path,
      imageCount: directory.image_count || 0,
      accessible: directory.path_exists !== false,
    }))
    .sort((left, right) => (
      left.libraryName.localeCompare(right.libraryName)
      || left.name.localeCompare(right.name, undefined, { numeric: true })
      || left.directoryId - right.directoryId
    ))
}

export function buildWd14RequestDirectories(options, selectedKeys) {
  return options
    .filter(option => selectedKeys.has(option.key))
    .map(option => ({
      library_id: option.libraryId,
      directory_id: option.directoryId,
    }))
}

export function wd14ConfirmationMessage(operation, overwrite, selectedCount) {
  if (operation === 'absorb') {
    return `Absorb WD14 sidecars from ${selectedCount} selected director${selectedCount === 1 ? 'y' : 'ies'}?\n\nTags are imported first. Each sidecar is permanently deleted only after all matching media records are updated successfully.`
  }
  if (operation === 'export' && overwrite) {
    return `Export and replace existing WD14 sidecars in ${selectedCount} selected director${selectedCount === 1 ? 'y' : 'ies'}?\n\nExisting same-stem .txt files will be atomically overwritten.`
  }
  return null
}

const NON_FAILURE_STATUSES = new Set([
  'imported',
  'absorbed',
  'exported',
  'skipped_missing',
  'skipped_exists',
])

export function getWd14Failures(results = []) {
  return results.filter(result => result.error || !NON_FAILURE_STATUSES.has(result.status))
}

export function getWd14SummaryItems(summary = {}) {
  return [
    ['Directories', summary.directories || 0],
    ['Media candidates', summary.media_candidates || 0],
    ['Sidecars found', summary.sidecars_found || 0],
    ['Succeeded', summary.sidecars_succeeded || 0],
    ['Skipped', summary.sidecars_skipped || 0],
    ['Failed', summary.sidecars_failed || 0],
    ['Tags parsed', summary.tags_parsed || 0],
    ['Tags added', summary.tags_added || 0],
    ['Files written', summary.sidecars_written || 0],
    ['Files removed', summary.sidecars_removed || 0],
  ]
}

export function wd14StatusLabel(status) {
  return String(status || 'unknown')
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}
