export function getInstallProgressPercent(progress) {
  if (!progress || !Number.isFinite(progress.total_bytes) || progress.total_bytes <= 0) {
    return null
  }
  const completed = Number.isFinite(progress.completed_bytes) ? progress.completed_bytes : 0
  return Math.max(0, Math.min(100, (completed / progress.total_bytes) * 100))
}

export function canInstallManagedAddon({ acceptedLicense, busy, transitional }) {
  return acceptedLicense === true && !busy && transitional !== true
}

export function canCancelManagedInstall(action) {
  return action === 'install' || action === 'repair'
}

export function shouldOfferManagedInstallCancellation(action, progress) {
  return !!progress && (!action || canCancelManagedInstall(action) || action === 'cancelling')
}
