export function remoteMediaProxyUrl(path, { tauri, remoteServerUrl, localServerBase }) {
  if (!path || !tauri || !remoteServerUrl) return null
  const cleanPath = path.startsWith('/') ? path : `/${path}`
  return `${localServerBase}/remote${cleanPath}`
}
