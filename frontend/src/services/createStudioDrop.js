const IMAGE_TYPES = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif', bmp: 'image/bmp', tif: 'image/tiff', tiff: 'image/tiff' }

export function studioDropFile(file) {
  const extension = file?.name?.split('.').pop().toLowerCase()
  const type = extension === 'json' ? 'application/json' : IMAGE_TYPES[extension] || file?.type
  if (!(file instanceof File) || !['application/json', ...Object.values(IMAGE_TYPES)].includes(type)
    || file.size <= 0 || file.size > (type === 'application/json' ? 4 : 32) * 1024 * 1024) {
    throw new Error('Drop one workflow JSON up to 4 MiB or an image up to 32 MiB.')
  }
  return file.type === type ? file : new File([file], file.name, { type, lastModified: file.lastModified })
}

// Native drag events provide file paths, not browser Files. The IPC command
// consumes a short-lived grant created by the main window's actual OS drop.
export async function listenStudioDesktopDrops(onDrop, onError) {
  if (!window.__TAURI_INTERNALS__) return () => {}
  const [{ getCurrentWindow }, { invoke }] = await Promise.all([
    import('@tauri-apps/api/window'), import('@tauri-apps/api/core'),
  ])
  return getCurrentWindow().listen('donut-create-file-drop', event => {
    onDrop(event.payload, async path => {
      const file = await invoke('read_create_drop_file', { path })
      return studioDropFile(new File([new Uint8Array(file.bytes)], file.name, { type: file.mime }))
    }).catch(onError)
  })
}
