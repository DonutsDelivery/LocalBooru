import { act, useLayoutEffect, useRef } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import MediaItem from './MediaItem'

vi.mock('../api', () => ({
  fetchPreviewFrames: vi.fn(),
  getMediaUrl: path => path,
  uploadImage: vi.fn(),
}))

vi.mock('../tauriAPI', () => ({
  getDesktopAPI: () => null,
}))

vi.mock('./Toast', () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}))

globalThis.IS_REACT_ACT_ENVIRONMENT = true

const still = {
  id: 7,
  library_id: 'primary',
  directory_id: 3,
  filename: 'photo.png',
  original_filename: 'photo.png',
  thumbnail_url: '/api/images/7/thumbnail',
  url: '/api/images/7/file',
  file_status: 'available',
  rating: 'pg',
}

function CachedImageHarness({ image, useFullImage = false }) {
  const host = useRef(null)
  useLayoutEffect(() => {
    host.current?.querySelector('img')?.dispatchEvent(new Event('load'))
  }, [image, useFullImage])
  return <div ref={host}><MediaItem image={image} useFullImage={useFullImage} /></div>
}

describe('MediaItem masonry source transitions', () => {
  let container
  let root

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
  })

  it('replaces a full image with a new thumbnail element when tile size decreases', async () => {
    await act(async () => root.render(<MediaItem image={still} useFullImage />))
    const fullImage = container.querySelector('img')
    expect(fullImage.getAttribute('src')).toBe(still.url)

    await act(async () => root.render(<MediaItem image={still} useFullImage={false} />))
    const thumbnail = container.querySelector('img')
    expect(thumbnail.getAttribute('src')).toBe(still.thumbnail_url)
    expect(thumbnail).not.toBe(fullImage)
  })

  it('keys loaded state to the current URL without waiting for an effect', async () => {
    await act(async () => root.render(<MediaItem image={still} useFullImage />))
    const fullImage = container.querySelector('img')
    await act(async () => fullImage.dispatchEvent(new Event('load')))
    expect(container.querySelector('.media-item').classList.contains('loaded')).toBe(true)

    flushSync(() => root.render(<MediaItem image={still} useFullImage={false} />))
    expect(container.querySelector('.media-item').classList.contains('loading')).toBe(true)

    const thumbnail = container.querySelector('img')
    await act(async () => thumbnail.dispatchEvent(new Event('load')))
    expect(container.querySelector('.media-item').classList.contains('loaded')).toBe(true)
  })

  it('does not overwrite a cached image load fired before passive effects', async () => {
    await act(async () => root.render(<CachedImageHarness image={still} />))
    expect(container.querySelector('.media-item').classList.contains('loaded')).toBe(true)
    expect(container.querySelector('.loading-placeholder')).toBeNull()
  })
})
