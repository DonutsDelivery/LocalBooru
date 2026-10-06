// SPDX-License-Identifier: GPL-3.0-only
import {afterEach, expect, test, vi} from 'vitest'
import {cleanup, fireEvent, render} from '@testing-library/react'
import MasonryGrid from './MasonryGrid'

vi.mock('../api', () => ({getMediaUrl: path => path, fetchPreviewFrames: vi.fn(), uploadImage: vi.fn()}))
vi.mock('../hooks/useImageWorkflow', () => ({useImageWorkflow: () => ({available:false})}))
afterEach(cleanup)

test('remote videos without artwork remain playable from the shared grid', () => {
  vi.stubGlobal('IntersectionObserver', class { observe() {} unobserve() {} disconnect() {} })
  const video = {_remote:true, remote_source_id:'community', remote_post_id:'18', id:'remote:community:18',
    original_filename:'remote-18.mp4', url:'/protected-original', thumbnail_url:'', width:1920,height:1080}
  const click=vi.fn()
  const {container}=render(<MasonryGrid images={[video]} loading={false} hasMore={false} onImageClick={click} />)
  expect(container.querySelector('img')).toBeNull()
  expect(container.textContent).toContain('Video · Open to play')
  fireEvent.click(container.querySelector('.media-item'))
  expect(click).toHaveBeenCalledWith(video)
  vi.unstubAllGlobals()
})

test('shared masonry renders remote media after an empty load and survives clearing results', () => {
  vi.stubGlobal('IntersectionObserver', class { observe() {} unobserve() {} disconnect() {} })
  const image = {_remote:true, remote_source_id:'community', remote_post_id:'17', id:'remote:community:17',
    original_filename:'remote-17.webp', url:'/original', thumbnail_url:'/thumbnail', width:800,height:1200}
  const click=vi.fn()
  const {container,rerender}=render(<MasonryGrid images={[]} loading={false} hasMore={false} onImageClick={click} />)
  rerender(<MasonryGrid images={[image]} loading={false} hasMore={false} onImageClick={click} />)
  expect(container.querySelector('.masonry-grid')).toBeTruthy()
  expect(container.querySelector('.media-item').dataset.imageKey).toBe('remote:community:17')
  expect(container.querySelector('img').getAttribute('src')).toBe('/thumbnail')
  fireEvent.click(container.querySelector('.media-item'))
  expect(click).toHaveBeenCalledWith(image)
  rerender(<MasonryGrid images={[]} loading={false} hasMore={false} onImageClick={click} />)
  expect(container.querySelector('.masonry-empty')).toBeTruthy()
  vi.unstubAllGlobals()
})
