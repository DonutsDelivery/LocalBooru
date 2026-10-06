// SPDX-License-Identifier: GPL-3.0-only
import {afterEach, expect, test, vi} from 'vitest'
import {cleanup, fireEvent, render, screen} from '@testing-library/react'
import {MemoryRouter} from 'react-router-dom'
const api = vi.hoisted(() => ({fetchDirectories:vi.fn().mockResolvedValue({directories:[]}),
  fetchCollections:vi.fn().mockResolvedValue({collections:[]}), getSavedSearches:vi.fn().mockResolvedValue({searches:[]}),
  getFileDimensions:vi.fn(), createSavedSearch:vi.fn(), deleteSavedSearch:vi.fn()}))
vi.mock('../../api', () => api)
vi.mock('../FamilyModeLock', () => ({default:()=>null}))
vi.mock('../Music/PersistentMusicPlayer', () => ({default:()=>null}))
vi.mock('../SidebarNavigation', () => ({default:()=>null}))
import Sidebar from './Sidebar'
afterEach(cleanup)

test('shared lightbox details use remote metadata without probing a local path or showing local filters', async () => {
  const tagClick=vi.fn()
  const image={_remote:true,id:'remote:community:42',remote_post_id:'42',remote_source_name:'DonutBooru',
    width:2304,height:1536,rating:'pg',title:'Community title',tags:[{name:'landscape'}]}
  render(<MemoryRouter initialEntries={['/shared/community']}><Sidebar selectedImage={image}
    tags={[]} currentTags="" lightboxMode collapsed={false} onTagClick={tagClick} /></MemoryRouter>)
  expect(await screen.findByText('Media Info')).toBeTruthy()
  expect(screen.getByText('2304x1536')).toBeTruthy()
  expect(screen.getByText('#42')).toBeTruthy()
  expect(screen.getByText('DonutBooru')).toBeTruthy()
  expect(screen.getByText('Community title')).toBeTruthy()
  expect(screen.queryByText('AI Generation')).toBeNull()
  expect(screen.queryByPlaceholderText('Search tags...')).toBeNull()
  fireEvent.click(screen.getByRole('button',{name:'landscape'}))
  expect(tagClick).toHaveBeenCalledWith('landscape')
  expect(api.getFileDimensions).not.toHaveBeenCalled()
})
