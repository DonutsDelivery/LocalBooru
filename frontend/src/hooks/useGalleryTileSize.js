// SPDX-License-Identifier: GPL-3.0-only
import { useEffect, useState } from 'react'
export function useGalleryTileSize() {
  const [tileSize, setTileSize] = useState(() => {
    const saved = Number(localStorage.getItem('localbooru_tileSize'))
    return Number.isInteger(saved) && saved >= 1 && saved <= 5 ? saved : 3
  })
  useEffect(() => { localStorage.setItem('localbooru_tileSize', String(tileSize)) }, [tileSize])
  return [tileSize, setTileSize]
}
