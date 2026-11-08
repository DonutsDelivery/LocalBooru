// Live gallery updates must not move what the user is looking at. The gallery
// scrolls inside `.content > .masonry-container`, not the window, so scroll
// anchoring has to be done against that element.

export const GALLERY_SCROLLER_SELECTOR = '.content > .masonry-container'

export function findGalleryScroller(doc = document) {
  return doc.querySelector(GALLERY_SCROLLER_SELECTOR) || doc.scrollingElement || null
}

function scrollerTop(scroller) {
  return scroller === scroller.ownerDocument?.scrollingElement ? 0 : scroller.getBoundingClientRect().top
}

// Remember the first tile that is (partly) visible and where it sits, or null
// when the user is at the very top (new tiles should then appear in view) and
// no lightbox is open.
export function captureScrollAnchor(scroller, { force = false } = {}) {
  if (!scroller || (!force && scroller.scrollTop <= 0)) return null
  const top = scrollerTop(scroller)
  const visible = Array.from(scroller.querySelectorAll('[data-image-key]'))
    .map(element => ({ element, rect: element.getBoundingClientRect() }))
    .filter(({ rect }) => rect.bottom > top)
    .sort((a, b) => a.rect.top - b.rect.top)[0]
  if (!visible) return null
  return { key: visible.element.dataset.imageKey, offset: visible.rect.top - top }
}

// Scroll so the anchored tile is back at the same offset. Looks the tile up by
// identity because its DOM node may have been replaced. Returns whether it
// moved the scroller.
export function restoreScrollAnchor(scroller, anchor) {
  if (!scroller || !anchor) return false
  const element = Array.from(scroller.querySelectorAll('[data-image-key]'))
    .find(candidate => candidate.dataset.imageKey === anchor.key)
  if (!element) return false
  const delta = element.getBoundingClientRect().top - scrollerTop(scroller) - anchor.offset
  if (Math.abs(delta) < 1) return false
  scroller.scrollTop += delta
  return true
}
