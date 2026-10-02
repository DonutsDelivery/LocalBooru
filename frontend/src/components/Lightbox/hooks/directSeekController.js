// Merge input within one display frame, while preserving the requested relative
// position until the native seek settles. Never wait for seeked to accept input.
export function createDirectSeekController(getVideo, onError) {
  let owner = null
  const clear = () => {
    if (!owner) return
    if (owner.frame !== null) cancelAnimationFrame(owner.frame)
    owner.video.removeEventListener('seeked', owner.complete)
    owner.video.removeEventListener('emptied', clear)
    owner.video.removeEventListener('error', clear)
    owner = null
  }
  const current = () => {
    if (owner && (getVideo() !== owner.video || owner.video.src !== owner.source)) clear()
    return owner
  }
  const request = (time, precise = false) => {
    const video = getVideo()
    if (!video) { clear(); return }
    let state = current()
    if (!state) {
      state = { video, source: video.src, frame: null }
      state.complete = () => {
        if (current() === state && state.frame === null && !video.seeking) clear()
      }
      owner = state
      video.addEventListener('seeked', state.complete)
      video.addEventListener('emptied', clear)
      video.addEventListener('error', clear)
    }
    state.target = time
    state.precise = precise
    if (state.frame !== null) return
    state.frame = requestAnimationFrame(() => {
      if (current() !== state) return
      state.frame = null
      if (state.target === video.currentTime && !video.seeking) { clear(); return }
      try {
        if (!state.precise && video.fastSeek) video.fastSeek(state.target)
        else video.currentTime = state.target
        if (!video.seeking) clear()
      } catch (error) {
        clear()
        onError(error)
      }
    })
  }
  return { request, clear, pendingTime: () => current()?.target ?? null }
}
