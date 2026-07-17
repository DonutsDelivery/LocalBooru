let nextTransitionGeneration = Date.now() * 1000

export function capturePlaybackIntent(video, absoluteTime, imageKey) {
  const position = Number.isFinite(absoluteTime) ? absoluteTime : 0
  const ended = Boolean(video?.ended)
  return {
    imageKey,
    media: video || null,
    position,
    shouldPlay: Boolean(video && !video.paused && !ended),
    ended,
  }
}

export function createPlaybackTransitionOwner() {
  let generation = 0
  let intent = null
  let controller = null

  return {
    begin(nextIntent, { reuseActiveIntent = true } = {}) {
      controller?.abort()
      controller = new AbortController()
      generation = ++nextTransitionGeneration

      if (!reuseActiveIntent || !intent || intent.imageKey !== nextIntent.imageKey) {
        intent = nextIntent
      }

      return {
        generation,
        intent,
        signal: controller.signal,
      }
    },

    isCurrent(owner, imageKey = owner?.intent?.imageKey, media = owner?.intent?.media) {
      return Boolean(
        owner
        && owner.generation === generation
        && owner.intent.imageKey === imageKey
        && owner.intent.media === media
      )
    },

    currentIntent() {
      return intent
    },

    currentGeneration() {
      return generation
    },

    finish(owner) {
      if (owner?.generation !== generation) return false
      intent = null
      controller = null
      return true
    },

    invalidate() {
      controller?.abort()
      controller = null
      intent = null
      generation = ++nextTransitionGeneration
      return generation
    },
  }
}

export function nextPlaybackSourceRevision(revision) {
  return revision + 1
}
