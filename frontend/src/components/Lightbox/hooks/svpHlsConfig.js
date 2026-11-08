// hls.js settings for the SVP HLS stream played by mobile and web clients.
export const SVP_HLS_CONFIG = Object.freeze({
  enableWorker: true,
  autoStartLoad: true,
  lowLatencyMode: false,
  startPosition: 0,
  // Same window as the transcode player. The SVP playlist is a single level with
  // no BANDWIDTH, so hls.js cannot scale these by bitrate: 120 s ahead of 60 fps
  // video is 100+ MB that a phone WebView would download, demux and hold.
  // Segments stay on the server, so seeking back re-fetches them instead of
  // restarting the stream.
  backBufferLength: 30,
  maxBufferLength: 30,
  maxMaxBufferLength: 60,
  // Retry manifest loading while SVP produces initial segments
  manifestLoadingMaxRetry: 30,
  manifestLoadingRetryDelay: 500,
  manifestLoadingMaxRetryTimeout: 60000,
  levelLoadingMaxRetry: 10,
  levelLoadingRetryDelay: 500,
  fragLoadingMaxRetry: 10,
  fragLoadingRetryDelay: 500,
})

// hls.js writes derived load policies into the config it is given, so each
// player needs its own unfrozen copy.
export function svpHlsConfig() {
  return { ...SVP_HLS_CONFIG }
}
