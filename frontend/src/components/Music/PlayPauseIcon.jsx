export default function PlayPauseIcon({ playing }) {
  return <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
    {playing
      ? <path d="M6.5 5h3v14h-3zm8 0h3v14h-3z" fill="currentColor" />
      : <path d="M8 5.75c0-.78.85-1.26 1.52-.86l10.1 6.25a1 1 0 0 1 0 1.72l-10.1 6.25c-.67.4-1.52-.08-1.52-.86V5.75Z" fill="currentColor" />}
  </svg>
}
