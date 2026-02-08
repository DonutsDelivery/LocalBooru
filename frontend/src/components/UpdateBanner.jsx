/**
 * Update banner for mobile app — shown below the title bar when an update is available.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { isMobileApp } from '../serverManager';
import { checkForUpdate, downloadApk, installApk } from '../services/appUpdater';

// States: idle | checking | available | downloading | ready | error | dismissed
export default function UpdateBanner() {
  const [state, setState] = useState('idle');
  const [updateInfo, setUpdateInfo] = useState(null);
  const [progress, setProgress] = useState(0);
  const [apkUri, setApkUri] = useState(null);
  const [error, setError] = useState(null);
  const checkedRef = useRef(false);

  const doCheck = useCallback(async () => {
    if (state === 'downloading') return; // don't interrupt a download
    try {
      setState('checking');
      const info = await checkForUpdate();
      if (info.available) {
        setUpdateInfo(info);
        setState('available');
      } else {
        setState('idle');
      }
    } catch (e) {
      console.warn('[UpdateBanner] check failed:', e.message);
      setState('idle'); // Silently fail — user doesn't need to see check errors
    }
  }, [state]);

  // Check once on mount
  useEffect(() => {
    if (!isMobileApp() || checkedRef.current) return;
    checkedRef.current = true;
    // Delay check to avoid slowing down app startup
    const timer = setTimeout(doCheck, 5000);
    return () => clearTimeout(timer);
  }, [doCheck]);

  // Re-check when app resumes from background
  useEffect(() => {
    if (!isMobileApp()) return;

    const handleVisibility = () => {
      if (document.visibilityState === 'visible' && state !== 'downloading' && state !== 'ready') {
        doCheck();
      }
    };

    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, [doCheck, state]);

  const handleDownload = async () => {
    try {
      setState('downloading');
      setProgress(0);
      const uri = await downloadApk(updateInfo.downloadUrl, (p) => {
        setProgress(Math.round(p.percent));
      });
      setApkUri(uri);
      setState('ready');
    } catch (e) {
      setError(e.message);
      setState('error');
    }
  };

  const handleInstall = async () => {
    try {
      const launched = await installApk(apkUri);
      if (!launched) {
        // User was sent to permission settings — they'll come back and tap again
        setError('Please grant install permission, then tap Install again.');
        setState('ready'); // Keep in ready state so they can retry
      }
    } catch (e) {
      setError(e.message);
      setState('error');
    }
  };

  const handleDismiss = () => setState('dismissed');

  if (!isMobileApp() || state === 'idle' || state === 'checking' || state === 'dismissed') {
    return null;
  }

  return (
    <div className="update-banner">
      {state === 'available' && (
        <>
          <span className="update-banner-text">
            v{updateInfo.latestVersion} available
          </span>
          <div className="update-banner-actions">
            <button className="update-banner-btn primary" onClick={handleDownload}>
              Update
            </button>
            <button className="update-banner-btn" onClick={handleDismiss}>
              Dismiss
            </button>
          </div>
        </>
      )}

      {state === 'downloading' && (
        <>
          <span className="update-banner-text">Downloading... {progress}%</span>
          <div className="update-banner-progress">
            <div className="update-banner-progress-bar" style={{ width: `${progress}%` }} />
          </div>
        </>
      )}

      {state === 'ready' && (
        <>
          <span className="update-banner-text">
            {error || 'Ready to install'}
          </span>
          <button className="update-banner-btn primary" onClick={handleInstall}>
            Install
          </button>
        </>
      )}

      {state === 'error' && (
        <>
          <span className="update-banner-text update-banner-error">
            Update failed: {error}
          </span>
          <button className="update-banner-btn" onClick={handleDismiss}>
            Dismiss
          </button>
        </>
      )}
    </div>
  );
}
