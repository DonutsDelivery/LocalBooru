/**
 * Custom Title Bar Component
 * Replaces the native OS title bar for a consistent look
 * Only renders in Tauri or the mobile app.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { isMobileApp } from '../serverManager';
import { getDesktopAPI, isTauri } from '../tauriAPI';
import { WINDOW_RESIZE_HANDLES, startWindowResize } from '../utils/windowResize.js';
import UpdateBanner from './UpdateBanner';
import './TitleBar.css';

const TITLE_BAR_HEIGHT = 32;
const MOBILE_TITLE_BAR_HEIGHT = 44;

export default function TitleBar({ onSwitchServer, onOpenFile }) {
  const [isMaximized, setIsMaximized] = useState(null);
  const isTauriApp = isTauri();
  const isDesktop = isTauriApp;
  const isMobile = isMobileApp();
  const apiRef = useRef(null);

  // Get desktop API on mount
  useEffect(() => {
    apiRef.current = getDesktopAPI();
  }, []);

  // Set the live title bar offset and desktop transparency class.
  useEffect(() => {
    if (isMobile) {
      document.documentElement.style.setProperty(
        '--title-bar-height',
        `calc(${MOBILE_TITLE_BAR_HEIGHT}px + var(--safe-top))`
      );
    } else if (isDesktop) {
      document.documentElement.style.setProperty('--title-bar-height', `${TITLE_BAR_HEIGHT}px`);
      document.documentElement.classList.add('desktop-app');
    } else {
      document.documentElement.style.setProperty('--title-bar-height', '0px');
    }
  }, [isDesktop, isMobile]);

  // Keep resize chrome synchronized with native maximize/restore actions.
  useEffect(() => {
    if (!isDesktop) return;
    let cancelled = false;
    let unlisten = () => {};
    let syncTimer = null;
    let syncGeneration = 0;
    const syncMaximized = async () => {
      const generation = ++syncGeneration;
      const maximized = await apiRef.current?.isMaximized?.();
      if (!cancelled && generation === syncGeneration) setIsMaximized(Boolean(maximized));
    };
    const scheduleMaximizedSync = () => {
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(syncMaximized, 80);
    };
    apiRef.current?.onWindowResized?.(scheduleMaximizedSync).then(stopListening => {
      if (cancelled) stopListening();
      else {
        unlisten = stopListening;
        syncMaximized();
      }
    });
    return () => {
      cancelled = true;
      if (syncTimer) clearTimeout(syncTimer);
      unlisten();
    };
  }, [isDesktop]);


  // Programmatic drag for Tauri (data-tauri-drag-region only works on direct element, not children)
  const handleDragMouseDown = useCallback((e) => {
    // Only handle left mouse button
    if (e.button !== 0) return;
    const api = apiRef.current;
    if (api?.startDragging) {
      api.startDragging();
    }
  }, []);

  const handleResizeMouseDown = useCallback((event, direction) => {
    startWindowResize({
      event,
      direction,
      isDesktop,
      isMaximized,
      startResizeDragging: (resizeDirection) => apiRef.current?.startResizeDragging?.(resizeDirection),
    });
  }, [isDesktop, isMaximized]);

  // On mobile app, show minimal title bar with switch server button
  if (isMobile) {
    return (
      <>
        <div className="title-bar mobile">
          <div className="title-bar-drag">
            <div className="title-bar-icon">
              <svg width="18" height="18" viewBox="0 0 64 64" fill="none">
                <rect x="10" y="10" width="44" height="44" rx="6" fill="var(--bg-tertiary)" stroke="currentColor" strokeWidth="3"/>
                <circle cx="22" cy="22" r="6" fill="currentColor"/>
                <path d="M10 46 L26 28 L34 38 L46 24 L54 46 Z" fill="currentColor" opacity="0.85"/>
              </svg>
            </div>
            <span className="title-bar-title">LocalBooru</span>
          </div>

          <div className="title-bar-controls">
            <button
              className="title-bar-btn switch-server"
              onClick={onSwitchServer}
              title="Switch Server"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>
                <rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>
                <line x1="6" y1="6" x2="6.01" y2="6"/>
                <line x1="6" y1="18" x2="6.01" y2="18"/>
              </svg>
            </button>
          </div>
        </div>
        <UpdateBanner />
      </>
    );
  }

  // Only render the full title bar in Tauri.
  if (!isDesktop) {
    return null;
  }

  const handleMinimize = async () => {
    const api = apiRef.current;
    if (api?.minimizeWindow) {
      await api.minimizeWindow();
    }
  };

  const handleMaximize = async () => {
    const api = apiRef.current;
    if (api?.maximizeWindow) {
      const maximized = await api.maximizeWindow();
      setIsMaximized(maximized);
    }
  };

  const handleClose = async () => {
    const api = apiRef.current;
    if (api?.closeWindow) {
      await api.closeWindow();
    }
  };

  const handleQuit = async () => {
    const api = apiRef.current;
    if (api?.quitApp) {
      await api.quitApp();
    }
  };

  return (
    <>
      {isMaximized === false && createPortal(
        WINDOW_RESIZE_HANDLES.map(({ direction, edge }) => (
          <div
            key={direction}
            className={`window-resize-handle ${edge}`}
            onMouseDown={(event) => handleResizeMouseDown(event, direction)}
            aria-hidden="true"
          />
        )),
        document.body
      )}
      <div className="title-bar">
      <div
        className="title-bar-drag"
        onMouseDown={isTauriApp ? handleDragMouseDown : undefined}
      >
        <div className="title-bar-icon">
          <svg width="18" height="18" viewBox="0 0 64 64" fill="none">
            <rect x="10" y="10" width="44" height="44" rx="6" fill="var(--bg-tertiary)" stroke="currentColor" strokeWidth="3"/>
            <circle cx="22" cy="22" r="6" fill="currentColor"/>
            <path d="M10 46 L26 28 L34 38 L46 24 L54 46 Z" fill="currentColor" opacity="0.85"/>
          </svg>
        </div>
        <span className="title-bar-title">LocalBooru</span>
      </div>

      <div className="title-bar-controls">
        <button
          className="title-bar-btn"
          onClick={onOpenFile}
          title="Open video file (Ctrl+O)"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M3 6.5h7l2 2h9v10H3z"/>
            <path d="M3 8.5V5h7l2 2"/>
          </svg>
        </button>
        <button
          className="title-bar-btn quit"
          onClick={handleQuit}
          title="Quit (fully exit app)"
        >
          <svg width="12" height="12" viewBox="0 0 12 12">
            <path d="M6 1v5M3 3.5A4.5 4.5 0 1 0 9 3.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" fill="none"/>
          </svg>
        </button>

        <button
          className="title-bar-btn minimize"
          onClick={handleMinimize}
          title="Minimize to tray"
        >
          <svg width="12" height="12" viewBox="0 0 12 12">
            <rect x="2" y="5.5" width="8" height="1" fill="currentColor"/>
          </svg>
        </button>

        <button
          className="title-bar-btn maximize"
          onClick={handleMaximize}
          title={isMaximized ? "Restore" : "Maximize"}
        >
          {isMaximized ? (
            <svg width="12" height="12" viewBox="0 0 12 12">
              <rect x="2.5" y="4" width="6" height="5.5" fill="none" stroke="currentColor" strokeWidth="1"/>
              <path d="M4 4V2.5h6v5.5h-1.5" fill="none" stroke="currentColor" strokeWidth="1"/>
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 12 12">
              <rect x="2" y="2" width="8" height="8" fill="none" stroke="currentColor" strokeWidth="1.2"/>
            </svg>
          )}
        </button>

        <button
          className="title-bar-btn close"
          onClick={handleClose}
          title="Close"
        >
          <svg width="12" height="12" viewBox="0 0 12 12">
            <path d="M2 2L10 10M10 2L2 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
      </div>
      </div>
    </>
  );
}
