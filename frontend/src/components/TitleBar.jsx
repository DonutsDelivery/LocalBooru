/**
 * Custom Title Bar Component
 * Replaces the native OS title bar for a consistent look
 * Only renders in Electron environment or mobile app
 */
import { useState, useEffect } from 'react';
import { isMobileApp } from '../serverManager';
import './TitleBar.css';

const TITLE_BAR_HEIGHT = 32;

export default function TitleBar({ onSwitchServer }) {
  const [isMaximized, setIsMaximized] = useState(false);
  const isElectron = window.electronAPI?.isElectron;
  const isMobile = isMobileApp();

  // Set CSS variable for title bar height offset
  useEffect(() => {
    if (isElectron || isMobile) {
      document.documentElement.style.setProperty('--title-bar-height', `${TITLE_BAR_HEIGHT}px`);
    } else {
      document.documentElement.style.setProperty('--title-bar-height', '0px');
    }
  }, [isElectron, isMobile]);

  // Check initial maximized state
  useEffect(() => {
    if (!isElectron) return;
    const checkMaximized = async () => {
      const maximized = await window.electronAPI.isMaximized();
      setIsMaximized(maximized);
    };
    checkMaximized();
  }, [isElectron]);

  // Check for updates when window gains focus
  useEffect(() => {
    if (!isElectron) return;

    const handleFocus = () => {
      window.electronAPI.checkForUpdate?.();
    };

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, [isElectron]);

  // On mobile app, show minimal title bar with switch server button
  if (isMobile) {
    return (
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
    );
  }

  // Only render full title bar in Electron
  if (!isElectron) {
    return null;
  }

  const handleMinimize = () => {
    window.electronAPI.minimizeWindow();
  };

  const handleMaximize = async () => {
    const maximized = await window.electronAPI.maximizeWindow();
    setIsMaximized(maximized);
  };

  const handleClose = () => {
    window.electronAPI.closeWindow();
  };

  const handleQuit = () => {
    window.electronAPI.quitApp();
  };

  return (
    <div className="title-bar">
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
  );
}
