/**
 * LocalBooru Preload Script
 * Exposes safe APIs to the renderer process via contextBridge
 */
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  // Get API URL for backend communication
  getApiUrl: () => ipcRenderer.invoke('get-api-url'),

  // Open native folder picker dialog
  addDirectory: () => ipcRenderer.invoke('add-directory'),

  // Get backend server status
  getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),

  // Restart backend server
  restartBackend: () => ipcRenderer.invoke('restart-backend'),

  // Open URL in system browser
  openExternal: (url) => ipcRenderer.invoke('open-external', url),

  // Show file in native file explorer
  showInFolder: (filePath) => ipcRenderer.invoke('show-in-folder', filePath),

  // Auto-updater
  checkForUpdate: () => ipcRenderer.invoke('updater:check'),
  downloadUpdate: () => ipcRenderer.invoke('updater:download'),
  installUpdate: () => ipcRenderer.invoke('updater:install'),
  getVersion: () => ipcRenderer.invoke('updater:get-version'),
  onUpdaterStatus: (callback) => {
    const handler = (_, data) => callback(data);
    ipcRenderer.on('updater:status', handler);
    return () => ipcRenderer.removeListener('updater:status', handler);
  },

  // Platform info
  platform: process.platform,
  isElectron: true,

  // Window controls for custom title bar
  minimizeWindow: () => ipcRenderer.invoke('minimize-window'),
  maximizeWindow: () => ipcRenderer.invoke('maximize-window'),
  closeWindow: () => ipcRenderer.invoke('close-window'),
  isMaximized: () => ipcRenderer.invoke('is-maximized'),

  // App lifecycle
  quitApp: () => ipcRenderer.invoke('quit-app'),
  checkForUpdates: () => ipcRenderer.invoke('check-for-updates')
});

// Log that preload script loaded
console.log('[LocalBooru] Preload script loaded');
