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

  // Platform info
  platform: process.platform,
  isElectron: true
});

// Log that preload script loaded
console.log('[LocalBooru] Preload script loaded');
