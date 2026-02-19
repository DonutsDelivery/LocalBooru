/**
 * Auto-updater module for LocalBooru
 * Uses electron-updater with GitHub releases
 */
const { autoUpdater } = require('electron-updater');
const { ipcMain } = require('electron');

let mainWindow = null;

/**
 * Initialize the auto-updater
 */
function initUpdater(window) {
  mainWindow = window;

  // Configure updater
  autoUpdater.autoDownload = false;
  autoUpdater.autoInstallOnAppQuit = true;

  // Check for updates on startup (with delay)
  setTimeout(() => {
    autoUpdater.checkForUpdates().catch((err) => {
      console.log('[Updater] Check failed:', err.message);
    });
  }, 5000);

  // Update available
  autoUpdater.on('update-available', (info) => {
    console.log('[Updater] Update available:', info.version);
    mainWindow?.webContents.send('updater:status', {
      status: 'available',
      version: info.version,
      releaseNotes: info.releaseNotes
    });
  });

  // No update available
  autoUpdater.on('update-not-available', (info) => {
    console.log('[Updater] Up to date:', info.version);
    mainWindow?.webContents.send('updater:status', {
      status: 'up-to-date',
      version: info.version
    });
  });

  // Download progress
  autoUpdater.on('download-progress', (progress) => {
    console.log(`[Updater] Download progress: ${progress.percent.toFixed(1)}%`);
    mainWindow?.webContents.send('updater:status', {
      status: 'downloading',
      progress: progress.percent,
      bytesPerSecond: progress.bytesPerSecond,
      transferred: progress.transferred,
      total: progress.total
    });
  });

  // Update downloaded
  autoUpdater.on('update-downloaded', (info) => {
    console.log('[Updater] Update downloaded:', info.version);
    mainWindow?.webContents.send('updater:status', {
      status: 'downloaded',
      version: info.version
    });
  });

  // Error
  autoUpdater.on('error', (err) => {
    console.error('[Updater] Error:', err.message);
    mainWindow?.webContents.send('updater:status', {
      status: 'error',
      error: err.message
    });
  });
}

// IPC handlers
ipcMain.handle('updater:check', async () => {
  try {
    const result = await autoUpdater.checkForUpdates();
    return { success: true, version: result?.updateInfo?.version };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('updater:download', async () => {
  try {
    await autoUpdater.downloadUpdate();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('updater:install', () => {
  autoUpdater.quitAndInstall(false, true);
});

ipcMain.handle('updater:get-version', () => {
  const { app } = require('electron');
  return app.getVersion();
});

module.exports = { initUpdater };
