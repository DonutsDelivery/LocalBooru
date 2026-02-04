/**
 * LocalBooru Electron Main Process
 * Entry point - manages app lifecycle, backend server, and directory watcher
 */
const { app, BrowserWindow } = require('electron');
const path = require('path');
const fs = require('fs');
const BackendManager = require('./backend');
const DirectoryWatcher = require('./directoryWatcher');
const { initUpdater } = require('./updater');
const windowModule = require('./window');
const ipcModule = require('./ipc');
const menuModule = require('./menu');

// Detect portable mode EARLY - before single instance lock
// This ensures portable and system installs use separate locks/userData
function detectPortableModeEarly() {
  if (!app.isPackaged) return null;

  const appDir = path.dirname(app.getPath('exe'));
  const portableDataPath = path.join(appDir, 'data');
  const useAppdataMarker = path.join(appDir, '.use-appdata');

  // Check for .use-appdata marker
  if (fs.existsSync(useAppdataMarker)) return null;

  // Check for Program Files (Windows)
  if (process.platform === 'win32') {
    const programFiles = process.env.ProgramFiles || 'C:\\Program Files';
    const programFilesX86 = process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)';
    if (appDir.startsWith(programFiles) || appDir.startsWith(programFilesX86)) return null;
  }

  // Portable mode - ensure data folder exists
  if (!fs.existsSync(portableDataPath)) {
    fs.mkdirSync(portableDataPath, { recursive: true });
  }
  return portableDataPath;
}

/**
 * Apply pending portable update on startup
 * This copies files from the extracted update to the app directory
 */
function applyPendingUpdate() {
  if (!portableDataDir) return false;

  const pendingPath = path.join(portableDataDir, 'updates', 'pending.json');
  if (!fs.existsSync(pendingPath)) return false;

  try {
    const pending = JSON.parse(fs.readFileSync(pendingPath, 'utf8'));
    const extractedDir = path.join(portableDataDir, 'updates', 'extracted');
    const appDir = path.dirname(process.execPath);

    console.log(`[Updater] Found pending update to v${pending.version}`);
    console.log(`[Updater] Source: ${extractedDir}`);
    console.log(`[Updater] Dest: ${appDir}`);

    if (!fs.existsSync(extractedDir)) {
      console.error('[Updater] Extracted directory not found');
      return false;
    }

    // Find the actual content directory (might be nested)
    let sourceDir = extractedDir;
    const entries = fs.readdirSync(extractedDir);
    if (entries.length === 1) {
      const nested = path.join(extractedDir, entries[0]);
      if (fs.statSync(nested).isDirectory()) {
        sourceDir = nested;
      }
    }

    // Copy all files from source to app directory
    const copyRecursive = (src, dest) => {
      const stats = fs.statSync(src);
      if (stats.isDirectory()) {
        if (!fs.existsSync(dest)) {
          fs.mkdirSync(dest, { recursive: true });
        }
        for (const entry of fs.readdirSync(src)) {
          copyRecursive(path.join(src, entry), path.join(dest, entry));
        }
      } else {
        try {
          fs.copyFileSync(src, dest);
        } catch (err) {
          // File might be locked (like the current exe), skip it
          console.log(`[Updater] Skipped locked file: ${dest}`);
        }
      }
    };

    console.log('[Updater] Applying update...');
    copyRecursive(sourceDir, appDir);
    console.log('[Updater] Update applied successfully');

    // Clean up
    fs.rmSync(path.join(portableDataDir, 'updates'), { recursive: true, force: true });
    console.log('[Updater] Cleaned up update files');

    return true;
  } catch (err) {
    console.error('[Updater] Failed to apply update:', err.message);
    return false;
  }
}

/**
 * Clean up the updates folder and stale update scripts on startup (only if no pending update)
 */
function cleanupUpdatesFolder() {
  if (!portableDataDir) return;

  const pendingPath = path.join(portableDataDir, 'updates', 'pending.json');
  // Don't clean up if there's a pending update
  if (fs.existsSync(pendingPath)) return;

  const updatesPath = path.join(portableDataDir, 'updates');
  if (fs.existsSync(updatesPath)) {
    try {
      fs.rmSync(updatesPath, { recursive: true, force: true });
      console.log('[Updater] Cleaned up updates folder');
    } catch (err) {
      console.error('[Updater] Failed to cleanup updates folder:', err.message);
    }
  }

  // Clean up stale update scripts (left behind if the script crashed before self-deleting)
  for (const scriptName of ['apply-update.bat', 'apply-update.sh']) {
    const scriptPath = path.join(portableDataDir, scriptName);
    if (fs.existsSync(scriptPath)) {
      try {
        fs.unlinkSync(scriptPath);
        console.log(`[Updater] Cleaned up stale ${scriptName}`);
      } catch (err) {
        // Ignore - might still be running
      }
    }
  }
}

const portableDataDir = detectPortableModeEarly();

// Apply pending update FIRST, before anything else
const updateApplied = applyPendingUpdate();

// Clean up any leftover update files (only if no pending update was found)
if (!updateApplied) {
  cleanupUpdatesFolder();
}

// Set userData path for portable mode BEFORE single instance lock
// This ensures portable and system installs have separate locks
if (portableDataDir) {
  app.setPath('userData', portableDataDir);
}

// Debug logging to file
const logFile = path.join(app.getPath('userData'), 'debug.log');
const log = (msg) => {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  console.log(msg);
  fs.appendFileSync(logFile, line);
};
log('=== App starting ===');
log(`[App] Mode: ${portableDataDir ? 'portable' : 'system'}`);

// Single instance lock - prevent multiple instances
// Now uses separate lock files for portable vs system installs
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  // Another instance is already running, quit this one
  app.quit();
} else {
  // This is the primary instance
  app.on('second-instance', async () => {
    // Someone tried to run a second instance, focus our window
    const mainWindow = windowModule.getMainWindow();
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      if (!mainWindow.isVisible()) mainWindow.show();
      mainWindow.focus();
    } else {
      // Window was destroyed, create a new one
      windowModule.setMainWindow(null);
      await windowModule.createWindow();
    }
  });
}

// Keep references to prevent garbage collection
let backendManager = null;
let directoryWatcher = null;

// Only enable dev mode when explicitly set via LOCALBOORU_DEV
// This avoids issues with NODE_ENV being set in the shell environment
const isDev = process.env.LOCALBOORU_DEV === 'true';
const DEFAULT_PORT = 8790;

// Get the current API port (from backendManager if available, otherwise default)
function getApiPort() {
  return backendManager ? backendManager.port : DEFAULT_PORT;
}

// Initialize modules with dependencies
windowModule.init({
  getApiPort,
  isDev,
  log
});

ipcModule.init({
  getApiPort,
  getMainWindow: windowModule.getMainWindow,
  backendManager: null  // Will be set after initialization
});

menuModule.init({
  getApiPort,
  getMainWindow: windowModule.getMainWindow,
  createWindow: windowModule.createWindow,
  reloadMainWindow: windowModule.reloadMainWindow,
  isDev,
  log
});

/**
 * Initialize backend and services
 */
async function initializeApp() {
  console.log('[LocalBooru] Initializing...');

  // Start backend server (port determined by settings and portable mode)
  backendManager = new BackendManager();
  await backendManager.start();

  // Update modules with backend manager reference
  ipcModule.setBackendManager(backendManager);
  menuModule.setBackendManager(backendManager);

  // Initialize directory watcher with the backend's port
  directoryWatcher = new DirectoryWatcher(backendManager.port);
  await directoryWatcher.loadWatchDirectories();

  console.log('[LocalBooru] Initialization complete, port:', backendManager.port);
}

// App event handlers
app.whenReady().then(async () => {
  log('[App] whenReady fired');
  ipcModule.setupIPC();
  log('[App] IPC setup complete');

  // Create tray FIRST - ensures user always has a way to quit/interact
  // even if backend or window initialization fails
  menuModule.createTray();
  log('[App] createTray complete');

  let backendStarted = false;
  let backendError = null;
  try {
    await initializeApp();
    backendStarted = true;
    log('[App] initializeApp complete');
  } catch (err) {
    log(`[App] initializeApp FAILED: ${err.message}`);
    backendError = err;
  }

  try {
    // If backend failed, create window without trying to load URL (saves 5+ seconds)
    const mainWindow = await windowModule.createWindow({ skipUrlLoad: !backendStarted });
    log('[App] createWindow complete');

    // If backend failed, show detailed error in window
    if (!backendStarted && mainWindow) {
      windowModule.showBackendErrorPage(mainWindow, backendError, getApiPort());
    }
  } catch (err) {
    log(`[App] createWindow FAILED: ${err.message}`);
  }

  // Initialize auto-updater
  const mainWindow = windowModule.getMainWindow();
  if (mainWindow) {
    initUpdater(mainWindow, { portableDataDir });
  }

  app.on('activate', async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      await windowModule.createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  // Don't quit on macOS
  if (process.platform !== 'darwin') {
    // Keep running in background with tray
  }
});

app.on('before-quit', async () => {
  console.log('[LocalBooru] Shutting down...');

  // Stop directory watcher
  if (directoryWatcher) {
    await directoryWatcher.stopAll();
  }

  // Stop backend
  if (backendManager) {
    await backendManager.stop();
  }
});

// CRITICAL: Synchronous cleanup on process exit
// This catches cases where before-quit doesn't fire (crashes, SIGKILL, etc.)
process.on('exit', () => {
  console.log('[LocalBooru] Process exit - forcing backend cleanup');
  if (backendManager) {
    backendManager.forceKill();
  }
});

// Handle SIGTERM (kill command, system shutdown)
process.on('SIGTERM', () => {
  console.log('[LocalBooru] Received SIGTERM');
  if (backendManager) {
    backendManager.forceKill();
  }
  process.exit(0);
});

// Handle SIGINT (Ctrl+C)
process.on('SIGINT', () => {
  console.log('[LocalBooru] Received SIGINT');
  if (backendManager) {
    backendManager.forceKill();
  }
  process.exit(0);
});

// Handle uncaught exceptions - cleanup before crashing
process.on('uncaughtException', (error) => {
  console.error('[LocalBooru] Uncaught exception:', error);
  if (backendManager) {
    backendManager.forceKill();
  }
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('[LocalBooru] Unhandled rejection:', reason);
});
