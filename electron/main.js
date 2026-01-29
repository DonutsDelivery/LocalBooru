/**
 * LocalBooru Electron Main Process
 * Manages the app lifecycle, backend server, and directory watcher
 */
const { app, BrowserWindow, ipcMain, Tray, Menu, dialog, shell, clipboard, nativeImage, session } = require('electron');
const path = require('path');
const fs = require('fs');
const BackendManager = require('./backendManager');
const DirectoryWatcher = require('./directoryWatcher');
const { initUpdater } = require('./updater');

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
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      if (!mainWindow.isVisible()) mainWindow.show();
      mainWindow.focus();
    } else {
      // Window was destroyed, create a new one
      mainWindow = null;
      await createWindow();
    }
  });
}

// Keep references to prevent garbage collection
let mainWindow = null;
let tray = null;
let backendManager = null;
let directoryWatcher = null;
let isCreatingWindow = false;  // Guard against concurrent window creation

// Only enable dev mode when explicitly set via LOCALBOORU_DEV
// This avoids issues with NODE_ENV being set in the shell environment
const isDev = process.env.LOCALBOORU_DEV === 'true';
const DEFAULT_PORT = 8790;

// Get the current API port (from backendManager if available, otherwise default)
function getApiPort() {
  return backendManager ? backendManager.port : DEFAULT_PORT;
}

/**
 * Create the main application window
 * @param {Object} options - Options for window creation
 * @param {boolean} options.skipUrlLoad - Skip loading the backend URL (used when backend failed)
 */
async function createWindow(options = {}) {
  const { skipUrlLoad = false } = options;

  // Prevent concurrent window creation (race condition guard)
  if (isCreatingWindow) {
    log('[createWindow] Already creating window, skipping...');
    return;
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    log('[createWindow] Window already exists, skipping...');
    return;
  }

  isCreatingWindow = true;
  log('[createWindow] Starting...');

  try {
    mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    frame: false,  // Custom title bar
    backgroundColor: '#141414',  // Match --bg-primary to prevent white flash
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    },
    icon: path.join(__dirname, '../assets/icon.png'),
    show: false // Show when ready
  });
  log('[createWindow] BrowserWindow created');

  // CRITICAL: Prevent blank windows from opening
  // This handles window.open(), target="_blank" links, and any other new window requests
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    log(`[Window] setWindowOpenHandler called for: ${url}`);

    // Allow same-origin URLs (internal app navigation) - but open in same window, not new
    const appHost = `127.0.0.1:${getApiPort()}`;
    const devHost = 'localhost:5174';

    if (url.includes(appHost) || url.includes(devHost)) {
      // Internal URL - load in current window instead of opening new one
      mainWindow.loadURL(url);
      return { action: 'deny' };
    }

    // External URLs - open in system browser
    if (url.startsWith('http://') || url.startsWith('https://')) {
      log(`[Window] Opening external URL in browser: ${url}`);
      shell.openExternal(url);
      return { action: 'deny' };
    }

    // Deny all other window creation requests (prevents blank windows)
    log(`[Window] Denying window open for: ${url}`);
    return { action: 'deny' };
  });

  // Prevent navigation to about:blank or other problematic URLs
  mainWindow.webContents.on('will-navigate', (event, url) => {
    log(`[Window] will-navigate to: ${url}`);

    // Block navigation to about:blank (common source of blank windows)
    if (url === 'about:blank' || url === '') {
      log('[Window] Blocking navigation to about:blank');
      event.preventDefault();
      return;
    }

    // Allow internal navigation
    const appHost = `127.0.0.1:${getApiPort()}`;
    const devHost = 'localhost:5174';
    if (url.includes(appHost) || url.includes(devHost)) {
      return; // Allow
    }

    // External URLs - open in browser and prevent navigation
    if (url.startsWith('http://') || url.startsWith('https://')) {
      log(`[Window] Redirecting external navigation to browser: ${url}`);
      event.preventDefault();
      shell.openExternal(url);
    }
  });

  // Register event listeners BEFORE loading URL to catch all events
  mainWindow.webContents.on('did-start-loading', () => {
    log('[Window] Started loading...');
  });
  mainWindow.webContents.on('did-finish-load', () => {
    log('[Window] Finished loading');
  });
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    log(`[Window] Failed to load: ${errorCode} ${errorDescription}`);
  });
  mainWindow.webContents.on('render-process-gone', (event, details) => {
    log(`[Window] Render process gone: ${details.reason}`);
  });
  mainWindow.webContents.on('unresponsive', () => {
    log('[Window] Webcontents unresponsive');
  });
  mainWindow.webContents.on('responsive', () => {
    log('[Window] Webcontents responsive again');
  });

  // Clear cache to ensure fresh CSS/JS is loaded
  await session.defaultSession.clearCache();

  // Load the frontend from backend server (same as browser access)
  const loadWithRetry = async (url, maxRetries = 5) => {
    for (let i = 0; i < maxRetries; i++) {
      log(`[Window] Loading URL attempt ${i + 1}: ${url}`);
      try {
        await mainWindow.loadURL(url);
        log('[Window] loadURL completed successfully');
        return true;
      } catch (err) {
        log(`[Window] loadURL failed attempt ${i + 1}: ${err.message}`);
        if (i < maxRetries - 1) {
          await new Promise(r => setTimeout(r, 1000));
        }
      }
    }
    return false;
  };

  // Skip URL loading if backend already failed (will show error page later)
  if (skipUrlLoad) {
    log('[Window] Skipping URL load (backend not started)');
    // Load a minimal page so window isn't completely blank
    mainWindow.loadURL(`data:text/html,<html><body style="background:#141414;"></body></html>`);
  } else if (isDev) {
    const loaded = await loadWithRetry('http://localhost:5174');
    if (!loaded) {
      log('[Window] Dev server load failed, showing error page');
      mainWindow.loadURL(`data:text/html,<html><body style="background:#141414;color:white;font-family:system-ui;padding:40px;"><h1>Failed to connect</h1><p>Could not connect to dev server at localhost:5174</p><p>Make sure the frontend dev server is running.</p><button onclick="location.reload()" style="padding:10px 20px;cursor:pointer;">Retry</button></body></html>`);
    }
  } else {
    log(`[Window] Backend should be at http://127.0.0.1:${getApiPort()}`);
    const loaded = await loadWithRetry(`http://127.0.0.1:${getApiPort()}`);
    if (!loaded) {
      log('[Window] All load attempts failed, showing error page');
      // Error page with draggable title bar since frame:false means no native title bar
      mainWindow.loadURL(`data:text/html,<html><head><style>
        body{background:#141414;color:white;font-family:system-ui;margin:0;padding:0;}
        .titlebar{-webkit-app-region:drag;background:#1a1a1a;padding:8px 16px;display:flex;justify-content:space-between;align-items:center;}
        .titlebar button{-webkit-app-region:no-drag;background:#dc3545;border:none;color:white;padding:4px 12px;cursor:pointer;border-radius:4px;}
        .content{padding:40px;}
        .retry{padding:10px 20px;cursor:pointer;background:#3b82f6;border:none;color:white;border-radius:4px;margin-right:10px;}
      </style></head><body>
        <div class="titlebar"><span>LocalBooru - Connection Error</span><button onclick="window.close()">Quit</button></div>
        <div class="content">
          <h1>Failed to connect</h1>
          <p>Could not connect to backend at 127.0.0.1:${getApiPort()}</p>
          <p>The backend server may have failed to start, or the port is occupied by another process.</p>
          <p style="color:#888;margin-top:20px;">Try: <code>pkill -9 -f uvicorn</code> then restart</p>
          <div style="margin-top:20px;">
            <button class="retry" onclick="location.href='http://127.0.0.1:${getApiPort()}'">Retry</button>
            <button class="retry" style="background:#666;" onclick="window.close()">Quit</button>
          </div>
        </div>
      </body></html>`);
    }
  }

  // Show window when ready - but also show after timeout as fallback
  const showTimeout = setTimeout(() => {
    log('[Window] Timeout - showing window anyway');
    if (mainWindow && !mainWindow.isVisible()) {
      mainWindow.show();
    }
  }, 5000);

  mainWindow.once('ready-to-show', () => {
    log('[Window] ready-to-show fired');
    clearTimeout(showTimeout);
    mainWindow.show();
  });

  // Handle window close - minimize to tray instead
  mainWindow.on('close', (event) => {
    if (!app.isQuitting) {
      event.preventDefault();
      mainWindow.hide();
      return false;
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  } finally {
    isCreatingWindow = false;
  }
}

/**
 * Create system tray icon
 */
function createTray() {
  // Prevent duplicate tray creation
  if (tray && !tray.isDestroyed()) {
    log('[createTray] Tray already exists, skipping...');
    return;
  }

  const iconPath = path.join(__dirname, '../assets/tray-icon.png');
  tray = new Tray(iconPath);

  // Helper to safely show or recreate window
  const showOrCreateWindow = async () => {
    // Check if window exists and is not destroyed
    if (mainWindow && !mainWindow.isDestroyed()) {
      log('[Tray] Window exists, showing...');
      if (mainWindow.isMinimized()) {
        mainWindow.restore();
      }
      mainWindow.show();
      mainWindow.setAlwaysOnTop(true);
      mainWindow.focus();
      mainWindow.setAlwaysOnTop(false);

      // Check if webContents is in a bad state and needs reload
      const needsReload = mainWindow.webContents.isCrashed() ||
        mainWindow.webContents.getURL() === '' ||
        mainWindow.webContents.getURL() === 'about:blank';

      if (needsReload) {
        log(`[Tray] WebContents needs reload (crashed: ${mainWindow.webContents.isCrashed()}, url: ${mainWindow.webContents.getURL()})`);
        const url = isDev ? 'http://localhost:5174' : `http://127.0.0.1:${getApiPort()}`;
        mainWindow.loadURL(url);
      }
    } else {
      log('[Tray] No valid window, creating...');
      // Clear reference if it was destroyed
      if (mainWindow) {
        mainWindow = null;
      }
      await createWindow();
    }
  };

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Open LocalBooru',
      click: async () => {
        await showOrCreateWindow();
      }
    },
    {
      label: 'Open in Browser',
      click: () => {
        shell.openExternal(`http://127.0.0.1:${getApiPort()}`);
      }
    },
    {
      label: 'Open Library Folder',
      click: async () => {
        const dataDir = app.getPath('userData');
        shell.openPath(dataDir);
      }
    },
    { type: 'separator' },
    {
      label: backendManager?.isRunning() ? 'Backend: Running' : 'Backend: Stopped',
      enabled: false
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        app.isQuitting = true;
        app.quit();
      }
    }
  ]);

  tray.setToolTip('LocalBooru');

  // On Windows, left-click shows window, right-click shows menu
  // On other platforms, use default behavior
  if (process.platform === 'win32') {
    tray.on('click', async () => {
      log('[Tray] Left-click detected');
      await showOrCreateWindow();
    });

    tray.on('right-click', () => {
      tray.popUpContextMenu(contextMenu);
    });
  } else {
    // macOS/Linux: use standard context menu behavior
    tray.setContextMenu(contextMenu);

    tray.on('double-click', async () => {
      await showOrCreateWindow();
    });
  }
}

/**
 * Initialize backend and services
 */
async function initializeApp() {
  console.log('[LocalBooru] Initializing...');

  // Start backend server (port determined by settings and portable mode)
  backendManager = new BackendManager();
  await backendManager.start();

  // Initialize directory watcher with the backend's port
  directoryWatcher = new DirectoryWatcher(backendManager.port);
  await directoryWatcher.loadWatchDirectories();

  console.log('[LocalBooru] Initialization complete, port:', backendManager.port);
}

/**
 * Setup IPC handlers for renderer process communication
 */
function setupIPC() {
  // Get API URL
  ipcMain.handle('get-api-url', () => {
    return `http://127.0.0.1:${getApiPort()}`;
  });

  // Add watch directory
  ipcMain.handle('add-directory', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory'],
      title: 'Select folder to watch'
    });

    if (!result.canceled && result.filePaths.length > 0) {
      return result.filePaths[0];
    }
    return null;
  });

  // Get backend status
  ipcMain.handle('get-backend-status', () => {
    return {
      running: backendManager?.isRunning() ?? false,
      port: getApiPort()
    };
  });

  // Restart backend
  ipcMain.handle('restart-backend', async () => {
    await backendManager.restart();
    return { success: true };
  });

  // Open external link
  ipcMain.handle('open-external', (event, url) => {
    shell.openExternal(url);
  });

  // Show file in folder
  ipcMain.handle('show-in-folder', (event, filePath) => {
    shell.showItemInFolder(filePath);
  });

  // Window control handlers for custom title bar
  ipcMain.handle('minimize-window', () => {
    mainWindow?.minimize();
  });

  ipcMain.handle('maximize-window', () => {
    if (mainWindow?.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow?.maximize();
    }
    return mainWindow?.isMaximized();
  });

  ipcMain.handle('close-window', () => {
    mainWindow?.close();
  });

  ipcMain.handle('quit-app', () => {
    app.isQuitting = true;
    app.quit();
  });

  // Copy image to clipboard
  ipcMain.handle('copy-image-to-clipboard', async (event, imageUrl) => {
    try {
      // Fetch the image data
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error('Failed to fetch image');

      const arrayBuffer = await response.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);

      // Create native image and copy to clipboard
      const image = nativeImage.createFromBuffer(buffer);
      if (image.isEmpty()) throw new Error('Invalid image data');

      clipboard.writeImage(image);
      return { success: true };
    } catch (error) {
      console.error('[Clipboard] Failed to copy image:', error);
      return { success: false, error: error.message };
    }
  });

  // Show image context menu
  ipcMain.handle('show-image-context-menu', async (event, { imageUrl, filePath, isVideo }) => {
    const menuTemplate = [];

    if (!isVideo) {
      menuTemplate.push({
        label: 'Copy Image',
        click: async () => {
          try {
            const response = await fetch(imageUrl);
            if (!response.ok) throw new Error('Failed to fetch image');
            const arrayBuffer = await response.arrayBuffer();
            const buffer = Buffer.from(arrayBuffer);
            const image = nativeImage.createFromBuffer(buffer);
            if (image.isEmpty()) throw new Error('Invalid image data');
            clipboard.writeImage(image);
          } catch (error) {
            console.error('[Clipboard] Failed to copy image:', error);
          }
        }
      });
    }

    if (filePath) {
      menuTemplate.push({
        label: 'Show in Folder',
        click: () => {
          shell.showItemInFolder(filePath);
        }
      });
    }

    if (menuTemplate.length > 0) {
      const menu = Menu.buildFromTemplate(menuTemplate);
      menu.popup({ window: mainWindow });
    }
  });

  // Note: check-for-updates is handled by updater.js as 'updater:check'

  ipcMain.handle('is-maximized', () => {
    return mainWindow?.isMaximized() ?? false;
  });
}

// App event handlers
app.whenReady().then(async () => {
  log('[App] whenReady fired');
  setupIPC();
  log('[App] IPC setup complete');

  // Create tray FIRST - ensures user always has a way to quit/interact
  // even if backend or window initialization fails
  createTray();
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
    await createWindow({ skipUrlLoad: !backendStarted });
    log('[App] createWindow complete');

    // If backend failed, show detailed error in window
    if (!backendStarted && mainWindow) {
      let errorHtml;

      if (backendError?.code === 'PORT_CONFLICT') {
        // Port conflict - show detailed info about what's using the port
        const portUser = backendError.portUser;
        const processInfo = portUser
          ? `<p><strong>Process using port:</strong> ${portUser.name} (PID: ${portUser.pid})</p>`
          : '<p>Could not identify the process using the port.</p>';

        const killCmd = process.platform === 'win32'
          ? `taskkill /F /PID ${portUser?.pid || 'PID'}`
          : `kill -9 ${portUser?.pid || 'PID'}`;

        const lsofCmd = process.platform === 'win32'
          ? `netstat -ano | findstr :${getApiPort()}`
          : `lsof -i :${getApiPort()}`;

        errorHtml = `<html><head><style>
          body{background:#141414;color:white;font-family:system-ui;margin:0;padding:0;}
          .titlebar{-webkit-app-region:drag;background:#1a1a1a;padding:8px 16px;display:flex;justify-content:space-between;align-items:center;}
          .titlebar button{-webkit-app-region:no-drag;background:#dc3545;border:none;color:white;padding:4px 12px;cursor:pointer;border-radius:4px;}
          .content{padding:40px;}
          h1{color:#f87171;margin-top:0;}
          code{background:#333;padding:2px 8px;border-radius:4px;font-size:0.9em;}
          .fix-steps{background:#1a1a1a;padding:20px;border-radius:8px;margin:20px 0;}
          .fix-steps li{margin:10px 0;}
          .retry{padding:10px 20px;cursor:pointer;background:#3b82f6;border:none;color:white;border-radius:4px;margin-right:10px;}
        </style></head><body>
          <div class="titlebar"><span>LocalBooru - Port Conflict</span><button onclick="window.close()">Quit</button></div>
          <div class="content">
            <h1>Port ${getApiPort()} is already in use</h1>
            <p>LocalBooru cannot start because another application is using port ${getApiPort()}.</p>
            ${processInfo}
            <div class="fix-steps">
              <strong>How to fix:</strong>
              <ol>
                <li>Close the other application using the port, OR</li>
                <li>Kill the process manually:<br><code>${killCmd}</code></li>
                <li>Or find what's using the port:<br><code>${lsofCmd}</code></li>
              </ol>
            </div>
            <p style="color:#888;">This often happens when LocalBooru didn't shut down cleanly, or another instance is running.</p>
            <div style="margin-top:20px;">
              <button class="retry" onclick="location.reload()">Retry</button>
              <button class="retry" style="background:#666;" onclick="window.close()">Quit</button>
            </div>
          </div>
        </body></html>`;
      } else {
        // Generic backend error
        errorHtml = `<html><head><style>
          body{background:#141414;color:white;font-family:system-ui;margin:0;padding:0;}
          .titlebar{-webkit-app-region:drag;background:#1a1a1a;padding:8px 16px;display:flex;justify-content:space-between;align-items:center;}
          .titlebar button{-webkit-app-region:no-drag;background:#dc3545;border:none;color:white;padding:4px 12px;cursor:pointer;border-radius:4px;}
          .content{padding:40px;}
          h1{color:#f87171;margin-top:0;}
          code{background:#333;padding:2px 8px;border-radius:4px;font-size:0.9em;}
          .retry{padding:10px 20px;cursor:pointer;background:#3b82f6;border:none;color:white;border-radius:4px;margin-right:10px;}
        </style></head><body>
          <div class="titlebar"><span>LocalBooru - Error</span><button onclick="window.close()">Quit</button></div>
          <div class="content">
            <h1>Backend Failed to Start</h1>
            <p>The LocalBooru backend server could not start.</p>
            <p><strong>Error:</strong> ${backendError?.message || 'Unknown error'}</p>
            <p style="color:#888;margin-top:20px;">Check the debug.log file for more details.</p>
            <div style="margin-top:20px;">
              <button class="retry" onclick="location.reload()">Retry</button>
              <button class="retry" style="background:#666;" onclick="window.close()">Quit</button>
            </div>
          </div>
        </body></html>`;
      }

      mainWindow.loadURL(`data:text/html,${encodeURIComponent(errorHtml)}`);
    }
  } catch (err) {
    log(`[App] createWindow FAILED: ${err.message}`);
  }

  // Initialize auto-updater
  if (mainWindow) {
    initUpdater(mainWindow, { portableDataDir });
  }

  app.on('activate', async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      await createWindow();
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
