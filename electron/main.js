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

// Debug logging to file
const logFile = path.join(app.getPath('userData'), 'debug.log');
const log = (msg) => {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  console.log(msg);
  fs.appendFileSync(logFile, line);
};
log('=== App starting ===');

// Single instance lock - prevent multiple instances
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

// Only enable dev mode when explicitly set via LOCALBOORU_DEV
// This avoids issues with NODE_ENV being set in the shell environment
const isDev = process.env.LOCALBOORU_DEV === 'true';
const API_PORT = 8790;

/**
 * Create the main application window
 */
async function createWindow() {
  log('[createWindow] Starting...');
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

  if (isDev) {
    const loaded = await loadWithRetry('http://localhost:5174');
    if (!loaded) {
      log('[Window] Dev server load failed, showing error page');
      mainWindow.loadURL(`data:text/html,<html><body style="background:#141414;color:white;font-family:system-ui;padding:40px;"><h1>Failed to connect</h1><p>Could not connect to dev server at localhost:5174</p><p>Make sure the frontend dev server is running.</p><button onclick="location.reload()" style="padding:10px 20px;cursor:pointer;">Retry</button></body></html>`);
    }
    mainWindow.webContents.openDevTools();
  } else {
    log(`[Window] Backend should be at http://127.0.0.1:${API_PORT}`);
    const loaded = await loadWithRetry(`http://127.0.0.1:${API_PORT}`);
    if (!loaded) {
      log('[Window] All load attempts failed, showing error page');
      mainWindow.loadURL(`data:text/html,<html><body style="background:#141414;color:white;font-family:system-ui;padding:40px;"><h1>Failed to connect</h1><p>Could not connect to backend at 127.0.0.1:${API_PORT}</p><p>The backend server may have failed to start.</p><button onclick="location.href='http://127.0.0.1:${API_PORT}'" style="padding:10px 20px;cursor:pointer;">Retry</button></body></html>`);
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
}

/**
 * Create system tray icon
 */
function createTray() {
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
        const url = isDev ? 'http://localhost:5174' : `http://127.0.0.1:${API_PORT}`;
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
        shell.openExternal(`http://127.0.0.1:${API_PORT}`);
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

  // Start backend server
  backendManager = new BackendManager(API_PORT);
  await backendManager.start();

  // Initialize directory watcher
  directoryWatcher = new DirectoryWatcher(API_PORT);
  await directoryWatcher.loadWatchDirectories();

  console.log('[LocalBooru] Initialization complete');
}

/**
 * Setup IPC handlers for renderer process communication
 */
function setupIPC() {
  // Get API URL
  ipcMain.handle('get-api-url', () => {
    return `http://127.0.0.1:${API_PORT}`;
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
      port: API_PORT
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
  await initializeApp();
  log('[App] initializeApp complete');
  await createWindow();
  log('[App] createWindow complete');
  createTray();
  log('[App] createTray complete');

  // Initialize auto-updater
  if (mainWindow) {
    initUpdater(mainWindow);
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

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('[LocalBooru] Uncaught exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('[LocalBooru] Unhandled rejection:', reason);
});
