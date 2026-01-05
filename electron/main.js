/**
 * LocalBooru Electron Main Process
 * Manages the app lifecycle, backend server, and directory watcher
 */
const { app, BrowserWindow, ipcMain, Tray, Menu, dialog, shell, clipboard, nativeImage } = require('electron');
const path = require('path');
const BackendManager = require('./backendManager');
const DirectoryWatcher = require('./directoryWatcher');
const { initUpdater } = require('./updater');

// Single instance lock - prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  // Another instance is already running, quit this one
  app.quit();
} else {
  // This is the primary instance
  app.on('second-instance', () => {
    // Someone tried to run a second instance, focus our window
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      if (!mainWindow.isVisible()) mainWindow.show();
      mainWindow.focus();
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
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    frame: false,  // Custom title bar
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    },
    icon: path.join(__dirname, '../assets/icon.png'),
    show: false // Show when ready
  });

  // Clear cache to prevent stale HTML after updates
  mainWindow.webContents.session.clearCache();

  // Load the frontend from backend server (same as browser access)
  if (isDev) {
    mainWindow.loadURL('http://localhost:5174');
    mainWindow.webContents.openDevTools();
  } else {
    // Load from backend - synced with browser access
    mainWindow.loadURL(`http://127.0.0.1:${API_PORT}`);
  }

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
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

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Open LocalBooru',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        } else {
          createWindow();
        }
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
    tray.on('click', () => {
      console.log('[Tray] Left-click detected');
      if (mainWindow) {
        console.log('[Tray] Window exists, showing...');
        if (mainWindow.isMinimized()) {
          mainWindow.restore();
        }
        mainWindow.show();
        mainWindow.setAlwaysOnTop(true);
        mainWindow.focus();
        mainWindow.setAlwaysOnTop(false);
      } else {
        console.log('[Tray] No window, creating...');
        createWindow();
      }
    });

    tray.on('right-click', () => {
      tray.popUpContextMenu(contextMenu);
    });
  } else {
    // macOS/Linux: use standard context menu behavior
    tray.setContextMenu(contextMenu);

    tray.on('double-click', () => {
      if (mainWindow) {
        mainWindow.show();
        mainWindow.focus();
      }
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
  setupIPC();
  await initializeApp();
  createWindow();
  createTray();

  // Initialize auto-updater
  if (mainWindow) {
    initUpdater(mainWindow);
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
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
