/**
 * LocalBooru Electron Main Process
 * Manages the app lifecycle, backend server, and directory watcher
 */
const { app, BrowserWindow, ipcMain, Tray, Menu, dialog, shell } = require('electron');
const path = require('path');
const BackendManager = require('./backendManager');
const DirectoryWatcher = require('./directoryWatcher');
const { initUpdater } = require('./updater');

// Keep references to prevent garbage collection
let mainWindow = null;
let tray = null;
let backendManager = null;
let directoryWatcher = null;

const isDev = process.env.NODE_ENV === 'development';
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
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    },
    icon: path.join(__dirname, '../assets/icon.png'),
    show: false // Show when ready
  });

  // Load the frontend from backend server (same as browser access)
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
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
  tray.setContextMenu(contextMenu);

  tray.on('double-click', () => {
    if (mainWindow) {
      mainWindow.show();
      mainWindow.focus();
    }
  });
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
