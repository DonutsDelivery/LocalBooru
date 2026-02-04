/**
 * Menu and tray management for LocalBooru Electron app
 * Handles system tray icon and context menus
 */
const { Tray, Menu, shell, app } = require('electron');
const path = require('path');

// Module state
let tray = null;

// External dependencies (injected via init)
let getApiPort = null;
let getMainWindow = null;
let createWindow = null;
let reloadMainWindow = null;
let backendManager = null;
let isDev = false;
let log = console.log;

/**
 * Initialize the menu module with dependencies
 * @param {Object} deps - Dependencies
 * @param {Function} deps.getApiPort - Function to get current API port
 * @param {Function} deps.getMainWindow - Function to get main window instance
 * @param {Function} deps.createWindow - Function to create a new window
 * @param {Function} deps.reloadMainWindow - Function to reload main window
 * @param {boolean} deps.isDev - Whether running in dev mode
 * @param {Function} deps.log - Logging function
 */
function init(deps) {
  getApiPort = deps.getApiPort;
  getMainWindow = deps.getMainWindow;
  createWindow = deps.createWindow;
  reloadMainWindow = deps.reloadMainWindow;
  isDev = deps.isDev;
  log = deps.log || console.log;
}

/**
 * Update the backend manager reference (called after backend is initialized)
 * @param {Object} manager
 */
function setBackendManager(manager) {
  backendManager = manager;
}

/**
 * Get the tray instance
 * @returns {Tray|null}
 */
function getTray() {
  return tray;
}

/**
 * Helper to safely show or recreate window
 */
async function showOrCreateWindow() {
  const mainWindow = getMainWindow();

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
    reloadMainWindow();
  } else {
    log('[Tray] No valid window, creating...');
    await createWindow();
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

module.exports = {
  init,
  createTray,
  getTray,
  setBackendManager
};
