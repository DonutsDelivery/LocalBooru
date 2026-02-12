/**
 * Window management for LocalBooru Electron app
 * Handles main window creation, URL loading, and window event handlers
 */
const { BrowserWindow, session, shell } = require('electron');
const path = require('path');

// Module state
let mainWindow = null;
let isCreatingWindow = false;

// External dependencies (injected via init)
let getApiPort = null;
let isDev = false;
let log = console.log;

/**
 * Initialize the window module with dependencies
 * @param {Object} deps - Dependencies
 * @param {Function} deps.getApiPort - Function to get current API port
 * @param {boolean} deps.isDev - Whether running in dev mode
 * @param {Function} deps.log - Logging function
 */
function init(deps) {
  getApiPort = deps.getApiPort;
  isDev = deps.isDev;
  log = deps.log || console.log;
}

/**
 * Get the main window instance
 * @returns {BrowserWindow|null}
 */
function getMainWindow() {
  return mainWindow;
}

/**
 * Set the main window instance (used for cleanup)
 * @param {BrowserWindow|null} window
 */
function setMainWindow(window) {
  mainWindow = window;
}

/**
 * Create the main application window
 * @param {Object} options - Options for window creation
 * @param {boolean} options.skipUrlLoad - Skip loading the backend URL (used when backend failed)
 * @returns {Promise<BrowserWindow>}
 */
async function createWindow(options = {}) {
  const { skipUrlLoad = false } = options;

  // Prevent concurrent window creation (race condition guard)
  if (isCreatingWindow) {
    log('[createWindow] Already creating window, skipping...');
    return mainWindow;
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    log('[createWindow] Window already exists, skipping...');
    return mainWindow;
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
      transparent: true,  // Transparent window for rounded corners
      webPreferences: {
        preload: path.join(__dirname, 'preload.js'),
        nodeIntegration: false,
        contextIsolation: true
      },
      icon: path.join(__dirname, '../assets/icon.png'),
      show: false // Show when ready
    });
    log('[createWindow] BrowserWindow created');

    // Setup window handlers
    setupWindowOpenHandler(mainWindow);
    setupNavigationHandler(mainWindow);
    setupWebContentsEvents(mainWindow);

    // Clear cache to ensure fresh CSS/JS is loaded
    await session.defaultSession.clearCache();

    // Load the frontend
    await loadWindowContent(mainWindow, skipUrlLoad);

    // Setup window display
    setupWindowDisplay(mainWindow);

    // Handle window close - minimize to tray instead
    mainWindow.on('close', (event) => {
      const { app } = require('electron');
      if (!app.isQuitting) {
        event.preventDefault();
        mainWindow.hide();
        return false;
      }
    });

    mainWindow.on('closed', () => {
      mainWindow = null;
    });

    return mainWindow;
  } finally {
    isCreatingWindow = false;
  }
}

/**
 * Setup window open handler to prevent blank windows
 * @param {BrowserWindow} window
 */
function setupWindowOpenHandler(window) {
  window.webContents.setWindowOpenHandler(({ url }) => {
    log(`[Window] setWindowOpenHandler called for: ${url}`);

    // Allow same-origin URLs (internal app navigation) - but open in same window, not new
    const appHost = `127.0.0.1:${getApiPort()}`;
    const devHost = 'localhost:5174';

    if (url.includes(appHost) || url.includes(devHost)) {
      // Internal URL - load in current window instead of opening new one
      window.loadURL(url);
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
}

/**
 * Setup navigation handler to block problematic URLs
 * @param {BrowserWindow} window
 */
function setupNavigationHandler(window) {
  window.webContents.on('will-navigate', (event, url) => {
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
}

/**
 * Setup webContents event logging
 * @param {BrowserWindow} window
 */
function setupWebContentsEvents(window) {
  window.webContents.on('did-start-loading', () => {
    log('[Window] Started loading...');
  });
  window.webContents.on('did-finish-load', () => {
    log('[Window] Finished loading');
  });
  window.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    log(`[Window] Failed to load: ${errorCode} ${errorDescription}`);
  });
  window.webContents.on('render-process-gone', (event, details) => {
    log(`[Window] Render process gone: ${details.reason}`);
  });
  window.webContents.on('unresponsive', () => {
    log('[Window] Webcontents unresponsive');
  });
  window.webContents.on('responsive', () => {
    log('[Window] Webcontents responsive again');
  });
}

/**
 * Load content into the window
 * @param {BrowserWindow} window
 * @param {boolean} skipUrlLoad
 */
async function loadWindowContent(window, skipUrlLoad) {
  const loadWithRetry = async (url, maxRetries = 5) => {
    for (let i = 0; i < maxRetries; i++) {
      log(`[Window] Loading URL attempt ${i + 1}: ${url}`);
      try {
        await window.loadURL(url);
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
    window.loadURL(`data:text/html,<html><body style="background:#141414;"></body></html>`);
  } else if (isDev) {
    const loaded = await loadWithRetry('http://localhost:5174');
    if (!loaded) {
      log('[Window] Dev server load failed, showing error page');
      window.loadURL(`data:text/html,<html><body style="background:#141414;color:white;font-family:system-ui;padding:40px;"><h1>Failed to connect</h1><p>Could not connect to dev server at localhost:5174</p><p>Make sure the frontend dev server is running.</p><button onclick="location.reload()" style="padding:10px 20px;cursor:pointer;">Retry</button></body></html>`);
    }
  } else {
    log(`[Window] Backend should be at http://127.0.0.1:${getApiPort()}`);
    const loaded = await loadWithRetry(`http://127.0.0.1:${getApiPort()}`);
    if (!loaded) {
      log('[Window] All load attempts failed, showing error page');
      // Error page with draggable title bar since frame:false means no native title bar
      window.loadURL(`data:text/html,<html><head><style>
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
}

/**
 * Setup window display with show timeout fallback
 * @param {BrowserWindow} window
 */
function setupWindowDisplay(window) {
  // Show window when ready - but also show after timeout as fallback
  const showTimeout = setTimeout(() => {
    log('[Window] Timeout - showing window anyway');
    if (window && !window.isVisible()) {
      window.show();
    }
  }, 5000);

  window.once('ready-to-show', () => {
    log('[Window] ready-to-show fired');
    clearTimeout(showTimeout);
    window.show();
  });
}

/**
 * Show error page for backend failure
 * @param {BrowserWindow} window
 * @param {Error} backendError
 * @param {number} port
 */
function showBackendErrorPage(window, backendError, port) {
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
      ? `netstat -ano | findstr :${port}`
      : `lsof -i :${port}`;

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
        <h1>Port ${port} is already in use</h1>
        <p>LocalBooru cannot start because another application is using port ${port}.</p>
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

  window.loadURL(`data:text/html,${encodeURIComponent(errorHtml)}`);
}

/**
 * Reload the main window URL (used by tray when webContents is in bad state)
 */
function reloadMainWindow() {
  if (!mainWindow || mainWindow.isDestroyed()) return;

  const needsReload = mainWindow.webContents.isCrashed() ||
    mainWindow.webContents.getURL() === '' ||
    mainWindow.webContents.getURL() === 'about:blank';

  if (needsReload) {
    log(`[Window] WebContents needs reload (crashed: ${mainWindow.webContents.isCrashed()}, url: ${mainWindow.webContents.getURL()})`);
    const url = isDev ? 'http://localhost:5174' : `http://127.0.0.1:${getApiPort()}`;
    mainWindow.loadURL(url);
  }
}

module.exports = {
  init,
  createWindow,
  getMainWindow,
  setMainWindow,
  showBackendErrorPage,
  reloadMainWindow
};
