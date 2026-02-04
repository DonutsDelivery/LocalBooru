/**
 * IPC handlers for LocalBooru Electron app
 * Handles communication between renderer and main process
 */
const { ipcMain, dialog, shell, clipboard, nativeImage, Menu, app } = require('electron');

// External dependencies (injected via init)
let getApiPort = null;
let getMainWindow = null;
let backendManager = null;

/**
 * Initialize the IPC module with dependencies
 * @param {Object} deps - Dependencies
 * @param {Function} deps.getApiPort - Function to get current API port
 * @param {Function} deps.getMainWindow - Function to get main window instance
 * @param {Object} deps.backendManager - Backend manager instance (can be set later)
 */
function init(deps) {
  getApiPort = deps.getApiPort;
  getMainWindow = deps.getMainWindow;
  backendManager = deps.backendManager;
}

/**
 * Update the backend manager reference (called after backend is initialized)
 * @param {Object} manager
 */
function setBackendManager(manager) {
  backendManager = manager;
}

/**
 * Setup all IPC handlers for renderer process communication
 */
function setupIPC() {
  // Get API URL
  ipcMain.handle('get-api-url', () => {
    return `http://127.0.0.1:${getApiPort()}`;
  });

  // Add watch directory
  ipcMain.handle('add-directory', async () => {
    const mainWindow = getMainWindow();
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
    const mainWindow = getMainWindow();
    mainWindow?.minimize();
  });

  ipcMain.handle('maximize-window', () => {
    const mainWindow = getMainWindow();
    if (mainWindow?.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow?.maximize();
    }
    return mainWindow?.isMaximized();
  });

  ipcMain.handle('close-window', () => {
    const mainWindow = getMainWindow();
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
    const mainWindow = getMainWindow();
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
    const mainWindow = getMainWindow();
    return mainWindow?.isMaximized() ?? false;
  });
}

module.exports = {
  init,
  setupIPC,
  setBackendManager
};
