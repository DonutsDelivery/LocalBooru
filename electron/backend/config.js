/**
 * Backend Configuration
 * Handles portable mode detection, settings, and path resolution
 */
const path = require('path');
const fs = require('fs');
const os = require('os');
const { app } = require('electron');

/**
 * Detect if running in portable mode
 * Portable mode is DEFAULT for packaged apps, unless:
 * - Running from Program Files (Windows installer location)
 * - A '.use-appdata' marker file exists next to the executable
 * @returns {string|null} Portable data directory path or null
 */
function detectPortableMode() {
  try {
    // Get the directory containing the app
    let appDir;
    if (app.isPackaged) {
      // Packaged app: use the directory containing the executable
      appDir = path.dirname(app.getPath('exe'));
    } else {
      // Development: don't use portable mode
      return null;
    }

    const portableDataPath = path.join(appDir, 'data');
    const useAppdataMarker = path.join(appDir, '.use-appdata');

    // Check if we should use AppData instead of portable mode
    if (fs.existsSync(useAppdataMarker)) {
      console.log('[Backend] Found .use-appdata marker, using AppData');
      return null;
    }

    // Check if installed in Program Files (Windows)
    if (process.platform === 'win32') {
      const programFiles = process.env.ProgramFiles || 'C:\\Program Files';
      const programFilesX86 = process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)';
      if (appDir.startsWith(programFiles) || appDir.startsWith(programFilesX86)) {
        console.log('[Backend] Running from Program Files, using AppData');
        return null;
      }
    }

    // Default: use portable mode - create data folder next to exe
    if (!fs.existsSync(portableDataPath)) {
      fs.mkdirSync(portableDataPath, { recursive: true });
    }
    console.log('[Backend] Portable mode enabled, data:', portableDataPath);
    return portableDataPath;
  } catch (e) {
    console.log('[Backend] Error detecting portable mode:', e.message);
    return null;
  }
}

/**
 * Get LocalBooru data directory (matches Python API location)
 * @param {string|null} portableDataDir - Portable data directory or null
 * @returns {string} Data directory path
 */
function getDataDir(portableDataDir) {
  // Use portable data directory if in portable mode
  if (portableDataDir) {
    return portableDataDir;
  }

  // Default: AppData (Windows) or home (Linux/Mac)
  if (process.platform === 'win32') {
    const appData = process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming');
    return path.join(appData, '.localbooru');
  }
  // Linux/Mac
  return path.join(os.homedir(), '.localbooru');
}

/**
 * Get settings.json path
 * @param {string|null} portableDataDir - Portable data directory or null
 * @returns {string} Settings file path
 */
function getSettingsPath(portableDataDir) {
  return path.join(getDataDir(portableDataDir), 'settings.json');
}

/**
 * Load network settings from settings.json
 * Note: local_port default is not set here - it's determined by portable mode in constructor
 * @param {string|null} portableDataDir - Portable data directory or null
 * @returns {Object} Network settings object
 */
function getNetworkSettings(portableDataDir) {
  const defaults = {
    local_network_enabled: false,
    public_network_enabled: false,
    // local_port intentionally omitted - defaults differ by mode (portable=8791, system=8790)
    public_port: 8791,
    auth_required_level: 'none',
    upnp_enabled: false
  };

  try {
    const settingsPath = getSettingsPath(portableDataDir);
    if (fs.existsSync(settingsPath)) {
      const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
      return { ...defaults, ...(settings.network || {}) };
    }
  } catch (e) {
    console.log('[Backend] Error reading settings:', e.message);
  }

  return defaults;
}

/**
 * Get the local IP address
 * @returns {string} Local IP address
 */
function getLocalIP() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      // Skip internal/loopback and IPv6
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return '127.0.0.1';
}

/**
 * Determine the host to bind to based on network settings
 * @param {string|null} portableDataDir - Portable data directory or null
 * @returns {string} Host to bind to
 */
function getBindHost(portableDataDir) {
  const networkSettings = getNetworkSettings(portableDataDir);

  // If local network or public is enabled, bind to all interfaces
  if (networkSettings.local_network_enabled || networkSettings.public_network_enabled) {
    return '0.0.0.0';
  }

  // Default: localhost only
  return '127.0.0.1';
}

/**
 * Get persistent packages directory for pip packages
 * @returns {string} Packages directory path
 */
function getPackagesDir() {
  const dataDir = app.getPath('userData');
  return path.join(dataDir, 'packages');
}

module.exports = {
  detectPortableMode,
  getDataDir,
  getSettingsPath,
  getNetworkSettings,
  getLocalIP,
  getBindHost,
  getPackagesDir
};
