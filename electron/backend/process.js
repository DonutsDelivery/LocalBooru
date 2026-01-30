/**
 * Backend Process Management
 * Handles Python executable detection, environment setup, and process spawning
 */
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const { app } = require('electron');
const { getPackagesDir } = require('./config');

/**
 * Get the Python executable path based on platform
 * @returns {string} Path to Python executable
 */
function getPythonPath() {
  const isPackaged = app.isPackaged;

  if (process.platform === 'win32') {
    if (isPackaged) {
      // Bundled Python in resources folder
      return path.join(process.resourcesPath, 'python-embed', 'python.exe');
    } else {
      // Development: check for local python-embed folder
      const localEmbed = path.join(__dirname, '..', '..', 'python-embed', 'python.exe');
      if (fs.existsSync(localEmbed)) {
        return localEmbed;
      }
      // Development: check for standard .venv folder
      const dotVenv = path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe');
      if (fs.existsSync(dotVenv)) {
        return dotVenv;
      }
      // Fall back to system Python
      return 'python';
    }
  } else {
    // Linux/macOS
    if (isPackaged) {
      // Check for bundled venv in resources folder
      const bundledPython = path.join(process.resourcesPath, 'python-venv', 'bin', 'python');
      if (fs.existsSync(bundledPython)) {
        return bundledPython;
      }
    } else {
      // Development: check for local python-venv-linux folder
      const localVenv = path.join(__dirname, '..', '..', 'python-venv-linux', 'bin', 'python');
      if (fs.existsSync(localVenv)) {
        return localVenv;
      }
      // Development: check for standard .venv folder
      const dotVenv = path.join(__dirname, '..', '..', '.venv', 'bin', 'python');
      if (fs.existsSync(dotVenv)) {
        return dotVenv;
      }
    }
    // Fall back to system Python
    return 'python';
  }
}

/**
 * Get the working directory for the Python process
 * @returns {string} Working directory path
 */
function getWorkingDirectory() {
  if (app.isPackaged) {
    // Check if we have bundled resources (api folder in resources/)
    const bundledApiPath = path.join(process.resourcesPath, 'api');
    if (fs.existsSync(bundledApiPath)) {
      // Windows or Linux with bundled venv: api folder is in resources/
      return process.resourcesPath;
    }
    // Fallback: api folder is in resources/app (unbundled Linux/macOS)
    return path.join(process.resourcesPath, 'app');
  }
  return path.join(__dirname, '..', '..');
}

/**
 * Get environment variables for Python subprocess
 * @param {string|null} portableDataDir - Portable data directory or null
 * @returns {Object} Environment variables object
 */
function getPythonEnv(portableDataDir) {
  const baseEnv = { ...process.env, PYTHONUNBUFFERED: '1' };
  const packagesDir = getPackagesDir();

  // Ensure packages directory exists
  if (!fs.existsSync(packagesDir)) {
    fs.mkdirSync(packagesDir, { recursive: true });
  }

  // Add portable data directory if in portable mode
  if (portableDataDir) {
    baseEnv.LOCALBOORU_PORTABLE_DATA = portableDataDir;
  }

  if (process.platform === 'win32') {
    const pythonPath = getPythonPath();
    const pythonDir = path.dirname(pythonPath);

    if (app.isPackaged || fs.existsSync(path.join(pythonDir, 'python.exe'))) {
      // Using bundled Python
      // Add onnxruntime capi folder to PATH for DLL loading
      const onnxCapi = path.join(packagesDir, 'onnxruntime', 'capi');
      const bundledOnnxCapi = path.join(pythonDir, 'Lib', 'site-packages', 'onnxruntime', 'capi');

      // Bundled video pipeline tools (ffmpeg, vapoursynth)
      const rootDir = app.isPackaged ? process.resourcesPath : path.join(__dirname, '..', '..');
      const ffmpegDir = path.join(rootDir, 'ffmpeg');
      const vsDir = path.join(rootDir, 'vapoursynth');
      const vsPluginsDir = path.join(vsDir, 'vs-plugins');

      // Build PATH with video tools prepended (if present)
      let pathPrefix = `${onnxCapi};${bundledOnnxCapi};${pythonDir};${path.join(pythonDir, 'Scripts')}`;
      if (fs.existsSync(ffmpegDir)) {
        pathPrefix = `${ffmpegDir};${pathPrefix}`;
      }
      if (fs.existsSync(vsDir)) {
        pathPrefix = `${vsDir};${pathPrefix}`;
      }

      const env = {
        ...baseEnv,
        PATH: `${pathPrefix};${process.env.PATH}`,
        PYTHONHOME: pythonDir,
        PYTHONPATH: `${packagesDir};${getWorkingDirectory()};${path.join(pythonDir, 'Lib', 'site-packages')}`,
        LOCALBOORU_PACKAGED: '1',
        LOCALBOORU_PACKAGES_DIR: packagesDir
      };

      // Set bundled video tool env vars if directories exist
      if (fs.existsSync(path.join(vsDir, 'python.exe'))) {
        env.LOCALBOORU_VS_PYTHON = path.join(vsDir, 'python.exe');
      }
      if (fs.existsSync(vsPluginsDir)) {
        env.LOCALBOORU_SVP_PLUGIN_PATH = vsPluginsDir;
        env.LOCALBOORU_VS_PLUGIN_PATH = vsPluginsDir;
      }

      return env;
    }
  } else {
    // Linux/macOS
    const pythonPath = getPythonPath();
    const pythonDir = path.dirname(path.dirname(pythonPath)); // go up from bin/python to venv root

    // Check if using bundled venv
    const isBundledVenv = pythonPath.includes('python-venv');

    if (isBundledVenv) {
      // Using bundled venv - find the site-packages directory
      const libDir = path.join(pythonDir, 'lib');
      let sitePackagesPath = '';

      // Find the python3.X directory inside lib
      if (fs.existsSync(libDir)) {
        const entries = fs.readdirSync(libDir);
        for (const entry of entries) {
          if (entry.startsWith('python3.')) {
            const spPath = path.join(libDir, entry, 'site-packages');
            if (fs.existsSync(spPath)) {
              sitePackagesPath = spPath;
              break;
            }
          }
        }
      }

      return {
        ...baseEnv,
        // Add bundled venv's bin to PATH
        PATH: `${path.join(pythonDir, 'bin')}:${process.env.PATH}`,
        // Include persistent packages, working directory, and bundled site-packages
        PYTHONPATH: `${packagesDir}:${getWorkingDirectory()}${sitePackagesPath ? ':' + sitePackagesPath : ''}`,
        LOCALBOORU_PACKAGED: app.isPackaged ? '1' : '',
        LOCALBOORU_PACKAGES_DIR: packagesDir
      };
    } else {
      // System Python with pyenv support
      const homeDir = process.env.HOME || process.env.USERPROFILE;
      const pyenvPath = `${homeDir}/.pyenv/shims:${homeDir}/.pyenv/bin`;
      return {
        ...baseEnv,
        PATH: `${pyenvPath}:${process.env.PATH}`,
        PYTHONPATH: `${packagesDir}:${getWorkingDirectory()}`,
        LOCALBOORU_PACKAGED: app.isPackaged ? '1' : '',
        LOCALBOORU_PACKAGES_DIR: packagesDir
      };
    }
  }

  return baseEnv;
}

/**
 * Spawn the uvicorn backend process
 * @param {Object} options - Spawn options
 * @param {string} options.bindHost - Host to bind to
 * @param {number} options.port - Port to bind to
 * @param {string|null} options.portableDataDir - Portable data directory or null
 * @returns {ChildProcess} Spawned process
 */
function spawnBackend({ bindHost, port, portableDataDir }) {
  const pythonPath = getPythonPath();
  const cwd = getWorkingDirectory();
  const env = getPythonEnv(portableDataDir);

  console.log('[Backend] Python path:', pythonPath);
  console.log('[Backend] Working directory:', cwd);
  console.log('[Backend] Binding to:', bindHost);

  // Spawn uvicorn via python -m for better compatibility
  return spawn(pythonPath, [
    '-m', 'uvicorn',
    'api.main:app',
    '--host', bindHost,
    '--port', String(port)
  ], {
    cwd: cwd,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: env,
    // Windows-specific: hide console window
    windowsHide: true
  });
}

/**
 * Kill uvicorn processes aggressively
 * Used for cleanup on startup and shutdown
 */
function killUvicornProcesses() {
  try {
    if (process.platform === 'win32') {
      execSync('taskkill /F /IM python.exe /FI "WINDOWTITLE eq uvicorn*" 2>nul', { stdio: 'ignore', timeout: 5000 });
    } else {
      // Kill any uvicorn process for api.main:app specifically
      execSync('pkill -9 -f "uvicorn api.main:app" 2>/dev/null || true', { stdio: 'ignore', timeout: 5000 });
    }
  } catch (e) {
    // Ignore errors - process might not exist
  }
}

/**
 * Force kill a specific process by PID
 * @param {number} pid - Process ID to kill
 */
function forceKillProcess(pid) {
  try {
    if (process.platform === 'win32') {
      execSync(`taskkill /pid ${pid} /T /F`, { stdio: 'ignore' });
    } else {
      process.kill(pid, 'SIGKILL');
    }
  } catch (e) {
    // Process may already be dead
  }
}

/**
 * Gracefully terminate a process (Windows)
 * @param {number} pid - Process ID to terminate
 */
function gracefulKillProcess(pid) {
  try {
    if (process.platform === 'win32') {
      // First try graceful tree kill (no /F flag)
      execSync(`taskkill /pid ${pid} /T`, { stdio: 'ignore', timeout: 3000 });
    }
  } catch (e) {
    // If that fails, the caller should handle force kill
  }
}

module.exports = {
  getPythonPath,
  getWorkingDirectory,
  getPythonEnv,
  spawnBackend,
  killUvicornProcesses,
  forceKillProcess,
  gracefulKillProcess
};
