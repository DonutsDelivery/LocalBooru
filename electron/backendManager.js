/**
 * Backend Manager
 * Manages the FastAPI backend as a subprocess
 */
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const os = require('os');
const { app } = require('electron');

class BackendManager {
  constructor(port = 8790) {
    this.port = port;
    this.process = null;
    this.healthCheckInterval = null;
    this.restartAttempts = 0;
    this.maxRestartAttempts = 5;
    this.portableDataDir = null;

    // Detect portable mode on construction
    this.detectPortableMode();
  }

  /**
   * Detect if running in portable mode
   * Portable mode is enabled when:
   * - A 'data' folder exists next to the app executable, OR
   * - A '.portable' marker file exists next to the executable
   */
  detectPortableMode() {
    try {
      // Get the directory containing the app
      let appDir;
      if (app.isPackaged) {
        // Packaged app: use the directory containing the executable
        appDir = path.dirname(app.getPath('exe'));
      } else {
        // Development: check project root
        appDir = path.join(__dirname, '..');
      }

      const portableDataPath = path.join(appDir, 'data');
      const portableMarker = path.join(appDir, '.portable');

      // Check for portable marker or data folder
      if (fs.existsSync(portableMarker) || fs.existsSync(portableDataPath)) {
        // Create data folder if it doesn't exist
        if (!fs.existsSync(portableDataPath)) {
          fs.mkdirSync(portableDataPath, { recursive: true });
        }
        this.portableDataDir = portableDataPath;
        console.log('[Backend] Portable mode enabled, data:', portableDataPath);
      }
    } catch (e) {
      console.log('[Backend] Error detecting portable mode:', e.message);
    }
  }

  /**
   * Check if running in portable mode
   */
  isPortable() {
    return this.portableDataDir !== null;
  }

  /**
   * Get LocalBooru data directory (matches Python API location)
   */
  getDataDir() {
    // Use portable data directory if in portable mode
    if (this.portableDataDir) {
      return this.portableDataDir;
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
   */
  getSettingsPath() {
    return path.join(this.getDataDir(), 'settings.json');
  }

  /**
   * Load network settings from settings.json
   */
  getNetworkSettings() {
    const defaults = {
      local_network_enabled: false,
      public_network_enabled: false,
      local_port: 8790,
      public_port: 8791,
      auth_required_level: 'none',
      upnp_enabled: false
    };

    try {
      const settingsPath = this.getSettingsPath();
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
   */
  getLocalIP() {
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
   */
  getBindHost() {
    const networkSettings = this.getNetworkSettings();

    // If local network or public is enabled, bind to all interfaces
    if (networkSettings.local_network_enabled || networkSettings.public_network_enabled) {
      return '0.0.0.0';
    }

    // Default: localhost only
    return '127.0.0.1';
  }

  /**
   * Get persistent packages directory for pip packages
   */
  getPackagesDir() {
    const dataDir = app.getPath('userData');
    return path.join(dataDir, 'packages');
  }

  /**
   * Kill any zombie processes on our port
   */
  async killZombieProcesses() {
    console.log('[Backend] Checking for zombie processes on port', this.port);

    try {
      if (process.platform === 'win32') {
        // Windows: synchronously find and kill processes on port
        // Using netstat output parsing
        try {
          const output = execSync(
            `netstat -ano | findstr :${this.port} | findstr LISTENING`,
            { encoding: 'utf-8', timeout: 5000, windowsHide: true }
          );
          // Parse PIDs from output (last column)
          const pids = new Set();
          for (const line of output.trim().split('\n')) {
            const parts = line.trim().split(/\s+/);
            const pid = parts[parts.length - 1];
            if (pid && /^\d+$/.test(pid) && pid !== '0') {
              pids.add(pid);
            }
          }
          for (const pid of pids) {
            console.log(`[Backend] Killing zombie process ${pid}`);
            try {
              execSync(`taskkill /F /PID ${pid}`, { stdio: 'ignore', timeout: 3000, windowsHide: true });
            } catch (e) {
              // Process may already be dead
            }
          }
        } catch (e) {
          // No process on port (findstr returns error if no match)
        }
      } else {
        // Linux/macOS: use fuser (fast and simple)
        try {
          execSync(`fuser -k ${this.port}/tcp 2>/dev/null`, { stdio: 'ignore', timeout: 3000 });
        } catch (e) {
          // Ignore - no process on port or fuser not available
        }
      }
    } catch (e) {
      console.log('[Backend] Zombie check error:', e.message);
    }

    // Wait for port to be released
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  /**
   * Get the Python executable path based on platform
   */
  getPythonPath() {
    const isPackaged = app.isPackaged;

    if (process.platform === 'win32') {
      if (isPackaged) {
        // Bundled Python in resources folder
        return path.join(process.resourcesPath, 'python-embed', 'python.exe');
      } else {
        // Development: check for local python-embed folder
        const localEmbed = path.join(__dirname, '..', 'python-embed', 'python.exe');
        if (fs.existsSync(localEmbed)) {
          return localEmbed;
        }
        // Fall back to system Python
        return 'python';
      }
    } else {
      // Linux/macOS - use system Python
      return 'python';
    }
  }

  /**
   * Get the working directory for the Python process
   */
  getWorkingDirectory() {
    if (app.isPackaged) {
      if (process.platform === 'win32') {
        // Windows: api folder is extracted to resources/
        return process.resourcesPath;
      }
      // Linux/macOS: api folder is in resources/app
      return path.join(process.resourcesPath, 'app');
    }
    return path.join(__dirname, '..');
  }

  /**
   * Get environment variables for Python subprocess
   */
  getPythonEnv() {
    const baseEnv = { ...process.env, PYTHONUNBUFFERED: '1' };
    const packagesDir = this.getPackagesDir();

    // Ensure packages directory exists
    if (!fs.existsSync(packagesDir)) {
      fs.mkdirSync(packagesDir, { recursive: true });
    }

    // Add portable data directory if in portable mode
    if (this.portableDataDir) {
      baseEnv.LOCALBOORU_PORTABLE_DATA = this.portableDataDir;
    }

    if (process.platform === 'win32') {
      const pythonPath = this.getPythonPath();
      const pythonDir = path.dirname(pythonPath);

      if (app.isPackaged || fs.existsSync(path.join(pythonDir, 'python.exe'))) {
        // Using bundled Python
        // Add onnxruntime capi folder to PATH for DLL loading
        const onnxCapi = path.join(packagesDir, 'onnxruntime', 'capi');
        const bundledOnnxCapi = path.join(pythonDir, 'Lib', 'site-packages', 'onnxruntime', 'capi');
        return {
          ...baseEnv,
          // Add persistent packages to PATH first (for DLLs), then bundled, then system
          PATH: `${onnxCapi};${bundledOnnxCapi};${pythonDir};${path.join(pythonDir, 'Scripts')};${process.env.PATH}`,
          PYTHONHOME: pythonDir,
          // Persistent packages first, then working directory, then bundled site-packages
          PYTHONPATH: `${packagesDir};${this.getWorkingDirectory()};${path.join(pythonDir, 'Lib', 'site-packages')}`,
          LOCALBOORU_PACKAGED: '1',
          LOCALBOORU_PACKAGES_DIR: packagesDir
        };
      }
    } else {
      // Linux/macOS with pyenv support
      const homeDir = process.env.HOME || process.env.USERPROFILE;
      const pyenvPath = `${homeDir}/.pyenv/shims:${homeDir}/.pyenv/bin`;
      return {
        ...baseEnv,
        PATH: `${pyenvPath}:${process.env.PATH}`,
        PYTHONPATH: `${packagesDir}:${this.getWorkingDirectory()}`,
        LOCALBOORU_PACKAGED: app.isPackaged ? '1' : '',
        LOCALBOORU_PACKAGES_DIR: packagesDir
      };
    }

    return baseEnv;
  }

  /**
   * Start the backend server
   */
  async start() {
    if (this.process) {
      console.log('[Backend] Already running');
      return;
    }

    // Kill any zombie processes first
    await this.killZombieProcesses();

    console.log('[Backend] Starting server on port', this.port);

    const pythonPath = this.getPythonPath();
    const cwd = this.getWorkingDirectory();
    const env = this.getPythonEnv();

    const bindHost = this.getBindHost();
    const networkSettings = this.getNetworkSettings();

    console.log('[Backend] Python path:', pythonPath);
    console.log('[Backend] Working directory:', cwd);
    console.log('[Backend] Binding to:', bindHost);
    if (bindHost === '0.0.0.0') {
      const localIP = this.getLocalIP();
      console.log('[Backend] Local network access:', networkSettings.local_network_enabled ? `enabled (http://${localIP}:${this.port})` : 'disabled');
      console.log('[Backend] Public access:', networkSettings.public_network_enabled ? 'enabled' : 'disabled');
    }

    // Spawn uvicorn via python -m for better compatibility
    this.process = spawn(pythonPath, [
      '-m', 'uvicorn',
      'api.main:app',
      '--host', bindHost,
      '--port', String(this.port)
    ], {
      cwd: cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: env,
      // Windows-specific: hide console window
      windowsHide: true
    });

    // Handle stdout
    this.process.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        console.log('[Backend]', output);
      }
    });

    // Handle stderr
    this.process.stderr.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        // Uvicorn logs to stderr, so not all are errors
        if (output.includes('ERROR') || output.includes('Exception')) {
          console.error('[Backend Error]', output);
        } else {
          console.log('[Backend]', output);
        }
      }
    });

    // Handle exit
    this.process.on('exit', (code, signal) => {
      console.log(`[Backend] Exited with code ${code}, signal ${signal}`);
      this.process = null;

      // Auto-restart if crashed unexpectedly
      if (code !== 0 && this.restartAttempts < this.maxRestartAttempts) {
        this.restartAttempts++;
        console.log(`[Backend] Restarting (attempt ${this.restartAttempts}/${this.maxRestartAttempts})...`);
        setTimeout(() => this.start(), 2000);
      }
    });

    // Handle error
    this.process.on('error', (err) => {
      console.error('[Backend] Failed to start:', err);
      this.process = null;
    });

    // Wait for backend to be ready
    await this.waitForReady();

    // Start health check
    this.startHealthCheck();

    // Reset restart counter on successful start
    this.restartAttempts = 0;
  }

  /**
   * Wait for the backend to respond to health checks
   */
  async waitForReady(timeout = 30000) {
    const startTime = Date.now();

    return new Promise((resolve, reject) => {
      const check = async () => {
        if (Date.now() - startTime > timeout) {
          reject(new Error('Backend startup timeout'));
          return;
        }

        try {
          const healthy = await this.healthCheck();
          if (healthy) {
            console.log('[Backend] Server ready');
            resolve();
            return;
          }
        } catch (e) {
          // Not ready yet
        }

        setTimeout(check, 500);
      };

      check();
    });
  }

  /**
   * Check if backend is healthy
   */
  healthCheck() {
    return new Promise((resolve) => {
      const req = http.get(`http://127.0.0.1:${this.port}/health`, (res) => {
        resolve(res.statusCode === 200);
      });

      req.on('error', () => resolve(false));
      req.setTimeout(2000, () => {
        req.destroy();
        resolve(false);
      });
    });
  }

  /**
   * Start periodic health checks
   */
  startHealthCheck() {
    this.healthCheckInterval = setInterval(async () => {
      if (this.process) {
        const healthy = await this.healthCheck();
        if (!healthy) {
          console.warn('[Backend] Health check failed');
        }
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Stop the backend server
   */
  async stop() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }

    if (!this.process) {
      console.log('[Backend] Not running');
      return;
    }

    console.log('[Backend] Stopping server...');

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        console.log('[Backend] Force killing...');
        this.forceKill();
        resolve();
      }, 5000);

      this.process.once('exit', () => {
        clearTimeout(timeout);
        this.process = null;
        console.log('[Backend] Stopped');
        resolve();
      });

      // Windows doesn't support SIGTERM well, use different approach
      if (process.platform === 'win32') {
        // Try graceful kill first, then force
        try {
          execSync(`taskkill /pid ${this.process.pid} /T`, { stdio: 'ignore' });
        } catch (e) {
          this.forceKill();
        }
      } else {
        this.process.kill('SIGTERM');
      }
    });
  }

  /**
   * Force kill the backend process
   */
  forceKill() {
    if (!this.process) return;

    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /pid ${this.process.pid} /T /F`, { stdio: 'ignore' });
      } else {
        this.process.kill('SIGKILL');
      }
    } catch (e) {
      // Process may already be dead
    }
    this.process = null;
  }

  /**
   * Restart the backend server
   */
  async restart() {
    await this.stop();
    await this.start();
  }

  /**
   * Check if backend is running
   */
  isRunning() {
    return this.process !== null;
  }
}

module.exports = BackendManager;
