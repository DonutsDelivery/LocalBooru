/**
 * Backend Manager
 * Manages the FastAPI backend as a subprocess
 */
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const net = require('net');
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
   * Portable mode is DEFAULT for packaged apps, unless:
   * - Running from Program Files (Windows installer location)
   * - A '.use-appdata' marker file exists next to the executable
   */
  detectPortableMode() {
    try {
      // Get the directory containing the app
      let appDir;
      if (app.isPackaged) {
        // Packaged app: use the directory containing the executable
        appDir = path.dirname(app.getPath('exe'));
      } else {
        // Development: don't use portable mode
        return;
      }

      const portableDataPath = path.join(appDir, 'data');
      const useAppdataMarker = path.join(appDir, '.use-appdata');

      // Check if we should use AppData instead of portable mode
      if (fs.existsSync(useAppdataMarker)) {
        console.log('[Backend] Found .use-appdata marker, using AppData');
        return;
      }

      // Check if installed in Program Files (Windows)
      if (process.platform === 'win32') {
        const programFiles = process.env.ProgramFiles || 'C:\\Program Files';
        const programFilesX86 = process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)';
        if (appDir.startsWith(programFiles) || appDir.startsWith(programFilesX86)) {
          console.log('[Backend] Running from Program Files, using AppData');
          return;
        }
      }

      // Default: use portable mode - create data folder next to exe
      if (!fs.existsSync(portableDataPath)) {
        fs.mkdirSync(portableDataPath, { recursive: true });
      }
      this.portableDataDir = portableDataPath;
      console.log('[Backend] Portable mode enabled, data:', portableDataPath);
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
   * Get information about what process is using a port
   */
  getPortUser() {
    try {
      if (process.platform === 'win32') {
        const output = execSync(
          `netstat -ano | findstr :${this.port} | findstr LISTENING`,
          { encoding: 'utf-8', timeout: 5000, windowsHide: true }
        );
        const lines = output.trim().split('\n');
        if (lines.length > 0) {
          const parts = lines[0].trim().split(/\s+/);
          const pid = parts[parts.length - 1];
          if (pid && /^\d+$/.test(pid)) {
            try {
              const taskInfo = execSync(`tasklist /FI "PID eq ${pid}" /FO CSV /NH`,
                { encoding: 'utf-8', timeout: 3000, windowsHide: true });
              const name = taskInfo.split(',')[0]?.replace(/"/g, '') || 'unknown';
              return { pid, name };
            } catch (e) {
              return { pid, name: 'unknown' };
            }
          }
        }
      } else {
        // Linux/macOS
        const output = execSync(`lsof -ti:${this.port}`, { encoding: 'utf-8', timeout: 5000 });
        const pid = output.trim().split('\n')[0];
        if (pid && /^\d+$/.test(pid)) {
          try {
            const name = execSync(`ps -p ${pid} -o comm=`, { encoding: 'utf-8', timeout: 3000 }).trim();
            return { pid, name };
          } catch (e) {
            return { pid, name: 'unknown' };
          }
        }
      }
    } catch (e) {
      // No process found or error
    }
    return null;
  }

  /**
   * Kill any zombie processes on our port
   * Throws PortConflictError if port cannot be freed
   */
  async killZombieProcesses() {
    console.log('[Backend] Checking for zombie processes on port', this.port);

    const killAttempt = async () => {
      try {
        if (process.platform === 'win32') {
          // Windows: synchronously find and kill processes on port
          try {
            const output = execSync(
              `netstat -ano | findstr :${this.port} | findstr LISTENING`,
              { encoding: 'utf-8', timeout: 5000, windowsHide: true }
            );
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
            // No process on port
          }
        } else {
          // Linux/macOS: use lsof + kill -9 directly (most reliable)
          try {
            const output = execSync(`lsof -ti:${this.port}`, { encoding: 'utf-8', timeout: 5000 });
            const pids = output.trim().split('\n').filter(p => p && /^\d+$/.test(p));
            for (const pid of pids) {
              console.log(`[Backend] Killing zombie process ${pid}`);
              try {
                execSync(`kill -9 ${pid}`, { stdio: 'ignore', timeout: 1000 });
              } catch (e) {
                // Process may already be dead
              }
            }
          } catch (e) {
            // No process on port (lsof returns error if no match)
          }
        }
      } catch (e) {
        console.log('[Backend] Zombie check error:', e.message);
      }
    };

    // Try killing up to 3 times with waits between
    for (let attempt = 1; attempt <= 3; attempt++) {
      await killAttempt();
      await new Promise(resolve => setTimeout(resolve, 500));

      const portFree = await this.isPortFree();
      if (portFree) {
        console.log('[Backend] Port is free');
        return;
      }
      console.log(`[Backend] Port still in use after attempt ${attempt}, retrying...`);
    }

    // Final check - if still occupied, throw error with details
    const portFree = await this.isPortFree();
    if (!portFree) {
      const portUser = this.getPortUser();
      const error = new Error(`Port ${this.port} is already in use`);
      error.code = 'PORT_CONFLICT';
      error.port = this.port;
      error.portUser = portUser;
      console.error('[Backend] CRITICAL: Port', this.port, 'occupied by:', portUser);
      throw error;
    }
  }

  /**
   * Check if port is free
   */
  isPortFree() {
    return new Promise((resolve) => {
      const server = net.createServer();
      server.once('error', () => resolve(false));
      server.once('listening', () => {
        server.close();
        resolve(true);
      });
      server.listen(this.port, '127.0.0.1');
    });
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

    // AGGRESSIVE cleanup - kill ALL uvicorn processes, not just on our port
    // This prevents zombie processes from previous sessions
    console.log('[Backend] Cleaning up any existing uvicorn processes...');
    try {
      if (process.platform === 'win32') {
        execSync('taskkill /F /IM python.exe /FI "WINDOWTITLE eq uvicorn*" 2>nul', { stdio: 'ignore', timeout: 5000 });
      } else {
        // Kill any uvicorn process for api.main:app specifically
        execSync('pkill -9 -f "uvicorn api.main:app" 2>/dev/null || true', { stdio: 'ignore', timeout: 5000 });
      }
      // Give OS time to release the port
      await new Promise(resolve => setTimeout(resolve, 1000));
    } catch (e) {
      // Ignore errors - process might not exist
    }

    // Now do the standard zombie kill on port
    try {
      await this.killZombieProcesses();
    } catch (e) {
      // Log but don't fail - we'll try to start anyway and let uvicorn fail if port is busy
      console.error('[Backend] Warning: Could not free port:', e.message);
    }

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
   * Check if backend is healthy and responding with valid content
   */
  healthCheck() {
    return new Promise((resolve) => {
      const req = http.get(`http://127.0.0.1:${this.port}/health`, (res) => {
        if (res.statusCode !== 200) {
          resolve(false);
          return;
        }

        // Read the response body to verify it's our backend
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            const json = JSON.parse(data);
            // Our health endpoint returns { "status": "healthy" }
            resolve(json.status === 'healthy');
          } catch (e) {
            // Invalid JSON response
            resolve(false);
          }
        });
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
   * Stop the backend server gracefully
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

    console.log('[Backend] Initiating graceful shutdown...');

    // First, try to notify the backend via a shutdown request
    // This gives FastAPI time to cleanup before SIGTERM
    try {
      await new Promise((resolve, reject) => {
        const req = http.request({
          hostname: '127.0.0.1',
          port: this.port,
          path: '/health',  // Just check it's responding
          method: 'GET',
          timeout: 1000
        }, () => resolve());
        req.on('error', () => resolve());  // Ignore errors
        req.on('timeout', () => { req.destroy(); resolve(); });
        req.end();
      });
    } catch (e) {
      // Backend may already be unresponsive
    }

    return new Promise((resolve) => {
      // Give more time for graceful shutdown (thread pools, db connections, etc.)
      const timeout = setTimeout(() => {
        console.log('[Backend] Graceful shutdown timeout, force killing...');
        this.forceKill();
        resolve();
      }, 10000);  // Increased from 5s to 10s

      this.process.once('exit', () => {
        clearTimeout(timeout);
        this.process = null;
        console.log('[Backend] Stopped gracefully');
        resolve();
      });

      // Send SIGINT first (Ctrl+C equivalent) for cleaner uvicorn shutdown
      // Then SIGTERM if needed
      if (process.platform === 'win32') {
        // Windows: CTRL_C_EVENT doesn't work well with spawn, use taskkill
        try {
          // First try graceful tree kill (no /F flag)
          execSync(`taskkill /pid ${this.process.pid} /T`, { stdio: 'ignore', timeout: 3000 });
        } catch (e) {
          // If that fails, the process exit handler will trigger force kill via timeout
        }
      } else {
        // Unix: Send SIGINT first (cleaner uvicorn shutdown)
        console.log('[Backend] Sending SIGINT...');
        this.process.kill('SIGINT');

        // If still running after 5s, escalate to SIGTERM
        setTimeout(() => {
          if (this.process) {
            console.log('[Backend] Escalating to SIGTERM...');
            this.process.kill('SIGTERM');
          }
        }, 5000);
      }
    });
  }

  /**
   * Force kill the backend process AND any zombie uvicorn processes
   */
  forceKill() {
    // Kill our tracked process
    if (this.process) {
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

    // ALSO kill any other uvicorn processes that might be zombies
    // This is aggressive but prevents the blank window issue
    try {
      if (process.platform === 'win32') {
        execSync('taskkill /F /IM python.exe /FI "WINDOWTITLE eq uvicorn*" 2>nul', { stdio: 'ignore', timeout: 3000 });
      } else {
        execSync('pkill -9 -f "uvicorn api.main:app" 2>/dev/null || true', { stdio: 'ignore', timeout: 3000 });
      }
    } catch (e) {
      // Ignore
    }

    // Kill anything on our port
    try {
      if (process.platform !== 'win32') {
        const output = execSync(`lsof -ti:${this.port}`, { encoding: 'utf-8', timeout: 3000 });
        const pids = output.trim().split('\n').filter(p => p && /^\d+$/.test(p));
        for (const pid of pids) {
          try { execSync(`kill -9 ${pid}`, { stdio: 'ignore', timeout: 1000 }); } catch (e) {}
        }
      }
    } catch (e) {
      // No process on port
    }
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
