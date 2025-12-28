/**
 * Backend Manager
 * Manages the FastAPI backend as a subprocess
 */
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { app } = require('electron');

class BackendManager {
  constructor(port = 8790) {
    this.port = port;
    this.process = null;
    this.healthCheckInterval = null;
    this.restartAttempts = 0;
    this.maxRestartAttempts = 5;
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

    if (process.platform === 'win32') {
      const pythonPath = this.getPythonPath();
      const pythonDir = path.dirname(pythonPath);

      if (app.isPackaged || fs.existsSync(path.join(pythonDir, 'python.exe'))) {
        // Using bundled Python
        // Add onnxruntime capi folder to PATH for DLL loading
        const onnxCapi = path.join(pythonDir, 'Lib', 'site-packages', 'onnxruntime', 'capi');
        return {
          ...baseEnv,
          PATH: `${onnxCapi};${pythonDir};${path.join(pythonDir, 'Scripts')};${process.env.PATH}`,
          PYTHONHOME: pythonDir,
          PYTHONPATH: this.getWorkingDirectory()
        };
      }
    } else {
      // Linux/macOS with pyenv support
      const homeDir = process.env.HOME || process.env.USERPROFILE;
      const pyenvPath = `${homeDir}/.pyenv/shims:${homeDir}/.pyenv/bin`;
      return {
        ...baseEnv,
        PATH: `${pyenvPath}:${process.env.PATH}`
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

    console.log('[Backend] Starting server on port', this.port);

    const pythonPath = this.getPythonPath();
    const cwd = this.getWorkingDirectory();
    const env = this.getPythonEnv();

    console.log('[Backend] Python path:', pythonPath);
    console.log('[Backend] Working directory:', cwd);

    // Spawn uvicorn via python -m for better compatibility
    this.process = spawn(pythonPath, [
      '-m', 'uvicorn',
      'api.main:app',
      '--host', '127.0.0.1',
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
