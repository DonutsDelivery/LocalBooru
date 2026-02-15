/**
 * Backend Manager
 * Manages the FastAPI backend as a subprocess
 * Main coordinator class that uses config, process, and health modules
 */
const {
  detectPortableMode,
  getDataDir,
  getSettingsPath,
  getNetworkSettings,
  getLocalIP,
  getBindHost
} = require('./config');

const {
  spawnBackend,
  forceKillProcess,
  gracefulKillProcess
} = require('./process');

const {
  healthCheck,
  waitForReady,
  isPortFree,
  getPortUser,
  killZombieProcesses,
  killProcessesOnPort
} = require('./health');

class BackendManager {
  constructor() {
    this.process = null;
    this.healthCheckInterval = null;
    this.restartAttempts = 0;
    this.maxRestartAttempts = 5;
    this.portableDataDir = null;

    // Detect portable mode on construction
    this.portableDataDir = detectPortableMode();

    // Different default ports: portable=8791, system=8790
    // This allows running both simultaneously without conflicts
    this.defaultPort = this.portableDataDir ? 8791 : 8790;

    // Read port from settings (user can override the default)
    const networkSettings = this.getNetworkSettings();
    this.port = networkSettings.local_port || this.defaultPort;
    console.log('[Backend] Mode:', this.portableDataDir ? 'portable' : 'system', '| Port:', this.port);
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
    return getDataDir(this.portableDataDir);
  }

  /**
   * Get settings.json path
   */
  getSettingsPath() {
    return getSettingsPath(this.portableDataDir);
  }

  /**
   * Load network settings from settings.json
   */
  getNetworkSettings() {
    return getNetworkSettings(this.portableDataDir);
  }

  /**
   * Get the local IP address
   */
  getLocalIP() {
    return getLocalIP();
  }

  /**
   * Determine the host to bind to based on network settings
   */
  getBindHost() {
    return getBindHost(this.portableDataDir);
  }

  /**
   * Get information about what process is using the port
   */
  getPortUser() {
    return getPortUser(this.port);
  }

  /**
   * Kill any zombie processes on our port
   * Throws PortConflictError if port cannot be freed
   */
  async killZombieProcesses() {
    return killZombieProcesses(this.port);
  }

  /**
   * Check if port is free
   */
  isPortFree() {
    return isPortFree(this.port);
  }

  /**
   * Start the backend server
   */
  async start() {
    if (this.process) {
      console.log('[Backend] Already running');
      return;
    }

    // Clean up zombie processes on OUR port only (not all uvicorn system-wide)
    // This allows multiple instances (portable + system) to coexist
    try {
      await this.killZombieProcesses();
    } catch (e) {
      // Log but don't fail - we'll try to start anyway and let uvicorn fail if port is busy
      console.error('[Backend] Warning: Could not free port:', e.message);
    }

    console.log('[Backend] Starting server on port', this.port);

    const bindHost = this.getBindHost();
    const networkSettings = this.getNetworkSettings();

    console.log('[Backend] Binding to:', bindHost);
    if (bindHost === '0.0.0.0') {
      const localIP = this.getLocalIP();
      console.log('[Backend] Local network access:', networkSettings.local_network_enabled ? `enabled (http://${localIP}:${this.port})` : 'disabled');
      console.log('[Backend] Public access:', networkSettings.public_network_enabled ? 'enabled' : 'disabled');
    }

    // Spawn the backend process
    this.process = spawnBackend({
      bindHost,
      port: this.port,
      portableDataDir: this.portableDataDir
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
    await waitForReady(this.port);

    // Start health check
    this.startHealthCheck();

    // Reset restart counter on successful start
    this.restartAttempts = 0;
  }

  /**
   * Wait for the backend to respond to health checks
   */
  async waitForReady(timeout = 30000) {
    return waitForReady(this.port, timeout);
  }

  /**
   * Check if backend is healthy and responding with valid content
   */
  healthCheck() {
    return healthCheck(this.port);
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
      const http = require('http');
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
        gracefulKillProcess(this.process.pid);
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
      forceKillProcess(this.process.pid);
      this.process = null;
    }

    // Kill anything on our port (not all uvicorn system-wide)
    killProcessesOnPort(this.port);
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
