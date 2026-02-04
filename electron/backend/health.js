/**
 * Backend Health Management
 * Handles health checks, port management, and zombie process cleanup
 */
const { execSync } = require('child_process');
const http = require('http');
const net = require('net');

/**
 * Check if backend is healthy and responding with valid content
 * @param {number} port - Port to check
 * @returns {Promise<boolean>} True if healthy
 */
function healthCheck(port) {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/health`, (res) => {
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
 * Wait for the backend to respond to health checks
 * @param {number} port - Port to check
 * @param {number} timeout - Timeout in milliseconds
 * @returns {Promise<void>}
 */
async function waitForReady(port, timeout = 30000) {
  const startTime = Date.now();

  return new Promise((resolve, reject) => {
    const check = async () => {
      if (Date.now() - startTime > timeout) {
        reject(new Error('Backend startup timeout'));
        return;
      }

      try {
        const healthy = await healthCheck(port);
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
 * Check if port is free
 * @param {number} port - Port to check
 * @returns {Promise<boolean>} True if port is free
 */
function isPortFree(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.once('listening', () => {
      server.close();
      resolve(true);
    });
    server.listen(port, '127.0.0.1');
  });
}

/**
 * Get information about what process is using a port
 * @param {number} port - Port to check
 * @returns {Object|null} Process info { pid, name } or null
 */
function getPortUser(port) {
  try {
    if (process.platform === 'win32') {
      const output = execSync(
        `netstat -ano | findstr :${port} | findstr LISTENING`,
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
      const output = execSync(`lsof -ti:${port}`, { encoding: 'utf-8', timeout: 5000 });
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
 * Kill any zombie processes on a port
 * Throws PortConflictError if port cannot be freed
 * @param {number} port - Port to free
 * @returns {Promise<void>}
 */
async function killZombieProcesses(port) {
  console.log('[Backend] Checking for zombie processes on port', port);

  const killAttempt = async () => {
    try {
      if (process.platform === 'win32') {
        // Windows: synchronously find and kill processes on port
        try {
          const output = execSync(
            `netstat -ano | findstr :${port} | findstr LISTENING`,
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
          const output = execSync(`lsof -ti:${port}`, { encoding: 'utf-8', timeout: 5000 });
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

    const portFree = await isPortFree(port);
    if (portFree) {
      console.log('[Backend] Port is free');
      return;
    }
    console.log(`[Backend] Port still in use after attempt ${attempt}, retrying...`);
  }

  // Final check - if still occupied, throw error with details
  const portFree = await isPortFree(port);
  if (!portFree) {
    const portUser = getPortUser(port);
    const error = new Error(`Port ${port} is already in use`);
    error.code = 'PORT_CONFLICT';
    error.port = port;
    error.portUser = portUser;
    console.error('[Backend] CRITICAL: Port', port, 'occupied by:', portUser);
    throw error;
  }
}

/**
 * Kill processes on a specific port
 * @param {number} port - Port to free
 */
function killProcessesOnPort(port) {
  try {
    if (process.platform !== 'win32') {
      const output = execSync(`lsof -ti:${port}`, { encoding: 'utf-8', timeout: 3000 });
      const pids = output.trim().split('\n').filter(p => p && /^\d+$/.test(p));
      for (const pid of pids) {
        try { execSync(`kill -9 ${pid}`, { stdio: 'ignore', timeout: 1000 }); } catch (e) {}
      }
    }
  } catch (e) {
    // No process on port
  }
}

module.exports = {
  healthCheck,
  waitForReady,
  isPortFree,
  getPortUser,
  killZombieProcesses,
  killProcessesOnPort
};
