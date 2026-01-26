/**
 * Auto-updater module for LocalBooru
 *
 * Supports two update modes:
 * 1. Standard installs: Uses electron-updater with GitHub releases
 * 2. Portable installs: Downloads ZIP, extracts to staging folder, uses script to apply
 */
const { autoUpdater } = require('electron-updater');
const { ipcMain, app } = require('electron');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
let isPortable = false;
let portableDataDir = null;
let pendingUpdate = null; // Stores info about downloaded portable update

// GitHub repo info (from package.json publish config)
const GITHUB_OWNER = 'DonutsDelivery';
const GITHUB_REPO = 'LocalBooru';

/**
 * Get the correct asset name for the current platform (portable ZIP)
 */
function getPortableAssetName() {
  switch (process.platform) {
    case 'win32':
      return 'LocalBooru-Windows.zip';
    case 'darwin':
      // macOS includes architecture in filename
      return `LocalBooru-macOS-${process.arch}.zip`;
    case 'linux':
      return 'LocalBooru-Linux.zip';
    default:
      return null;
  }
}

/**
 * Fetch JSON from a URL
 */
function fetchJson(url) {
  return new Promise((resolve, reject) => {
    const options = {
      headers: {
        'User-Agent': 'LocalBooru-Updater',
        'Accept': 'application/vnd.github.v3+json'
      }
    };

    https.get(url, options, (res) => {
      // Handle redirects
      if (res.statusCode === 301 || res.statusCode === 302) {
        return fetchJson(res.headers.location).then(resolve).catch(reject);
      }

      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode}`));
        return;
      }

      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    }).on('error', reject);
  });
}

/**
 * Download a file with progress reporting
 */
function downloadFile(url, destPath, onProgress) {
  return new Promise((resolve, reject) => {
    const options = {
      headers: {
        'User-Agent': 'LocalBooru-Updater',
        'Accept': 'application/octet-stream'
      }
    };

    const makeRequest = (requestUrl) => {
      https.get(requestUrl, options, (res) => {
        // Handle redirects (GitHub uses these for asset downloads)
        if (res.statusCode === 301 || res.statusCode === 302) {
          return makeRequest(res.headers.location);
        }

        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode}`));
          return;
        }

        const totalSize = parseInt(res.headers['content-length'], 10);
        let downloadedSize = 0;

        const fileStream = fs.createWriteStream(destPath);

        res.on('data', (chunk) => {
          downloadedSize += chunk.length;
          if (onProgress && totalSize) {
            onProgress({
              percent: (downloadedSize / totalSize) * 100,
              transferred: downloadedSize,
              total: totalSize
            });
          }
        });

        res.pipe(fileStream);

        fileStream.on('finish', () => {
          fileStream.close();
          resolve();
        });

        fileStream.on('error', (err) => {
          fs.unlink(destPath, () => {}); // Delete partial file
          reject(err);
        });
      }).on('error', reject);
    };

    makeRequest(url);
  });
}

/**
 * Extract a ZIP file using built-in tools
 * Windows: PowerShell Expand-Archive
 * Linux/macOS: unzip command
 */
function extractZip(zipPath, destPath) {
  return new Promise((resolve, reject) => {
    // Ensure destination exists
    fs.mkdirSync(destPath, { recursive: true });

    let command, args;

    if (process.platform === 'win32') {
      // Use PowerShell's Expand-Archive
      command = 'powershell.exe';
      args = [
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-Command',
        `Expand-Archive -Path "${zipPath}" -DestinationPath "${destPath}" -Force`
      ];
    } else {
      // Use unzip on Linux/macOS
      command = 'unzip';
      args = ['-o', zipPath, '-d', destPath];
    }

    const proc = spawn(command, args, { stdio: 'pipe' });

    let stderr = '';
    proc.stderr?.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Extraction failed (code ${code}): ${stderr}`));
      }
    });

    proc.on('error', reject);
  });
}

/**
 * Check for updates (portable mode)
 */
async function checkForPortableUpdate() {
  const assetName = getPortableAssetName();
  if (!assetName) {
    throw new Error('Unsupported platform for portable updates');
  }

  console.log('[Updater] Checking for portable update...');

  // Fetch latest release from GitHub API
  const releaseUrl = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases/latest`;
  const release = await fetchJson(releaseUrl);

  const currentVersion = app.getVersion();
  const latestVersion = release.tag_name.replace(/^v/, '');

  console.log(`[Updater] Current: ${currentVersion}, Latest: ${latestVersion}`);

  // Simple version comparison (assumes semver-like format)
  if (latestVersion === currentVersion) {
    return { available: false, version: currentVersion };
  }

  // Find the correct asset
  const asset = release.assets.find(a => a.name === assetName);
  if (!asset) {
    throw new Error(`Asset ${assetName} not found in release ${latestVersion}`);
  }

  return {
    available: true,
    version: latestVersion,
    releaseNotes: release.body,
    downloadUrl: asset.browser_download_url,
    assetName: asset.name,
    size: asset.size
  };
}

/**
 * Download and prepare portable update
 */
async function downloadPortableUpdate(updateInfo, onProgress) {
  const updatesDir = path.join(portableDataDir, 'updates');
  const extractedDir = path.join(updatesDir, 'extracted');
  const zipPath = path.join(updatesDir, updateInfo.assetName);

  // Create updates directory
  fs.mkdirSync(updatesDir, { recursive: true });

  console.log(`[Updater] Downloading ${updateInfo.assetName}...`);

  // Download the ZIP
  await downloadFile(updateInfo.downloadUrl, zipPath, onProgress);

  console.log('[Updater] Extracting update...');

  // Extract the ZIP
  await extractZip(zipPath, extractedDir);

  // Delete the ZIP to save space
  try {
    fs.unlinkSync(zipPath);
  } catch (e) {
    // Ignore deletion errors
  }

  // Save pending update info
  pendingUpdate = {
    version: updateInfo.version,
    extractedDir: extractedDir
  };

  // Write marker file (main.js will apply update on next start)
  const markerPath = path.join(updatesDir, 'pending.json');
  fs.writeFileSync(markerPath, JSON.stringify({
    version: updateInfo.version,
    timestamp: new Date().toISOString()
  }));

  console.log('[Updater] Update downloaded and ready to apply on restart');
  return pendingUpdate;
}

/**
 * Launch an external script to apply the update after the app fully exits.
 * This is necessary because the running executable (and DLLs) can't be
 * overwritten while the app is still running, especially on Windows.
 */
function launchUpdateScript(sourceDir, appDir, updatesDir) {
  const appExe = process.execPath;
  const pid = process.pid;

  if (process.platform === 'win32') {
    const scriptPath = path.join(portableDataDir, 'apply-update.bat');

    const script = [
      '@echo off',
      'setlocal',
      '',
      'REM Wait for the app process to exit',
      ':waitloop',
      `tasklist /FI "PID eq ${pid}" 2>nul | find /i "${pid}" >nul`,
      'if not errorlevel 1 (',
      '    timeout /t 1 /nobreak >nul',
      '    goto waitloop',
      ')',
      '',
      'REM Extra delay for file lock release',
      'timeout /t 2 /nobreak >nul',
      '',
      'REM Copy updated files over the app directory',
      `xcopy /s /e /y /q "${sourceDir}\\*" "${appDir}\\"`,
      '',
      'REM Remove the updates staging folder',
      `rmdir /s /q "${updatesDir}"`,
      '',
      'REM Restart the application',
      `start "" "${appExe}"`,
      '',
      'REM Delete this script',
      'del "%~f0"',
    ].join('\r\n');

    fs.writeFileSync(scriptPath, script);

    spawn('cmd.exe', ['/c', scriptPath], {
      detached: true,
      stdio: 'ignore',
      windowsHide: true
    }).unref();

  } else {
    const scriptPath = path.join(portableDataDir, 'apply-update.sh');

    const script = [
      '#!/bin/sh',
      '',
      '# Wait for app process to exit',
      `while kill -0 ${pid} 2>/dev/null; do sleep 1; done`,
      '',
      '# Extra delay for safety',
      'sleep 1',
      '',
      '# Copy updated files over the app directory',
      `cp -rf "${sourceDir}/." "${appDir}/"`,
      '',
      '# Remove the updates staging folder',
      `rm -rf "${updatesDir}"`,
      '',
      '# Restart the application',
      `"${appExe}" &`,
      '',
      '# Delete this script',
      `rm -f "${scriptPath}"`,
    ].join('\n');

    fs.writeFileSync(scriptPath, script, { mode: 0o755 });

    spawn('/bin/sh', [scriptPath], {
      detached: true,
      stdio: 'ignore'
    }).unref();
  }

  console.log('[Updater] Update script launched');
}

/**
 * Apply the portable update by launching an external script, then quitting.
 * The script waits for the app to exit, copies files, and restarts.
 */
function applyPortableUpdate() {
  // Get update info from memory or disk
  let extractedDir;

  if (pendingUpdate) {
    extractedDir = pendingUpdate.extractedDir;
  } else {
    // Fallback: check filesystem directly
    extractedDir = path.join(portableDataDir, 'updates', 'extracted');
    if (!fs.existsSync(extractedDir)) {
      throw new Error('No pending update to apply');
    }
  }

  if (!fs.existsSync(extractedDir)) {
    throw new Error('Extracted update directory not found');
  }

  // Find the actual content directory (might be nested in a single folder)
  let sourceDir = extractedDir;
  const entries = fs.readdirSync(extractedDir);
  if (entries.length === 1) {
    const nested = path.join(extractedDir, entries[0]);
    try {
      if (fs.statSync(nested).isDirectory()) {
        sourceDir = nested;
      }
    } catch (e) {
      // Not a directory, use extractedDir as-is
    }
  }

  const appDir = path.dirname(process.execPath);
  const updatesDir = path.join(portableDataDir, 'updates');

  console.log('[Updater] Launching external update script...');
  console.log(`[Updater] Source: ${sourceDir}`);
  console.log(`[Updater] Dest: ${appDir}`);

  // Launch external script to apply update after app exits
  launchUpdateScript(sourceDir, appDir, updatesDir);

  // Quit the app so the script can overwrite files (including the exe)
  app.isQuitting = true;
  app.quit();
}

/**
 * Initialize the auto-updater
 */
function initUpdater(window, options = {}) {
  mainWindow = window;
  isPortable = !!options.portableDataDir;
  portableDataDir = options.portableDataDir;

  console.log(`[Updater] Initializing (portable: ${isPortable})`);

  if (isPortable) {
    // Portable mode: use custom update logic
    initPortableUpdater();
  } else {
    // Standard mode: use electron-updater
    initStandardUpdater();
  }
}

/**
 * Initialize standard electron-updater
 */
function initStandardUpdater() {
  // Configure updater
  autoUpdater.autoDownload = false;
  autoUpdater.autoInstallOnAppQuit = true;

  // Check for updates on startup (with delay)
  setTimeout(() => {
    autoUpdater.checkForUpdates().catch((err) => {
      console.log('[Updater] Check failed:', err.message);
    });
  }, 5000);

  // Update available
  autoUpdater.on('update-available', (info) => {
    console.log('[Updater] Update available:', info.version);
    mainWindow?.webContents.send('updater:status', {
      status: 'available',
      version: info.version,
      releaseNotes: info.releaseNotes
    });
  });

  // No update available
  autoUpdater.on('update-not-available', (info) => {
    console.log('[Updater] Up to date:', info.version);
    mainWindow?.webContents.send('updater:status', {
      status: 'up-to-date',
      version: info.version
    });
  });

  // Download progress
  autoUpdater.on('download-progress', (progress) => {
    console.log(`[Updater] Download progress: ${progress.percent.toFixed(1)}%`);
    mainWindow?.webContents.send('updater:status', {
      status: 'downloading',
      progress: progress.percent,
      bytesPerSecond: progress.bytesPerSecond,
      transferred: progress.transferred,
      total: progress.total
    });
  });

  // Update downloaded
  autoUpdater.on('update-downloaded', (info) => {
    console.log('[Updater] Update downloaded:', info.version);
    mainWindow?.webContents.send('updater:status', {
      status: 'downloaded',
      version: info.version
    });
  });

  // Error
  autoUpdater.on('error', (err) => {
    console.error('[Updater] Error:', err.message);
    mainWindow?.webContents.send('updater:status', {
      status: 'error',
      error: err.message
    });
  });
}

/**
 * Initialize portable updater
 */
function initPortableUpdater() {
  // Check for updates on startup (with delay)
  setTimeout(async () => {
    try {
      // If update was already downloaded this session, keep that status
      if (pendingUpdate) {
        mainWindow?.webContents.send('updater:status', {
          status: 'downloaded',
          version: pendingUpdate.version
        });
        return;
      }

      const result = await checkForPortableUpdate();
      if (result.available) {
        mainWindow?.webContents.send('updater:status', {
          status: 'available',
          version: result.version,
          releaseNotes: result.releaseNotes
        });
      } else {
        mainWindow?.webContents.send('updater:status', {
          status: 'up-to-date',
          version: result.version
        });
      }
    } catch (err) {
      console.log('[Updater] Check failed:', err.message);
      mainWindow?.webContents.send('updater:status', {
        status: 'error',
        error: err.message
      });
    }
  }, 5000);
}

// IPC handlers
ipcMain.handle('updater:check', async () => {
  try {
    if (isPortable) {
      // If update was already downloaded this session, don't reset to 'available'
      if (pendingUpdate) {
        mainWindow?.webContents.send('updater:status', {
          status: 'downloaded',
          version: pendingUpdate.version
        });
        return { success: true, version: pendingUpdate.version, available: true };
      }

      const result = await checkForPortableUpdate();
      if (result.available) {
        mainWindow?.webContents.send('updater:status', {
          status: 'available',
          version: result.version,
          releaseNotes: result.releaseNotes
        });
      }
      return { success: true, version: result.version, available: result.available };
    } else {
      const result = await autoUpdater.checkForUpdates();
      return { success: true, version: result?.updateInfo?.version };
    }
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('updater:download', async () => {
  try {
    if (isPortable) {
      // If already downloaded, just re-emit the status
      if (pendingUpdate) {
        mainWindow?.webContents.send('updater:status', {
          status: 'downloaded',
          version: pendingUpdate.version
        });
        return { success: true };
      }

      const updateInfo = await checkForPortableUpdate();
      if (!updateInfo.available) {
        return { success: false, error: 'No update available' };
      }

      await downloadPortableUpdate(updateInfo, (progress) => {
        mainWindow?.webContents.send('updater:status', {
          status: 'downloading',
          progress: progress.percent,
          transferred: progress.transferred,
          total: progress.total
        });
      });

      mainWindow?.webContents.send('updater:status', {
        status: 'downloaded',
        version: updateInfo.version
      });

      return { success: true };
    } else {
      await autoUpdater.downloadUpdate();
      return { success: true };
    }
  } catch (err) {
    mainWindow?.webContents.send('updater:status', {
      status: 'error',
      error: err.message
    });
    return { success: false, error: err.message };
  }
});

ipcMain.handle('updater:install', () => {
  if (isPortable) {
    applyPortableUpdate();
  } else {
    autoUpdater.quitAndInstall(false, true);
  }
});

ipcMain.handle('updater:get-version', () => {
  return app.getVersion();
});

ipcMain.handle('updater:is-portable', () => {
  return isPortable;
});

module.exports = { initUpdater };
