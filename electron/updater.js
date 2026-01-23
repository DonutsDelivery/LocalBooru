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
 * Create the updater script that will apply the update after app exits
 */
function createUpdaterScript(extractedPath, appDir) {
  const isWindows = process.platform === 'win32';
  const scriptExt = isWindows ? '.bat' : '.sh';
  const scriptPath = path.join(portableDataDir, 'updates', `apply-update${scriptExt}`);

  // Find the actual content directory (might be nested in a folder)
  let sourceDir = extractedPath;
  const entries = fs.readdirSync(extractedPath);

  // If there's only one directory and no files, the content is nested
  if (entries.length === 1) {
    const nested = path.join(extractedPath, entries[0]);
    if (fs.statSync(nested).isDirectory()) {
      sourceDir = nested;
    }
  }

  // Get the executable name/path for restart
  const exePath = app.getPath('exe');
  const exeName = path.basename(exePath);

  let script;
  const logPath = path.join(portableDataDir, 'update.log').replace(/\\/g, '\\\\');

  if (isWindows) {
    // Windows batch script with logging
    script = `@echo off
setlocal enabledelayedexpansion

set LOGFILE="${logPath}"
echo [%date% %time%] Update script started > %LOGFILE%
echo [%date% %time%] Source: ${sourceDir} >> %LOGFILE%
echo [%date% %time%] Dest: ${appDir} >> %LOGFILE%
echo [%date% %time%] Exe: ${exePath} >> %LOGFILE%

echo Waiting for LocalBooru to exit...
echo [%date% %time%] Waiting for app to exit... >> %LOGFILE%
timeout /t 3 /nobreak >nul

:waitloop
tasklist /FI "IMAGENAME eq ${exeName}" 2>NUL | find /I /N "${exeName}" >NUL
if "%ERRORLEVEL%"=="0" (
    echo [%date% %time%] App still running, waiting... >> %LOGFILE%
    timeout /t 1 /nobreak >nul
    goto waitloop
)

echo [%date% %time%] App exited, applying update... >> %LOGFILE%
echo Applying update...

REM Check if source directory exists
if not exist "${sourceDir}" (
    echo [%date% %time%] ERROR: Source directory not found >> %LOGFILE%
    goto error
)

REM List source files for debugging
echo [%date% %time%] Source contents: >> %LOGFILE%
dir "${sourceDir}" >> %LOGFILE% 2>&1

xcopy /E /Y /I "${sourceDir}\\*" "${appDir}\\" >> %LOGFILE% 2>&1
if errorlevel 1 (
    echo [%date% %time%] ERROR: xcopy failed with errorlevel %errorlevel% >> %LOGFILE%
    goto error
)

echo [%date% %time%] Update applied successfully >> %LOGFILE%
echo Starting LocalBooru...
echo [%date% %time%] Starting app: ${exePath} >> %LOGFILE%
start "" "${exePath}"

echo [%date% %time%] Update complete >> %LOGFILE%
exit /b 0

:error
echo [%date% %time%] Update FAILED >> %LOGFILE%
pause
exit /b 1
`;
  } else {
    // Linux/macOS shell script with logging
    script = `#!/bin/bash

LOGFILE="${logPath}"
echo "[$(date)] Update script started" > "$LOGFILE"
echo "[$(date)] Source: ${sourceDir}" >> "$LOGFILE"
echo "[$(date)] Dest: ${appDir}" >> "$LOGFILE"
echo "[$(date)] Exe: ${exePath}" >> "$LOGFILE"

echo "Waiting for LocalBooru to exit..."
echo "[$(date)] Waiting for app to exit..." >> "$LOGFILE"
sleep 3

# Wait for the process to fully exit
while pgrep -f "${exeName}" > /dev/null 2>&1; do
    echo "[$(date)] App still running, waiting..." >> "$LOGFILE"
    sleep 1
done

echo "[$(date)] App exited, applying update..." >> "$LOGFILE"
echo "Applying update..."

# Check if source directory exists
if [ ! -d "${sourceDir}" ]; then
    echo "[$(date)] ERROR: Source directory not found" >> "$LOGFILE"
    exit 1
fi

# List source files for debugging
echo "[$(date)] Source contents:" >> "$LOGFILE"
ls -la "${sourceDir}" >> "$LOGFILE" 2>&1

cp -rf "${sourceDir}/"* "${appDir}/" >> "$LOGFILE" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: cp failed" >> "$LOGFILE"
    exit 1
fi

# Make sure the executable is still executable
chmod +x "${exePath}"

echo "[$(date)] Update applied successfully" >> "$LOGFILE"
echo "Starting LocalBooru..."
echo "[$(date)] Starting app: ${exePath}" >> "$LOGFILE"
"${exePath}" &

echo "[$(date)] Update complete" >> "$LOGFILE"
exit 0
`;
  }

  fs.writeFileSync(scriptPath, script, { mode: 0o755 });
  return scriptPath;
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

  // Create the updater script
  const appDir = path.dirname(app.getPath('exe'));
  const scriptPath = createUpdaterScript(extractedDir, appDir);

  // Save pending update info
  pendingUpdate = {
    version: updateInfo.version,
    scriptPath: scriptPath,
    extractedDir: extractedDir
  };

  // Write marker file
  const markerPath = path.join(updatesDir, 'pending.json');
  fs.writeFileSync(markerPath, JSON.stringify({
    version: updateInfo.version,
    scriptPath: scriptPath,
    timestamp: new Date().toISOString()
  }));

  console.log('[Updater] Update ready to apply');
  return pendingUpdate;
}

/**
 * Apply the portable update (launches script and quits app)
 */
function applyPortableUpdate() {
  if (!pendingUpdate) {
    throw new Error('No pending update to apply');
  }

  console.log('[Updater] Launching update script and quitting...');

  const isWindows = process.platform === 'win32';

  // Launch the updater script detached
  const scriptProcess = spawn(
    isWindows ? 'cmd.exe' : '/bin/bash',
    isWindows ? ['/c', pendingUpdate.scriptPath] : [pendingUpdate.scriptPath],
    {
      detached: true,
      stdio: 'ignore',
      windowsHide: true
    }
  );

  scriptProcess.unref();

  // Quit the app so the script can overwrite files
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
