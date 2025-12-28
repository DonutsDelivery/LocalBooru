#!/usr/bin/env node
/**
 * Prepares Windows embedded Python bundle with all dependencies
 * Run with: node scripts/prepare-python-bundle.js
 *
 * This script:
 * 1. Downloads Windows embeddable Python
 * 2. Extracts and enables pip
 * 3. Installs all dependencies from requirements.txt
 * 4. Cleans up unnecessary files to reduce size
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { execSync } = require('child_process');
const { createWriteStream, rmSync, mkdirSync, readdirSync, statSync, unlinkSync } = require('fs');

const PYTHON_VERSION = '3.11.9';
const PYTHON_SHORT = '311';
const BUNDLE_DIR = path.join(__dirname, '..', 'python-embed');
const PYTHON_URL = `https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip`;
const GET_PIP_URL = 'https://bootstrap.pypa.io/get-pip.py';

/**
 * Download a file from URL to destination
 */
function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    console.log(`Downloading: ${url}`);
    const file = createWriteStream(dest);

    const request = (urlToFetch) => {
      https.get(urlToFetch, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          // Follow redirect
          request(response.headers.location);
          return;
        }

        if (response.statusCode !== 200) {
          reject(new Error(`HTTP ${response.statusCode}`));
          return;
        }

        const total = parseInt(response.headers['content-length'], 10);
        let downloaded = 0;

        response.on('data', (chunk) => {
          downloaded += chunk.length;
          if (total) {
            const percent = ((downloaded / total) * 100).toFixed(1);
            process.stdout.write(`\rProgress: ${percent}%`);
          }
        });

        response.pipe(file);
        file.on('finish', () => {
          file.close();
          console.log('\nDownload complete');
          resolve();
        });
      }).on('error', (err) => {
        unlinkSync(dest);
        reject(err);
      });
    };

    request(url);
  });
}

/**
 * Recursively delete directories matching patterns
 */
function cleanDirectory(dir, patterns = ['__pycache__', 'tests', 'test', 'docs', 'doc']) {
  if (!fs.existsSync(dir)) return;

  let cleaned = 0;

  const walk = (currentDir) => {
    const entries = readdirSync(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);

      if (entry.isDirectory()) {
        if (patterns.includes(entry.name.toLowerCase())) {
          rmSync(fullPath, { recursive: true, force: true });
          cleaned++;
        } else {
          walk(fullPath);
        }
      } else if (entry.name.endsWith('.pyc') || entry.name.endsWith('.pyo')) {
        unlinkSync(fullPath);
        cleaned++;
      }
    }
  };

  walk(dir);
  return cleaned;
}

/**
 * Get directory size in MB
 */
function getDirSize(dir) {
  let size = 0;

  const walk = (currentDir) => {
    const entries = readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else {
        size += statSync(fullPath).size;
      }
    }
  };

  walk(dir);
  return (size / 1024 / 1024).toFixed(2);
}

async function main() {
  console.log('='.repeat(60));
  console.log('LocalBooru Windows Python Bundle Preparation');
  console.log('='.repeat(60));
  console.log(`Python version: ${PYTHON_VERSION}`);
  console.log(`Bundle directory: ${BUNDLE_DIR}`);
  console.log();

  // Clean existing bundle
  if (fs.existsSync(BUNDLE_DIR)) {
    console.log('Removing existing bundle...');
    rmSync(BUNDLE_DIR, { recursive: true });
  }
  mkdirSync(BUNDLE_DIR, { recursive: true });

  // 1. Download embedded Python
  const pythonZip = path.join(BUNDLE_DIR, 'python.zip');
  console.log('\n[1/5] Downloading Windows embeddable Python...');
  await downloadFile(PYTHON_URL, pythonZip);

  // 2. Extract Python
  console.log('\n[2/5] Extracting Python...');
  try {
    // Try unzip first (Linux/macOS)
    execSync(`unzip -q "${pythonZip}" -d "${BUNDLE_DIR}"`, { stdio: 'inherit' });
  } catch (e) {
    // Try PowerShell (Windows)
    try {
      execSync(`powershell -Command "Expand-Archive -Path '${pythonZip}' -DestinationPath '${BUNDLE_DIR}'"`, { stdio: 'inherit' });
    } catch (e2) {
      console.error('Failed to extract. Please install unzip or run on Windows.');
      process.exit(1);
    }
  }
  unlinkSync(pythonZip);

  // 3. Enable pip by modifying ._pth file
  console.log('\n[3/5] Configuring Python for pip...');
  const pthFile = path.join(BUNDLE_DIR, `python${PYTHON_SHORT}._pth`);
  let pthContent = fs.readFileSync(pthFile, 'utf8');
  // Uncomment import site and add Lib\site-packages
  pthContent = pthContent.replace('#import site', 'import site');
  if (!pthContent.includes('Lib\\site-packages')) {
    pthContent += '\nLib\\site-packages\n';
  }
  fs.writeFileSync(pthFile, pthContent);
  console.log('Modified python311._pth to enable site-packages');

  // 4. Download and install pip
  console.log('\n[4/5] Installing pip and dependencies...');
  const getPipPath = path.join(BUNDLE_DIR, 'get-pip.py');
  await downloadFile(GET_PIP_URL, getPipPath);

  // Determine python executable name
  const pythonExe = path.join(BUNDLE_DIR, 'python.exe');

  // On Linux, we can't actually run the Windows Python, so we just prepare it
  if (process.platform !== 'win32') {
    console.log('\n' + '='.repeat(60));
    console.log('NOTE: Running on Linux/macOS');
    console.log('The Python bundle has been prepared but dependencies');
    console.log('cannot be installed from here.');
    console.log('');
    console.log('To complete setup, either:');
    console.log('1. Run this script on Windows, OR');
    console.log('2. Use Wine: wine python.exe get-pip.py');
    console.log('3. Use a Windows CI/CD runner');
    console.log('='.repeat(60));

    // Create a batch script for Windows users
    const batchScript = `@echo off
cd /d "%~dp0"
echo Installing pip...
python.exe get-pip.py
echo Installing dependencies...
python.exe -m pip install -r ../requirements.txt --no-warn-script-location
echo Cleaning up...
del get-pip.py
echo Done!
pause
`;
    fs.writeFileSync(path.join(BUNDLE_DIR, 'install-deps.bat'), batchScript);
    console.log('\nCreated install-deps.bat for Windows setup');

  } else {
    // Running on Windows - can install directly
    console.log('Installing pip...');
    execSync(`"${pythonExe}" "${getPipPath}"`, { cwd: BUNDLE_DIR, stdio: 'inherit' });
    unlinkSync(getPipPath);

    // Install requirements
    const requirementsPath = path.join(__dirname, '..', 'requirements.txt');
    console.log('Installing Python dependencies (this may take a while)...');
    execSync(`"${pythonExe}" -m pip install -r "${requirementsPath}" --no-warn-script-location`, {
      cwd: BUNDLE_DIR,
      stdio: 'inherit'
    });
  }

  // 5. Clean up (can do on any platform)
  console.log('\n[5/5] Cleaning up unnecessary files...');
  const sitePackages = path.join(BUNDLE_DIR, 'Lib', 'site-packages');
  if (fs.existsSync(sitePackages)) {
    const cleaned = cleanDirectory(sitePackages);
    console.log(`Removed ${cleaned} cache/test directories and .pyc files`);
  }

  // Report final size
  const finalSize = getDirSize(BUNDLE_DIR);
  console.log('\n' + '='.repeat(60));
  console.log('Bundle preparation complete!');
  console.log(`Bundle size: ${finalSize} MB`);
  console.log(`Location: ${BUNDLE_DIR}`);
  console.log('='.repeat(60));
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
