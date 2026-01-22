#!/usr/bin/env node
/**
 * Prepares Linux Python virtual environment bundle with all dependencies
 * Run with: node scripts/prepare-python-bundle-linux.js
 *
 * This script:
 * 1. Creates a Python virtual environment
 * 2. Installs all dependencies from requirements.txt
 * 3. Cleans up unnecessary files to reduce size
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { rmSync, mkdirSync, readdirSync, statSync, unlinkSync } = require('fs');

const BUNDLE_DIR = path.join(__dirname, '..', 'python-venv-linux');
const REQUIREMENTS_PATH = path.join(__dirname, '..', 'requirements.txt');

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

/**
 * Find Python 3 executable
 */
function findPython() {
  const candidates = ['python3.11', 'python3', 'python'];

  for (const cmd of candidates) {
    try {
      const version = execSync(`${cmd} --version 2>&1`, { encoding: 'utf-8' });
      if (version.includes('Python 3.')) {
        // Verify it's at least Python 3.10
        const match = version.match(/Python 3\.(\d+)/);
        if (match && parseInt(match[1]) >= 10) {
          console.log(`Found: ${cmd} (${version.trim()})`);
          return cmd;
        }
      }
    } catch (e) {
      // Command not found, continue
    }
  }

  throw new Error('Python 3.10+ not found. Please install Python 3.10 or newer.');
}

async function main() {
  console.log('='.repeat(60));
  console.log('LocalBooru Linux Python Bundle Preparation');
  console.log('='.repeat(60));

  if (process.platform === 'win32') {
    console.error('This script is for Linux/macOS only.');
    console.error('Use prepare-python-bundle.js for Windows.');
    process.exit(1);
  }

  // Find Python
  console.log('\nLocating Python...');
  const pythonCmd = findPython();

  console.log(`Bundle directory: ${BUNDLE_DIR}`);
  console.log();

  // Clean existing bundle
  if (fs.existsSync(BUNDLE_DIR)) {
    console.log('Removing existing bundle...');
    rmSync(BUNDLE_DIR, { recursive: true });
  }

  // 1. Create virtual environment
  console.log('\n[1/4] Creating virtual environment...');
  execSync(`${pythonCmd} -m venv "${BUNDLE_DIR}"`, { stdio: 'inherit' });

  // Get venv Python path
  const venvPython = path.join(BUNDLE_DIR, 'bin', 'python');

  // 2. Upgrade pip
  console.log('\n[2/4] Upgrading pip...');
  execSync(`"${venvPython}" -m pip install --upgrade pip`, { stdio: 'inherit' });

  // 3. Install dependencies
  console.log('\n[3/4] Installing dependencies from requirements.txt...');
  console.log('This may take a while...\n');

  // Read requirements and install
  if (!fs.existsSync(REQUIREMENTS_PATH)) {
    throw new Error(`requirements.txt not found at ${REQUIREMENTS_PATH}`);
  }

  // Install requirements
  execSync(`"${venvPython}" -m pip install -r "${REQUIREMENTS_PATH}" --no-warn-script-location`, {
    stdio: 'inherit'
  });

  // 4. Clean up
  console.log('\n[4/4] Cleaning up unnecessary files...');
  const sitePackages = path.join(BUNDLE_DIR, 'lib');

  // Find actual site-packages (could be python3.X)
  const libDirs = readdirSync(sitePackages);
  for (const libDir of libDirs) {
    const spPath = path.join(sitePackages, libDir, 'site-packages');
    if (fs.existsSync(spPath)) {
      const cleaned = cleanDirectory(spPath);
      console.log(`Removed ${cleaned} cache/test directories and .pyc files`);
    }
  }

  // Remove pip cache in venv
  const pipCache = path.join(BUNDLE_DIR, '.cache');
  if (fs.existsSync(pipCache)) {
    rmSync(pipCache, { recursive: true, force: true });
  }

  // Remove unnecessary executables (keep python and pip)
  const binDir = path.join(BUNDLE_DIR, 'bin');
  const keepBins = ['python', 'python3', 'pip', 'pip3', 'activate'];
  const binEntries = readdirSync(binDir);
  let removedBins = 0;
  for (const entry of binEntries) {
    // Keep python*, pip*, and activate scripts
    if (!keepBins.some(k => entry.startsWith(k) || entry === k)) {
      const binPath = path.join(binDir, entry);
      try {
        unlinkSync(binPath);
        removedBins++;
      } catch (e) {
        // May be a directory or symlink
      }
    }
  }
  if (removedBins > 0) {
    console.log(`Removed ${removedBins} unnecessary executables`);
  }

  // Make the venv relocatable by making paths relative
  console.log('\nMaking venv relocatable...');
  makeVenvRelocatable(BUNDLE_DIR);

  // Report final size
  const finalSize = getDirSize(BUNDLE_DIR);
  console.log('\n' + '='.repeat(60));
  console.log('Bundle preparation complete!');
  console.log(`Bundle size: ${finalSize} MB`);
  console.log(`Location: ${BUNDLE_DIR}`);
  console.log('='.repeat(60));
}

/**
 * Make the venv relocatable by patching scripts
 */
function makeVenvRelocatable(venvDir) {
  const binDir = path.join(venvDir, 'bin');

  // Patch the activate script to use relative paths
  const activatePath = path.join(binDir, 'activate');
  if (fs.existsSync(activatePath)) {
    let content = fs.readFileSync(activatePath, 'utf8');
    // The VIRTUAL_ENV line contains the absolute path
    // We'll replace it with a dynamic one
    content = content.replace(
      /^VIRTUAL_ENV=.*$/m,
      'VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'
    );
    fs.writeFileSync(activatePath, content);
  }

  // Patch pip and other scripts in bin to use relative python
  const scripts = readdirSync(binDir);
  for (const script of scripts) {
    const scriptPath = path.join(binDir, script);
    try {
      const stat = fs.lstatSync(scriptPath);
      if (stat.isFile() && !stat.isSymbolicLink()) {
        let content = fs.readFileSync(scriptPath, 'utf8');
        // Check if it has a shebang pointing to the venv python
        if (content.startsWith('#!') && content.includes(venvDir)) {
          // Replace absolute path with /usr/bin/env python3 or relative
          content = content.replace(
            /^#!.*python.*$/m,
            '#!/usr/bin/env python3'
          );
          fs.writeFileSync(scriptPath, content);
        }
      }
    } catch (e) {
      // Skip on error
    }
  }

  // Create a pyvenv.cfg that works with relocation
  const pyvenvCfg = path.join(venvDir, 'pyvenv.cfg');
  if (fs.existsSync(pyvenvCfg)) {
    let content = fs.readFileSync(pyvenvCfg, 'utf8');
    // Keep the config but note it may need adjustment at runtime
    // The key setting is include-system-site-packages = false
    fs.writeFileSync(pyvenvCfg, content);
  }
}

main().catch((err) => {
  console.error('Error:', err.message);
  process.exit(1);
});
