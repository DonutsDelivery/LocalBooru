/**
 * App auto-updater for Capacitor (Android) builds.
 * Checks GitHub releases, downloads APK, and triggers install via native plugin.
 */
import { App } from '@capacitor/app';
import { Filesystem, Directory } from '@capacitor/filesystem';
import { registerPlugin } from '@capacitor/core';

const ApkInstaller = registerPlugin('ApkInstaller');

const GITHUB_OWNER = 'DonutsDelivery';
const GITHUB_REPO = 'LocalBooru';
const APK_FILENAME = 'LocalBooru.apk';

/**
 * Compare two semver strings. Returns:
 *  -1 if a < b, 0 if equal, 1 if a > b
 */
function compareSemver(a, b) {
  const pa = a.split('.').map(Number);
  const pb = b.split('.').map(Number);
  for (let i = 0; i < 3; i++) {
    const va = pa[i] || 0;
    const vb = pb[i] || 0;
    if (va < vb) return -1;
    if (va > vb) return 1;
  }
  return 0;
}

/**
 * Check GitHub for a newer release.
 * Returns { available, currentVersion, latestVersion, downloadUrl, releaseNotes } or throws.
 */
export async function checkForUpdate() {
  const info = await App.getInfo();
  const currentVersion = info.version; // e.g. "0.3.10"

  const res = await fetch(
    `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases/latest`,
    { headers: { 'Accept': 'application/vnd.github.v3+json' } }
  );
  if (!res.ok) throw new Error(`GitHub API returned ${res.status}`);
  const release = await res.json();

  const latestVersion = release.tag_name.replace(/^v/, '');

  if (compareSemver(currentVersion, latestVersion) >= 0) {
    return { available: false, currentVersion, latestVersion };
  }

  // Find the APK asset
  const asset = release.assets.find(a => a.name === APK_FILENAME);
  if (!asset) {
    throw new Error(`${APK_FILENAME} not found in release ${latestVersion}`);
  }

  return {
    available: true,
    currentVersion,
    latestVersion,
    downloadUrl: asset.browser_download_url,
    size: asset.size,
    releaseNotes: release.body,
  };
}

/**
 * Download the APK with progress callback.
 * Returns the native file path for installApk().
 */
export async function downloadApk(url, onProgress) {
  // Download via XMLHttpRequest for progress events
  const blob = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'blob';

    xhr.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress({ loaded: e.loaded, total: e.total, percent: (e.loaded / e.total) * 100 });
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response);
      } else {
        reject(new Error(`Download failed: HTTP ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error('Download failed: network error'));
    xhr.send();
  });

  // Convert blob to base64 and write via Filesystem
  const base64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      // Strip the data:...;base64, prefix
      const result = reader.result.split(',')[1];
      resolve(result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });

  await Filesystem.writeFile({
    path: APK_FILENAME,
    data: base64,
    directory: Directory.Cache,
  });

  // Get the native URI so we can pass it to the install intent
  const stat = await Filesystem.stat({
    path: APK_FILENAME,
    directory: Directory.Cache,
  });

  return stat.uri;
}

/**
 * Trigger APK installation. Handles the install-from-unknown-sources permission flow.
 * Returns true if install intent was launched, false if user needs to grant permission.
 */
export async function installApk(uri) {
  // Check if we can install from unknown sources
  const { value: canInstall } = await ApkInstaller.canRequestInstall();

  if (!canInstall) {
    // Open system settings so user can grant the permission
    await ApkInstaller.openInstallPermissionSettings();
    return false; // Caller should retry after user returns
  }

  // Convert content:// URI to a file path the plugin can use
  // Filesystem.stat returns a URI like file:///... — strip the file:// prefix
  const filePath = uri.replace(/^file:\/\//, '');

  await ApkInstaller.installApk({ path: filePath });
  return true;
}
