/**
 * App auto-updater for Capacitor (Android) builds.
 * Checks the connected LocalBooru server for a newer APK and installs it.
 */
import { App } from '@capacitor/app';
import { Filesystem, Directory } from '@capacitor/filesystem';
import { registerPlugin } from '@capacitor/core';
import { getApiBaseUrl } from '../serverManager';

const ApkInstaller = registerPlugin('ApkInstaller');

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
 * Check the connected server for a newer APK.
 * Returns { available, currentVersion, latestVersion, downloadUrl }.
 */
export async function checkForUpdate() {
  const baseUrl = await getApiBaseUrl();
  if (!baseUrl) throw new Error('No server connected');

  const info = await App.getInfo();
  const currentVersion = info.version;

  const res = await fetch(
    `${baseUrl}/app/update/check?platform=android&current_version=${currentVersion}`
  );
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  const data = await res.json();

  const serverVersion = data.version;
  const hasApk = data.apk_available;

  if (!hasApk || compareSemver(currentVersion, serverVersion) >= 0) {
    return { available: false, currentVersion, latestVersion: serverVersion };
  }

  return {
    available: true,
    currentVersion,
    latestVersion: serverVersion,
    downloadUrl: `${baseUrl}/app/update/download`,
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
  const { value: canInstall } = await ApkInstaller.canRequestInstall();

  if (!canInstall) {
    await ApkInstaller.openInstallPermissionSettings();
    return false;
  }

  const filePath = uri.replace(/^file:\/\//, '');
  await ApkInstaller.installApk({ path: filePath });
  return true;
}
