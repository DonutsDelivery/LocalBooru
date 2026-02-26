/**
 * App auto-updater stub.
 * Previously used Capacitor native plugins for APK download + install.
 * TODO: Implement Tauri-native updater (tauri-plugin-updater or custom Kotlin plugin).
 */

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
 * Check the connected server for a newer version.
 * Returns { available: false } — updater not yet ported to Tauri.
 */
export async function checkForUpdate() {
  // Updater not yet implemented for Tauri mobile
  return { available: false, currentVersion: '2.0.0', latestVersion: '2.0.0' };
}

/**
 * Download APK — stub, not yet implemented for Tauri.
 */
export async function downloadApk(url, onProgress) {
  throw new Error('APK download not yet implemented for Tauri mobile');
}

/**
 * Install APK — stub, not yet implemented for Tauri.
 */
export async function installApk(uri) {
  throw new Error('APK install not yet implemented for Tauri mobile');
}
