/**
 * Directory Watcher
 * Uses chokidar to watch directories for new images and notify the backend
 */
const chokidar = require('chokidar');
const path = require('path');
const http = require('http');

// Supported image extensions
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']);
const VIDEO_EXTENSIONS = new Set(['.webm', '.mp4', '.mov']);
const ALL_EXTENSIONS = new Set([...IMAGE_EXTENSIONS, ...VIDEO_EXTENSIONS]);

class DirectoryWatcher {
  constructor(apiPort = 8787) {
    this.apiPort = apiPort;
    this.watchers = new Map(); // path -> { watcher, directoryId }
    this.pendingFiles = new Set(); // Debounce rapid changes
    this.apiBaseUrl = `http://127.0.0.1:${apiPort}`;
  }

  /**
   * Load watch directories from the API and start watching
   */
  async loadWatchDirectories() {
    try {
      const directories = await this.fetchDirectories();

      for (const dir of directories) {
        if (dir.enabled && dir.path_exists) {
          await this.addDirectory(dir.id, dir.path, {
            recursive: dir.recursive,
            autoTag: dir.auto_tag
          });
        }
      }

      console.log(`[Watcher] Watching ${this.watchers.size} directories`);
    } catch (error) {
      console.error('[Watcher] Failed to load directories:', error.message);
    }
  }

  /**
   * Fetch directories from API
   */
  async fetchDirectories() {
    return new Promise((resolve, reject) => {
      http.get(`${this.apiBaseUrl}/directories`, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            resolve(parsed.directories || []);
          } catch (e) {
            reject(e);
          }
        });
      }).on('error', reject);
    });
  }

  /**
   * Add a directory to watch
   */
  async addDirectory(directoryId, dirPath, options = {}) {
    const { recursive = true, autoTag = true } = options;

    if (this.watchers.has(dirPath)) {
      console.log(`[Watcher] Already watching: ${dirPath}`);
      return;
    }

    console.log(`[Watcher] Adding: ${dirPath} (recursive=${recursive})`);

    const watcher = chokidar.watch(dirPath, {
      persistent: true,
      ignoreInitial: true, // Don't process existing files (handled by scan)
      depth: recursive ? undefined : 0,
      awaitWriteFinish: {
        stabilityThreshold: 2000, // Wait for file writes to complete
        pollInterval: 100
      },
      ignored: [
        /(^|[\/\\])\../, // Ignore dotfiles
        /node_modules/,
        /\.git/
      ]
    });

    watcher
      .on('add', (filePath) => this.onFileAdded(filePath, directoryId, autoTag))
      .on('unlink', (filePath) => this.onFileRemoved(filePath))
      .on('error', (error) => console.error(`[Watcher] Error for ${dirPath}:`, error));

    this.watchers.set(dirPath, { watcher, directoryId, options });
  }

  /**
   * Remove a directory from watching
   */
  async removeDirectory(dirPath) {
    const entry = this.watchers.get(dirPath);
    if (entry) {
      await entry.watcher.close();
      this.watchers.delete(dirPath);
      console.log(`[Watcher] Removed: ${dirPath}`);
    }
  }

  /**
   * Check if a file is a supported media file
   */
  isMediaFile(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    return ALL_EXTENSIONS.has(ext);
  }

  /**
   * Handle new file detected
   */
  async onFileAdded(filePath, directoryId, autoTag) {
    if (!this.isMediaFile(filePath)) {
      return;
    }

    // Debounce - file might still be writing
    if (this.pendingFiles.has(filePath)) {
      return;
    }
    this.pendingFiles.add(filePath);

    // Wait a bit then process
    setTimeout(async () => {
      this.pendingFiles.delete(filePath);
      await this.importFile(filePath, directoryId, autoTag);
    }, 500);
  }

  /**
   * Handle file removed
   */
  async onFileRemoved(filePath) {
    if (!this.isMediaFile(filePath)) {
      return;
    }

    console.log(`[Watcher] File removed: ${filePath}`);

    // Notify backend to mark file as missing
    try {
      await this.notifyFileMissing(filePath);
    } catch (error) {
      console.error(`[Watcher] Failed to mark file missing:`, error.message);
    }
  }

  /**
   * Import a file via API
   */
  async importFile(filePath, directoryId, autoTag) {
    console.log(`[Watcher] Importing: ${filePath}`);

    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({
        file_path: filePath,
        watch_directory_id: directoryId,
        auto_tag: autoTag
      });

      const req = http.request({
        hostname: '127.0.0.1',
        port: this.apiPort,
        path: '/library/import-file',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData)
        }
      }, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          if (res.statusCode >= 200 && res.statusCode < 300) {
            console.log(`[Watcher] Imported: ${filePath}`);
            resolve(JSON.parse(data));
          } else {
            reject(new Error(`Import failed: ${res.statusCode}`));
          }
        });
      });

      req.on('error', reject);
      req.write(postData);
      req.end();
    });
  }

  /**
   * Notify backend that a file is missing
   */
  async notifyFileMissing(filePath) {
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({ file_path: filePath });

      const req = http.request({
        hostname: '127.0.0.1',
        port: this.apiPort,
        path: '/library/file-missing',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData)
        }
      }, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => resolve(JSON.parse(data)));
      });

      req.on('error', reject);
      req.write(postData);
      req.end();
    });
  }

  /**
   * Stop all watchers
   */
  async stopAll() {
    console.log('[Watcher] Stopping all watchers...');

    const closePromises = [];
    for (const [dirPath, entry] of this.watchers) {
      closePromises.push(entry.watcher.close());
    }

    await Promise.all(closePromises);
    this.watchers.clear();

    console.log('[Watcher] All watchers stopped');
  }

  /**
   * Refresh watchers from API
   */
  async refresh() {
    await this.stopAll();
    await this.loadWatchDirectories();
  }
}

module.exports = DirectoryWatcher;
