// Donut Create browser adaptation of DonutUI 0.1.0.
// Source hashes and attribution are in runtime.json and INSTALL.md.
// Browser assets are bundled; renderers are never loaded from a local folder.
(function () {
  'use strict';
  if (window.__donutui_injected) return;
  window.__donutui_injected = true;
  window.DonutUI = {
    version: '0.1.0-dmc', plugins: new Map(), pluginMeta: new Map(),
    _bridgeReady: false, _readyCallbacks: [],
    onReady(fn) { if (this._bridgeReady) fn(); else this._readyCallbacks.push(fn); },
    getPlugin(type) { return this.plugins.get(type) || null; },
    log(message, ...args) { console.log('[DonutUI]', message, ...args); },
    warn(message, ...args) { console.warn('[DonutUI]', message, ...args); },
    error(message, ...args) { console.error('[DonutUI]', message, ...args); },
  };
})();

// =========================================================================
// DonutUI Bridge — hooks into LiteGraph.js via ComfyUI's extension API
// =========================================================================
// This module waits for ComfyUI's `app` global to become available, then
// registers a ComfyUI extension that intercepts node type registration.
// When a DonutUI plugin exists for a node type, it patches the node's
// drawing methods to call the plugin renderer.

(function () {
  'use strict';

  const POLL_INTERVAL = 200;
  const MAX_WAIT = 30000;

  // Wait for ComfyUI's app object to be available
  function waitForApp() {
    return new Promise((resolve, reject) => {
      const start = Date.now();
      const check = () => {
        if (window.app && window.app.registerExtension) {
          resolve(window.app);
          return;
        }
        if (Date.now() - start > MAX_WAIT) {
          reject(new Error('ComfyUI app not found after ' + MAX_WAIT + 'ms'));
          return;
        }
        setTimeout(check, POLL_INTERVAL);
      };
      check();
    });
  }

  // Cache LiteGraph reference once at init
  let LG = null;

  // Frame counter for cache-per-frame state — only ticks when plugins are active
  let frameId = 0;
  let rAfRunning = false;
  function tickFrame() {
    frameId = (frameId + 1) | 0;
    if (rAfRunning) requestAnimationFrame(tickFrame);
  }
  function startFrameCounter() {
    if (rAfRunning) return;
    rAfRunning = true;
    requestAnimationFrame(tickFrame);
  }
  // Exposed so the plugin system can start it when plugins load
  DonutUI._startFrameCounter = startFrameCounter;

  // Build a plugin-friendly state object from a LiteGraph node
  function buildNodeState(node) {
    if (!LG) LG = window.LiteGraph;
    return {
      id: node.id,
      title: node.title || node.type,
      type: node.type,
      width: node.size ? node.size[0] : 200,
      height: node.size ? node.size[1] : 100,
      headerHeight: (LG && LG.NODE_TITLE_HEIGHT) || 30,
      widgets: (node.widgets || []).map(w => ({
        name: w.name,
        type: w.type,
        value: w.value,
      })),
      inputs: (node.inputs || []).map(i => ({
        name: i.name,
        type: i.type,
        link: i.link,
      })),
      outputs: (node.outputs || []).map(o => ({
        name: o.name,
        type: o.type,
        links: o.links,
      })),
      properties: node.properties || {},
      flags: node.flags || {},
      mode: node.mode,
      executing: !!node.is_executing,
      bgColor: node.bgcolor,
      color: node.color,
    };
  }

  // Cache-per-frame: build state at most once per node per frame
  function getOrBuildState(node) {
    if (node._donutFrame === frameId) return node._donutState;
    node._donutFrame = frameId;
    node._donutState = buildNodeState(node);
    return node._donutState;
  }

  // Override DonutUI.getNodeState with the bridge's implementation
  DonutUI.getNodeState = buildNodeState;

  // Track patched node types to avoid double-patching
  const patchedTypes = new Set();

  // Patch a single node type's drawing methods to call plugin renderers
  function patchNodeType(nodeType, nodeTypeName, plugin) {
    if (patchedTypes.has(nodeTypeName)) return;
    patchedTypes.add(nodeTypeName);

    const proto = nodeType.prototype;

    // Patch onDrawBackground
    if (plugin.render || plugin.renderFull) {
      const origDrawBg = proto.onDrawBackground;
      proto.onDrawBackground = function (ctx, canvas) {
        try {
          if (plugin.renderFull) {
            // Full replacement — skip original
            const state = getOrBuildState(this);
            plugin.renderFull(ctx, state);
            return;
          }
          // Call original first, then overlay
          if (origDrawBg) origDrawBg.call(this, ctx, canvas);
          const state = getOrBuildState(this);
          plugin.render(ctx, state);
        } catch (err) {
          DonutUI.error(`Plugin render error on ${nodeTypeName}:`, err);
          reportPluginError(plugin, nodeTypeName);
          // Fall back to original on error
          if (origDrawBg) origDrawBg.call(this, ctx, canvas);
        }
      };
    }

    // Patch onDrawForeground
    if (plugin.renderForeground) {
      const origDrawFg = proto.onDrawForeground;
      proto.onDrawForeground = function (ctx, canvas) {
        if (origDrawFg) origDrawFg.call(this, ctx, canvas);
        try {
          const state = getOrBuildState(this);
          plugin.renderForeground(ctx, state);
        } catch (err) {
          DonutUI.error(`Plugin renderForeground error on ${nodeTypeName}:`, err);
          reportPluginError(plugin, nodeTypeName);
        }
      };
    }
  }

  // Track errors per plugin for auto-disable
  const errorCounts = new Map();
  const ERROR_BUDGET = 5;

  function reportPluginError(plugin, nodeTypeName) {
    const key = plugin._donutName || nodeTypeName;
    const count = (errorCounts.get(key) || 0) + 1;
    errorCounts.set(key, count);

    if (count >= ERROR_BUDGET) {
      DonutUI.warn(`Plugin "${key}" exceeded error budget (${ERROR_BUDGET}), disabling.`);
      // Remove plugin from the registry for all its node types
      for (const [nt, p] of DonutUI.plugins.entries()) {
        if (p === plugin) {
          DonutUI.plugins.delete(nt);
        }
      }
      // Update metadata
      const meta = DonutUI.pluginMeta.get(key);
      if (meta) meta.status = 'error';
    }
  }

  // Expose error reporting for other modules
  DonutUI._reportPluginError = reportPluginError;

  // Register the DonutUI extension with ComfyUI
  async function initBridge() {
    try {
      const app = await waitForApp();
      DonutUI.log('ComfyUI app found, registering extension...');

      // Cache LiteGraph reference now that ComfyUI is loaded
      LG = window.LiteGraph;

      // Diagnostic: check if the main canvas has willReadFrequently set
      // (this silently disables GPU canvas acceleration in WebKitGTK)
      try {
        const canvasEl = document.querySelector('canvas.liteGraph, canvas');
        if (canvasEl) {
          const attrs = canvasEl.getContext('2d');
          // getContextAttributes() available in modern browsers
          if (attrs && typeof canvasEl.getContext === 'function') {
            const testCtx = canvasEl.getContext('2d');
            if (testCtx && testCtx.getContextAttributes) {
              const ctxAttrs = testCtx.getContextAttributes();
              if (ctxAttrs && ctxAttrs.willReadFrequently) {
                DonutUI.warn('Canvas has willReadFrequently=true — GPU canvas acceleration is DISABLED. This is the #1 performance killer.');
              } else {
                DonutUI.log('Canvas willReadFrequently: false (GPU acceleration OK)');
              }
            }
          }
        }
      } catch (diagErr) {
        // Non-critical diagnostic, ignore errors
      }

      app.registerExtension({
        name: 'DonutUI.PluginBridge',

        async beforeRegisterNodeDef(nodeType, nodeData, app) {
          const nodeTypeName = nodeData.name;
          const plugin = DonutUI.getPlugin(nodeTypeName);
          if (!plugin) return;

          DonutUI.log(`Patching node type: ${nodeTypeName}`);
          patchNodeType(nodeType, nodeTypeName, plugin);
        },
      });

      // Also expose a method to patch already-registered nodes
      // (for plugins loaded after ComfyUI has finished init)
      DonutUI.patchExistingNodes = function () {
        if (!LG) LG = window.LiteGraph;
        if (!LG || !LG.registered_node_types) return;

        let count = 0;
        for (const [typeName, nodeType] of Object.entries(LG.registered_node_types)) {
          const plugin = DonutUI.getPlugin(typeName);
          if (plugin && !patchedTypes.has(typeName)) {
            patchNodeType(nodeType, typeName, plugin);
            count++;
          }
        }
        if (count > 0) {
          DonutUI.log(`Patched ${count} existing node type(s)`);
        }
      };

      DonutUI._bridgeReady = true;
      DonutUI.log('Bridge registered');

      // Adaptive FPS throttle: 30fps idle, 60fps during interaction
      // LiteGraph redraws at display refresh rate even when nothing changes —
      // halving idle FPS cuts CPU rendering work significantly for large workflows.
      setupFPSThrottle(app);

      // Bitmap pan cache: render workflow once, blit shifted bitmap during pan
      setupPanCache(app);

      // Fire ready callbacks
      for (const fn of DonutUI._readyCallbacks) {
        try { fn(); } catch (e) { DonutUI.error('Ready callback error:', e); }
      }
      DonutUI._readyCallbacks = [];

    } catch (err) {
      DonutUI.error('Bridge init failed:', err);
    }
  }

  // Adaptive FPS throttle for LiteGraph's canvas render loop
  function setupFPSThrottle(app) {
    const lgCanvas = app.canvas;
    if (!lgCanvas) {
      DonutUI.warn('FPS throttle: no LGraphCanvas found');
      return;
    }

    const IDLE_FPS = 30;
    const ACTIVE_FPS = 60;
    let minFrameTime = 1000 / IDLE_FPS;
    let lastDrawTime = 0;

    const origDraw = lgCanvas.draw;
    lgCanvas.draw = function (force_canvas, force_bgcanvas) {
      const now = performance.now();
      if (now - lastDrawTime < minFrameTime) return;
      lastDrawTime = now;
      return origDraw.call(this, force_canvas, force_bgcanvas);
    };

    // Boost to 60fps during pointer interaction, drop back after
    const canvasEl = lgCanvas.canvas;
    if (canvasEl) {
      canvasEl.addEventListener('pointerdown', () => {
        minFrameTime = 1000 / ACTIVE_FPS;
      });
      window.addEventListener('pointerup', () => {
        setTimeout(() => { minFrameTime = 1000 / IDLE_FPS; }, 500);
      });
      // Scroll-wheel zoom doesn't fire pointerdown — boost FPS on wheel too
      let wheelTimer = null;
      canvasEl.addEventListener('wheel', () => {
        minFrameTime = 1000 / ACTIVE_FPS;
        clearTimeout(wheelTimer);
        wheelTimer = setTimeout(() => { minFrameTime = 1000 / IDLE_FPS; }, 500);
      }, { passive: true });

      // Compositor hint: give the canvas its own layer so static UI isn't re-rasterized
      canvasEl.style.willChange = 'transform';
    }

    DonutUI.log(`FPS throttle: ${IDLE_FPS}fps idle / ${ACTIVE_FPS}fps active`);
  }

  // Pan/zoom cache: avoids full LiteGraph redraws during interaction.
  //
  // On pan/zoom start, instantly copies the visible canvas into a cache.
  // During interaction, blits shifted + scaled copies (zero-cost frames).
  // Grey edges appear if panning/zooming beyond the cached area — never freezes.
  //
  // The expensive oversized (1.5x) render ONLY runs during idle:
  //  - 300ms after interaction stops
  // If an oversized cache exists when the next interaction starts,
  // the extra 25% margin on each side delays grey edges.
  //
  // State machine: idle → cached → idle
  function setupPanCache(app) {
    const lgCanvas = app.canvas;
    if (!lgCanvas) {
      DonutUI.warn('Pan cache: no LGraphCanvas found');
      return;
    }

    const protoDraw = Object.getPrototypeOf(lgCanvas).draw;

    const cacheCanvas = document.createElement('canvas');
    const cacheCtx = cacheCanvas.getContext('2d');

    let mode = 'idle'; // 'idle' | 'cached'
    let snapOffset = [0, 0]; // offset when cache was captured
    let cacheScale = 0;
    let renderedScale = 0;   // scale of last origDraw frame
    let renderedOffset = [0, 0]; // offset of last origDraw frame
    let lastFrameScale = 0;  // scale on previous draw call (for change detection)
    let settleTimer = null;
    let idleRenderTimer = null;
    let cacheValid = false;

    const PAN_MULTIPLIER = 1.5;
    let marginX = 0, marginY = 0;
    let allocW = 0, allocH = 0;

    // Instant viewport copy — just pixel copy, no LiteGraph redraw.
    // Records renderedScale as the cache's scale (the scale the pixels
    // were actually drawn at, not the current ds.scale which may differ).
    function snapshotViewport(self) {
      const el = self.canvas;
      const w = el.width;
      const h = el.height;
      if (allocW !== w || allocH !== h) {
        cacheCanvas.width = w;
        cacheCanvas.height = h;
        allocW = w;
        allocH = h;
      }
      cacheCtx.drawImage(el, 0, 0);
      marginX = 0;
      marginY = 0;
      cacheScale = renderedScale || self.ds.scale;
      cacheValid = true;
    }

    // Expensive oversized render — ONLY call during idle.
    function renderOversized(self) {
      const el = self.canvas;
      const viewW = el.width;
      const viewH = el.height;
      const cacheW = Math.round(viewW * PAN_MULTIPLIER);
      const cacheH = Math.round(viewH * PAN_MULTIPLIER);

      marginX = Math.round((cacheW - viewW) / 2);
      marginY = Math.round((cacheH - viewH) / 2);

      if (allocW !== cacheW || allocH !== cacheH) {
        cacheCanvas.width = cacheW;
        cacheCanvas.height = cacheH;
        allocW = cacheW;
        allocH = cacheH;
      }

      const origCanvas = self.canvas;
      const origCtx = self.ctx;
      const origBgCanvas = self.bgcanvas;
      const origBgCtx = self.bgctx;
      const origDsElement = self.ds.element;
      const origOffsetX = self.ds.offset[0];
      const origOffsetY = self.ds.offset[1];
      const origDrawFg = self.onDrawForeground;

      self.canvas = cacheCanvas;
      self.ctx = cacheCtx;
      self.bgcanvas = cacheCanvas;
      self.bgctx = null;
      self.ds.element = cacheCanvas;

      const scale = self.ds.scale;
      const dpr = window.devicePixelRatio || 1;
      self.ds.offset[0] = origOffsetX + marginX / (scale * dpr);
      self.ds.offset[1] = origOffsetY + marginY / (scale * dpr);
      self.onDrawForeground = null;

      self.dirty_canvas = true;
      self.dirty_bgcanvas = true;
      protoDraw.call(self, true, true);

      self.canvas = origCanvas;
      self.ctx = origCtx;
      self.bgcanvas = origBgCanvas;
      self.bgctx = origBgCtx;
      self.ds.element = origDsElement;
      self.ds.offset[0] = origOffsetX;
      self.ds.offset[1] = origOffsetY;
      self.onDrawForeground = origDrawFg;

      // Record the offset at which the cache center sits
      snapOffset[0] = origOffsetX;
      snapOffset[1] = origOffsetY;
      cacheScale = scale;
      cacheValid = true;
    }

    const origDraw = lgCanvas.draw;

    lgCanvas.draw = function (force_canvas, force_bgcanvas) {
      const t0 = performance.now();
      const el = this.canvas;
      const DPR = window.devicePixelRatio || 1;
      const panning = !!this.dragging_canvas;
      const scale = this.ds.scale;
      const scaleChanged = lastFrameScale !== 0 && scale !== lastFrameScale;
      lastFrameScale = scale;

      const interacting = panning || scaleChanged;

      // Reset settle timer on every interaction frame
      if (interacting) {
        clearTimeout(settleTimer);
        clearTimeout(idleRenderTimer);
        settleTimer = setTimeout(() => {
          if (mode !== 'cached') return;
          mode = 'idle';
          lgCanvas.dirty_canvas = true;
          lgCanvas.dirty_bgcanvas = true;
          // Build oversized cache well after interaction stops.
          // Long delay (2s) avoids blocking the next interaction.
          idleRenderTimer = setTimeout(() => {
            if (mode !== 'idle') return;
            DonutUI.log('Pan cache: building oversized cache...');
            const rt0 = performance.now();
            renderOversized(lgCanvas);
            DonutUI.log(`Pan cache: oversized render took ${(performance.now() - rt0).toFixed(0)}ms`);
          }, 2000);
        }, 300);
      }

      // Enter cached mode on first interaction frame
      if (interacting && mode === 'idle') {
        mode = 'cached';
        DonutUI.log('Pan cache: entering cached mode' + (cacheValid ? ' (cache warm)' : ' (viewport snap)'));
        DonutUI.log(`DIAG enter: DPR=${window.devicePixelRatio} canvas=${el.width}x${el.height} ` +
          `cssRect=${Math.round(el.getBoundingClientRect().width)}x${Math.round(el.getBoundingClientRect().height)} ` +
          `offset=[${this.ds.offset[0].toFixed(1)},${this.ds.offset[1].toFixed(1)}] ` +
          `renderedOffset=[${renderedOffset[0].toFixed(1)},${renderedOffset[1].toFixed(1)}] ` +
          `scale=${scale.toFixed(3)} cacheScale=${cacheScale.toFixed(3)} ` +
          `snapOffset=[${snapOffset[0].toFixed(1)},${snapOffset[1].toFixed(1)}] ` +
          `margin=[${marginX},${marginY}] cacheValid=${cacheValid}`);

        if (!cacheValid) {
          const st0 = performance.now();
          snapshotViewport(this);
          DonutUI.log(`Pan cache: viewport snapshot took ${(performance.now() - st0).toFixed(1)}ms`);
          // Viewport pixels were drawn at renderedOffset, not current offset
          snapOffset[0] = renderedOffset[0];
          snapOffset[1] = renderedOffset[1];
          DonutUI.log(`DIAG snap: snapOffset now=[${snapOffset[0].toFixed(1)},${snapOffset[1].toFixed(1)}]`);
        }
        // If cacheValid (warm oversized cache), snapOffset is already
        // correct from renderOversized — do NOT overwrite it.
      }

      // --- CACHED MODE: blit with pan + zoom transform ---
      if (mode === 'cached') {
        const ctx = el.getContext('2d');

        // Reset to clean identity transform — LiteGraph may leave
        // a DPR-scaled transform on the context between frames.
        ctx.setTransform(1, 0, 0, 1, 0, 0);

        const zoomRatio = scale / cacheScale;
        const dx = (this.ds.offset[0] - snapOffset[0]) * scale * DPR;
        const dy = (this.ds.offset[1] - snapOffset[1]) * scale * DPR;

        ctx.fillStyle = '#202020';
        ctx.fillRect(0, 0, el.width, el.height);

        ctx.drawImage(
          cacheCanvas,
          0, 0, allocW, allocH,
          dx - marginX * zoomRatio, dy - marginY * zoomRatio,
          allocW * zoomRatio, allocH * zoomRatio
        );

        if (this.graph) this.ds.computeVisibleArea(this.viewport);
        this.onDrawForeground?.(ctx, this.visible_area);
        this.dirty_canvas = false;
        this.dirty_bgcanvas = false;

        // DIAGNOSTIC: red dot at graph origin using current ds state.
        // If this dot aligns with graph origin in the blit content, blit is correct.
        // If the dot is offset from the blit content, the blit math is wrong.
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        const dotX = this.ds.offset[0] * scale * DPR;
        const dotY = this.ds.offset[1] * scale * DPR;
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.beginPath();
        ctx.arc(dotX, dotY, 8, 0, Math.PI * 2);
        ctx.fill();

        return;
      }

      // --- IDLE: normal LiteGraph draw ---
      origDraw.call(this, force_canvas, force_bgcanvas);
      renderedScale = this.ds.scale;
      renderedOffset[0] = this.ds.offset[0];
      renderedOffset[1] = this.ds.offset[1];

      // DIAGNOSTIC: green dot at graph origin — should match red dot position
      {
        const ctx = this.canvas.getContext('2d');
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        const DPR2 = window.devicePixelRatio || 1;
        const dotX = this.ds.offset[0] * this.ds.scale * DPR2;
        const dotY = this.ds.offset[1] * this.ds.scale * DPR2;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.beginPath();
        ctx.arc(dotX, dotY, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    };

    DonutUI.log('Pan cache: enabled (pan+zoom, non-blocking)');
  }

  initBridge();
})();

// =========================================================================
// DonutUI API — public helpers and type documentation for plugin authors
// =========================================================================
// This module exposes convenience functions on the DonutUI global that
// plugin authors can use in their renderers.

(function () {
  'use strict';

  // ---- Drawing Helpers ----

  /**
   * Draw a rounded rectangle.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   * @param {number} r - corner radius
   */
  DonutUI.drawRoundRect = function (ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  };

  /**
   * Draw a progress bar inside a node.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   * @param {number} progress - 0 to 1
   * @param {string} [color='#ff6b9d']
   * @param {string} [bgColor='rgba(255,255,255,0.1)']
   */
  DonutUI.drawProgressBar = function (ctx, x, y, w, h, progress, color, bgColor) {
    color = color || '#ff6b9d';
    bgColor = bgColor || 'rgba(255,255,255,0.1)';
    progress = Math.max(0, Math.min(1, progress));

    ctx.fillStyle = bgColor;
    DonutUI.drawRoundRect(ctx, x, y, w, h, h / 2);
    ctx.fill();

    if (progress > 0) {
      ctx.fillStyle = color;
      DonutUI.drawRoundRect(ctx, x, y, w * progress, h, h / 2);
      ctx.fill();
    }
  };

  /**
   * Draw text with ellipsis truncation.
   * @param {CanvasRenderingContext2D} ctx
   * @param {string} text
   * @param {number} x
   * @param {number} y
   * @param {number} maxWidth
   */
  DonutUI.drawEllipsisText = function (ctx, text, x, y, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) {
      ctx.fillText(text, x, y);
      return;
    }
    // Binary search for the longest prefix that fits with ellipsis
    let lo = 0, hi = text.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ctx.measureText(text.slice(0, mid) + '...').width <= maxWidth) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    ctx.fillText(text.slice(0, lo) + '...', x, y);
  };

  /**
   * Get a widget's value from the node state by name.
   * @param {Object} state - DonutUI node state
   * @param {string} widgetName
   * @returns {*} the widget value, or undefined
   */
  DonutUI.getWidgetValue = function (state, widgetName) {
    const w = state.widgets.find(w => w.name === widgetName);
    return w ? w.value : undefined;
  };

  // ---- Theme Colors ----

  DonutUI.theme = {
    accent: '#ff6b9d',
    accentDim: 'rgba(255,107,157,0.3)',
    bg: '#1a1a2e',
    bgLight: '#2a2a4a',
    text: '#e0e0e0',
    textDim: '#888888',
    success: '#4caf50',
    error: '#f44336',
    warning: '#ff9800',
  };

  DonutUI.log('API helpers loaded');
})();


// Studio bootstrap uses the real ComfyUI editor, widget bindings and Run path.
(function () {
  'use strict';
  const config = window.__DMC_CREATE__;
  if (!config?.prefix || window.__DMC_CREATE_ADAPTER__) return;
  window.__DMC_CREATE_ADAPTER__ = true;
  const base = new URL(config.prefix, location.origin);
  const draftKey = 'dmc-create:v5:draft:' + config.backendKey;
  let app, api, initialized = false, loading = false, lastSaved = '';
  let ready = false, ignoreInitialDefault = false;
  let defaultGraph;
  let lastError = '', parentOrigin = null, bridgeQueue = Promise.resolve(), queueAttempt = null;
  let latestRun = {revision: 0, workflow: null}, loadedRunRevision = 0;
  const uploadedReferences = new Set();

  function notice(message, error = false) {
    let bar = document.getElementById('donut-create-status');
    if (!bar) {
      bar = document.createElement('div'); bar.id = 'donut-create-status';
      bar.setAttribute('role', 'status'); bar.setAttribute('aria-live', 'polite');
      const title = document.createElement('strong'); title.textContent = 'Donut Create · v5';
      const text = document.createElement('span'); text.className = 'message';
      const reset = document.createElement('button'); reset.type = 'button';
      reset.textContent = 'Load setup preset';
      reset.addEventListener('click', () => { bridgeQueue = bridgeQueue.then(loadPreset).catch(showError); });
      bar.append(title, text, reset); document.body.append(bar);
    }
    bar.classList.toggle('error', error);
    bar.querySelector('.message').textContent = message;
  }
  const showError = error => { lastError = error?.message || String(error); notice(lastError, true); };

  function savedDraft() {
    try { return JSON.parse(localStorage.getItem(draftKey) || 'null'); }
    catch (error) { showError(new Error('The saved draft could not be read. Load the setup preset to start a new draft.')); return null; }
  }
  function saveDraft() {
    if (!initialized || loading || !app?.graph) return;
    try {
      const value = JSON.stringify(app.graph.serialize());
      if (value === lastSaved) return;
      localStorage.setItem(draftKey, value);
      lastSaved = value;
    } catch (error) { showError(new Error('The browser could not save this studio draft. Keep the studio open or save the workflow from ComfyUI.')); }
  }
  function stockWorkflow(graph) {
    if (!graph || !graph.nodes?.length) return true;
    if (!defaultGraph) return false;
    const signature = value => JSON.stringify({ nodes: (value.nodes || []).map(node => [node.id, node.type, node.widgets_values]), links: value.links || [] });
    return signature(graph) === signature(defaultGraph);
  }
  async function preset() {
    const response = await fetch(new URL('workflow.json', base));
    if (!response.ok) throw new Error('The v5 workflow could not be loaded. Retry setup.');
    return response.json();
  }
  async function checkCapabilities() {
    // Inspect the serialized execution prompt, so promoted/subgraph bindings
    // are checked exactly as ComfyUI will submit them from the Run button.
    const graph = await app.graphToPrompt();
    const info = await (await api.fetchApi('/object_info')).json();
    const missingNodes = new Set(), missingModels = new Set(), catalogs = new Map();
    const fileWidgets = {
      UNETLoader: ['diffusion_models', 'unet_name'], CLIPLoader: ['text_encoders', 'clip_name'],
      VAELoader: ['vae', 'vae_name'], DonutVAELoader: ['vae', 'vae_name'],
      UpscaleModelLoader: ['upscale_models', 'model_name'], SAMLoader: ['sams', 'model_name'],
      UltralyticsDetectorProvider: ['ultralytics', 'model_name'],
    };
    async function hasModel(folder, filename) {
      if (!filename || filename === 'None') return;
      if (!catalogs.has(folder)) {
        const response = await api.fetchApi('/models/' + folder);
        if (!response.ok) throw new Error('Model discovery failed for ' + folder + '. Check backend readiness.');
        catalogs.set(folder, await response.json());
      }
      if (!catalogs.get(folder).includes(filename)) missingModels.add(folder + '/' + filename);
    }
    for (const node of Object.values(graph.output || {})) {
      const type = node.class_type, inputs = node.inputs || {};
      if (!info[type]) missingNodes.add(type || 'Unregistered workflow node');
      const field = fileWidgets[type];
      if (field && typeof inputs[field[1]] === 'string') await hasModel(field[0], inputs[field[1]]);
      if (type === 'DonutLoRALoader') {
        const rows = JSON.parse(inputs.slots_json || '[]');
        for (const row of rows) if (row.enabled && row.lora_name) await hasModel('loras', row.lora_name);
      }
      if (type === 'DonutEditStudio' && inputs.enabled) {
        await hasModel('loras', inputs.lora_name);
        if (inputs.use_reference_b && inputs.mask_b_mode === 'Auto subject') {
          await hasModel('background_removal', inputs.mask_b_model);
          for (const nodeType of ['LoadBackgroundRemovalModel', 'RemoveBackground']) if (!info[nodeType]) missingNodes.add(nodeType);
        }
      }
      if (type === 'DonutVAELoader' && inputs.vae_name === 'Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors' && !info.VAEUtils_PatchWanUpscaleVAE) missingNodes.add('VAEUtils_PatchWanUpscaleVAE');
      if (type === 'DonutSampler' && inputs.sda_enabled) await hasModel('loras', 'krea2/krea2_turbo_sda_v1.0_comfy.safetensors');
      if (type === 'DonutToneLab' && inputs.enabled) await hasModel('donut_tone', inputs.model_name);
      if ((type === 'DonutTiledUpscale' && inputs.upscale_engine === 'SeedVR2') || (type === 'DonutSeedVR2Upscale' && inputs.enabled)) {
        for (const nodeType of ['SeedVR2Preprocess', 'SeedVR2Conditioning', 'SeedVR2PostProcessing']) if (!info[nodeType]) missingNodes.add(nodeType);
        await hasModel('diffusion_models', inputs.seedvr2_model_name);
        await hasModel('vae', inputs.seedvr2_vae_name);
      }
    }
    if (missingNodes.size || missingModels.size) {
      throw new Error(['Missing node types: ' + [...missingNodes].join(', '), 'Missing models: ' + [...missingModels].join(', ')].filter(value => !value.endsWith(': ')).join('. ') + '. Install the selected models and node packs before generating.');
    }
  }
  async function finishLoad() {
    initialized = true;
    try { await checkCapabilities(); lastError = ''; notice('Draft is saved locally for this backend.'); }
    catch (error) { showError(error); }
    saveDraft();
  }
  async function loadPreset() {
    if (!ready) throw new Error('The ComfyUI editor is still loading.');
    loading = true;
    try { if (await app._donutCreateLoadGraphData(await preset()) === false) throw new Error('The setup workflow could not be loaded. Check the Advanced editor.'); }
    finally { loading = false; }
    await finishLoad();
  }
  async function readLatest(revisionOnly = false) {
    const response = await api.fetchApi('/dmc/workflow' + (revisionOnly ? '?revision_only=1' : ''));
    if (!response.ok) {
      if (response.status === 404) return {revision: 0, workflow: null};
      throw new Error('The latest shared workflow could not be read. Check the studio connection.');
    }
    const value = await response.json();
    // Older isolated backends do not publish shared workflow revisions.
    if (!value || !Number.isSafeInteger(value.revision) || value.revision < 0) return {revision: 0, workflow: null};
    return {revision: value.revision, workflow: value.workflow || null};
  }
  async function loadLatest() {
    requireEditor();
    const latest = await readLatest();
    if (!latest.workflow) throw new Error('No workflow has been run in this workspace yet.');
    loading = true;
    try {
      if (await app._donutCreateLoadGraphData(latest.workflow) === false) throw new Error('The latest workflow could not be loaded. Check the Advanced editor.');
      latestRun = latest; loadedRunRevision = latest.revision;
    }
    finally { loading = false; }
    await finishLoad();
  }
  async function loadWorkflow(payload) {
    requireEditor();
    const workflow = payload?.workflow;
    if (!object(workflow) || !Array.isArray(workflow.nodes) || !workflow.nodes.length
      || JSON.stringify(workflow).length > 4 * 1024 * 1024) throw new Error('This image has no valid editable workflow.');
    const before = (app.rootGraph || app.graph).serialize();
    loading = true;
    try {
      if (await app._donutCreateLoadGraphData(workflow) === false) throw new Error('The attached workflow could not be loaded.');
      // This is a saved image recipe, not an acknowledgement of the shared run.
      loadedRunRevision = 0;
    } catch (error) {
      await app._donutCreateLoadGraphData(before);
      throw error;
    } finally { loading = false; }
    await finishLoad();
  }

  // Only these named v5 controls are exposed to the parent. The controls panel
  // supplies live paths/options; verified paths also support its initial load.
  const basicFields = {
    prompt: ['Text', [1138, 1126], 'text', 'Prompt · subject and scene'],
    stylePrompt: ['Text', [1138, 1125], 'text', 'Face and general style'],
    negativePrompt: ['Text', [1138, 1127], 'text', 'Negative prompt'],
    model: ['unet_name', [1138, 1122], 'select', 'Primary model file'],
    secondaryModel: ['unet_name', [1138, 1120], 'select', 'Secondary model file'],
    modelMode: ['model_mode', [1138, 1128, 1124], 'select', 'Model mode'],
    width: ['width', [881], 'number', 'Custom output width'],
    height: ['height', [881], 'number', 'Custom output height'],
    aspectRatio: ['aspect_ratio', [881], 'select', 'Output aspect ratio'],
    resolutionMode: ['resolution_mode', [881], 'select', 'Output sizing mode'],
    megapixels: ['megapixels', [881], 'number', 'Output megapixels'],
    batchSize: ['batch_size', [1014, 999], 'number', 'Batch size'],
    steps: ['steps', [1014], 'number', 'Steps'],
    guidance: ['cfg_start', [1014], 'number', 'Cfg start'],
    seed: ['seed', [1138, 1137], 'number', 'Shared seed'],
    seedMode: ['fixed', [1138, 1137], 'select', 'After generation'],
    editing: ['enabled', [881], 'boolean'],
    editPrompt: ['prompt', [881], 'text'],
    referenceA: ['image_a', [881], 'text'],
    inpaint: ['inpaint_enabled', [881], 'boolean'],
    editMask: ['mask_data', [881], 'text'],
    maskFeather: ['mask_feather', [881], 'number'],
  };
  const integerFields = new Set(['width', 'height', 'batchSize', 'steps', 'seed', 'maskFeather']);
  const channel = 'donut-create-basic-v1';
  const actions = new Set(['snapshot', 'patch', 'generate', 'upload-reference', 'load-preset', 'load-latest', 'load-workflow']);
  const referencePattern = /^donutref:[a-f0-9]{64}$/;
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const graphNodes = graph => graph?.nodes || graph?._nodes || [];

  function liveNodes(graph = app?.rootGraph || app?.graph, seen = new Set()) {
    if (!graph || seen.has(graph)) return [];
    seen.add(graph);
    return Array.from(graphNodes(graph)).flatMap(node => [node, ...liveNodes(node.subgraph, seen)]);
  }
  function resolveControl(path) {
    let graph = app?.rootGraph || app?.graph, node;
    for (const id of path) {
      node = graph?.getNodeById?.(id) || Array.from(graphNodes(graph)).find(item => String(item.id) === String(id));
      graph = node?.subgraph;
    }
    return node;
  }
  function bindings() {
    const controls = liveNodes().flatMap(node => (node.properties?.donut_app_controls?.groups || []).flatMap(group => group.controls || []));
    const result = {};
    for (const [key, [name, path, kind, title]] of Object.entries(basicFields)) {
      const control = controls.find(item => item.widget === name && item.path?.length === path.length && item.path.every((id, index) => String(id) === String(path[index])))
        || (title && controls.find(item => item.widget === name && item.title === title));
      let node = resolveControl(control?.path || path);
      if (!node && control?.fallback_type) {
        const graph = resolveControl(control.path.slice(0, -1))?.subgraph;
        node = Array.from(graphNodes(graph)).find(item => item.type === control.fallback_type || item.properties?.['Node name for S&R'] === control.fallback_type);
      }
      const widget = node?.widgets?.find(item => item.name === name);
      if (widget) result[key] = {node, widget, control, kind};
    }
    // V5 executes the shared scene/style conditioning, rather than the unused
    // Edit Studio prompt output. Basic edit instructions edit that same scene.
    if (resolveControl([881])?.properties?.donut_shared_prompt && result.prompt) result.editPrompt = result.prompt;
    return result;
  }
  function descriptor(key, binding) {
    const kind = basicFields[key][2];
    const result = {value: null, kind, options: [], min: null, max: null, step: null, available: Boolean(binding)};
    if (!binding) return result;
    const {widget, control} = binding;
    const value = widget.value;
    // A legacy reference path stays in the advanced graph but never crosses
    // this narrow bridge. New references are controller-scoped opaque tokens.
    result.value = key === 'referenceA' ? (referencePattern.test(value) ? value : '')
      : kind === 'boolean' ? Boolean(value)
      : ['string', 'number', 'boolean'].includes(typeof value) ? value : null;
    if (key === 'editMask' && value) {
      try { validateMask(value, binding.node.widgets.find(item => item.name === 'image_a')?.value, true); }
      catch { result.value = ''; }
    }
    if (kind === 'select') {
      let values = control?.choices || widget.options?.values;
      if (typeof values === 'function') values = values.call(widget);
      if (Array.isArray(values)) result.options = values.filter(item => ['string', 'number', 'boolean'].includes(typeof item));
      result.available = result.options.length > 0;
    }
    if (kind === 'number') {
      result.min = Number.isFinite(widget.options?.min) ? widget.options.min : 0;
      result.max = Number.isFinite(widget.options?.max) ? widget.options.max : Number.MAX_SAFE_INTEGER;
      if (key === 'seed') result.max = Math.min(result.max, Number.MAX_SAFE_INTEGER);
      if (key === 'maskFeather') { result.min = Math.max(0, result.min); result.max = Math.min(128, result.max); }
      // LiteGraph number widgets store increments scaled by ten.
      const step = widget.options?.step2 ?? (widget.options?.step / 10);
      result.step = integerFields.has(key) ? 1 : Number.isFinite(step) && step > 0 ? step : 0.01;
    }
    return result;
  }
  function snapshot() {
    const current = bindings();
    return {ready: Boolean(ready && initialized && !loading),
      fields: Object.fromEntries(Object.keys(basicFields).map(key => [key, descriptor(key, current[key])])),
      latestRunRevision: latestRun.revision, latestAvailable: latestRun.revision > loadedRunRevision,
      ...(lastError ? {error: lastError} : {})};
  }
  function requireEditor() {
    if (!ready || !initialized || loading || !app?.graph) throw new Error('The ComfyUI editor is still loading.');
  }
  function validateMask(value, reference, advanced = false) {
    if (value === '') return null;
    if (typeof value !== 'string' || value.length > 2 * 1024 * 1024) throw new Error('The selection is too large. Use fewer brush strokes.');
    let mask;
    try { mask = JSON.parse(value); } catch { throw new Error('The selection is invalid. Paint it again.'); }
    if (!object(mask) || mask.version !== 1 || !referencePattern.test(reference) || mask.image !== reference || !Array.isArray(mask.strokes)
      || mask.strokes.length > 1000 || (mask.inverted !== undefined && typeof mask.inverted !== 'boolean')
      || Object.keys(mask).some(key => !['version', 'image', 'strokes', 'inverted', ...(advanced ? ['outpaint'] : [])].includes(key))) {
      throw new Error('Paint a selection for the current reference image.');
    }
    let points = 0;
    for (const stroke of mask.strokes) {
      if (!object(stroke) || !Number.isFinite(stroke.size) || stroke.size <= 0 || !advanced && stroke.size < 0.01 || stroke.size > 1
        || (stroke.erase === undefined ? !advanced : typeof stroke.erase !== 'boolean') || !Array.isArray(stroke.points) || !stroke.points.length
        || Object.keys(stroke).some(key => !['size', 'erase', 'points', ...(advanced ? ['shape'] : [])].includes(key))
        || (stroke.shape !== undefined && (!advanced || stroke.shape !== 'rectangle' || stroke.points.length !== 2))) {
        throw new Error('The selection contains an invalid brush stroke.');
      }
      points += stroke.points.length;
      if (points > 10000 || stroke.points.some(point => !Array.isArray(point) || point.length !== 2 || point.some(coordinate => !Number.isFinite(coordinate) || coordinate < 0 || coordinate > 1))) {
        throw new Error('The selection contains invalid or too many brush points.');
      }
    }
    if (mask.outpaint !== undefined) {
      const settings = mask.outpaint;
      if (!advanced || !object(settings) || Object.keys(settings).some(key => !['scale', 'x', 'y', 'overlap'].includes(key)) || [['scale', 0.1, 1], ['x', 0, 1], ['y', 0, 1], ['overlap', 0, 128]]
        .some(([key, min, max]) => !Number.isFinite(settings[key]) || settings[key] < min || settings[key] > max)) {
        throw new Error('The advanced outpaint selection is invalid. Repair it in the Advanced editor.');
      }
    }
    return mask;
  }
  function hasSelection(mask) {
    if (!mask) return false;
    if (mask.outpaint) return true; // Advanced placement is checked by the backend.
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 128;
    const context = canvas.getContext?.('2d', {willReadFrequently: true});
    if (!context) return Boolean(mask.inverted || mask.strokes.some(stroke => !stroke.erase));
    for (const stroke of mask.strokes) {
      context.globalCompositeOperation = stroke.erase ? 'destination-out' : 'source-over';
      context.fillStyle = context.strokeStyle = '#fff';
      context.lineWidth = Math.max(1, stroke.size * 128); context.lineCap = context.lineJoin = 'round';
      const points = stroke.points.map(([x, y]) => [x * 127, y * 127]);
      if (stroke.shape === 'rectangle') {
        const [[x1, y1], [x2, y2]] = points;
        context.fillRect(Math.min(x1, x2), Math.min(y1, y2), Math.max(1, Math.abs(x1 - x2)), Math.max(1, Math.abs(y1 - y2)));
      } else {
        context.beginPath(); context.moveTo(...points[0]);
        for (const point of points.slice(1)) context.lineTo(...point);
        context.stroke();
        for (const [x, y] of points) { context.beginPath(); context.arc(x, y, context.lineWidth / 2, 0, Math.PI * 2); context.fill(); }
      }
    }
    const pixels = context.getImageData(0, 0, 128, 128).data;
    for (let index = 3; index < pixels.length; index += 4) if (mask.inverted ? pixels[index] < 255 : pixels[index] > 0) return true;
    return false;
  }
  async function applyPatch(payload) {
    requireEditor();
    if (!object(payload) || !Object.keys(payload).length) throw new Error('Choose a supported control to change.');
    const current = bindings(), updates = new Map();
    const reference = Object.hasOwn(payload, 'referenceA') ? payload.referenceA : current.referenceA?.widget.value;
    for (const [key, value] of Object.entries(payload)) {
      if (!Object.hasOwn(basicFields, key)) throw new Error('Unsupported control: ' + key);
      const binding = current[key], field = descriptor(key, binding);
      if (!field.available) throw new Error('This workflow does not expose ' + key + '. Open the Advanced editor.');
      if (field.kind === 'boolean' && typeof value !== 'boolean'
        || field.kind === 'text' && (typeof value !== 'string' || value.length > (key === 'editMask' ? 2 * 1024 * 1024 : 65536))
        || field.kind === 'select' && !field.options.includes(value)
        || field.kind === 'number' && (typeof value !== 'number' || !Number.isFinite(value) || value < field.min || value > field.max || integerFields.has(key) && !Number.isSafeInteger(value))) {
        throw new Error('Invalid value for ' + key + '.');
      }
      if (key === 'referenceA' && value !== '' && (!referencePattern.test(value) || value !== binding.widget.value && !uploadedReferences.has(value))) {
        throw new Error('Upload a reference in this studio before selecting it.');
      }
      if (key === 'editMask') validateMask(value, reference);
      if (updates.has(binding.widget) && updates.get(binding.widget).value !== value) throw new Error('Prompt and edit instruction share the same workflow control. Choose one value.');
      updates.set(binding.widget, {...binding, value});
    }
    if (Object.hasOwn(payload, 'referenceA') && payload.referenceA !== current.referenceA?.widget.value) {
      if (current.editMask) updates.set(current.editMask.widget, {...current.editMask, value: Object.hasOwn(payload, 'editMask') ? payload.editMask : ''});
      if (current.inpaint && !Object.hasOwn(payload, 'inpaint')) updates.set(current.inpaint.widget, {...current.inpaint, value: false});
      const node = current.referenceA.node, crop = node.widgets.find(item => item.name === 'crop_data_a');
      if (crop) updates.set(crop, {node, widget: crop, value: ''});
    }
    // Prevalidation above is complete before touching any live widget. Keep a
    // graph rollback for callback failures, including promoted sibling updates.
    const before = (app.rootGraph || app.graph).serialize();
    try {
      for (const {node, widget, value} of updates.values()) {
        if (widget.value === value) continue;
        node.graph?.beforeChange?.();
        try { widget.value = value; widget.callback?.(value, app.canvas, node); }
        finally { node.graph?.afterChange?.(); }
        node.setDirtyCanvas?.(true, true);
      }
      await Promise.resolve();
      for (const node of new Set([...updates.values()].map(item => item.node))) node._donutEditStudio?.render?.();
    } catch (error) {
      loading = true;
      try { await app._donutCreateLoadGraphData(before); } finally { loading = false; }
      throw error;
    }
    lastError = ''; saveDraft();
  }
  async function uploadReference(payload) {
    requireEditor();
    if (!object(payload) || Object.keys(payload).some(key => key !== 'file') || typeof File === 'undefined' || !(payload.file instanceof File)
      || !/^image\/(png|jpeg|webp|gif|bmp|tiff)$/.test(payload.file.type) || payload.file.size <= 0 || payload.file.size > 32 * 1024 * 1024) {
      throw new Error('Choose an image file up to 32 MiB.');
    }
    const current = bindings();
    if (!current.referenceA || !current.editing) throw new Error('This workflow has no reference editing controls. Load the setup preset.');
    const form = new FormData(); form.append('image', payload.file, 'reference');
    const response = await api.fetchApi('/donut/edit-studio/reference', {method: 'POST', body: form});
    if (!response.ok) throw new Error('The reference upload failed (' + response.status + ').');
    const saved = await response.json();
    if (!referencePattern.test(saved?.reference)) throw new Error('The backend did not return a scoped reference.');
    uploadedReferences.add(saved.reference);
    await applyPatch({referenceA: saved.reference, editing: true,
      ...(current.editMask ? {editMask: ''} : {}), ...(current.inpaint ? {inpaint: false} : {})});
  }
  async function generate() {
    requireEditor();
    const current = bindings();
    if (current.editing?.widget.value) {
      if (!referencePattern.test(current.referenceA?.widget.value)) throw new Error('Upload a reference image before editing.');
      if (current.inpaint?.widget.value) {
        const mask = validateMask(current.editMask?.widget.value, current.referenceA.widget.value, true);
        if (!hasSelection(mask)) throw new Error('Paint an area to edit, or turn the selection off.');
      }
    }
    if (typeof app.queuePrompt !== 'function') throw new Error('The ComfyUI Run action is not available yet.');
    await checkCapabilities(); saveDraft();
    const attempt = {accepted: 0, error: null}; queueAttempt = attempt;
    try {
      // The app Run path calls before/afterQueued widget hooks. It can swallow
      // API failures, so our API wrapper also records the actual queue outcome.
      await app.queuePrompt(0, 1);
      if (attempt.error) throw attempt.error;
      if (!attempt.accepted) throw new Error('ComfyUI did not queue this workflow. Check the Advanced editor for validation errors.');
      lastError = ''; saveDraft();
    } finally { queueAttempt = null; }
  }
  window.addEventListener('message', event => {
    const message = event.data;
    // Tauri/Wry's registered native scheme uses a tuple origin, like HTTP.
    // Keep the exact observed origin; opaque parents cannot receive a reply.
    if (event.source !== window.parent || !object(message) || message.channel !== channel || message.sessionId !== config.sessionId
      || typeof message.requestId !== 'string' || !message.requestId.length || message.requestId.length > 256 || !actions.has(message.action)
      || !event.origin || event.origin === 'null') return;
    if (parentOrigin === null) {
      if (message.action !== 'snapshot') return;
      parentOrigin = event.origin;
    }
    if (event.origin !== parentOrigin) return;
    const reply = value => event.source.postMessage({channel, sessionId: config.sessionId, requestId: message.requestId, ...value}, event.origin);
    bridgeQueue = bridgeQueue.then(async () => {
      try {
        if (message.action === 'snapshot' && initialized && !loading && !queueAttempt) {
          const latest = await readLatest(true);
          if (latest.revision > latestRun.revision) {
            latestRun = latest;
            notice('A newer run is available in this workspace. Load it from the basic controls to replace this draft.');
          }
        }
        else if (message.action === 'patch') await applyPatch(message.payload);
        else if (message.action === 'generate') await generate();
        else if (message.action === 'upload-reference') await uploadReference(message.payload);
        else if (message.action === 'load-preset') await loadPreset();
        else if (message.action === 'load-latest') await loadLatest();
        else if (message.action === 'load-workflow') await loadWorkflow(message.payload);
        reply({ok: true, snapshot: snapshot()});
      } catch (error) { showError(error); reply({ok: false, error: error?.message || String(error)}); }
    }).catch(showError);
  });
  async function init() {
    ({ app } = await import(new URL('scripts/app.js', base).href));
    ({ api } = await import(new URL('scripts/api.js', base).href));
    window.app = app;
    // ComfyUI computes api_base from location.pathname. Keep it aligned with
    // the scoped proxy even when a host supplies a custom studio prefix.
    api.api_base = base.pathname.replace(/\/$/, '');
    // Scoped extension links already include the API base. Newer ComfyUI
    // helpers otherwise prepend it a second time during module loading.
    for (const helper of ['apiURL', 'fileURL']) {
      if (typeof api[helper] !== 'function') continue;
      const originalURL = api[helper].bind(api);
      api[helper] = function (route, ...args) {
        return typeof route === 'string' && route.startsWith(base.pathname)
          ? route : originalURL(route, ...args);
      };
    }
    try {
      const defaults = await import(new URL('scripts/defaultGraph.js', base).href);
      defaultGraph = defaults.defaultGraph;
    } catch (error) { /* Empty first graphs remain supported. */ }
    try { latestRun = await readLatest(); } catch (error) { showError(error); }
    const original = app.loadGraphData.bind(app);
    app._donutCreateLoadGraphData = original;
    app.loadGraphData = async function (graphData, ...args) {
      if (loading) return original(graphData, ...args);
      if (initialized) {
        // A delayed stock restore after our empty-graph bootstrap cannot
        // replace the studio. All explicit workflow uploads still use ComfyUI.
        if (ignoreInitialDefault && stockWorkflow(graphData)) { ignoreInitialDefault = false; return; }
        const result = await original(graphData, ...args); return result;
      }
      loading = true;
      try {
        const own = savedDraft();
        // Shared runs are the cross-device baseline. Without a submitted run,
        // preserve meaningful graphs and restore local drafts for stock graphs.
        const chosen = latestRun.workflow || (stockWorkflow(graphData) ? (own || await preset()) : graphData);
        const result = await original(chosen, ...args);
        if (result === false) throw new Error('The studio workflow could not be loaded. Check the Advanced editor.');
        ready = true;
        if (latestRun.workflow) loadedRunRevision = latestRun.revision;
        await finishLoad(); return result;
      } finally { loading = false; saveDraft(); }
    };
    const originalQueue = api.queuePrompt.bind(api);
    api.queuePrompt = async function (...args) {
      const attempt = queueAttempt;
      try {
        await checkCapabilities(); saveDraft();
        const result = await originalQueue(...args);
        if (result?.error || result?.node_errors && Object.keys(result.node_errors).length) throw new Error(result.error?.message || 'The backend rejected this workflow. Check the Advanced editor for node errors.');
        if (attempt && result?.prompt_id) attempt.accepted++;
        if (result?.prompt_id) {
          try {
            const latest = await readLatest();
            const canonical = value => JSON.stringify(value, (key, item) => object(item) ? Object.fromEntries(Object.keys(item).sort().map(name => [name, item[name]])) : item);
            if (latest.workflow && canonical(latest.workflow) === canonical(args[1]?.workflow)) loadedRunRevision = latest.revision;
            latestRun = {revision: latest.revision, workflow: null};
          }
          catch (error) { showError(error); }
        }
        return result;
      } catch (error) { if (attempt) attempt.error = error; showError(error); throw error; }
    };
    app.registerExtension({
      name: 'DonutCreate.Studio',
      async setup() { ready = true; },
    });
    const started = Date.now();
    const wait = setInterval(async () => {
      if (!app.graph || !app.canvas || !window.LiteGraph?.registered_node_types?.DonutWorkflowPanel) {
        if (Date.now() - started > 60000) { clearInterval(wait); showError(new Error('The v5 editor did not finish loading. Required DonutNodes browser panels may be missing.')); }
        return;
      }
      clearInterval(wait); ready = true;
      if (!initialized && app.graph._nodes?.length) {
        const current = app.graph.serialize();
        if (latestRun.workflow || stockWorkflow(current)) await app.loadGraphData(current).catch(showError);
        else { initialized = true; await finishLoad(); }
      }
      else if (!initialized) {
        // Before ComfyUI's first restore, our wrapper owns initial loading.
        ignoreInitialDefault = true;
        await app.loadGraphData().catch(showError);
      }
    }, 200);
    setInterval(saveDraft, 1500);
    window.addEventListener('pagehide', saveDraft);
    document.addEventListener('visibilitychange', () => { if (document.hidden) saveDraft(); });
    notice('Loading the v5 editor…');
  }
  init().catch(showError);
})();
