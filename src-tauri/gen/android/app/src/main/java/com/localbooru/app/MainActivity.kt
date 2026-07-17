package com.localbooru.app

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebSettings
import android.webkit.WebView
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

class MainActivity : TauriActivity() {
  private var isImmersive = false
  private var appWebView: WebView? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge(
      statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
      navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT)
    )
    super.onCreate(savedInstanceState)
    // Enable Chrome DevTools remote debugging (chrome://inspect) in debug builds
    // only — leaving it on in release lets anyone with USB/ADB inspect the WebView.
    if (BuildConfig.DEBUG) {
      WebView.setWebContentsDebuggingEnabled(true)
    }
  }

  override fun onWebViewCreate(webView: WebView) {
    appWebView = webView

    // Allow mixed content: the embedded axum server runs on http://127.0.0.1:8790
    // while the WebView serves from https://tauri.localhost. Without this, all
    // HTTP requests (XHR, fetch, img src, video src) would be blocked.
    webView.settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
    Log.i("LocalBooru", "onWebViewCreate: mixedContentMode set to ALWAYS_ALLOW")

    // LocalBooru drives video playback from its own controls/autoplay state.
    // Android WebView otherwise requires a gesture for media with audio, which
    // can make direct-play video appear to run with muted or blocked audio.
    webView.settings.mediaPlaybackRequiresUserGesture = false
    Log.i("LocalBooru", "onWebViewCreate: mediaPlaybackRequiresUserGesture=false")

    // Inject system bar insets as CSS variables since env(safe-area-inset-*) is
    // unreliable in Android WebView with edge-to-edge. The listener re-fires on
    // rotation and when immersive mode hides the bars, so the values stay live.
    ViewCompat.setOnApplyWindowInsetsListener(webView) { v, insets ->
      val bars = insets.getInsets(
        WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
      )
      val density = resources.displayMetrics.density
      val top = (bars.top / density).toInt()
      val bottom = (bars.bottom / density).toInt()
      val left = (bars.left / density).toInt()
      val right = (bars.right / density).toInt()
      webView.evaluateJavascript(
        """
        (function(s){
          s.setProperty('--android-inset-top', '${top}px');
          s.setProperty('--android-inset-bottom', '${bottom}px');
          s.setProperty('--android-inset-left', '${left}px');
          s.setProperty('--android-inset-right', '${right}px');
        })(document.documentElement.style)
        """.trimIndent(),
        null
      )
      Log.i("LocalBooru", "Insets (dp): top=$top bottom=$bottom left=$left right=$right")
      insets
    }
    // Apply current insets immediately (the listener only fires on changes), and
    // re-dispatch a few times so the values land after the page finishes loading
    webView.requestApplyInsets()
    for (delayMs in longArrayOf(500, 1500, 4000)) {
      webView.postDelayed({ webView.requestApplyInsets() }, delayMs)
    }

    // Expose native immersive mode toggle to JavaScript
    webView.addJavascriptInterface(ImmersiveBridge(), "AndroidImmersive")
  }

  override fun onResume() {
    super.onResume()
    appWebView?.post { appWebView?.requestApplyInsets() }
  }

  override fun onWindowFocusChanged(hasFocus: Boolean) {
    super.onWindowFocusChanged(hasFocus)
    if (hasFocus && isImmersive) {
      applyImmersiveMode(true)
    }
  }

  private fun applyImmersiveMode(immersive: Boolean) {
    val controller = WindowCompat.getInsetsController(window, window.decorView)
    controller.systemBarsBehavior =
      WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    if (immersive) {
      controller.hide(WindowInsetsCompat.Type.systemBars())
    } else {
      controller.show(WindowInsetsCompat.Type.systemBars())
    }
    isImmersive = immersive
    appWebView?.post { appWebView?.requestApplyInsets() }
    Log.i("LocalBooru", if (immersive) "Entered immersive mode" else "Exited immersive mode")
  }

  inner class ImmersiveBridge {
    @JavascriptInterface
    fun enter() {
      runOnUiThread { applyImmersiveMode(true) }
    }

    @JavascriptInterface
    fun exit() {
      runOnUiThread { applyImmersiveMode(false) }
    }

    @JavascriptInterface
    fun isActive(): Boolean {
      return isImmersive
    }
  }
}
