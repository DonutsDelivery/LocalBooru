package com.localbooru.app

import android.os.Bundle
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebSettings
import android.webkit.WebView
import androidx.activity.enableEdgeToEdge
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

class MainActivity : TauriActivity() {
  private var isImmersive = false

  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
    // Enable Chrome DevTools remote debugging (chrome://inspect)
    WebView.setWebContentsDebuggingEnabled(true)
  }

  override fun onWebViewCreate(webView: WebView) {
    // Allow mixed content: the embedded axum server runs on http://127.0.0.1:8790
    // while the WebView serves from https://tauri.localhost. Without this, all
    // HTTP requests (XHR, fetch, img src, video src) would be blocked.
    webView.settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
    Log.i("LocalBooru", "onWebViewCreate: mixedContentMode set to ALWAYS_ALLOW")

    // Inject status bar height as CSS variable since env(safe-area-inset-top) is
    // unreliable in Android WebView with edge-to-edge
    val statusBarHeightPx = getStatusBarHeight()
    val density = resources.displayMetrics.density
    val statusBarHeightDp = (statusBarHeightPx / density).toInt()
    webView.evaluateJavascript(
      "document.documentElement.style.setProperty('--android-status-bar-height', '${statusBarHeightDp}px')",
      null
    )
    Log.i("LocalBooru", "Status bar height: ${statusBarHeightDp}dp (${statusBarHeightPx}px)")

    // Expose native immersive mode toggle to JavaScript
    webView.addJavascriptInterface(ImmersiveBridge(), "AndroidImmersive")
  }

  private fun getStatusBarHeight(): Int {
    val resourceId = resources.getIdentifier("status_bar_height", "dimen", "android")
    return if (resourceId > 0) resources.getDimensionPixelSize(resourceId) else 0
  }

  inner class ImmersiveBridge {
    @JavascriptInterface
    fun enter() {
      runOnUiThread {
        val controller = WindowCompat.getInsetsController(window, window.decorView)
        controller.systemBarsBehavior =
          WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        controller.hide(WindowInsetsCompat.Type.systemBars())
        isImmersive = true
        Log.i("LocalBooru", "Entered immersive mode")
      }
    }

    @JavascriptInterface
    fun exit() {
      runOnUiThread {
        val controller = WindowCompat.getInsetsController(window, window.decorView)
        controller.show(WindowInsetsCompat.Type.systemBars())
        isImmersive = false
        Log.i("LocalBooru", "Exited immersive mode")
      }
    }

    @JavascriptInterface
    fun isActive(): Boolean {
      return isImmersive
    }
  }
}
