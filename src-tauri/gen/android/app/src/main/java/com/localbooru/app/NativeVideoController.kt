package com.localbooru.app

import android.graphics.Color
import android.media.MediaCodecList
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.annotation.OptIn
import androidx.core.view.doOnLayout
import androidx.media3.common.C
import androidx.media3.common.Format
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.Presentation
import androidx.media3.exoplayer.DefaultRenderersFactory
import androidx.media3.exoplayer.DecoderReuseEvaluation
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.analytics.AnalyticsListener
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Owns only Android video decoding and pixels. The transparent Tauri WebView
 * remains above this TextureView and continues to own LocalBooru's controls.
 */
@OptIn(UnstableApi::class)
class NativeVideoController(
  private val activity: MainActivity,
  private val webView: WebView,
) : Player.Listener, AnalyticsListener, SurfaceHolder.Callback {
  private val mainHandler = Handler(Looper.getMainLooper())
  private val surfaceView = SurfaceView(activity)
  private var player: ExoPlayer? = null
  private var activeGeneration = 0L
  private var inputWidth = 0
  private var inputHeight = 0
  private var inputMimeType: String? = null
  private var inputCodecs: String? = null
  private var inputFrameRate = 0f
  private var reportedVideoWidth = 0
  private var reportedVideoHeight = 0
  private var effectOutputWidth = 0
  private var effectOutputHeight = 0
  private var effectHeight = 0
  private var droppedFrames = 0L
  private var decoderName: String? = null
  private var decoderHardwareAccelerated: Boolean? = null
  private var requestedVolume = 1f
  private var muted = false

  private val positionReporter = object : Runnable {
    override fun run() {
      val currentPlayer = player ?: return
      if (activeGeneration == 0L) return
      emit(
        "position",
        JSONObject()
          .put("position", currentPlayer.currentPosition.coerceAtLeast(0L) / 1000.0)
          .put("duration", durationSeconds(currentPlayer))
          .put("buffered", currentPlayer.bufferedPosition.coerceAtLeast(0L) / 1000.0)
      )
      mainHandler.postDelayed(this, POSITION_INTERVAL_MS)
    }
  }

  init {
    surfaceView.visibility = View.GONE
    surfaceView.setZOrderOnTop(false)
    surfaceView.holder.addCallback(this)
    webView.post { attachBehindWebView() }
  }

  private fun attachBehindWebView() {
    if (surfaceView.parent != null) return
    val parent = webView.parent as? ViewGroup
      ?: throw IllegalStateException("Tauri WebView has no native parent")
    val webViewIndex = parent.indexOfChild(webView).coerceAtLeast(0)
    parent.addView(surfaceView, webViewIndex, ViewGroup.LayoutParams(1, 1))
  }

  private fun ensurePlayer(): ExoPlayer {
    player?.let { return it }
    val httpDataSourceFactory = DefaultHttpDataSource.Factory()
      .setUserAgent("LocalBooru/2.0.3 Android")
      .setAllowCrossProtocolRedirects(false)
    val mediaSourceFactory = DefaultMediaSourceFactory(httpDataSourceFactory)
    val renderersFactory = DefaultRenderersFactory(activity)
      .setEnableDecoderFallback(true)
    return ExoPlayer.Builder(activity)
      .setMediaSourceFactory(mediaSourceFactory)
      .setRenderersFactory(renderersFactory)
      .build()
      .also { created ->
      created.addListener(this)
      created.addAnalyticsListener(this)
      created.setVideoSurfaceView(surfaceView)
      player = created
    }
  }

  @JavascriptInterface
  fun available(): Boolean = true

  @JavascriptInterface
  fun open(
    generation: Long,
    url: String,
    startPosition: Double,
    autoplay: Boolean,
    maxOutputHeight: Int,
    viewportLeft: Double,
    viewportTop: Double,
    viewportWidth: Double,
    viewportHeight: Double,
    devicePixelRatio: Double,
  ) {
    activity.runOnUiThread {
      try {
        val mediaUri = validateMediaUri(url)
        activeGeneration = generation
        inputWidth = 0
        inputHeight = 0
        inputMimeType = null
        inputCodecs = null
        inputFrameRate = 0f
        reportedVideoWidth = 0
        reportedVideoHeight = 0
        effectOutputWidth = 0
        effectOutputHeight = 0
        effectHeight = maxOutputHeight.coerceAtLeast(0)
        droppedFrames = 0
        decoderName = null
        decoderHardwareAccelerated = null
        attachBehindWebView()
        applyViewport(
          viewportLeft,
          viewportTop,
          viewportWidth,
          viewportHeight,
          devicePixelRatio,
        )
        // Hardware-accelerated WebView content is normally composited as an
        // opaque layer even where the DOM background is transparent. Render
        // the control layer in software while native video owns the aperture
        // so its alpha is blended over the sibling SurfaceView.
        webView.setBackgroundColor(Color.TRANSPARENT)
        webView.setLayerType(View.LAYER_TYPE_SOFTWARE, null)
        surfaceView.visibility = View.VISIBLE

        emit("preparing")
        mainHandler.removeCallbacks(positionReporter)
        mainHandler.post(positionReporter)
        surfaceView.doOnLayout {
          if (generation != activeGeneration) return@doOnLayout
          try {
            require(surfaceView.width > 1 && surfaceView.height > 1) {
              "Native video viewport did not complete layout"
            }
            val currentPlayer = ensurePlayer()
            applyVideoEffect(currentPlayer)
            Log.i(
              TAG,
              "open generation=$generation source=${mediaUri.scheme}://${mediaUri.host}:${mediaUri.port}${mediaUri.path} " +
                "requestedEffectHeight=$effectHeight viewport=${surfaceView.width}x${surfaceView.height} " +
                "encoder=false outputFile=false",
            )
            currentPlayer.setMediaItem(
              MediaItem.fromUri(mediaUri),
              (startPosition.coerceAtLeast(0.0) * 1000.0).roundToInt().toLong(),
            )
            currentPlayer.playWhenReady = autoplay
            currentPlayer.prepare()
          } catch (error: Throwable) {
            Log.e(TAG, "Failed to prepare native video after layout", error)
            emit("error", error.message ?: "Native player failed after layout")
            closeActivePlayer()
          }
        }
      } catch (error: Throwable) {
        Log.e(TAG, "Failed to open native video", error)
        emit("error", error.message ?: "Native player failed")
        closeActivePlayer()
      }
    }
  }

  private fun validateMediaUri(rawUrl: String): Uri {
    val uri = Uri.parse(rawUrl)
    val isLoopbackHttp = (uri.scheme == "http" || uri.scheme == "https") &&
      (uri.host == "127.0.0.1" || uri.host == "localhost")
    require(isLoopbackHttp) { "Native playback accepts only LocalBooru loopback media URLs" }
    return uri
  }

  private fun applyVideoEffect(currentPlayer: ExoPlayer) {
    val presentation = if (effectHeight > 0) {
      Presentation.createForHeight(effectHeight)
    } else {
      null
    }
    val effects = presentation?.let(::listOf) ?: emptyList()
    currentPlayer.setVideoEffects(effects)
    if (presentation != null && inputWidth > 0 && inputHeight > 0) {
      val configured = presentation.configure(inputWidth, inputHeight)
      effectOutputWidth = configured.width
      effectOutputHeight = configured.height
    } else if (effectHeight == 0 && inputWidth > 0 && inputHeight > 0) {
      effectOutputWidth = inputWidth
      effectOutputHeight = inputHeight
    }
    Log.i(TAG, "setVideoEffects generation=$activeGeneration requestedHeight=$effectHeight effects=${effects.size}")
  }

  @JavascriptInterface
  fun play(generation: Long) = withGeneration(generation) { it.play() }

  @JavascriptInterface
  fun pause(generation: Long) = withGeneration(generation) { it.pause() }

  @JavascriptInterface
  fun seek(generation: Long, seconds: Double) = withGeneration(generation) {
    it.seekTo((seconds.coerceAtLeast(0.0) * 1000.0).roundToInt().toLong())
  }

  @JavascriptInterface
  fun setVolume(generation: Long, value: Double) = withGeneration(generation) {
    requestedVolume = value.coerceIn(0.0, 1.0).toFloat()
    if (!muted) it.volume = requestedVolume
  }

  @JavascriptInterface
  fun setMuted(generation: Long, muted: Boolean) = withGeneration(generation) {
    this.muted = muted
    it.volume = if (muted) 0f else requestedVolume
  }

  @JavascriptInterface
  fun setSpeed(generation: Long, speed: Double) = withGeneration(generation) {
    it.setPlaybackSpeed(speed.coerceIn(0.25, 4.0).toFloat())
  }

  @JavascriptInterface
  fun setDisplayMode(generation: Long, mode: String) = withGeneration(generation) {
    it.videoScalingMode = if (mode == "fill") {
      C.VIDEO_SCALING_MODE_SCALE_TO_FIT_WITH_CROPPING
    } else {
      C.VIDEO_SCALING_MODE_SCALE_TO_FIT
    }
  }

  @JavascriptInterface
  fun setReduction(generation: Long, maxOutputHeight: Int) = withGeneration(generation) {
    effectHeight = maxOutputHeight.coerceAtLeast(0)
    applyVideoEffect(it)
    emitVideoSize()
  }

  /** Coordinates are CSS pixels relative to the WebView. */
  @JavascriptInterface
  fun setViewport(
    generation: Long,
    left: Double,
    top: Double,
    width: Double,
    height: Double,
    devicePixelRatio: Double,
  ) {
    activity.runOnUiThread {
      if (generation != activeGeneration || activeGeneration == 0L) return@runOnUiThread
      attachBehindWebView()
      applyViewport(left, top, width, height, devicePixelRatio)
    }
  }

  private fun applyViewport(
    left: Double,
    top: Double,
    width: Double,
    height: Double,
    devicePixelRatio: Double,
  ) {
    require(width > 0.0 && height > 0.0) { "Native video viewport must be non-empty" }
    val scale = devicePixelRatio.coerceAtLeast(0.25)
    surfaceView.x = webView.x + (left * scale).toFloat()
    surfaceView.y = webView.y + (top * scale).toFloat()
    surfaceView.layoutParams = surfaceView.layoutParams.apply {
      this.width = max(1, (width * scale).roundToInt())
      this.height = max(1, (height * scale).roundToInt())
    }
    val parent = surfaceView.parent as? ViewGroup
    Log.i(
      TAG,
      "viewport generation=$activeGeneration css=${width}x$height scale=$scale " +
        "native=${surfaceView.layoutParams.width}x${surfaceView.layoutParams.height} " +
        "surfaceIndex=${parent?.indexOfChild(surfaceView)} webViewIndex=${parent?.indexOfChild(webView)}",
    )
    emitVideoSize()
  }

  @JavascriptInterface
  fun close(generation: Long) {
    activity.runOnUiThread {
      if (generation != activeGeneration && generation != 0L) return@runOnUiThread
      closeActivePlayer()
    }
  }

  private fun closeActivePlayer() {
    mainHandler.removeCallbacks(positionReporter)
    activeGeneration = 0L
    player?.stop()
    player?.clearMediaItems()
    surfaceView.visibility = View.GONE
    webView.setLayerType(View.LAYER_TYPE_HARDWARE, null)
    // Native video is the only feature that needs a transparent WebView.
    // Restore an opaque surface so WebView-owned video elements such as the
    // QR scanner preview composite in front of the application again.
    webView.setBackgroundColor(Color.BLACK)
  }

  private fun withGeneration(generation: Long, action: (ExoPlayer) -> Unit) {
    activity.runOnUiThread {
      val currentPlayer = player
      if (generation == activeGeneration && activeGeneration != 0L && currentPlayer != null) {
        action(currentPlayer)
      }
    }
  }

  private fun durationSeconds(currentPlayer: Player): Double =
    if (currentPlayer.duration == C.TIME_UNSET || currentPlayer.duration < 0) 0.0
    else currentPlayer.duration / 1000.0

  private fun emit(type: String, value: Any? = null) {
    val generation = activeGeneration
    if (generation == 0L && type != "error") return
    val detail = JSONObject()
      .put("generation", generation)
      .put("type", type)
      .put("value", value ?: JSONObject.NULL)
    val script =
      "window.dispatchEvent(new CustomEvent('localbooru-native-video',{detail:${detail}}));"
    webView.evaluateJavascript(script, null)
  }

  private fun emitVideoSize() {
    emit(
      "video-size",
      JSONObject()
        .put("sourceWidth", inputWidth)
        .put("sourceHeight", inputHeight)
        .put("inputMimeType", inputMimeType ?: JSONObject.NULL)
        .put("inputCodecs", inputCodecs ?: JSONObject.NULL)
        .put("inputFrameRate", inputFrameRate)
        .put("requestedEffectHeight", effectHeight)
        .put("effectOutputWidth", effectOutputWidth)
        .put("effectOutputHeight", effectOutputHeight)
        .put("reportedVideoWidth", reportedVideoWidth)
        .put("reportedVideoHeight", reportedVideoHeight)
        .put("surfaceWidth", surfaceView.width)
        .put("surfaceHeight", surfaceView.height)
        .put("decoder", decoderName ?: JSONObject.NULL)
        .put("decoderHardwareAccelerated", decoderHardwareAccelerated ?: JSONObject.NULL)
        .put("droppedFrames", droppedFrames)
    )
  }

  override fun onPlaybackStateChanged(playbackState: Int) {
    when (playbackState) {
      Player.STATE_BUFFERING -> emit("buffering", true)
      Player.STATE_READY -> {
        emit("buffering", false)
        emit("ready", JSONObject().put("duration", player?.let(::durationSeconds) ?: 0.0))
      }
      Player.STATE_ENDED -> emit("ended")
    }
  }

  override fun onIsPlayingChanged(isPlaying: Boolean) {
    emit("playing", isPlaying)
  }

  override fun onPlayerError(error: PlaybackException) {
    surfaceView.visibility = View.GONE
    webView.setLayerType(View.LAYER_TYPE_HARDWARE, null)
    emit("error", error.message ?: "Android playback failed")
  }

  override fun onVideoSizeChanged(videoSize: VideoSize) {
    reportedVideoWidth = videoSize.width
    reportedVideoHeight = videoSize.height
    Log.i(
      TAG,
      "videoSize generation=$activeGeneration reported=${reportedVideoWidth}x$reportedVideoHeight " +
        "effectOutput=${effectOutputWidth}x$effectOutputHeight requestedHeight=$effectHeight",
    )
    emitVideoSize()
  }

  override fun onRenderedFirstFrame() {
    if (surfaceView.width <= 1 || surfaceView.height <= 1) {
      Log.e(TAG, "Rejecting first frame for invalid surface ${surfaceView.width}x${surfaceView.height}")
      emit("error", "Native video rendered without a visible viewport")
      return
    }
    Log.i(
      TAG,
      "firstFrame generation=$activeGeneration input=${inputWidth}x$inputHeight " +
        "effectOutput=${effectOutputWidth}x$effectOutputHeight " +
        "reportedVideo=${reportedVideoWidth}x$reportedVideoHeight " +
        "surface=${surfaceView.width}x${surfaceView.height} " +
        "decoder=$decoderName hardware=$decoderHardwareAccelerated",
    )
    emit("first-frame")
  }

  override fun onVideoInputFormatChanged(
    eventTime: AnalyticsListener.EventTime,
    format: Format,
    decoderReuseEvaluation: DecoderReuseEvaluation?,
  ) {
    inputWidth = format.width.coerceAtLeast(0)
    inputHeight = format.height.coerceAtLeast(0)
    inputMimeType = format.sampleMimeType
    inputCodecs = format.codecs
    inputFrameRate = format.frameRate.coerceAtLeast(0f)
    applyVideoEffect(player ?: return)
    Log.i(TAG, "videoInput generation=$activeGeneration ${Format.toLogString(format)}")
    emitVideoSize()
  }

  override fun onVideoDecoderInitialized(
    eventTime: AnalyticsListener.EventTime,
    decoderName: String,
    initializedTimestampMs: Long,
    initializationDurationMs: Long,
  ) {
    this.decoderName = decoderName
    decoderHardwareAccelerated = isHardwareAcceleratedDecoder(decoderName)
    Log.i(
      TAG,
      "decoder generation=$activeGeneration name=$decoderName hardware=$decoderHardwareAccelerated " +
        "initializationMs=$initializationDurationMs",
    )
    emitVideoSize()
  }

  override fun onDroppedVideoFrames(
    eventTime: AnalyticsListener.EventTime,
    droppedFrames: Int,
    elapsedMs: Long,
  ) {
    this.droppedFrames += droppedFrames
    emitVideoSize()
  }

  override fun surfaceCreated(holder: SurfaceHolder) {
    Log.i(TAG, "surfaceCreated generation=$activeGeneration valid=${holder.surface.isValid}")
    emit("surface", JSONObject().put("available", true))
  }

  override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
    Log.i(TAG, "surfaceChanged generation=$activeGeneration size=${width}x$height format=$format")
    emitVideoSize()
  }

  override fun surfaceDestroyed(holder: SurfaceHolder) {
    Log.i(TAG, "surfaceDestroyed generation=$activeGeneration")
  }

  private fun isHardwareAcceleratedDecoder(name: String): Boolean? {
    return try {
      val codecInfo = MediaCodecList(MediaCodecList.ALL_CODECS).codecInfos
        .firstOrNull { !it.isEncoder && it.name == name }
        ?: return null
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        codecInfo.isHardwareAccelerated
      } else {
        val lowerName = codecInfo.name.lowercase()
        !lowerName.startsWith("omx.google.") &&
          !lowerName.startsWith("c2.android.") &&
          !lowerName.contains("software") &&
          !lowerName.contains("sw.")
      }
    } catch (error: Throwable) {
      Log.w(TAG, "Could not classify decoder $name", error)
      null
    }
  }

  fun destroy() {
    mainHandler.removeCallbacks(positionReporter)
    activeGeneration = 0L
    player?.removeListener(this)
    player?.removeAnalyticsListener(this)
    player?.clearVideoSurfaceView(surfaceView)
    player?.release()
    player = null
    surfaceView.holder.removeCallback(this)
  }

  companion object {
    private const val TAG = "LocalBooruNativeVideo"
    private const val POSITION_INTERVAL_MS = 250L
  }
}