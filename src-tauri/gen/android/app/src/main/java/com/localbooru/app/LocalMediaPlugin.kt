package com.localbooru.app

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import app.tauri.annotation.InvokeArg
import app.tauri.plugin.JSArray
import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.os.Build
import android.provider.MediaStore
import android.text.Editable
import android.text.TextWatcher
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.core.content.ContextCompat
import app.tauri.annotation.Command
import app.tauri.annotation.Permission
import app.tauri.annotation.PermissionCallback
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

@InvokeArg
class LocalMediaFileArgs {
  lateinit var path: String
  var output: String? = null
  var size: Int = 400
  var count: Int = 8
}

@TauriPlugin(permissions = [
  Permission(strings = ["android.permission.READ_EXTERNAL_STORAGE"], alias = "legacy"),
  Permission(strings = ["android.permission.READ_MEDIA_IMAGES"], alias = "images"),
  Permission(strings = ["android.permission.READ_MEDIA_VIDEO"], alias = "videos"),
  Permission(strings = ["android.permission.READ_MEDIA_AUDIO"], alias = "music"),
  Permission(strings = ["android.permission.READ_MEDIA_VISUAL_USER_SELECTED"], alias = "selected")
])
class LocalMediaPlugin(private val activity: Activity) : Plugin(activity) {
  private val picking = AtomicBoolean(false)
  private var requested = booleanArrayOf(true, true, true)
  private val types = arrayOf("Images", "Videos", "Music")

  private fun granted(permission: String): Boolean =
    ContextCompat.checkSelfPermission(activity, permission) == PackageManager.PERMISSION_GRANTED

  private fun finish(invoke: Invoke, directory: JSObject? = null) {
    picking.set(false)
    invoke.resolve(JSObject().put("directory", directory))
  }

  @Command
  fun pickDirectory(invoke: Invoke) {
    if (!picking.compareAndSet(false, true)) {
      invoke.reject("A media folder picker is already open.")
      return
    }
    activity.runOnUiThread {
      requested = booleanArrayOf(true, true, true)
      val dialog = AlertDialog.Builder(activity)
        .setTitle("Media to add")
        .setMultiChoiceItems(types, requested) { _, index, checked -> requested[index] = checked }
        .setNegativeButton("Cancel") { _, _ -> finish(invoke) }
        .setPositiveButton("Choose folder", null)
        .setOnCancelListener { finish(invoke) }
        .create()
      dialog.setOnShowListener {
        val next = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
        next.setOnClickListener {
          if (requested.none { it }) return@setOnClickListener
          dialog.dismiss()
          val aliases = if (Build.VERSION.SDK_INT < 33) arrayOf("legacy") else {
            val selected = mutableListOf<String>()
            if (requested[0]) selected.add("images")
            if (requested[1]) selected.add("videos")
            if (requested[2]) selected.add("music")
            if (Build.VERSION.SDK_INT >= 34 && (requested[0] || requested[1])) selected.add("selected")
            selected.toTypedArray()
          }
          requestPermissionForAliases(aliases, invoke, "mediaPermissionsResult")
        }
      }
      dialog.show()
    }
  }

  @PermissionCallback
  fun mediaPermissionsResult(invoke: Invoke) {
    val selectedPhotos = Build.VERSION.SDK_INT >= 34 && granted("android.permission.READ_MEDIA_VISUAL_USER_SELECTED")
    val available = if (Build.VERSION.SDK_INT < 33) {
      requested.map { it && granted(Manifest.permission.READ_EXTERNAL_STORAGE) }.toBooleanArray()
    } else booleanArrayOf(
      requested[0] && (granted("android.permission.READ_MEDIA_IMAGES") || selectedPhotos),
      requested[1] && (granted("android.permission.READ_MEDIA_VIDEO") || selectedPhotos),
      requested[2] && granted("android.permission.READ_MEDIA_AUDIO")
    )
    if (available.none { it }) {
      picking.set(false)
      invoke.reject("Media access was not granted. You can allow access in Android Settings and try again.")
      return
    }
    // Only MediaStore-authorized existing paths; no URI guessing or copying.
    val cancelled = AtomicBoolean(false)
    var loading: AlertDialog? = null
    activity.runOnUiThread {
      val layout = LinearLayout(activity).apply {
        orientation = LinearLayout.HORIZONTAL
        val padding = (24 * resources.displayMetrics.density).toInt()
        setPadding(padding, padding, padding, padding)
        addView(ProgressBar(activity))
        addView(TextView(activity).apply { text = "Finding accessible media folders…"; setPadding(padding, 0, 0, 0) })
      }
      loading = AlertDialog.Builder(activity).setView(layout)
        .setNegativeButton("Cancel") { _, _ -> if (cancelled.compareAndSet(false, true)) finish(invoke) }
        .setOnCancelListener { if (cancelled.compareAndSet(false, true)) finish(invoke) }.create()
      loading?.show()
    }
    Thread {
      try {
        val folders = sortedMapOf<String, Int>()
        val collections = arrayOf(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
          MediaStore.Video.Media.EXTERNAL_CONTENT_URI, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI)
        for (index in available.indices) {
          if (cancelled.get()) return@Thread
          if (!available[index]) continue
          activity.contentResolver.query(collections[index], arrayOf(MediaStore.MediaColumns.DATA), null, null, null)?.use { cursor ->
            val column = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA)
            while (!cancelled.get() && cursor.moveToNext()) {
              val path = cursor.getString(column) ?: continue
              val file = File(path)
              if (!file.isAbsolute || !file.canRead()) continue
              val parent = file.parentFile ?: continue
              if (parent.isDirectory) {
                var folder: File? = parent
                while (folder != null && (folder.path.startsWith("/storage/") || folder.path.startsWith("/sdcard/"))) {
                  if (folder.path == "/storage/emulated") break
                  folders[folder.path] = (folders[folder.path] ?: 0) + 1
                  folder = folder.parentFile
                }
              }
            }
          }
        }
        activity.runOnUiThread {
          loading?.dismiss()
          if (cancelled.get()) return@runOnUiThread
          if (folders.isEmpty()) {
            picking.set(false)
            invoke.reject("No accessible local media folders were found for these media types.")
          } else showFolders(invoke, folders, available)
        }
      } catch (error: Exception) {
        activity.runOnUiThread {
          loading?.dismiss()
          if (cancelled.get()) return@runOnUiThread
          picking.set(false)
          invoke.reject(if (error is SecurityException)
            "Android media access changed. Choose the folder again to refresh access."
            else "Android could not list local media folders. Please try again.")
        }
      }
    }.start()
  }

  private fun showFolders(invoke: Invoke, folders: Map<String, Int>, available: BooleanArray) {
    val allPaths = folders.keys.toList()
    var paths = allPaths
    val adapter = ArrayAdapter(activity, android.R.layout.simple_list_item_1,
      paths.map { "${File(it).name} · ${folders[it]} files\n$it" }.toMutableList())
    val filter = EditText(activity).apply { hint = "Search folders"; isSingleLine = true }
    val list = ListView(activity).apply { this.adapter = adapter }
    val layout = LinearLayout(activity).apply {
      orientation = LinearLayout.VERTICAL
      val padding = (16 * resources.displayMetrics.density).toInt()
      setPadding(padding, 0, padding, 0)
      addView(filter)
      addView(list, LinearLayout.LayoutParams(-1, (360 * resources.displayMetrics.density).toInt()))
    }
    val dialog = AlertDialog.Builder(activity).setTitle("Add local media folder")
      .setView(layout).setNegativeButton("Cancel") { _, _ -> finish(invoke) }
      .setOnCancelListener { finish(invoke) }.create()
    filter.addTextChangedListener(object : TextWatcher {
      override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
      override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
        paths = allPaths.filter { it.contains(s?.toString() ?: "", ignoreCase = true) }
        adapter.clear()
        adapter.addAll(paths.map { "${File(it).name} · ${folders[it]} files\n$it" })
        adapter.notifyDataSetChanged()
      }
      override fun afterTextChanged(s: Editable?) {}
    })
    list.setOnItemClickListener { _, _, position, _ ->
      val path = paths[position]
      dialog.dismiss()
      finish(invoke, JSObject().put("path", path).put("name", File(path).name)
        .put("show_images", available[0]).put("show_videos", available[1]).put("show_music", available[2]))
    }
    dialog.show()
  }

  private fun withVideo(invoke: Invoke, work: (LocalMediaFileArgs, MediaMetadataRetriever) -> JSObject) {
    val args = invoke.parseArgs(LocalMediaFileArgs::class.java)
    Thread {
      val retriever = MediaMetadataRetriever()
      try {
        retriever.setDataSource(args.path)
        invoke.resolve(work(args, retriever))
      } catch (_: Exception) {
        invoke.reject("This video could not be read by Android. Check access or use a supported video format.")
      } finally { runCatching { retriever.release() } }
    }.start()
  }

  @Command
  fun mediaInfo(invoke: Invoke) = withVideo(invoke) { _, retriever ->
    var width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
    var height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
    val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
    if (rotation == 90 || rotation == 270) { val temporary = width; width = height; height = temporary }
    val duration = (retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toDoubleOrNull() ?: 0.0) / 1000.0
    JSObject().put("width", width).put("height", height).put("duration", duration)
  }

  private fun thumbnailFrame(retriever: MediaMetadataRetriever, timeUs: Long, size: Int): Bitmap? {
    if (Build.VERSION.SDK_INT >= 27) {
      val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: size
      val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: size
      val scale = minOf(1.0, size.toDouble() / maxOf(1, width, height))
      return retriever.getScaledFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
        maxOf(1, (width * scale).toInt()), maxOf(1, (height * scale).toInt()))
    }
    val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: return null
    val scale = minOf(1.0, size.toDouble() / maxOf(frame.width, frame.height))
    val resized = Bitmap.createScaledBitmap(frame, maxOf(1, (frame.width * scale).toInt()), maxOf(1, (frame.height * scale).toInt()), true)
    if (resized !== frame) frame.recycle()
    return resized
  }

  private fun cacheOutput(path: String): File {
    val file = File(path).canonicalFile
    require(file.path.startsWith(File(activity.applicationInfo.dataDir).canonicalPath + File.separator)) { "Output must stay in application storage" }
    require(generateSequence(file.parentFile) { it.parentFile }.any { it.name == "thumbnails" || it.name == "previews" }) { "Output must be a media cache" }
    file.parentFile?.mkdirs()
    return file
  }

  private fun saveFrame(frame: Bitmap, destination: File): Boolean {
    try {
      return destination.outputStream().use { output ->
        val format = if (Build.VERSION.SDK_INT >= 30) Bitmap.CompressFormat.WEBP_LOSSY else Bitmap.CompressFormat.WEBP
        frame.compress(format, 80, output)
      }
    } finally { frame.recycle() }
  }

  @Command
  fun videoThumbnail(invoke: Invoke) = withVideo(invoke) { args, retriever ->
    val output = cacheOutput(requireNotNull(args.output))
    val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
    val frame = thumbnailFrame(retriever, duration * 500, args.size.coerceIn(64, 2048))
    JSObject().put("saved", frame != null && saveFrame(frame, output))
  }

  @Command
  fun videoPreviews(invoke: Invoke) = withVideo(invoke) { args, retriever ->
    require(args.count in 1..8)
    val directory = cacheOutput(requireNotNull(args.output) + "/frame_0.webp").parentFile!!
    val durationUs = (retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L) * 1000
    val files = JSArray()
    for (index in 0 until args.count) {
      val timeUs = (durationUs * (0.05 + 0.9 * index / args.count)).toLong()
      val frame = thumbnailFrame(retriever, timeUs, args.size.coerceIn(64, 2048)) ?: break
      val output = File(directory, "frame_${index}.webp")
      if (!saveFrame(frame, output)) break
      files.put(output.path)
    }
    JSObject().put("files", files)
  }
}
