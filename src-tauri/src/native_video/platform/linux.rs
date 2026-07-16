use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::os::fd::RawFd;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use gdk::prelude::{GdkContextExt, WindowExtManual};
use gtk::prelude::*;
use tauri::{Emitter, Manager};

use crate::native_video::commands::NativeVideoState;
use crate::native_video::display_geometry::{
    compute_display_geometry, DisplayGeometry, DisplayGeometryInput, DisplayMode,
};
use crate::native_video::protocol::{NativeVideoCommand, NativeVideoEvent, TrackInfo};
use crate::native_video::shm_surface_consumer::{ShmFrameView, ShmSurfaceConsumer};
use crate::native_video::surface_channel::{ReceivedSurfaceMessage, SurfaceChannelMessage};
use crate::native_video::surface_protocol::{
    SurfaceColorRange, SurfaceColorSpace, SurfaceDescriptor, SurfaceFrameRelease, DRM_FORMAT_YUV420,
};

thread_local! {
    static VIEWPORT: RefCell<Option<NativeViewport>> = const { RefCell::new(None) };
    static SHM_CONSUMER: RefCell<ShmSurfaceConsumer> = RefCell::new(ShmSurfaceConsumer::default());
    static DISPLAY_MODE: Cell<DisplayMode> = const { Cell::new(DisplayMode::Fit) };
}

static DMA_BUF_SHADER_FRAMES: AtomicU64 = AtomicU64::new(0);
static DMA_BUF_ACCEPTED_FRAMES: AtomicU64 = AtomicU64::new(0);
static DMA_BUF_PRESENTED_FRAMES: AtomicU64 = AtomicU64::new(0);
static DMA_BUF_DISCARDED_FRAMES: AtomicU64 = AtomicU64::new(0);
static DMA_BUF_GENERATION: AtomicU64 = AtomicU64::new(0);
static SHM_ACCEPTED_FRAMES: AtomicU64 = AtomicU64::new(0);
static SHM_SUBMITTED_FRAMES: AtomicU64 = AtomicU64::new(0);
static SHM_PRESENTED_FRAMES: AtomicU64 = AtomicU64::new(0);
static SHM_DISCARDED_FRAMES: AtomicU64 = AtomicU64::new(0);
static SHM_QUEUE_DEPTH: AtomicU64 = AtomicU64::new(0);
static SHM_QUEUE_LATENCY_US: AtomicU64 = AtomicU64::new(0);
static SHM_GENERATION: AtomicU64 = AtomicU64::new(0);
static VIEWPORT_ATTACHED: AtomicBool = AtomicBool::new(false);
static HUD_SCREENSHOT_CAPTURED: AtomicBool = AtomicBool::new(false);
const MAX_PENDING_GL_FRAMES: usize = 3;

fn svg_image(svg: &[u8]) -> Result<gtk::Image, String> {
    let loader = gdk_pixbuf::PixbufLoader::with_type("svg")
        .map_err(|error| format!("failed to create SVG loader: {error}"))?;
    loader
        .write(svg)
        .map_err(|error| format!("failed to decode native HUD icon: {error}"))?;
    loader
        .close()
        .map_err(|error| format!("failed to finish native HUD icon: {error}"))?;
    let pixbuf = loader
        .pixbuf()
        .ok_or_else(|| "native HUD SVG produced no pixels".to_string())?
        .scale_simple(18, 18, gdk_pixbuf::InterpType::Bilinear)
        .ok_or_else(|| "failed to scale native HUD icon".to_string())?;
    Ok(gtk::Image::from_pixbuf(Some(&pixbuf)))
}

fn icon_button(svg: &[u8], tooltip: &str) -> Result<gtk::Button, String> {
    let button = gtk::Button::new();
    button.set_image(Some(&svg_image(svg)?));
    button.set_always_show_image(true);
    button.set_tooltip_text(Some(tooltip));
    Ok(button)
}

fn set_button_icon(button: &gtk::Button, svg: &[u8]) {
    match svg_image(svg) {
        Ok(image) => button.set_image(Some(&image)),
        Err(error) => log::warn!("[NativeVideo] Failed to update HUD icon: {error}"),
    }
}

fn maybe_capture_hud_screenshot(viewport: &NativeViewport) {
    let Some(path) = std::env::var_os("LOCALBOORU_NATIVE_HUD_SCREENSHOT") else {
        return;
    };
    if HUD_SCREENSHOT_CAPTURED.swap(true, Ordering::AcqRel) {
        return;
    }
    let Some(window) = viewport.host_window.window() else {
        HUD_SCREENSHOT_CAPTURED.store(false, Ordering::Release);
        return;
    };
    let width = viewport.host_window.allocated_width();
    let height = viewport.host_window.allocated_height();
    let Some(pixbuf) = window.pixbuf(0, 0, width, height) else {
        HUD_SCREENSHOT_CAPTURED.store(false, Ordering::Release);
        return;
    };
    if let Err(error) = pixbuf.savev(path.to_string_lossy().as_ref(), "png", &[]) {
        log::warn!("[NativeVideo] Failed to capture native HUD screenshot: {error}");
        HUD_SCREENSHOT_CAPTURED.store(false, Ordering::Release);
    }
}

fn push_bounded<T>(queue: &mut VecDeque<T>, item: T, limit: usize) -> Option<T> {
    if limit == 0 {
        return Some(item);
    }
    let dropped = (queue.len() >= limit).then(|| queue.pop_front()).flatten();
    queue.push_back(item);
    dropped
}

struct NativeViewport {
    host_window: gtk::Window,
    webview: webkit2gtk::WebView,
    bounds: Rc<Cell<ViewportBounds>>,
    desired_visible: Rc<Cell<bool>>,
    display_mode: Rc<Cell<DisplayMode>>,
    event_box: gtk::EventBox,
    rgba_gl_area: gtk::GLArea,
    _gl_area: gtk::GLArea,
    content_stack: gtk::Stack,
    hud: gtk::Box,
    play_pause: gtk::Button,
    timeline: gtk::Scale,
    time_label: gtk::Label,
    status_label: gtk::Label,
    diagnostics_label: gtk::Label,
    mute: gtk::ToggleButton,
    volume: gtk::Scale,
    speed: gtk::ComboBoxText,
    speed_badge: gtk::Label,
    interpolation: gtk::ComboBoxText,
    audio_tracks: gtk::ComboBoxText,
    subtitle_tracks: gtk::ComboBoxText,
    subtitle_delay: gtk::SpinButton,
    whisper_subtitles: gtk::Button,
    subtitle_label: gtk::Label,
    current_position: Rc<Cell<f64>>,
    paused_state: Rc<Cell<bool>>,
    muted_state: Rc<Cell<bool>>,
    speed_state: Rc<Cell<f64>>,
    updating_controls: Rc<Cell<bool>>,
    scrubbing: Rc<Cell<bool>>,
    _touch_video_gesture: gtk::GestureDrag,
    _touch_timeline_gesture: gtk::GestureDrag,
    _pending_draw: Rc<RefCell<Option<PendingDraw>>>,
    pending_rgba: Rc<RefCell<VecDeque<PendingRgba>>>,
    pending_dmabuf: Rc<RefCell<Option<PendingDmabuf>>>,
    in_flight_dmabuf: Rc<RefCell<Option<InFlightDmabuf>>>,
    clear_external_cache: Rc<Cell<bool>>,
}

struct PendingDraw {
    pixbuf: gdk_pixbuf::Pixbuf,
    on_presented: Option<Box<dyn FnOnce()>>,
    on_discarded: Option<Box<dyn FnOnce()>>,
}

struct PendingDmabuf {
    descriptor: SurfaceDescriptor,
    object_fds: Vec<RawFd>,
    completion: mpsc::SyncSender<Result<(), String>>,
}

enum PendingPixels {
    Owned(Vec<u8>),
    Shared(ShmFrameView),
}

impl PendingPixels {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Shared(frame) => frame.bytes(),
        }
    }
}

struct PendingRgba {
    accepted_at: Instant,
    buffer_id: u32,
    width: i32,
    height: i32,
    sample_aspect_ratio: f64,
    rotation_degrees: i32,
    color_space: Option<SurfaceColorSpace>,
    color_range: Option<SurfaceColorRange>,
    stride: i32,
    yuv420p: bool,
    strides: [i32; 3],
    offsets: [usize; 3],
    pixels: PendingPixels,
    on_presented: Option<Box<dyn FnOnce()>>,
    on_discarded: Option<Box<dyn FnOnce()>>,
}

struct InFlightRgba {
    fence: crate::native_video::gtk_gl_context::GpuFence,
    on_presented: Option<Box<dyn FnOnce()>>,
    on_discarded: Option<Box<dyn FnOnce()>>,
}

struct InFlightDmabuf {
    fence: crate::native_video::gtk_gl_context::GpuFence,
    completion: mpsc::SyncSender<Result<(), String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportBounds {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl ViewportBounds {
    pub fn normalized(self) -> Self {
        Self {
            x: self.x.max(0),
            y: self.y.max(0),
            width: self.width.max(1),
            height: self.height.max(1),
        }
    }
}

fn frame_geometry(
    source_width: i32,
    source_height: i32,
    sample_aspect_ratio: f64,
    rotation_degrees: i32,
    viewport_width: i32,
    viewport_height: i32,
    scale_factor: f64,
    mode: DisplayMode,
) -> Result<DisplayGeometry, String> {
    compute_display_geometry(DisplayGeometryInput {
        coded_width: u32::try_from(source_width)
            .map_err(|_| "source width does not fit display geometry".to_string())?,
        coded_height: u32::try_from(source_height)
            .map_err(|_| "source height does not fit display geometry".to_string())?,
        sample_aspect_ratio,
        rotation_degrees,
        viewport_width: f64::from(viewport_width.max(1)),
        viewport_height: f64::from(viewport_height.max(1)),
        scale_factor,
        mode,
        // The popup itself is already positioned below the desktop title bar.
        title_bar_safe_inset: 0.0,
    })
}

fn position_host(host: &gtk::Window, bounds: ViewportBounds) {
    if let Some(window) = host.window() {
        window.move_to_rect(
            &gdk::Rectangle::new(bounds.x, bounds.y, 1, 1),
            gdk::Gravity::NorthWest,
            gdk::Gravity::NorthWest,
            gdk::AnchorHints::empty(),
            0,
            0,
        );
    }
}

fn map_host_for_preroll(viewport: &NativeViewport) {
    if viewport.host_window.is_visible() {
        return;
    }
    // GTK does not dispatch GLArea frame-clock renders for an unmapped popup.
    // Map it fully transparent, render the first frame, then let the
    // draw-completion callback reveal it through set_visible().
    viewport.host_window.set_opacity(0.0);
    // Wayland fixes a popup's anchor when it is mapped; set the player
    // rectangle before show_all() so it cannot map over the title bar.
    position_host(&viewport.host_window, viewport.bounds.get());
    viewport.host_window.show_all();
    viewport.event_box.show();
}

fn bottom_workarea_inset(window_y: i32, height: i32, workarea: gdk::Rectangle) -> i32 {
    (window_y
        .saturating_add(height)
        .saturating_sub(workarea.y().saturating_add(workarea.height())))
    .max(0)
}

fn update_hud_workarea_inset(viewport: &NativeViewport) {
    let Some(window) = viewport.host_window.window() else {
        return;
    };
    let (has_origin, _, window_y) = window.origin();
    if has_origin == 0 {
        return;
    }
    let Some(monitor) = window.display().monitor_at_window(&window) else {
        return;
    };
    let inset = bottom_workarea_inset(window_y, viewport.bounds.get().height, monitor.workarea());
    viewport.hud.set_margin_bottom(12_i32.saturating_add(inset));
    viewport
        .subtitle_label
        .set_margin_bottom(92_i32.saturating_add(inset));
}

fn send_control(app: &tauri::AppHandle, command: NativeVideoCommand) {
    if let Err(error) = app
        .state::<NativeVideoState>()
        .send_runtime_control(command)
    {
        log::warn!("[NativeVideo] HUD control ignored: {error}");
    }
}

#[derive(Clone, Copy)]
enum SemanticAction {
    Previous,
    Next,
    Close,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PointerZone {
    Backward,
    Center,
    Forward,
}

fn pointer_zone(x: f64, width: f64) -> PointerZone {
    let fraction = x / width.max(1.0);
    if fraction < 0.33 {
        PointerZone::Backward
    } else if fraction > 0.67 {
        PointerZone::Forward
    } else {
        PointerZone::Center
    }
}

fn emit_semantic_event(app: &tauri::AppHandle, action: SemanticAction) {
    let generation = app.state::<NativeVideoState>().current_generation();
    let event = match action {
        SemanticAction::Previous => NativeVideoEvent::NavigatePrevious { generation },
        SemanticAction::Next => NativeVideoEvent::NavigateNext { generation },
        SemanticAction::Close => NativeVideoEvent::CloseRequested { generation },
    };
    let _ = app.emit("native-video-event", event);
}

fn format_time(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    if seconds >= 3600 {
        format!(
            "{}:{:02}:{:02}",
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        )
    } else {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    }
}

fn flash_seek_indicator(label: &gtk::Label, text: &str) {
    label.set_text(text);
    label.show();
    let expected = text.to_owned();
    let label = label.downgrade();
    gtk::glib::timeout_add_local_once(Duration::from_millis(600), move || {
        if let Some(label) = label.upgrade() {
            if label.text().as_str() == expected {
                label.hide();
            }
        }
    });
}

pub fn attach(window: &tauri::WebviewWindow) -> Result<(), String> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let app = window.app_handle().clone();
    window
        .with_webview(move |platform_webview| {
            let result = attach_to_webview(platform_webview.inner(), app);
            let _ = sender.send(result);
        })
        .map_err(|error| format!("failed to access native WebView: {error}"))?;
    receiver
        .recv_timeout(Duration::from_secs(2))
        .map_err(|error| format!("native viewport attachment timed out: {error}"))?
}

pub fn is_attached() -> bool {
    VIEWPORT_ATTACHED.load(Ordering::Acquire)
}

pub fn queue_dmabuf_render(
    descriptor: SurfaceDescriptor,
    object_fds: Vec<RawFd>,
    completion: mpsc::SyncSender<Result<(), String>>,
) -> Result<(), String> {
    if DMA_BUF_GENERATION.swap(descriptor.generation, Ordering::Relaxed) != descriptor.generation {
        DMA_BUF_SHADER_FRAMES.store(0, Ordering::Relaxed);
        DMA_BUF_ACCEPTED_FRAMES.store(0, Ordering::Relaxed);
        DMA_BUF_PRESENTED_FRAMES.store(0, Ordering::Relaxed);
        DMA_BUF_DISCARDED_FRAMES.store(0, Ordering::Relaxed);
    }
    DMA_BUF_ACCEPTED_FRAMES.fetch_add(1, Ordering::Relaxed);
    VIEWPORT.with(|slot| {
        let viewport = slot.borrow();
        let viewport = viewport
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        if let Some(previous) = viewport.pending_dmabuf.borrow_mut().replace(PendingDmabuf {
            descriptor,
            object_fds,
            completion,
        }) {
            DMA_BUF_DISCARDED_FRAMES.fetch_add(1, Ordering::Relaxed);
            let _ = previous.completion.send(Err(
                "DMA-BUF frame was replaced before GTK render".to_string()
            ));
        }
        if !viewport.host_window.is_visible() {
            viewport.host_window.set_opacity(0.0);
            viewport.event_box.show();
            viewport.host_window.show();
            position_host(&viewport.host_window, viewport.bounds.get());
        }
        viewport.content_stack.set_visible_child_name("dma-buf");
        viewport._gl_area.queue_render();
        Ok(())
    })
}

fn attach_to_webview(webview: webkit2gtk::WebView, app: tauri::AppHandle) -> Result<(), String> {
    if VIEWPORT.with(|slot| slot.borrow().is_some()) {
        return Ok(());
    }
    let owner = webview
        .toplevel()
        .ok_or_else(|| "Tauri WebView has no GtkWindow toplevel".to_string())?;
    let owner_type = owner.type_().name().to_string();
    let owner_window = owner
        .downcast::<gtk::Window>()
        .map_err(|_| format!("Tauri WebView toplevel is {owner_type}, not GtkWindow"))?;
    let host_window = gtk::Window::new(gtk::WindowType::Popup);
    host_window.set_transient_for(Some(&owner_window));
    host_window.set_attached_to(Some(&webview));
    host_window.set_decorated(false);
    host_window.set_resizable(false);
    host_window.set_skip_taskbar_hint(true);
    host_window.set_skip_pager_hint(true);
    host_window.set_accept_focus(false);
    host_window.set_focus_on_map(false);
    let bounds = Rc::new(Cell::new(ViewportBounds {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    }));
    let desired_visible = Rc::new(Cell::new(false));
    let display_mode = Rc::new(Cell::new(DISPLAY_MODE.with(Cell::get)));
    let container = gtk::Fixed::new();
    container.set_hexpand(true);
    container.set_vexpand(true);
    host_window.add(&container);

    let event_box = gtk::EventBox::new();
    event_box.set_visible_window(true);
    event_box.set_above_child(true);
    event_box.set_size_request(1, 1);
    event_box.hide();
    container.put(&event_box, 0, 0);

    let drawing_area = gtk::DrawingArea::new();
    drawing_area.set_hexpand(true);
    drawing_area.set_vexpand(true);
    let pending_draw = Rc::new(RefCell::new(None::<PendingDraw>));
    let draw_state = Rc::clone(&pending_draw);
    let draw_display_mode = Rc::clone(&display_mode);
    drawing_area.connect_draw(move |widget, context| {
        let mut callback = None;
        if let Some(frame) = draw_state.borrow_mut().as_mut() {
            let Ok(geometry) = frame_geometry(
                frame.pixbuf.width(),
                frame.pixbuf.height(),
                1.0,
                0,
                widget.allocated_width().max(1),
                widget.allocated_height().max(1),
                f64::from(widget.scale_factor()),
                draw_display_mode.get(),
            ) else {
                return gtk::glib::Propagation::Proceed;
            };
            let crop_width = f64::from(geometry.uv_crop.right - geometry.uv_crop.left)
                * f64::from(frame.pixbuf.width());
            let crop_height = f64::from(geometry.uv_crop.bottom - geometry.uv_crop.top)
                * f64::from(frame.pixbuf.height());
            let scale_x = geometry.content_rect.width / crop_width;
            let scale_y = geometry.content_rect.height / crop_height;
            context.save().ok();
            context.rectangle(
                geometry.content_rect.x,
                geometry.content_rect.y,
                geometry.content_rect.width,
                geometry.content_rect.height,
            );
            context.clip();
            context.translate(
                geometry.content_rect.x
                    - f64::from(geometry.uv_crop.left) * f64::from(frame.pixbuf.width()) * scale_x,
                geometry.content_rect.y
                    - f64::from(geometry.uv_crop.top) * f64::from(frame.pixbuf.height()) * scale_y,
            );
            context.scale(scale_x, scale_y);
            context.set_source_pixbuf(&frame.pixbuf, 0.0, 0.0);
            if context.paint().is_ok() {
                callback = frame.on_presented.take();
                frame.on_discarded.take();
            }
            context.restore().ok();
        }
        if let Some(callback) = callback {
            callback();
        }
        gtk::glib::Propagation::Proceed
    });
    let pending_rgba = Rc::new(RefCell::new(VecDeque::<PendingRgba>::new()));
    let rgba_render_state = Rc::clone(&pending_rgba);
    let rgba_display_mode = Rc::clone(&display_mode);
    let rgba_gl_area = gtk::GLArea::builder()
        // GTK may coalesce explicit queue_render() requests that arrive just
        // after a frame-clock phase, which halves a 60 fps surface stream on a
        // 75 Hz desktop. Auto-render remains display-clock driven (not a busy
        // loop) and lets the bounded queue supply one fresh frame per tick.
        .auto_render(true)
        .has_alpha(false)
        .use_es(true)
        .hexpand(true)
        .vexpand(true)
        .build();
    rgba_gl_area.set_required_version(2, 0);
    rgba_gl_area.connect_render(move |area, _| {
        let next_frame = {
            let mut queue = rgba_render_state.borrow_mut();
            let frame = queue.pop_front();
            SHM_QUEUE_DEPTH.store(queue.len() as u64, Ordering::Relaxed);
            frame
        };
        if let Some(mut frame) = next_frame {
            SHM_QUEUE_LATENCY_US.store(
                frame
                    .accepted_at
                    .elapsed()
                    .as_micros()
                    .min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
            if frame.yuv420p {
                let submitted = SHM_SUBMITTED_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                if submitted == 1 || submitted % 60 == 0 {
                    log::info!("[NativeVideo] GTK submitted SHM frame {submitted}");
                }
            }
            let geometry = match frame_geometry(
                frame.width,
                frame.height,
                frame.sample_aspect_ratio,
                frame.rotation_degrees,
                area.allocated_width().max(1),
                area.allocated_height().max(1),
                f64::from(area.scale_factor()),
                rgba_display_mode.get(),
            ) {
                Ok(geometry) => geometry,
                Err(error) => {
                    log::error!("[NativeVideo] invalid shared-memory display geometry: {error}");
                    if let Some(callback) = frame.on_discarded.take() {
                        callback();
                    }
                    return gtk::glib::Propagation::Stop;
                }
            };
            let viewport = geometry.physical_viewport;
            let uv_crop = [
                geometry.uv_crop.left,
                geometry.uv_crop.top,
                geometry.uv_crop.right,
                geometry.uv_crop.bottom,
            ];
            let upload = if frame.yuv420p {
                crate::native_video::gtk_gl_context::upload_yuv420_frame(
                    frame.buffer_id,
                    frame.pixels.bytes(),
                    frame.width,
                    frame.height,
                    frame.strides,
                    frame.offsets,
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    uv_crop,
                    geometry.rotation_quadrants,
                    frame.color_space,
                    frame.color_range,
                )
            } else {
                crate::native_video::gtk_gl_context::upload_rgba_frame(
                    frame.pixels.bytes(),
                    frame.width,
                    frame.height,
                    frame.stride,
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    uv_crop,
                    geometry.rotation_quadrants,
                )
            };
            match upload {
                Ok(fence) => {
                    let in_flight = Rc::new(RefCell::new(Some(InFlightRgba {
                        fence,
                        on_presented: frame.on_presented.take(),
                        on_discarded: frame.on_discarded.take(),
                    })));
                    let poll_area = area.clone();
                    let poll_state = Rc::clone(&in_flight);
                    gtk::glib::timeout_add_local(Duration::from_millis(1), move || {
                        poll_area.make_current();
                        let status = poll_state
                            .borrow()
                            .as_ref()
                            .map(|frame| frame.fence.is_signaled());
                        match status {
                            Some(Ok(true)) => {
                                if let Some(mut frame) = poll_state.borrow_mut().take() {
                                    if let Some(callback) = frame.on_presented.take() {
                                        callback();
                                    }
                                    frame.on_discarded.take();
                                }
                                gtk::glib::ControlFlow::Break
                            }
                            Some(Ok(false)) => gtk::glib::ControlFlow::Continue,
                            Some(Err(error)) => {
                                log::error!("[NativeVideo] RGBA GPU fence failed: {error}");
                                if let Some(mut frame) = poll_state.borrow_mut().take() {
                                    if let Some(callback) = frame.on_discarded.take() {
                                        callback();
                                    }
                                }
                                gtk::glib::ControlFlow::Break
                            }
                            None => gtk::glib::ControlFlow::Break,
                        }
                    });
                }
                Err(error) => {
                    log::error!("[NativeVideo] RGBA GL upload failed: {error}");
                    if let Some(callback) = frame.on_discarded.take() {
                        callback();
                    }
                }
            }
        } else if let Err(error) = crate::native_video::gtk_gl_context::clear_probe() {
            log::error!("[NativeVideo] RGBA GLArea render failed: {error}");
        }
        if !rgba_render_state.borrow().is_empty() {
            area.queue_render();
        }
        gtk::glib::Propagation::Stop
    });

    let pending_dmabuf = Rc::new(RefCell::new(None::<PendingDmabuf>));
    let render_state = Rc::clone(&pending_dmabuf);
    let dmabuf_display_mode = Rc::clone(&display_mode);
    let in_flight_dmabuf = Rc::new(RefCell::new(None::<InFlightDmabuf>));
    let render_in_flight = Rc::clone(&in_flight_dmabuf);
    let clear_external_cache = Rc::new(Cell::new(false));
    let render_clear_external_cache = Rc::clone(&clear_external_cache);
    let gl_area = gtk::GLArea::builder()
        .auto_render(false)
        .has_alpha(false)
        .use_es(true)
        .hexpand(true)
        .vexpand(true)
        .build();
    gl_area.set_required_version(2, 0);
    gl_area.connect_realize(|area| {
        area.make_current();
        if let Some(error) = area.error() {
            log::error!("[NativeVideo] GTK GLArea realization failed: {error}");
            return;
        }
        match crate::native_video::gtk_gl_context::current_capabilities() {
            Ok(capabilities) => log::info!(
                "[NativeVideo] GTK GLArea ready EGL={} {} GL={} dmabuf_import={} dmabuf_modifiers={}",
                capabilities.egl_vendor,
                capabilities.egl_version,
                capabilities.gl_version,
                capabilities.dmabuf_import,
                capabilities.dmabuf_modifiers
            ),
            Err(error) => {
                log::error!("[NativeVideo] GTK GLArea capability probe failed: {error}")
            }
        }
    });
    gl_area.connect_render(move |area, _| {
        if render_in_flight.borrow().is_some() {
            return gtk::glib::Propagation::Stop;
        }
        if let Some(frame) = render_state.borrow_mut().take() {
            let destination_width = area.allocated_width().max(1);
            let destination_height = area.allocated_height().max(1);
            let geometry = match frame_geometry(
                frame.descriptor.width as i32,
                frame.descriptor.height as i32,
                frame.descriptor.sample_aspect_ratio,
                frame.descriptor.rotation_degrees,
                destination_width,
                destination_height,
                f64::from(area.scale_factor()),
                dmabuf_display_mode.get(),
            ) {
                Ok(geometry) => geometry,
                Err(error) => {
                    DMA_BUF_DISCARDED_FRAMES.fetch_add(1, Ordering::Relaxed);
                    let _ = frame.completion.send(Err(error));
                    return gtk::glib::Propagation::Stop;
                }
            };
            let viewport = geometry.physical_viewport;
            let result = crate::native_video::gtk_gl_context::import_nv12_frame(
                &frame.descriptor,
                &frame.object_fds,
                viewport.x,
                viewport.y,
                viewport.width,
                viewport.height,
                [
                    geometry.uv_crop.left,
                    geometry.uv_crop.top,
                    geometry.uv_crop.right,
                    geometry.uv_crop.bottom,
                ],
                geometry.rotation_quadrants,
            );
            match result {
                Ok(fence) => {
                    let frame_count =
                        DMA_BUF_SHADER_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                    if frame_count == 1 || frame_count % 120 == 0 {
                        log::info!(
                            "[NativeVideo] GTK submitted DMA-BUF frame {frame_count} with asynchronous GPU fence"
                        );
                    }
                    *render_in_flight.borrow_mut() = Some(InFlightDmabuf {
                        fence,
                        completion: frame.completion,
                    });
                    let poll_area = area.clone();
                    let poll_state = Rc::clone(&render_in_flight);
                    let poll_pending = Rc::clone(&render_state);
                    let poll_clear_external_cache = Rc::clone(&render_clear_external_cache);
                    // Do not busy-spin the GTK main loop while waiting for the
                    // asynchronous GPU fence. A 1 ms timer keeps completion
                    // latency below a frame while allowing the UI thread to sleep.
                    gtk::glib::timeout_add_local(Duration::from_millis(1), move || {
                        poll_area.make_current();
                        if let Some(error) = poll_area.error() {
                            if let Some(frame) = poll_state.borrow_mut().take() {
                                DMA_BUF_DISCARDED_FRAMES.fetch_add(1, Ordering::Relaxed);
                                let _ = frame.completion.send(Err(format!(
                                    "GTK GLArea became unavailable while waiting for GPU completion: {error}"
                                )));
                            }
                            if poll_pending.borrow().is_some() {
                                poll_area.queue_render();
                            }
                            return gtk::glib::ControlFlow::Break;
                        }
                        let fence_status = poll_state
                            .borrow()
                            .as_ref()
                            .map(|frame| frame.fence.is_signaled());
                        match fence_status {
                            Some(Ok(true)) => {
                                if let Some(frame) = poll_state.borrow_mut().take() {
                                    DMA_BUF_PRESENTED_FRAMES.fetch_add(1, Ordering::Relaxed);
                                    let _ = frame.completion.send(Ok(()));
                                }
                                if poll_clear_external_cache.replace(false) {
                                    crate::native_video::gtk_gl_context::clear_external_image_cache();
                                }
                                if poll_pending.borrow().is_some() {
                                    poll_area.queue_render();
                                }
                                gtk::glib::ControlFlow::Break
                            }
                            Some(Ok(false)) => gtk::glib::ControlFlow::Continue,
                            Some(Err(error)) => {
                                if let Some(frame) = poll_state.borrow_mut().take() {
                                    DMA_BUF_DISCARDED_FRAMES.fetch_add(1, Ordering::Relaxed);
                                    let _ = frame.completion.send(Err(error));
                                }
                                if poll_clear_external_cache.replace(false) {
                                    crate::native_video::gtk_gl_context::clear_external_image_cache();
                                }
                                if poll_pending.borrow().is_some() {
                                    poll_area.queue_render();
                                }
                                gtk::glib::ControlFlow::Break
                            }
                            None => gtk::glib::ControlFlow::Break,
                        }
                    });
                }
                Err(error) => {
                    DMA_BUF_DISCARDED_FRAMES.fetch_add(1, Ordering::Relaxed);
                    let _ = frame.completion.send(Err(error));
                }
            }
        } else if let Err(error) = crate::native_video::gtk_gl_context::clear_probe() {
            log::error!("[NativeVideo] GTK GLArea render failed: {error}");
        } else if std::env::var_os("LOCALBOORU_NATIVE_GLAREA_SPIKE").is_some() {
            log::info!("[NativeVideo] GTK GLArea spike clear submitted");
        }
        gtk::glib::Propagation::Stop
    });

    let content_stack = gtk::Stack::new();
    content_stack.set_hexpand(true);
    content_stack.set_vexpand(true);
    content_stack.add_named(&drawing_area, "shared-memory-cairo");
    content_stack.add_named(&rgba_gl_area, "shared-memory");
    content_stack.add_named(&gl_area, "dma-buf");
    content_stack.set_visible_child_name("shared-memory");

    let media_overlay = gtk::Overlay::new();
    let video_event_box = gtk::EventBox::new();
    video_event_box.add(&content_stack);
    media_overlay.add(&video_event_box);
    let hud = gtk::Box::new(gtk::Orientation::Vertical, 6);
    hud.set_widget_name("native-video-hud");
    hud.set_halign(gtk::Align::Fill);
    hud.set_valign(gtk::Align::End);
    hud.set_margin_start(22);
    hud.set_margin_end(22);
    hud.set_margin_bottom(12);
    let subtitle_label = gtk::Label::new(None);
    subtitle_label.set_widget_name("native-video-subtitle");
    subtitle_label.set_halign(gtk::Align::Center);
    subtitle_label.set_valign(gtk::Align::End);
    subtitle_label.set_margin_start(40);
    subtitle_label.set_margin_end(40);
    subtitle_label.set_margin_bottom(92);
    subtitle_label.set_line_wrap(true);
    subtitle_label.set_justify(gtk::Justification::Center);
    let seek_indicator = gtk::Label::new(None);
    seek_indicator.set_widget_name("native-video-seek-indicator");
    seek_indicator.set_halign(gtk::Align::Center);
    seek_indicator.set_valign(gtk::Align::Center);
    seek_indicator.hide();

    let previous = icon_button(
        include_bytes!("../../../assets/native-video-controls/previous.svg"),
        "Previous",
    )?;
    let play_pause = icon_button(
        include_bytes!("../../../assets/native-video-controls/pause.svg"),
        "Pause / Play (Space)",
    )?;
    let next = icon_button(
        include_bytes!("../../../assets/native-video-controls/next.svg"),
        "Next",
    )?;
    let timeline = gtk::Scale::with_range(gtk::Orientation::Horizontal, 0.0, 1.0, 0.1);
    timeline.set_draw_value(false);
    timeline.set_size_request(240, -1);
    timeline.set_has_tooltip(true);
    timeline.connect_query_tooltip(|scale, x, _, keyboard_mode, tooltip| {
        let position = if keyboard_mode {
            scale.value()
        } else {
            let width = scale.allocated_width().max(1) as f64;
            (x as f64 / width * scale.adjustment().upper())
                .clamp(scale.adjustment().lower(), scale.adjustment().upper())
        };
        tooltip.set_text(Some(&format_time(position)));
        true
    });
    let time_label = gtk::Label::new(Some("0:00 / 0:00"));
    let status_label = gtk::Label::new(None);
    status_label.set_widget_name("native-video-status");
    status_label.hide();
    let mute = gtk::ToggleButton::new();
    mute.set_image(Some(&svg_image(include_bytes!(
        "../../../assets/native-video-controls/volume.svg"
    ))?));
    mute.set_always_show_image(true);
    mute.set_tooltip_text(Some("Mute / Unmute (M)"));
    let volume = gtk::Scale::with_range(gtk::Orientation::Horizontal, 0.0, 1.0, 0.05);
    volume.set_value(1.0);
    volume.set_size_request(110, -1);
    let speed = gtk::ComboBoxText::new();
    for value in ["0.5×", "1×", "1.5×", "2×"] {
        speed.append_text(value);
    }
    speed.set_active(Some(1));
    let audio_tracks = gtk::ComboBoxText::new();
    audio_tracks.append(Some("default"), "Audio");
    audio_tracks.set_active(Some(0));
    audio_tracks.set_sensitive(false);
    audio_tracks.set_tooltip_text(Some("Native audio track switching is not available yet"));
    let subtitle_tracks = gtk::ComboBoxText::new();
    subtitle_tracks.append(Some("__off__"), "Subtitles off");
    subtitle_tracks.set_active(Some(0));
    subtitle_tracks.set_sensitive(false);
    subtitle_tracks.set_tooltip_text(Some("Subtitle track"));
    let subtitle_delay = gtk::SpinButton::with_range(-10.0, 10.0, 0.25);
    subtitle_delay.set_value(0.0);
    subtitle_delay.set_sensitive(false);
    let whisper_subtitles = gtk::Button::with_label("CC+");
    whisper_subtitles.set_tooltip_text(Some("Generate durable Whisper subtitles"));
    subtitle_delay.set_tooltip_text(Some("Subtitle delay in seconds"));
    let display_mode_button = icon_button(
        include_bytes!("../../../assets/native-video-controls/display-mode.svg"),
        "Display mode: Fit",
    )?;
    let fullscreen = icon_button(
        include_bytes!("../../../assets/native-video-controls/fullscreen.svg"),
        "Fullscreen (F)",
    )?;
    let close = icon_button(
        include_bytes!("../../../assets/native-video-controls/close.svg"),
        "Close (Esc)",
    )?;
    let diagnostics_label = gtk::Label::new(Some("GPU -- fps"));
    diagnostics_label.set_widget_name("native-video-diagnostics");
    let speed_badge = gtk::Label::new(None);
    speed_badge.set_widget_name("native-video-speed-badge");
    speed_badge.hide();
    let quality_badge = gtk::Label::new(Some("Original"));
    quality_badge.set_widget_name("native-video-quality-badge");
    let interpolation = gtk::ComboBoxText::new();
    interpolation.append(Some("off"), "Interpolation off");
    interpolation.set_active_id(Some("off"));
    interpolation.set_sensitive(false);
    let timeline_row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    timeline_row.pack_start(&timeline, true, true, 0);
    timeline_row.pack_start(&time_label, false, false, 0);
    timeline_row.pack_start(&status_label, false, false, 0);
    timeline_row.pack_start(&diagnostics_label, false, false, 0);
    let controls_row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    controls_row.pack_start(&previous, false, false, 0);
    controls_row.pack_start(&play_pause, false, false, 0);
    controls_row.pack_start(&next, false, false, 0);
    controls_row.pack_start(&mute, false, false, 0);
    controls_row.pack_start(&volume, true, true, 0);
    controls_row.pack_start(&speed, false, false, 0);
    controls_row.pack_start(&interpolation, false, false, 0);
    controls_row.pack_start(&audio_tracks, false, false, 0);
    controls_row.pack_start(&subtitle_tracks, false, false, 0);
    controls_row.pack_start(&subtitle_delay, false, false, 0);
    controls_row.pack_start(&whisper_subtitles, false, false, 0);
    controls_row.pack_start(&quality_badge, false, false, 0);
    controls_row.pack_start(&speed_badge, false, false, 0);
    controls_row.pack_start(&display_mode_button, false, false, 0);
    controls_row.pack_start(&fullscreen, false, false, 0);
    controls_row.pack_start(&close, false, false, 0);
    hud.pack_start(&timeline_row, false, false, 0);
    hud.pack_start(&controls_row, false, false, 0);
    let paused_state = Rc::new(Cell::new(false));
    let muted_state = Rc::new(Cell::new(false));
    let fullscreen_state = Rc::new(Cell::new(false));
    let current_position = Rc::new(Cell::new(0.0_f64));
    let updating_controls = Rc::new(Cell::new(false));
    let scrubbing = Rc::new(Cell::new(false));
    let speed_state = Rc::new(Cell::new(1.0_f64));
    let last_pointer_activity = Rc::new(Cell::new(Instant::now()));

    {
        let app = app.clone();
        previous.connect_clicked(move |_| emit_semantic_event(&app, SemanticAction::Previous));
    }
    {
        let app = app.clone();
        next.connect_clicked(move |_| emit_semantic_event(&app, SemanticAction::Next));
    }
    {
        let app = app.clone();
        let updating_controls = Rc::clone(&updating_controls);
        let scrubbing = Rc::clone(&scrubbing);
        timeline.connect_change_value(move |_, _, position| {
            if !updating_controls.get() && !scrubbing.get() {
                send_control(&app, NativeVideoCommand::Seek { position });
            }
            gtk::glib::Propagation::Proceed
        });
    }
    {
        let scrubbing = Rc::clone(&scrubbing);
        timeline.connect_button_press_event(move |_, _| {
            scrubbing.set(true);
            gtk::glib::Propagation::Proceed
        });
    }
    {
        let app = app.clone();
        let scrubbing = Rc::clone(&scrubbing);
        timeline.connect_button_release_event(move |scale, _| {
            scrubbing.set(false);
            send_control(
                &app,
                NativeVideoCommand::Seek {
                    position: scale.value(),
                },
            );
            gtk::glib::Propagation::Proceed
        });
    }
    let touch_timeline_gesture = gtk::GestureDrag::new(&timeline);
    touch_timeline_gesture.set_touch_only(true);
    touch_timeline_gesture.set_propagation_phase(gtk::PropagationPhase::Capture);
    let touch_timeline_start = Rc::new(Cell::new(0.0_f64));
    {
        let timeline = timeline.clone();
        let scrubbing = Rc::clone(&scrubbing);
        let start = Rc::clone(&touch_timeline_start);
        touch_timeline_gesture.connect_drag_begin(move |_, _, _| {
            start.set(timeline.value());
            scrubbing.set(true);
        });
    }
    {
        let timeline = timeline.clone();
        let start = Rc::clone(&touch_timeline_start);
        touch_timeline_gesture.connect_drag_update(move |_, offset_x, _| {
            let width = timeline.allocated_width().max(1) as f64;
            let duration = timeline.adjustment().upper();
            timeline.set_value((start.get() + offset_x / width * duration).clamp(0.0, duration));
        });
    }
    {
        let app = app.clone();
        let timeline = timeline.clone();
        let scrubbing = Rc::clone(&scrubbing);
        touch_timeline_gesture.connect_drag_end(move |_, _, _| {
            scrubbing.set(false);
            send_control(
                &app,
                NativeVideoCommand::Seek {
                    position: timeline.value(),
                },
            );
        });
    }
    {
        let app = app.clone();
        let paused_state = Rc::clone(&paused_state);
        play_pause.connect_clicked(move |_| {
            let next_paused = !paused_state.get();
            send_control(
                &app,
                NativeVideoCommand::SetPaused {
                    paused: next_paused,
                },
            );
        });
    }
    {
        let app = app.clone();
        let muted_state = Rc::clone(&muted_state);
        let updating_controls = Rc::clone(&updating_controls);
        mute.connect_toggled(move |button| {
            if updating_controls.get() {
                return;
            }
            let requested_muted = button.is_active();
            updating_controls.set(true);
            button.set_active(muted_state.get());
            updating_controls.set(false);
            send_control(
                &app,
                NativeVideoCommand::SetMuted {
                    muted: requested_muted,
                },
            );
        });
    }
    {
        let app = app.clone();
        let updating_controls = Rc::clone(&updating_controls);
        volume.connect_value_changed(move |scale| {
            if updating_controls.get() {
                return;
            }
            send_control(
                &app,
                NativeVideoCommand::SetVolume {
                    volume: scale.value(),
                },
            );
        });
    }
    {
        let app = app.clone();
        let speed_badge = speed_badge.clone();
        let speed_state = Rc::clone(&speed_state);
        let updating_controls = Rc::clone(&updating_controls);
        speed.connect_changed(move |combo| {
            if updating_controls.get() {
                return;
            }
            let speed = match combo.active() {
                Some(0) => 0.5,
                Some(2) => 1.5,
                Some(3) => 2.0,
                _ => 1.0,
            };
            speed_state.set(speed);
            if (speed - 1.0_f64).abs() < f64::EPSILON {
                speed_badge.hide();
            } else {
                speed_badge.set_text(&format!("{speed:.2}×"));
                speed_badge.show();
            }
            send_control(&app, NativeVideoCommand::SetSpeed { speed });
        });
    }

    {
        let app = app.clone();
        let display_mode = Rc::clone(&display_mode);
        display_mode_button.connect_clicked(move |button| {
            let next = match display_mode.get() {
                DisplayMode::Fit => DisplayMode::Fill,
                DisplayMode::Fill => DisplayMode::Original,
                DisplayMode::Original => DisplayMode::Fit,
            };
            display_mode.set(next);
            DISPLAY_MODE.with(|mode| mode.set(next));
            button.set_tooltip_text(Some(match next {
                DisplayMode::Fit => "Display mode: Fit",
                DisplayMode::Fill => "Display mode: Fill",
                DisplayMode::Original => "Display mode: Original",
            }));
            let _ = set_display_mode(next);
            app.state::<crate::native_video::commands::NativeVideoState>()
                .set_display_mode(next);
        });
    }
    {
        let app = app.clone();
        let updating_controls = Rc::clone(&updating_controls);
        interpolation.connect_changed(move |combo| {
            if updating_controls.get() {
                return;
            }
            let Some(engine) = combo.active_id() else {
                return;
            };
            let engine = engine.to_string();
            send_control(
                &app,
                NativeVideoCommand::SetInterpolation {
                    preset: (engine == "svp").then(|| "balanced".to_string()),
                    engine,
                    target_fps: 60,
                },
            );
        });
    }
    {
        let app = app.clone();
        let updating_controls = Rc::clone(&updating_controls);
        audio_tracks.connect_changed(move |combo| {
            if updating_controls.get() {
                return;
            }
            if let Some(track_id) = combo.active_id() {
                send_control(
                    &app,
                    NativeVideoCommand::SelectAudioTrack {
                        track_id: track_id.to_string(),
                    },
                );
            }
        });
    }
    {
        let app = app.clone();
        let updating_controls = Rc::clone(&updating_controls);
        subtitle_tracks.connect_changed(move |combo| {
            if updating_controls.get() {
                return;
            }
            let track_id = combo.active_id().and_then(|id| {
                let id = id.to_string();
                (id != "__off__").then_some(id)
            });
            send_control(&app, NativeVideoCommand::SelectSubtitleTrack { track_id });
        });
    }
    {
        let app = app.clone();
        let updating_controls = Rc::clone(&updating_controls);
        subtitle_delay.connect_value_changed(move |control| {
            if !updating_controls.get() {
                send_control(
                    &app,
                    NativeVideoCommand::SetSubtitleDelay {
                        seconds: control.value(),
                    },
                );
            }
        });
    }
    {
        let app = app.clone();
        whisper_subtitles.connect_clicked(move |button| {
            let generation = app.state::<NativeVideoState>().current_generation();
            button.set_sensitive(false);
            button.set_label("CC…");
            if let Err(error) =
                crate::native_video::commands::request_whisper_subtitles(&app, generation)
            {
                log::warn!("[NativeVideo] Whisper generation ignored: {error}");
                button.set_label("CC!");
                button.set_sensitive(true);
            }
        });
    }
    {
        let owner = owner_window.clone();
        let fullscreen_state = Rc::clone(&fullscreen_state);
        fullscreen.connect_clicked(move |_| {
            let fullscreen = !fullscreen_state.get();
            fullscreen_state.set(fullscreen);
            if fullscreen {
                owner.fullscreen();
            } else {
                owner.unfullscreen();
            }
        });
    }
    {
        let app = app.clone();
        close.connect_clicked(move |_| emit_semantic_event(&app, SemanticAction::Close));
    }
    {
        let app = app.clone();
        let owner = owner_window.clone();
        let fullscreen_state = Rc::clone(&fullscreen_state);
        let paused_state = Rc::clone(&paused_state);
        let current_position = Rc::clone(&current_position);
        let seek_indicator = seek_indicator.clone();
        video_event_box.connect_button_press_event(move |widget, event| {
            if event.button() != 1 {
                return gtk::glib::Propagation::Proceed;
            }
            if event.event_type() == gdk::EventType::DoubleButtonPress {
                let fullscreen = !fullscreen_state.get();
                fullscreen_state.set(fullscreen);
                if fullscreen {
                    owner.fullscreen();
                } else {
                    owner.unfullscreen();
                }
                return gtk::glib::Propagation::Stop;
            }
            let zone = pointer_zone(event.position().0, widget.allocated_width() as f64);
            if zone == PointerZone::Backward {
                flash_seek_indicator(&seek_indicator, "−10s");
                send_control(
                    &app,
                    NativeVideoCommand::Seek {
                        position: (current_position.get() - 10.0).max(0.0),
                    },
                );
            } else if zone == PointerZone::Forward {
                flash_seek_indicator(&seek_indicator, "+10s");
                send_control(
                    &app,
                    NativeVideoCommand::Seek {
                        position: current_position.get() + 10.0,
                    },
                );
            } else {
                flash_seek_indicator(
                    &seek_indicator,
                    if paused_state.get() { "Play" } else { "Pause" },
                );
                send_control(
                    &app,
                    NativeVideoCommand::SetPaused {
                        paused: !paused_state.get(),
                    },
                );
            }
            gtk::glib::Propagation::Stop
        });
    }
    let touch_video_gesture = gtk::GestureDrag::new(&video_event_box);
    touch_video_gesture.set_touch_only(true);
    touch_video_gesture.set_propagation_phase(gtk::PropagationPhase::Capture);
    {
        let app = app.clone();
        let current_position = Rc::clone(&current_position);
        let hud = hud.clone();
        let last_pointer_activity = Rc::clone(&last_pointer_activity);
        let seek_indicator = seek_indicator.clone();
        touch_video_gesture.connect_drag_end(move |_, offset_x, offset_y| {
            last_pointer_activity.set(Instant::now());
            hud.show();
            if offset_x.abs() < 20.0 || offset_x.abs() <= offset_y.abs() * 1.5 {
                return;
            }
            let delta = offset_x / 50.0 * 10.0;
            flash_seek_indicator(&seek_indicator, &format!("{delta:+.0}s"));
            send_control(
                &app,
                NativeVideoCommand::Seek {
                    position: (current_position.get() + delta).max(0.0),
                },
            );
        });
    }
    {
        let app = app.clone();
        let owner = owner_window.clone();
        let paused_state = Rc::clone(&paused_state);
        let muted_state = Rc::clone(&muted_state);
        let fullscreen_state = Rc::clone(&fullscreen_state);
        let current_position = Rc::clone(&current_position);
        let volume = volume.clone();
        let speed_state = Rc::clone(&speed_state);
        let speed_badge = speed_badge.clone();
        let diagnostics_label = diagnostics_label.clone();
        let subtitle_tracks = subtitle_tracks.clone();
        let hud = hud.clone();
        let last_pointer_activity = Rc::clone(&last_pointer_activity);
        event_box.connect_key_press_event(move |_, event| {
            last_pointer_activity.set(Instant::now());
            hud.show();
            let control = event.state().contains(gdk::ModifierType::CONTROL_MASK);
            let shift = event.state().contains(gdk::ModifierType::SHIFT_MASK);
            let handled = match event.keyval() {
                gdk::keys::constants::space => {
                    let paused = !paused_state.get();
                    send_control(&app, NativeVideoCommand::SetPaused { paused });
                    true
                }
                gdk::keys::constants::Left => {
                    let delta = if control {
                        30.0
                    } else if shift {
                        1.0
                    } else {
                        5.0
                    };
                    send_control(
                        &app,
                        NativeVideoCommand::Seek {
                            position: (current_position.get() - delta).max(0.0),
                        },
                    );
                    true
                }
                gdk::keys::constants::Right => {
                    let delta = if control {
                        30.0
                    } else if shift {
                        1.0
                    } else {
                        5.0
                    };
                    send_control(
                        &app,
                        NativeVideoCommand::Seek {
                            position: current_position.get() + delta,
                        },
                    );
                    true
                }
                gdk::keys::constants::Up => {
                    volume.set_value((volume.value() + 0.05).min(1.0));
                    true
                }
                gdk::keys::constants::Down => {
                    volume.set_value((volume.value() - 0.05).max(0.0));
                    true
                }
                gdk::keys::constants::m | gdk::keys::constants::M => {
                    let muted = !muted_state.get();
                    send_control(&app, NativeVideoCommand::SetMuted { muted });
                    true
                }
                gdk::keys::constants::f | gdk::keys::constants::F => {
                    let fullscreen = !fullscreen_state.get();
                    fullscreen_state.set(fullscreen);
                    if fullscreen {
                        owner.fullscreen();
                    } else {
                        owner.unfullscreen();
                    }
                    true
                }
                gdk::keys::constants::plus
                | gdk::keys::constants::equal
                | gdk::keys::constants::bracketright => {
                    let speed = (speed_state.get() + 0.25).min(4.0);
                    speed_state.set(speed);
                    speed_badge.set_text(&format!("{speed:.2}×"));
                    speed_badge.show();
                    send_control(&app, NativeVideoCommand::SetSpeed { speed });
                    true
                }
                gdk::keys::constants::minus | gdk::keys::constants::bracketleft => {
                    let speed = (speed_state.get() - 0.25).max(0.25);
                    speed_state.set(speed);
                    speed_badge.set_text(&format!("{speed:.2}×"));
                    speed_badge.show();
                    send_control(&app, NativeVideoCommand::SetSpeed { speed });
                    true
                }
                gdk::keys::constants::BackSpace => {
                    speed_state.set(1.0);
                    speed_badge.hide();
                    send_control(&app, NativeVideoCommand::SetSpeed { speed: 1.0 });
                    true
                }
                gdk::keys::constants::e | gdk::keys::constants::E => {
                    send_control(
                        &app,
                        NativeVideoCommand::Seek {
                            position: current_position.get() + 1.0 / 60.0,
                        },
                    );
                    true
                }
                gdk::keys::constants::c | gdk::keys::constants::C => {
                    if subtitle_tracks.active_id().as_deref() == Some("__off__") {
                        subtitle_tracks.set_active(Some(1));
                    } else {
                        subtitle_tracks.set_active_id(Some("__off__"));
                    }
                    true
                }
                gdk::keys::constants::i | gdk::keys::constants::I => {
                    if diagnostics_label.is_visible() {
                        diagnostics_label.hide();
                    } else {
                        diagnostics_label.show();
                    }
                    true
                }
                gdk::keys::constants::Escape => {
                    emit_semantic_event(&app, SemanticAction::Close);
                    true
                }
                _ => false,
            };
            if handled {
                gtk::glib::Propagation::Stop
            } else {
                gtk::glib::Propagation::Proceed
            }
        });
    }

    media_overlay.add_overlay(&subtitle_label);
    media_overlay.add_overlay(&seek_indicator);
    media_overlay.add_overlay(&hud);
    event_box.add(&media_overlay);
    drawing_area.show();
    rgba_gl_area.show();
    gl_area.show();
    content_stack.show();
    video_event_box.show();
    hud.show_all();
    status_label.hide();
    subtitle_label.show();
    media_overlay.show();

    event_box.add_events(gdk::EventMask::POINTER_MOTION_MASK);
    {
        let hud = hud.clone();
        let last_pointer_activity = Rc::clone(&last_pointer_activity);
        event_box.connect_motion_notify_event(move |_, _| {
            last_pointer_activity.set(Instant::now());
            hud.show();
            gtk::glib::Propagation::Proceed
        });
    }
    {
        let hud = hud.downgrade();
        let last_pointer_activity = Rc::clone(&last_pointer_activity);
        let scrubbing = Rc::clone(&scrubbing);
        gtk::glib::timeout_add_local(Duration::from_millis(500), move || {
            let Some(hud) = hud.upgrade() else {
                return gtk::glib::ControlFlow::Break;
            };
            if !scrubbing.get()
                && hud.focus_child().is_none()
                && last_pointer_activity.get().elapsed() >= Duration::from_secs(3)
            {
                hud.hide();
            }
            gtk::glib::ControlFlow::Continue
        });
    }

    let provider = gtk::CssProvider::new();
    provider
        .load_from_data(include_bytes!("../../../assets/native-video-hud.css"))
        .map_err(|error| format!("failed to style native viewport: {error}"))?;
    event_box
        .style_context()
        .add_provider(&provider, gtk::STYLE_PROVIDER_PRIORITY_APPLICATION);
    container.show();
    webview.show();
    event_box.hide();
    host_window.hide();

    {
        let host = host_window.clone();
        let bounds = Rc::clone(&bounds);
        owner_window.connect_configure_event(move |_, _| {
            position_host(&host, bounds.get());
            false
        });
    }
    {
        let host = host_window.clone();
        owner_window.connect_hide(move |_| host.hide());
    }
    {
        let host = host_window.clone();
        let bounds = Rc::clone(&bounds);
        let desired_visible = Rc::clone(&desired_visible);
        owner_window.connect_show(move |_| {
            if desired_visible.get() {
                position_host(&host, bounds.get());
                host.show();
            }
        });
    }
    {
        let host = host_window.clone();
        owner_window.connect_destroy(move |_| {
            VIEWPORT_ATTACHED.store(false, Ordering::Release);
            host.close();
        });
    }

    VIEWPORT.with(|slot| {
        *slot.borrow_mut() = Some(NativeViewport {
            host_window,
            webview,
            bounds,
            desired_visible,
            display_mode,
            event_box,
            rgba_gl_area: rgba_gl_area.clone(),
            _gl_area: gl_area.clone(),
            content_stack: content_stack.clone(),
            hud,
            play_pause,
            timeline,
            time_label,
            status_label,
            diagnostics_label,
            mute,
            volume,
            speed,
            speed_badge,
            interpolation,
            audio_tracks,
            subtitle_tracks,
            subtitle_delay,
            whisper_subtitles,
            subtitle_label,
            current_position,
            paused_state,
            muted_state,
            speed_state,
            updating_controls,
            scrubbing,
            _touch_video_gesture: touch_video_gesture,
            _touch_timeline_gesture: touch_timeline_gesture,
            _pending_draw: pending_draw,
            pending_rgba,
            pending_dmabuf,
            in_flight_dmabuf,
            clear_external_cache,
        });
    });
    VIEWPORT_ATTACHED.store(true, Ordering::Release);
    #[cfg(debug_assertions)]
    if std::env::var_os("LOCALBOORU_NATIVE_GLAREA_SPIKE").is_some() {
        set_bounds(ViewportBounds {
            x: 100,
            y: 100,
            width: 640,
            height: 360,
        })?;
        set_visible(true)?;
        content_stack.set_visible_child_name("dma-buf");
        gl_area.queue_render();
    }
    #[cfg(debug_assertions)]
    if std::env::var_os("LOCALBOORU_NATIVE_VIEWPORT_SPIKE").is_some() {
        set_bounds(ViewportBounds {
            x: 100,
            y: 100,
            width: 640,
            height: 360,
        })?;
        set_visible(true)?;
        let mut pixels = vec![0_u8; 640 * 360 * 4];
        for (index, pixel) in pixels.chunks_exact_mut(4).enumerate() {
            let x = index % 640;
            let y = index / 640;
            let light = ((x / 40) + (y / 40)) % 2 == 0;
            pixel.copy_from_slice(if light {
                &[0x18, 0x8c, 0xff, 0xff]
            } else {
                &[0x07, 0x16, 0x29, 0xff]
            });
        }
        present_rgba(
            640,
            360,
            1.0,
            0,
            640 * 4,
            pixels,
            || eprintln!("[NativeVideo] GTK viewport spike frame presented"),
            || eprintln!("[NativeVideo] GTK viewport spike frame discarded"),
        )?;
    }
    Ok(())
}

pub fn set_bounds(bounds: ViewportBounds) -> Result<(), String> {
    let bounds = bounds.normalized();
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        if viewport.bounds.get() != bounds {
            log::info!(
                "[NativeVideo] viewport x={} y={} width={} height={}",
                bounds.x,
                bounds.y,
                bounds.width,
                bounds.height
            );
        }
        viewport.bounds.set(bounds);
        viewport
            .event_box
            .set_size_request(bounds.width, bounds.height);
        viewport.host_window.resize(bounds.width, bounds.height);
        if viewport.host_window.is_visible() {
            position_host(&viewport.host_window, bounds);
            update_hud_workarea_inset(viewport);
        }
        Ok(())
    })
}

pub fn set_display_mode(mode: DisplayMode) -> Result<(), String> {
    DISPLAY_MODE.with(|current| current.set(mode));
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let Some(viewport) = slot.as_ref() else {
            return Ok(());
        };
        if viewport.display_mode.replace(mode) != mode {
            log::info!("[NativeVideo] display mode changed to {mode:?}");
            viewport.rgba_gl_area.queue_render();
            viewport._gl_area.queue_render();
            if let Some(window) = viewport.event_box.window() {
                window.invalidate_rect(None, false);
            }
        }
        Ok(())
    })
}

pub fn handle_surface_message(
    received: ReceivedSurfaceMessage,
    on_presented: impl FnOnce(SurfaceFrameRelease) + 'static,
    on_discarded: impl FnOnce(SurfaceFrameRelease) + 'static,
) -> Result<bool, String> {
    match &received.message {
        SurfaceChannelMessage::SurfaceCreated { descriptor } => {
            if SHM_GENERATION.swap(descriptor.generation, Ordering::Relaxed)
                != descriptor.generation
            {
                SHM_ACCEPTED_FRAMES.store(0, Ordering::Relaxed);
                SHM_SUBMITTED_FRAMES.store(0, Ordering::Relaxed);
                SHM_PRESENTED_FRAMES.store(0, Ordering::Relaxed);
                SHM_DISCARDED_FRAMES.store(0, Ordering::Relaxed);
                SHM_QUEUE_DEPTH.store(0, Ordering::Relaxed);
                SHM_QUEUE_LATENCY_US.store(0, Ordering::Relaxed);
            }
            SHM_CONSUMER.with(|consumer| consumer.borrow_mut().register(received))?;
            Ok(false)
        }
        SurfaceChannelMessage::FrameReady { .. } => {
            let frame = SHM_CONSUMER.with(|consumer| consumer.borrow().frame(received))?;
            let descriptor = frame.descriptor().clone();
            let release = frame.release();
            let width = i32::try_from(descriptor.width)
                .map_err(|_| "shared frame width does not fit GTK".to_string())?;
            let height = i32::try_from(descriptor.height)
                .map_err(|_| "shared frame height does not fit GTK".to_string())?;
            let discarded_release = release.clone();
            if descriptor.fourcc == DRM_FORMAT_YUV420 {
                let strides = descriptor
                    .planes
                    .iter()
                    .map(|plane| i32::try_from(plane.stride))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "shared YUV stride does not fit GTK".to_string())?;
                let offsets = descriptor
                    .planes
                    .iter()
                    .map(|plane| usize::try_from(plane.offset))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| "shared YUV offset does not fit GTK".to_string())?;
                present_yuv420(
                    release.buffer_id,
                    width,
                    height,
                    descriptor.sample_aspect_ratio,
                    descriptor.rotation_degrees,
                    [strides[0], strides[1], strides[2]],
                    [offsets[0], offsets[1], offsets[2]],
                    descriptor.color_space,
                    descriptor.color_range,
                    frame,
                    move || {
                        let presented = SHM_PRESENTED_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                        if presented == 1 || presented % 60 == 0 {
                            log::info!("[NativeVideo] GTK draw-completed SHM frame {presented}");
                        }
                        on_presented(release)
                    },
                    move || {
                        let discarded = SHM_DISCARDED_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                        if discarded == 1 || discarded % 60 == 0 {
                            log::info!("[NativeVideo] GTK discarded SHM frame {discarded}");
                        }
                        on_discarded(discarded_release)
                    },
                )?;
            } else {
                let stride = i32::try_from(descriptor.planes[0].stride)
                    .map_err(|_| "shared frame stride does not fit GTK".to_string())?;
                present_rgba(
                    width,
                    height,
                    descriptor.sample_aspect_ratio,
                    descriptor.rotation_degrees,
                    stride,
                    frame.bytes().to_vec(),
                    move || on_presented(release),
                    move || on_discarded(discarded_release),
                )?;
            }
            Ok(true)
        }
        SurfaceChannelMessage::FrameRelease { .. } => {
            Err("consumer received an unexpected frame_release message".to_string())
        }
    }
}

pub fn present_rgba(
    width: i32,
    height: i32,
    sample_aspect_ratio: f64,
    rotation_degrees: i32,
    stride: i32,
    rgba: Vec<u8>,
    on_presented: impl FnOnce() + 'static,
    on_discarded: impl FnOnce() + 'static,
) -> Result<(), String> {
    if width <= 0 || height <= 0 || stride < width.saturating_mul(4) {
        return Err("native viewport frame geometry is invalid".to_string());
    }
    let required = usize::try_from(stride)
        .ok()
        .and_then(|stride| stride.checked_mul(usize::try_from(height).ok()?))
        .ok_or_else(|| "native viewport frame size overflow".to_string())?;
    if rgba.len() < required {
        return Err("native viewport frame data is shorter than its stride".to_string());
    }
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        viewport
            .content_stack
            .set_visible_child_name("shared-memory");
        let mut pending = viewport.pending_rgba.borrow_mut();
        let dropped = push_bounded(
            &mut pending,
            PendingRgba {
                accepted_at: Instant::now(),
                buffer_id: 0,
                width,
                height,
                sample_aspect_ratio,
                rotation_degrees,
                color_space: None,
                color_range: None,
                stride,
                yuv420p: false,
                strides: [stride, 0, 0],
                offsets: [0, 0, 0],
                pixels: PendingPixels::Owned(rgba),
                on_presented: Some(Box::new(on_presented)),
                on_discarded: Some(Box::new(on_discarded)),
            },
            MAX_PENDING_GL_FRAMES,
        );
        drop(pending);
        if let Some(mut dropped) = dropped {
            if let Some(on_discarded) = dropped.on_discarded.take() {
                on_discarded();
            }
        }
        map_host_for_preroll(viewport);
        viewport.rgba_gl_area.queue_render();
        Ok(())
    })
}

pub fn present_yuv420(
    buffer_id: u32,
    width: i32,
    height: i32,
    sample_aspect_ratio: f64,
    rotation_degrees: i32,
    strides: [i32; 3],
    offsets: [usize; 3],
    color_space: Option<SurfaceColorSpace>,
    color_range: Option<SurfaceColorRange>,
    yuv: ShmFrameView,
    on_presented: impl FnOnce() + 'static,
    on_discarded: impl FnOnce() + 'static,
) -> Result<(), String> {
    if width <= 0 || height <= 0 || strides != [width, width / 2, width / 2] {
        return Err("native YUV frame geometry is invalid".to_string());
    }
    let required = offsets[2]
        .checked_add((strides[2] as usize).saturating_mul((height / 2) as usize))
        .ok_or_else(|| "native YUV frame size overflow".to_string())?;
    if yuv.bytes().len() < required {
        return Err("native YUV frame data is shorter than its planes".to_string());
    }
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        viewport
            .content_stack
            .set_visible_child_name("shared-memory");
        let mut pending = viewport.pending_rgba.borrow_mut();
        let accepted = SHM_ACCEPTED_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
        if accepted == 1 || accepted % 60 == 0 {
            log::info!(
                "[NativeVideo] GTK accepted SHM frame {accepted} queue_depth={}",
                pending.len() + 1
            );
        }
        let dropped = push_bounded(
            &mut pending,
            PendingRgba {
                accepted_at: Instant::now(),
                buffer_id,
                width,
                height,
                sample_aspect_ratio,
                rotation_degrees,
                color_space,
                color_range,
                stride: strides[0],
                yuv420p: true,
                strides,
                offsets,
                pixels: PendingPixels::Shared(yuv),
                on_presented: Some(Box::new(on_presented)),
                on_discarded: Some(Box::new(on_discarded)),
            },
            MAX_PENDING_GL_FRAMES,
        );
        SHM_QUEUE_DEPTH.store(pending.len() as u64, Ordering::Relaxed);
        drop(pending);
        if let Some(mut dropped) = dropped {
            if let Some(on_discarded) = dropped.on_discarded.take() {
                on_discarded();
            }
        }
        map_host_for_preroll(viewport);
        viewport.rgba_gl_area.queue_render();
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub fn update_playback(
    position: f64,
    duration: f64,
    paused: bool,
    volume: f64,
    muted: bool,
    speed: f64,
    selected_audio_track: Option<&str>,
    selected_subtitle_track: Option<&str>,
    subtitle_delay: f64,
    interpolation_engine: &str,
) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        let duration = duration.max(0.0);
        viewport.updating_controls.set(true);
        viewport.current_position.set(position.max(0.0));
        viewport.paused_state.set(paused);
        viewport.muted_state.set(muted);
        viewport.speed_state.set(speed);
        viewport.mute.set_active(muted);
        viewport.volume.set_value(volume.clamp(0.0, 1.0));
        viewport.speed.set_active(match speed {
            value if (value - 0.5).abs() < 0.001 => Some(0),
            value if (value - 1.0).abs() < 0.001 => Some(1),
            value if (value - 1.5).abs() < 0.001 => Some(2),
            value if (value - 2.0).abs() < 0.001 => Some(3),
            _ => None,
        });
        if (speed - 1.0).abs() < 0.001 {
            viewport.speed_badge.hide();
        } else {
            viewport.speed_badge.set_text(&format!("{speed:.2}×"));
            viewport.speed_badge.show();
        }
        if let Some(track_id) = selected_audio_track {
            viewport.audio_tracks.set_active_id(Some(track_id));
        }
        viewport
            .subtitle_tracks
            .set_active_id(Some(selected_subtitle_track.unwrap_or("__off__")));
        viewport.subtitle_delay.set_value(subtitle_delay);
        viewport
            .interpolation
            .set_active_id(Some(interpolation_engine));
        viewport.timeline.set_range(0.0, duration.max(1.0));
        if !viewport.scrubbing.get() {
            viewport
                .timeline
                .set_value(position.clamp(0.0, duration.max(1.0)));
        }
        viewport.time_label.set_text(&format!(
            "{} / {}",
            format_time(position),
            format_time(duration)
        ));
        set_button_icon(
            &viewport.play_pause,
            if paused {
                include_bytes!("../../../assets/native-video-controls/play.svg")
            } else {
                include_bytes!("../../../assets/native-video-controls/pause.svg")
            },
        );
        if position >= 0.5 {
            maybe_capture_hud_screenshot(viewport);
        }
        viewport.updating_controls.set(false);
        Ok(())
    })
}

pub fn update_status(message: Option<&str>, error: bool) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        if let Some(message) = message.filter(|message| !message.is_empty()) {
            viewport.status_label.set_widget_name(if error {
                "native-video-error"
            } else {
                "native-video-status"
            });
            viewport.status_label.set_text(message);
            viewport.status_label.show();
            viewport.hud.show();
        } else {
            viewport.status_label.hide();
        }
        Ok(())
    })
}

pub fn update_tracks(audio: &[TrackInfo], subtitles: &[TrackInfo]) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        viewport.updating_controls.set(true);
        viewport.audio_tracks.remove_all();
        for track in audio {
            viewport.audio_tracks.append(Some(&track.id), &track.label);
        }
        if !audio.is_empty() {
            let active = audio.iter().position(|track| track.is_default).unwrap_or(0) as u32;
            viewport.audio_tracks.set_active(Some(active));
        }

        viewport.subtitle_tracks.remove_all();
        viewport
            .subtitle_tracks
            .append(Some("__off__"), "Subtitles off");
        for track in subtitles {
            viewport
                .subtitle_tracks
                .append(Some(&track.id), &track.label);
        }
        let active = subtitles
            .iter()
            .position(|track| track.is_default || track.is_forced)
            .map(|index| index as u32 + 1)
            .unwrap_or(0);
        viewport.subtitle_tracks.set_active(Some(active));
        viewport
            .subtitle_tracks
            .set_sensitive(!subtitles.is_empty());
        viewport.subtitle_delay.set_value(0.0);
        viewport.subtitle_delay.set_sensitive(!subtitles.is_empty());
        viewport.whisper_subtitles.set_label("CC+");
        viewport.whisper_subtitles.set_sensitive(true);
        viewport.updating_controls.set(false);
        Ok(())
    })
}

pub fn update_whisper_status(status: &str) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        match status {
            "completed" => {
                viewport.whisper_subtitles.set_label("CC✓");
                viewport.whisper_subtitles.set_sensitive(true);
            }
            "failed" => {
                viewport.whisper_subtitles.set_label("CC!");
                viewport.whisper_subtitles.set_sensitive(true);
            }
            _ => {
                viewport.whisper_subtitles.set_label("CC…");
                viewport.whisper_subtitles.set_sensitive(false);
            }
        }
        Ok(())
    })
}

pub fn add_subtitle_track(track: &TrackInfo, select: bool) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        viewport.updating_controls.set(true);
        viewport
            .subtitle_tracks
            .append(Some(&track.id), &track.label);
        viewport.subtitle_tracks.set_sensitive(true);
        viewport.subtitle_delay.set_sensitive(true);
        if select {
            viewport.subtitle_tracks.set_active_id(Some(&track.id));
        }
        viewport.updating_controls.set(false);
        Ok(())
    })
}

pub fn update_subtitle_text(lines: &[String]) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        viewport.subtitle_label.set_text(&lines.join("\n"));
        viewport.subtitle_label.set_visible(!lines.is_empty());
        Ok(())
    })
}

pub fn update_interpolation_capabilities(
    interpolation_engines: &[String],
    svp_status: Option<&str>,
) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        let detected = interpolation_engines.iter().any(|engine| engine == "svp");
        viewport.updating_controls.set(true);
        viewport.interpolation.remove_all();
        viewport
            .interpolation
            .append(Some("off"), "Interpolation off");
        if detected {
            viewport.interpolation.append(Some("svp"), "SVP 60 fps");
        }
        let selected = if svp_status
            .is_some_and(|status| status == "selected_external" || status == "active_external")
        {
            "svp"
        } else {
            "off"
        };
        viewport.interpolation.set_active_id(Some(selected));
        viewport.interpolation.set_sensitive(detected);
        viewport
            .interpolation
            .set_tooltip_text(Some(svp_status.unwrap_or("SVP runtime unavailable")));
        viewport.updating_controls.set(false);
        Ok(())
    })
}

pub fn enrich_diagnostics(event: &mut NativeVideoEvent) {
    let NativeVideoEvent::Diagnostics {
        surface_mode,
        queue_depth,
        queue_latency_ms,
        accepted_frames,
        draw_completed_frames,
        dropped_frames,
        ..
    } = event
    else {
        return;
    };
    let mode = surface_mode.as_deref().unwrap_or_default();
    if mode.contains("shared_memory") {
        *queue_depth = Some(SHM_QUEUE_DEPTH.load(Ordering::Relaxed));
        *queue_latency_ms = Some(SHM_QUEUE_LATENCY_US.load(Ordering::Relaxed) as f64 / 1000.0);
        *accepted_frames = Some(SHM_ACCEPTED_FRAMES.load(Ordering::Relaxed));
        *draw_completed_frames = Some(SHM_PRESENTED_FRAMES.load(Ordering::Relaxed));
        *dropped_frames =
            dropped_frames.saturating_add(SHM_DISCARDED_FRAMES.load(Ordering::Relaxed));
    } else if mode.contains("dmabuf") || mode.contains("dma_buf") {
        *accepted_frames = Some(DMA_BUF_ACCEPTED_FRAMES.load(Ordering::Relaxed));
        *draw_completed_frames = Some(DMA_BUF_PRESENTED_FRAMES.load(Ordering::Relaxed));
        *dropped_frames =
            dropped_frames.saturating_add(DMA_BUF_DISCARDED_FRAMES.load(Ordering::Relaxed));
    }
}

pub fn update_diagnostics(
    produced_fps: f64,
    presented_fps: f64,
    dropped_frames: u64,
    zero_cpu_copy: bool,
    decoder: Option<&str>,
    hardware_device: Option<&str>,
    source_fps: Option<f64>,
    queue_depth: Option<u64>,
    interpolation_engine: Option<&str>,
    surface_mode: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    av_drift_ms: Option<f64>,
    first_frame_latency_ms: Option<f64>,
    seek_latency_ms: Option<f64>,
) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        let surface = match surface_mode {
            Some("dma_buf_external_oes") => "DMA",
            Some("shared_memory_rgba") => "SHM",
            Some("svp_yuv420p_shared_memory_to_gpu") => "SVP-SHM/GPU",
            Some(_) => "CPU",
            None if zero_cpu_copy => "GPU",
            None => "SHM",
        };
        let drift = av_drift_ms
            .map(|value| format!("{value:+.1}ms"))
            .unwrap_or_else(|| "n/a".to_string());
        let first_frame = first_frame_latency_ms
            .map(|value| format!("{value:.1}ms"))
            .unwrap_or_else(|| "pending".to_string());
        let seek = seek_latency_ms
            .map(|value| format!("{value:.1}ms"))
            .unwrap_or_else(|| "n/a".to_string());
        viewport.diagnostics_label.set_text(&format!(
            "{} {} {} {:.1}→{:.1}/{:.1} fps · q{} · {}×{} · {} · {} drop · A/V {} · first {} · seek {}",
            decoder.unwrap_or("decoder"),
            hardware_device.unwrap_or("cpu"),
            surface,
            source_fps.unwrap_or(0.0),
            produced_fps,
            presented_fps,
            queue_depth.unwrap_or(0),
            width.unwrap_or(0),
            height.unwrap_or(0),
            interpolation_engine.unwrap_or("off"),
            dropped_frames,
            drift,
            first_frame,
            seek
        ));
        Ok(())
    })
}

pub fn set_visible(visible: bool) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        if visible {
            viewport.desired_visible.set(true);
            viewport.host_window.set_accept_focus(true);
            viewport.host_window.set_focus_on_map(true);
            viewport.host_window.set_opacity(1.0);
            viewport.event_box.show();
            position_host(&viewport.host_window, viewport.bounds.get());
            viewport.host_window.show();
            viewport.host_window.present();
            update_hud_workarea_inset(viewport);
            viewport.event_box.grab_focus();
        } else {
            viewport.desired_visible.set(false);
            viewport.clear_external_cache.set(true);
            if viewport.in_flight_dmabuf.borrow().is_none() {
                viewport._gl_area.make_current();
                if viewport._gl_area.error().is_none() {
                    crate::native_video::gtk_gl_context::clear_external_image_cache();
                    viewport.clear_external_cache.set(false);
                }
            }
            viewport.host_window.set_accept_focus(false);
            viewport.host_window.set_focus_on_map(false);
            viewport.host_window.hide();
            viewport.event_box.hide();
            viewport.webview.grab_focus();
            if let Some(mut pending) = viewport._pending_draw.borrow_mut().take() {
                if let Some(discarded) = pending.on_discarded.take() {
                    discarded();
                }
            }
            if let Some(pending) = viewport.pending_dmabuf.borrow_mut().take() {
                let _ = pending.completion.send(Err(
                    "DMA-BUF frame discarded because the native viewport was hidden".to_string(),
                ));
            }
            for mut pending in viewport.pending_rgba.borrow_mut().drain(..) {
                if let Some(discarded) = pending.on_discarded.take() {
                    discarded();
                }
            }
            SHM_QUEUE_DEPTH.store(0, Ordering::Relaxed);
            viewport.subtitle_label.set_text("");
            viewport.subtitle_label.hide();
        }
        Ok(())
    })
}

pub fn set_fullscreen(fullscreen: bool) -> Result<(), String> {
    VIEWPORT.with(|slot| {
        let slot = slot.borrow();
        let viewport = slot
            .as_ref()
            .ok_or_else(|| "native viewport is not attached".to_string())?;
        if fullscreen {
            viewport.host_window.fullscreen();
        } else {
            viewport.host_window.unfullscreen();
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_geometry_letterboxes_without_distorting_aspect_ratio() {
        let geometry = frame_geometry(1920, 1080, 1.0, 0, 800, 800, 1.0, DisplayMode::Fit)
            .expect("fit geometry");
        assert_eq!(geometry.content_rect.x, 0.0);
        assert_eq!(geometry.content_rect.y, 175.0);
        assert_eq!(geometry.content_rect.width, 800.0);
        assert_eq!(geometry.content_rect.height, 450.0);
    }

    #[test]
    fn bounds_are_clamped_to_a_visible_surface() {
        assert_eq!(
            ViewportBounds {
                x: -2,
                y: -3,
                width: 0,
                height: -1,
            }
            .normalized(),
            ViewportBounds {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            }
        );
    }

    #[test]
    fn hud_stays_above_reserved_desktop_workarea() {
        let workarea = gdk::Rectangle::new(0, 0, 2560, 1376);
        assert_eq!(bottom_workarea_inset(78, 1362, workarea), 64);
        assert_eq!(bottom_workarea_inset(78, 1200, workarea), 0);
    }

    #[test]
    fn hud_time_labels_are_stable() {
        assert_eq!(format_time(-1.0), "0:00");
        assert_eq!(format_time(65.4), "1:05");
        assert_eq!(format_time(65.6), "1:06");
        assert_eq!(format_time(3661.0), "1:01:01");
    }

    #[test]
    fn pointer_zones_preserve_hud_control_hit_testing() {
        assert_eq!(pointer_zone(0.0, 300.0), PointerZone::Backward);
        assert_eq!(pointer_zone(98.0, 300.0), PointerZone::Backward);
        assert_eq!(pointer_zone(99.0, 300.0), PointerZone::Center);
        assert_eq!(pointer_zone(201.0, 300.0), PointerZone::Center);
        assert_eq!(pointer_zone(202.0, 300.0), PointerZone::Forward);
        assert_eq!(pointer_zone(300.0, 300.0), PointerZone::Forward);
    }

    #[test]
    fn pending_gl_queue_discards_the_oldest_frame_at_its_hard_limit() {
        let mut queue = VecDeque::from([1, 2, 3]);
        assert_eq!(push_bounded(&mut queue, 4, 3), Some(1));
        assert_eq!(queue, VecDeque::from([2, 3, 4]));
    }
}
