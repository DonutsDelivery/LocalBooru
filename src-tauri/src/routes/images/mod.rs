pub mod adjustments;
pub mod batch;
pub mod helpers;
pub mod list;
pub mod single;

use axum::routing::{delete, get, patch, post};
use axum::Router;

use crate::server::state::AppState;

/// Build the /api/images router with all sub-routes.
pub fn router() -> Router<AppState> {
    Router::new()
        // List endpoints
        .route("/", get(list::list_images))
        .route("/folders", get(list::list_folders))
        // Static routes (must be before /{image_id})
        .route("/media/file-info", get(single::get_file_info))
        .route("/upload", post(single::upload_image))
        // Batch endpoints
        .route("/batch/delete", post(batch::batch_delete))
        .route("/batch/retag", post(batch::batch_retag))
        .route("/batch/age-detect", post(batch::batch_age_detect))
        .route("/batch/extract-metadata", post(batch::batch_extract_metadata))
        .route("/batch/move", post(batch::batch_move))
        // Dynamic routes (/{image_id}/...)
        .route("/{image_id}", get(single::get_image))
        .route("/{image_id}", delete(single::delete_image))
        .route("/{image_id}/file", get(single::get_image_file))
        .route("/{image_id}/thumbnail", get(single::get_image_thumbnail))
        .route("/{image_id}/favorite", post(single::toggle_favorite))
        .route("/{image_id}/rating", patch(single::update_rating))
        .route("/{image_id}/preview-frames", get(single::get_preview_frames))
        .route("/{image_id}/preview-frame/{frame_index}", get(single::get_preview_frame))
        // Adjustment endpoints
        .route("/{image_id}/preview-adjust", post(adjustments::preview_adjust))
        .route("/{image_id}/preview", get(adjustments::get_preview))
        .route("/{image_id}/preview", delete(adjustments::discard_preview))
        .route("/{image_id}/adjust", post(adjustments::apply_adjust))
}
