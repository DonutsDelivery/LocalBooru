//! Backend Management Module
//!
//! Manages the FastAPI Python backend as a subprocess.
//! This mirrors the functionality of Electron's backendManager.js

pub mod config;
pub mod health;
pub mod process;
pub mod manager;

pub use manager::BackendManager;
