//! Core Foundation adapter. Access starts only after resolving a scoped bookmark.
use super::Backend;
use std::ffi::{c_void, CStr, OsStr};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};

type Ref = *const c_void;

// Signatures/options follow Apple's CFURL.h and CFData.h. No Objective-C
// objects or autoreleased pointers cross the adapter boundary.
#[cfg_attr(target_os = "macos", link(name = "CoreFoundation", kind = "framework"))]
extern "C" {
    fn CFRelease(value: Ref);
    fn CFURLCreateFromFileSystemRepresentation(
        allocator: Ref,
        bytes: *const u8,
        len: isize,
        directory: u8,
    ) -> Ref;
    fn CFURLCreateBookmarkData(
        allocator: Ref,
        url: Ref,
        options: usize,
        keys: Ref,
        relative: Ref,
        error: *mut Ref,
    ) -> Ref;
    fn CFURLCreateByResolvingBookmarkData(
        allocator: Ref,
        data: Ref,
        options: usize,
        relative: Ref,
        keys: Ref,
        stale: *mut u8,
        error: *mut Ref,
    ) -> Ref;
    fn CFURLGetFileSystemRepresentation(url: Ref, resolve: u8, bytes: *mut u8, len: isize) -> u8;
    fn CFURLStartAccessingSecurityScopedResource(url: Ref) -> u8;
    fn CFURLStopAccessingSecurityScopedResource(url: Ref);
    fn CFDataCreate(allocator: Ref, bytes: *const u8, len: isize) -> Ref;
    fn CFDataGetLength(data: Ref) -> isize;
    fn CFDataGetBytePtr(data: Ref) -> *const u8;
}

struct Owned(NonNull<c_void>);
impl Owned {
    fn created(value: Ref) -> Result<Self, String> {
        NonNull::new(value.cast_mut())
            .map(Self)
            .ok_or_else(|| "Folder permission is unavailable. Select the folder again.".into())
    }
    fn as_ref(&self) -> Ref {
        self.0.as_ptr().cast_const()
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { CFRelease(self.as_ref()) }
    }
}

pub struct Scope {
    url: Owned,
}
// CFURL is immutable; a scope's retain/start and stop/release are balanced
// exactly once. The enclosing Mutex serializes all adapter operations.
unsafe impl Send for Scope {}
impl Drop for Scope {
    fn drop(&mut self) {
        unsafe { CFURLStopAccessingSecurityScopedResource(self.url.as_ref()) }
    }
}

fn bookmark(url: &Owned) -> Result<Vec<u8>, String> {
    let data = Owned::created(unsafe {
        CFURLCreateBookmarkData(
            ptr::null(),
            url.as_ref(),
            1 << 11,
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
        )
    })?;
    let length = unsafe { CFDataGetLength(data.as_ref()) };
    let bytes = unsafe { CFDataGetBytePtr(data.as_ref()) };
    if !(1..=1024 * 1024).contains(&length) || bytes.is_null() {
        return Err("Folder permission could not be saved. Select the folder again.".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(bytes, length as usize) }.to_vec())
}

/// This function is called only with a Path returned by the native picker.
pub fn bookmark_from_selection(path: &Path) -> Result<Vec<u8>, String> {
    if !path.is_absolute() {
        return Err("Select a local folder".into());
    }
    let bytes = path.as_os_str().as_bytes();
    let url = Owned::created(unsafe {
        CFURLCreateFromFileSystemRepresentation(
            ptr::null(),
            bytes.as_ptr(),
            bytes.len() as isize,
            1,
        )
    })?;
    bookmark(&url)
}

fn resolved_path(url: &Owned) -> Result<PathBuf, String> {
    let mut buffer = vec![0u8; 4096];
    if unsafe {
        CFURLGetFileSystemRepresentation(
            url.as_ref(),
            1,
            buffer.as_mut_ptr(),
            buffer.len() as isize,
        )
    } == 0
    {
        return Err("Folder location could not be restored. Select it again.".into());
    }
    let bytes =
        CStr::from_bytes_until_nul(&buffer).map_err(|_| "Folder location could not be restored")?;
    Ok(PathBuf::from(OsStr::from_bytes(bytes.to_bytes())))
}

pub struct AppleBackend;
impl Backend for AppleBackend {
    type Scope = Scope;
    fn activate(&mut self, bookmark: &[u8]) -> Result<(Scope, bool), String> {
        let data = Owned::created(unsafe {
            CFDataCreate(ptr::null(), bookmark.as_ptr(), bookmark.len() as isize)
        })?;
        let mut stale = 0u8;
        let url = Owned::created(unsafe {
            // WithSecurityScope | WithoutUI | WithoutMounting: startup must
            // neither display dialogs nor automatically mount unavailable media.
            CFURLCreateByResolvingBookmarkData(
                ptr::null(),
                data.as_ref(),
                (1 << 10) | (1 << 8) | (1 << 9),
                ptr::null(),
                ptr::null(),
                &mut stale,
                ptr::null_mut(),
            )
        })?;
        if unsafe { CFURLStartAccessingSecurityScopedResource(url.as_ref()) } == 0 {
            return Err("Folder access was not granted. Select the folder again.".into());
        }
        let scope = Scope { url };
        if !resolved_path(&scope.url)?.is_dir() {
            return Err(
                "Folder is unavailable. Reconnect the drive or select the folder again.".into(),
            );
        }
        Ok((scope, stale != 0))
    }
    fn refresh(&mut self, scope: &Scope) -> Result<Vec<u8>, String> {
        bookmark(&scope.url)
    }
}
