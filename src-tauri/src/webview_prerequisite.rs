// The policy is independent of the native adapters so it can be tested without Windows.
#[derive(Debug, PartialEq)]
enum Probe<E> {
    Present,
    Missing,
    Error(E),
}

fn preflight<E>(
    probe: impl FnOnce() -> Probe<E>,
    ask: impl FnOnce() -> bool,
    open: impl FnOnce() -> Result<(), E>,
) -> Result<bool, E> {
    match probe() {
        Probe::Present => Ok(true),
        Probe::Missing => {
            if ask() {
                open()?;
            }
            Ok(false)
        }
        Probe::Error(error) => Err(error),
    }
}

#[cfg(windows)]
pub fn check() -> bool {
    use webview2_com_sys::Microsoft::Web::WebView2::Win32::GetAvailableCoreWebView2BrowserVersionString;
    use windows_core::{PCWSTR, PWSTR};
    use windows_sys::Win32::System::Com::CoTaskMemFree;
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, IDYES, MB_ICONERROR, MB_ICONWARNING, MB_YESNO,
    };
    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(Some(0)).collect()
    }
    fn message(s: &str, flags: u32) -> i32 {
        unsafe {
            MessageBoxW(
                std::ptr::null_mut(),
                wide(s).as_ptr(),
                wide("Donut Media Center").as_ptr(),
                flags,
            )
        }
    }
    let result = preflight(
        || {
            let mut version = PWSTR::null();
            match unsafe {
                GetAvailableCoreWebView2BrowserVersionString(PCWSTR::null(), &mut version)
            } {
                Ok(()) => {
                    unsafe { CoTaskMemFree(version.0.cast()) };
                    Probe::Present
                }
                Err(e) if e.code().0 as u32 == 0x80070002 => Probe::Missing,
                Err(e) => Probe::Error(format!("WebView2 availability check failed: {}", e)),
            }
        },
        || {
            message("Microsoft Edge WebView2 is required to start Donut Media Center. Open Microsoft's download page? Install the Evergreen Runtime, then relaunch the app.", MB_YESNO | MB_ICONWARNING) == IDYES
        },
        || {
            let result = unsafe {
                ShellExecuteW(
                    std::ptr::null_mut(),
                    wide("open").as_ptr(),
                    wide("https://developer.microsoft.com/microsoft-edge/webview2/").as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1,
                )
            } as isize;
            if result <= 32 {
                Err(format!(
                    "Could not open the WebView2 download page (ShellExecute error {}).",
                    result
                ))
            } else {
                Ok(())
            }
        },
    );
    match result {
        Ok(start) => start,
        Err(error) => {
            message(&error, MB_ICONERROR);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    #[test]
    fn present_starts_without_ui() {
        assert_eq!(
            preflight(|| Probe::<()>::Present, || panic!(), || panic!()),
            Ok(true)
        );
    }
    #[test]
    fn missing_accept_opens_and_exits() {
        let opened = Cell::new(false);
        assert_eq!(
            preflight(
                || Probe::<()>::Missing,
                || true,
                || {
                    opened.set(true);
                    Ok(())
                }
            ),
            Ok(false)
        );
        assert!(opened.get());
    }
    #[test]
    fn missing_cancel_exits_without_opening() {
        assert_eq!(
            preflight(|| Probe::<()>::Missing, || false, || panic!()),
            Ok(false)
        );
    }
    #[test]
    fn probe_error_is_preserved_without_ui() {
        assert_eq!(
            preflight(|| Probe::Error("probe"), || panic!(), || panic!()),
            Err("probe")
        );
    }
    #[test]
    fn open_error_is_preserved() {
        assert_eq!(
            preflight(|| Probe::Missing, || true, || Err("open")),
            Err("open")
        );
    }
}
