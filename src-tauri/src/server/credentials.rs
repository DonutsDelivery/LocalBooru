use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;

const CREDENTIAL_DIR: &str = ".credentials";
const JWT_SECRET_FILE: &str = "jwt-signing-secret";

trait SecretStore {
    fn load(&self) -> io::Result<Option<String>>;
    fn store(&self, secret: &str) -> io::Result<()>;
}

struct PlatformSecretStore {
    path: PathBuf,
}

impl PlatformSecretStore {
    fn new(data_dir: &Path) -> Self {
        Self {
            path: data_dir.join(CREDENTIAL_DIR).join(JWT_SECRET_FILE),
        }
    }
}

impl SecretStore for PlatformSecretStore {
    fn load(&self) -> io::Result<Option<String>> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let plaintext = unprotect_secret(&bytes)?;
        String::from_utf8(plaintext)
            .map(Some)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    fn store(&self, secret: &str) -> io::Result<()> {
        let protected = protect_secret(secret.as_bytes())?;
        write_private_atomic(&self.path, &protected)
    }
}

pub(crate) fn load_or_generate_jwt_secret(data_dir: &Path) -> io::Result<String> {
    load_or_migrate_jwt_secret(data_dir, &PlatformSecretStore::new(data_dir))
}

fn load_or_migrate_jwt_secret(data_dir: &Path, store: &dyn SecretStore) -> io::Result<String> {
    let settings_path = data_dir.join("settings.json");
    let mut settings = load_settings_object(&settings_path)?;
    let legacy_secret = settings
        .as_ref()
        .and_then(|value| value.get("jwt_secret"))
        .and_then(Value::as_str)
        .filter(|secret| !secret.is_empty())
        .map(str::to_owned);

    if let Some(secret) = legacy_secret {
        store.store(&secret)?;
        verify_stored_secret(store, &secret)?;
        remove_legacy_secret(&settings_path, &mut settings)?;
        return Ok(secret);
    }

    if let Some(secret) = store.load()?.filter(|secret| !secret.is_empty()) {
        remove_legacy_secret(&settings_path, &mut settings)?;
        return Ok(secret);
    }

    let secret = generate_jwt_secret();
    store.store(&secret)?;
    verify_stored_secret(store, &secret)?;
    remove_legacy_secret(&settings_path, &mut settings)?;
    Ok(secret)
}

fn remove_legacy_secret(path: &Path, settings: &mut Option<Value>) -> io::Result<()> {
    let Some(settings) = settings.as_mut() else {
        return Ok(());
    };
    let removed = settings
        .as_object_mut()
        .expect("settings object was validated")
        .remove("jwt_secret")
        .is_some();
    if removed {
        write_json_atomic(path, settings)?;
    }
    Ok(())
}

fn load_settings_object(path: &Path) -> io::Result<Option<Value>> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let value: Value = serde_json::from_str(&contents)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if !value.is_object() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "settings.json is not a JSON object",
        ));
    }
    Ok(Some(value))
}

fn verify_stored_secret(store: &dyn SecretStore, expected: &str) -> io::Result<()> {
    match store.load()? {
        Some(actual) if actual == expected => Ok(()),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stored signing credential could not be verified",
        )),
    }
}

fn generate_jwt_secret() -> String {
    use rand::Rng;
    let bytes: [u8; 32] = rand::thread_rng().gen();
    bytes.iter().map(|byte| format!("{:02x}", byte)).collect()
}

fn write_json_atomic(path: &Path, value: &Value) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write_atomic(path, &bytes, false)
}

fn write_private_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_atomic(path, bytes, true)
}

fn write_atomic(path: &Path, bytes: &[u8], private: bool) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    fs::create_dir_all(parent)?;
    set_private_directory_permissions(parent, private)?;

    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("secret"),
        uuid::Uuid::new_v4()
    ));
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        set_private_file_options(&mut options, private);
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        replace_file(&temporary, path)?;
        sync_parent_directory(parent);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &Path, private: bool) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if private {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &Path, _private: bool) -> io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_private_file_options(options: &mut OpenOptions, private: bool) {
    use std::os::unix::fs::OpenOptionsExt;
    if private {
        options.mode(0o600);
    }
}

#[cfg(not(unix))]
fn set_private_file_options(_options: &mut OpenOptions, _private: bool) {}

#[cfg(unix)]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(target_os = "windows")]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(source, destination)
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) {
    if let Ok(directory) = fs::File::open(path) {
        let _ = directory.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) {}

#[cfg(not(target_os = "windows"))]
fn protect_secret(secret: &[u8]) -> io::Result<Vec<u8>> {
    Ok(secret.to_vec())
}

#[cfg(not(target_os = "windows"))]
fn unprotect_secret(secret: &[u8]) -> io::Result<Vec<u8>> {
    Ok(secret.to_vec())
}

#[cfg(target_os = "windows")]
fn protect_secret(secret: &[u8]) -> io::Result<Vec<u8>> {
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    let mut input = CRYPT_INTEGER_BLOB {
        cbData: secret.len() as u32,
        pbData: secret.as_ptr() as *mut u8,
    };
    let entropy_bytes = b"LocalBooru JWT signing credential v1";
    let mut entropy = CRYPT_INTEGER_BLOB {
        cbData: entropy_bytes.len() as u32,
        pbData: entropy_bytes.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: null_mut(),
    };
    let success = unsafe {
        CryptProtectData(
            &mut input,
            null(),
            &mut entropy,
            null_mut(),
            null_mut(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if success == 0 {
        return Err(io::Error::last_os_error());
    }
    let protected =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) }.to_vec();
    unsafe {
        LocalFree(output.pbData.cast());
    }
    Ok(protected)
}

#[cfg(target_os = "windows")]
fn unprotect_secret(secret: &[u8]) -> io::Result<Vec<u8>> {
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    let mut input = CRYPT_INTEGER_BLOB {
        cbData: secret.len() as u32,
        pbData: secret.as_ptr() as *mut u8,
    };
    let entropy_bytes = b"LocalBooru JWT signing credential v1";
    let mut entropy = CRYPT_INTEGER_BLOB {
        cbData: entropy_bytes.len() as u32,
        pbData: entropy_bytes.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: null_mut(),
    };
    let success = unsafe {
        CryptUnprotectData(
            &mut input,
            null_mut(),
            &mut entropy,
            null_mut(),
            null_mut(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if success == 0 {
        return Err(io::Error::last_os_error());
    }
    let plaintext =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) }.to_vec();
    unsafe {
        LocalFree(output.pbData.cast());
    }
    Ok(plaintext)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::sync::Mutex;

    #[derive(Debug, Serialize, Deserialize)]
    struct TestClaims {
        sub: String,
        exp: usize,
    }

    #[derive(Default)]
    struct MemoryStore {
        secret: Mutex<Option<String>>,
        fail_store: bool,
    }

    impl SecretStore for MemoryStore {
        fn load(&self) -> io::Result<Option<String>> {
            Ok(self.secret.lock().unwrap().clone())
        }

        fn store(&self, secret: &str) -> io::Result<()> {
            if self.fail_store {
                return Err(io::Error::new(io::ErrorKind::Other, "store failed"));
            }
            *self.secret.lock().unwrap() = Some(secret.to_owned());
            Ok(())
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "localbooru-credentials-{}-{}",
            name,
            uuid::Uuid::new_v4()
        ))
    }

    // AC: @credential-storage ac-1
    #[test]
    fn legacy_secret_migrates_unchanged_and_is_removed_from_settings() {
        let data_dir = temp_dir("migrate");
        fs::create_dir_all(&data_dir).unwrap();
        let settings = data_dir.join("settings.json");
        fs::write(
            &settings,
            r#"{"jwt_secret":"existing","network":{"enabled":true}}"#,
        )
        .unwrap();
        let store = MemoryStore::default();
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::default(),
            &TestClaims {
                sub: "existing-user".into(),
                exp: 4_000_000_000,
            },
            &jsonwebtoken::EncodingKey::from_secret(b"existing"),
        )
        .unwrap();

        let secret = load_or_migrate_jwt_secret(&data_dir, &store).unwrap();

        assert_eq!(secret, "existing");
        jsonwebtoken::decode::<TestClaims>(
            &token,
            &jsonwebtoken::DecodingKey::from_secret(secret.as_bytes()),
            &jsonwebtoken::Validation::default(),
        )
        .unwrap();
        assert_eq!(store.load().unwrap().as_deref(), Some("existing"));
        let sanitized: Value = serde_json::from_slice(&fs::read(&settings).unwrap()).unwrap();
        assert!(sanitized.get("jwt_secret").is_none());
        assert_eq!(sanitized["network"]["enabled"], true);
        let _ = fs::remove_dir_all(data_dir);
    }

    // AC: @credential-storage ac-2
    #[test]
    fn failed_secure_store_leaves_legacy_settings_unchanged() {
        let data_dir = temp_dir("failure");
        fs::create_dir_all(&data_dir).unwrap();
        let settings = data_dir.join("settings.json");
        let original = br#"{"jwt_secret":"existing","other":1}"#;
        fs::write(&settings, original).unwrap();
        let store = MemoryStore {
            fail_store: true,
            ..Default::default()
        };

        assert!(load_or_migrate_jwt_secret(&data_dir, &store).is_err());
        assert_eq!(fs::read(&settings).unwrap(), original);
        let _ = fs::remove_dir_all(data_dir);
    }

    // AC: @credential-storage ac-2
    #[test]
    fn interrupted_cleanup_reuses_stored_legacy_secret() {
        let data_dir = temp_dir("retry");
        fs::create_dir_all(&data_dir).unwrap();
        let settings = data_dir.join("settings.json");
        fs::write(&settings, r#"{"jwt_secret":"existing","other":1}"#).unwrap();
        let store = MemoryStore {
            secret: Mutex::new(Some("existing".into())),
            fail_store: false,
        };

        assert_eq!(
            load_or_migrate_jwt_secret(&data_dir, &store).unwrap(),
            "existing"
        );
        let sanitized: Value = serde_json::from_slice(&fs::read(&settings).unwrap()).unwrap();
        assert!(sanitized.get("jwt_secret").is_none());
        let _ = fs::remove_dir_all(data_dir);
    }

    #[test]
    fn new_secret_is_generated_once_and_reused() {
        let data_dir = temp_dir("new");
        fs::create_dir_all(&data_dir).unwrap();
        let store = MemoryStore::default();

        let first = load_or_migrate_jwt_secret(&data_dir, &store).unwrap();
        let second = load_or_migrate_jwt_secret(&data_dir, &store).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        let _ = fs::remove_dir_all(data_dir);
    }

    // AC: @credential-storage ac-4
    #[cfg(unix)]
    #[test]
    fn unix_credential_storage_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let data_dir = temp_dir("permissions");
        fs::create_dir_all(&data_dir).unwrap();
        load_or_generate_jwt_secret(&data_dir).unwrap();
        let credential_dir = data_dir.join(CREDENTIAL_DIR);
        let credential_file = credential_dir.join(JWT_SECRET_FILE);

        assert_eq!(
            fs::metadata(&credential_dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&credential_file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let _ = fs::remove_dir_all(data_dir);
    }

    // AC: @credential-storage ac-3
    #[cfg(target_os = "windows")]
    #[test]
    fn windows_secret_is_dpapi_encrypted_and_round_trips() {
        let plaintext = b"not-visible-in-file";
        let protected = protect_secret(plaintext).unwrap();
        assert_ne!(protected, plaintext);
        assert_eq!(unprotect_secret(&protected).unwrap(), plaintext);
    }
}
