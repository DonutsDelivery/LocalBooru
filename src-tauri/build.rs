fn main() {
    // Tauri's dependency build script can remain cached across worktrees. Ensure
    // its generated activity is also present in this app's Android project.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        // Align both LOAD segments and the RELRO boundary for 16KB devices.
        println!("cargo:rustc-link-arg=-Wl,-z,max-page-size=16384");
        println!("cargo:rustc-link-arg=-Wl,-z,common-page-size=16384");
        for key in [
            "WRY_ANDROID_KOTLIN_FILES_OUT_DIR",
            "WRY_ANDROID_PACKAGE",
            "WRY_ANDROID_LIBRARY",
        ] {
            println!("cargo:rerun-if-env-changed={key}");
        }
        if let Ok(output) = std::env::var("WRY_ANDROID_KOTLIN_FILES_OUT_DIR") {
            let library_path = std::env::var("DEP_TAURI_ANDROID_LIBRARY_PATH")
                .expect("missing Tauri Android library path");
            let source = std::path::Path::new(&library_path)
                .parent()
                .expect("invalid Tauri mobile directory")
                .join("android-codegen");
            let package = std::env::var("WRY_ANDROID_PACKAGE").expect("missing Android package");
            let library = std::env::var("WRY_ANDROID_LIBRARY").expect("missing Android library");
            std::fs::create_dir_all(&output).expect("create Android codegen directory");
            for file in std::fs::read_dir(source).expect("read Tauri Android codegen") {
                let file = file.expect("read Android template entry");
                println!("cargo:rerun-if-changed={}", file.path().display());
                let content = std::fs::read_to_string(file.path())
                    .expect("read Android template")
                    .replace("{{package}}", &package)
                    .replace("{{library}}", &library);
                let target = std::path::Path::new(&output).join(file.file_name());
                if std::fs::read_to_string(&target).ok().as_deref() != Some(content.as_str()) {
                    std::fs::write(&target, content).expect("write Android template");
                }
                println!("cargo:rerun-if-changed={}", target.display());
            }
        }
    }
    tauri_build::build()
}
