"""
LocalBooru Bootstrap Script
Sets up persistent Python packages and model downloads.

This script handles:
1. Adding persistent package directory to sys.path
2. Installing missing dependencies on first run
3. Setting up model download paths
"""
import sys
import os
from pathlib import Path


def get_data_dir() -> Path:
    """Get LocalBooru data directory."""
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:  # Linux/Mac
        base = Path.home()
    return base / '.localbooru'


def get_packages_dir() -> Path:
    """Get persistent packages directory."""
    return get_data_dir() / 'packages'


def setup_paths():
    """Add persistent packages directory to sys.path."""
    packages_dir = get_packages_dir()
    packages_dir.mkdir(parents=True, exist_ok=True)

    # Add to beginning of path so it takes precedence
    packages_path = str(packages_dir)
    if packages_path not in sys.path:
        sys.path.insert(0, packages_path)

    # Also set PYTHONPATH for subprocesses
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if packages_path not in current_pythonpath:
        os.environ['PYTHONPATH'] = f"{packages_path}{os.pathsep}{current_pythonpath}"

    return packages_dir


def check_core_dependencies() -> bool:
    """Check if core dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import PIL
        return True
    except ImportError:
        return False


def get_required_packages() -> list[str]:
    """Get list of required pip packages."""
    return [
        'fastapi>=0.115.0',
        'uvicorn[standard]>=0.32.0',
        'sqlalchemy>=2.0.0',
        'aiosqlite>=0.20.0',
        'pillow>=10.0.0',
        'imagehash>=4.3.0',
        'onnxruntime>=1.18.0',
        'opencv-python-headless>=4.8.0',
        'numpy>=1.24.0,<2',  # Pin numpy <2 for compatibility
        'pydantic>=2.0.0',
        'pydantic-settings>=2.0.0',
        'python-multipart>=0.0.9',
        'httpx>=0.27.0',
        'watchdog>=4.0.0',
    ]


def install_dependencies(packages_dir: Path, progress_callback=None) -> bool:
    """
    Install dependencies to the persistent packages directory.

    Args:
        packages_dir: Target directory for packages
        progress_callback: Optional callback(message, percent)

    Returns:
        True if successful
    """
    import subprocess

    packages = get_required_packages()
    total = len(packages)

    if progress_callback:
        progress_callback("Checking dependencies...", 0)

    # Use pip to install to target directory
    for i, package in enumerate(packages):
        if progress_callback:
            progress_callback(f"Installing {package.split('>=')[0]}...", int((i / total) * 100))

        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install',
             '--target', str(packages_dir),
             '--upgrade', '--no-warn-script-location',
             package],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: Failed to install {package}: {result.stderr}")

    # Try to install insightface separately (may fail without C++ tools)
    if progress_callback:
        progress_callback("Installing face detection (optional)...", 95)

    try:
        # Try pre-built wheel first (Windows)
        if os.name == 'nt':
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install',
                 '--target', str(packages_dir),
                 '--no-warn-script-location',
                 'insightface',
                 '--find-links', 'https://github.com/abetlen/insightface-wheels/releases/latest'],
                capture_output=True
            )
        else:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install',
                 '--target', str(packages_dir),
                 '--no-warn-script-location',
                 'insightface>=0.7.3'],
                capture_output=True
            )
    except Exception:
        pass  # Face detection is optional

    if progress_callback:
        progress_callback("Setup complete!", 100)

    return True


def get_installed_version(packages_dir: Path) -> str:
    """Get version of installed packages (for update detection)."""
    version_file = packages_dir / '.localbooru_version'
    if version_file.exists():
        return version_file.read_text().strip()
    return ''


def set_installed_version(packages_dir: Path, version: str):
    """Set version marker for installed packages."""
    version_file = packages_dir / '.localbooru_version'
    version_file.write_text(version)


def needs_update(packages_dir: Path, current_version: str) -> bool:
    """Check if packages need to be updated."""
    installed = get_installed_version(packages_dir)
    return installed != current_version


def bootstrap(app_version: str = "0.1.30", force_reinstall: bool = False):
    """
    Main bootstrap function.
    Should be called before importing any other modules.

    Args:
        app_version: Current app version for update detection
        force_reinstall: Force reinstall even if packages exist
    """
    # Setup paths first
    packages_dir = setup_paths()

    # Check if we need to install/update
    if force_reinstall or not check_core_dependencies():
        print(f"[LocalBooru] Installing dependencies to {packages_dir}...")
        install_dependencies(packages_dir)
        set_installed_version(packages_dir, app_version)
    elif needs_update(packages_dir, app_version):
        print(f"[LocalBooru] Updating dependencies...")
        install_dependencies(packages_dir)
        set_installed_version(packages_dir, app_version)

    return packages_dir


# Auto-run when imported
if __name__ != '__main__':
    # Only run in packaged mode (not development)
    if getattr(sys, 'frozen', False) or os.environ.get('LOCALBOORU_PACKAGED'):
        setup_paths()
