# Build and verify LocalBooru's unsigned Windows x64 artifacts on native Windows.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Target = 'x86_64-pc-windows-msvc'
$TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Root 'src-tauri\target' }
$env:CARGO_TARGET_DIR = $TargetDir
$ReleaseDir = Join-Path $TargetDir "$Target\release"
$BundleDir = Join-Path $ReleaseDir 'bundle\nsis'
$DistDir = if ($env:LOCALBOORU_DIST_WINDOWS_DIR) { $env:LOCALBOORU_DIST_WINDOWS_DIR } else { Join-Path $Root 'dist-windows' }

if ($env:RUNNER_OS -and $env:RUNNER_OS -ne 'Windows') {
  throw 'Windows artifacts must be built on native Windows'
}

cargo tauri --version
rustc --version
node --version
npm --version
python (Join-Path $Root 'scripts\check-release-version.py')

Remove-Item -Recurse -Force $DistDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $DistDir | Out-Null

npm --prefix (Join-Path $Root 'frontend') ci
npm --prefix (Join-Path $Root 'frontend') test
npm --prefix (Join-Path $Root 'frontend') run build

cargo check --locked --manifest-path (Join-Path $Root 'src-tauri\Cargo.toml')
Push-Location $Root
try {
  cargo tauri build --ci --target $Target --bundles nsis
} finally {
  Pop-Location
}

git -C $Root diff --exit-code -- Cargo.lock

$Binary = Join-Path $ReleaseDir 'localbooru.exe'
$Installer = Get-ChildItem -Path $BundleDir -Filter '*.exe' -File | Select-Object -First 1
if (-not (Test-Path $Binary -PathType Leaf)) { throw "Missing standalone executable: $Binary" }
if (-not $Installer) { throw "Missing NSIS installer under $BundleDir" }

$BinaryBytes = [System.IO.File]::ReadAllBytes($Binary)
$Magic = $BinaryBytes[0..1]
if ($Magic[0] -ne 0x4d -or $Magic[1] -ne 0x5a) { throw 'Standalone is not a PE executable' }
$PeOffset = [System.BitConverter]::ToInt32($BinaryBytes, 0x3c)
$Machine = [System.BitConverter]::ToUInt16($BinaryBytes, $PeOffset + 4)
if ($Machine -ne 0x8664) { throw "Standalone PE machine is not x64: 0x$($Machine.ToString('x4'))" }

$Signature = Get-AuthenticodeSignature $Binary
Write-Host "Standalone Authenticode status: $($Signature.Status)"
$InstallerSignature = Get-AuthenticodeSignature $Installer.FullName
Write-Host "Installer Authenticode status: $($InstallerSignature.Status)"

$PortableDir = Join-Path $env:RUNNER_TEMP 'LocalBooru-Windows'
Remove-Item -Recurse -Force $PortableDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $PortableDir | Out-Null
Copy-Item $Binary (Join-Path $PortableDir 'LocalBooru.exe')
Copy-Item (Join-Path $Root 'LICENSE') $PortableDir

$Zip = Join-Path $DistDir 'LocalBooru-Windows.zip'
Compress-Archive -Path (Join-Path $PortableDir '*') -DestinationPath $Zip -CompressionLevel Optimal
Copy-Item $Installer.FullName (Join-Path $DistDir 'LocalBooru-Windows-Setup.exe')

$ExtractDir = Join-Path $env:RUNNER_TEMP 'LocalBooru-Windows-verify'
Remove-Item -Recurse -Force $ExtractDir -ErrorAction SilentlyContinue
Expand-Archive -Path $Zip -DestinationPath $ExtractDir
if (-not (Test-Path (Join-Path $ExtractDir 'LocalBooru.exe') -PathType Leaf)) { throw 'Portable ZIP is missing LocalBooru.exe' }
if (-not (Test-Path (Join-Path $ExtractDir 'LICENSE') -PathType Leaf)) { throw 'Portable ZIP is missing LICENSE' }

$Forbidden = @('/home/user', '/mnt/storage', '/build/worktree', '/source/', 'C:\a\LocalBooru\LocalBooru')
$BinaryText = [System.Text.Encoding]::ASCII.GetString($BinaryBytes)
foreach ($Needle in $Forbidden) {
  if ($BinaryText.Contains($Needle)) { throw "Standalone contains forbidden build path: $Needle" }
}

$Artifacts = @(
  (Join-Path $DistDir 'LocalBooru-Windows-Setup.exe'),
  $Zip
)
$Manifest = Join-Path $DistDir 'SHA256SUMS-Windows'
$Lines = foreach ($Artifact in $Artifacts) {
  $Hash = (Get-FileHash -Algorithm SHA256 $Artifact).Hash.ToLowerInvariant()
  "$Hash  $([System.IO.Path]::GetFileName($Artifact))"
}
$ManifestText = ($Lines -join "`n") + "`n"
[System.IO.File]::WriteAllText($Manifest, $ManifestText, [System.Text.UTF8Encoding]::new($false))

foreach ($Line in Get-Content $Manifest) {
  $Parts = $Line -split '  ', 2
  $Actual = (Get-FileHash -Algorithm SHA256 (Join-Path $DistDir $Parts[1])).Hash.ToLowerInvariant()
  if ($Actual -ne $Parts[0]) { throw "Checksum mismatch for $($Parts[1])" }
}

Write-Host 'Windows x64 artifacts verified:'
Get-ChildItem -Path $DistDir -File | Sort-Object Name | ForEach-Object {
  Write-Host "  $($_.Name) ($($_.Length) bytes)"
}
