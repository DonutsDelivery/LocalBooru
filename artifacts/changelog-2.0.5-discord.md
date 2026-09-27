# What's New in LocalBooru 2.0.5

## Image loading reliability

- Fixed a full-image route failure when identical files are imported from multiple paths under one image ID. The read-only route selects a copy matching the imported image version; it keeps library and directory identity checks intact. Thumbnails and full images now both remain readable when an earlier copy changes or disappears.
- Failed images in the Lightbox now show a reason where the server supplies a recognized error, plus a diagnostic HTTP recheck status and image/directory/library IDs. A successful recheck also records the media type and byte length without downloading the full image again. The recheck is labeled separately: it does not establish the status of the original request or prove a decoding failure.
- Diagnostic console messages no longer include token-bearing media URLs or private file paths.

These changes fix the verified duplicate-file route defect and make other image failures diagnosable. The specific Windows user report has not been matched to a failing request or original file, so this release does not claim every cause of that report is resolved.

## Downloads

- Linux: AppImage, DEB, RPM, and portable ZIP.
- Windows: installer and portable ZIP.
- macOS: universal preview DMG and ZIP (ad-hoc signed, not notarized; not stable real-Mac support).
