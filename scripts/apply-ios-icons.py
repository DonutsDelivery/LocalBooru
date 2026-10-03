#!/usr/bin/env python3
"""Apply the committed opaque DMC icons after Tauri creates the iOS project."""
import json
from pathlib import Path
import shutil
import struct

root = Path(__file__).resolve().parents[1]
source = root / 'src-tauri/icons/ios'
catalog = root / 'src-tauri/gen/apple/Assets.xcassets/AppIcon.appiconset'
entries = json.loads((catalog / 'Contents.json').read_text())['images']
copies = []
for entry in entries:
    name = entry.get('filename')
    if not name or Path(name).name != name:
        raise SystemExit('iOS icon catalog has an unexpected filename')
    icon = source / name
    data = icon.read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n' or data[25] != 2:
        raise SystemExit(f'{name}: expected an opaque RGB PNG')
    dimensions = struct.unpack('>II', data[16:24])
    expected = round(float(entry['size'].split('x')[0]) * float(entry['scale'].rstrip('x')))
    if dimensions != (expected, expected):
        raise SystemExit(f'{name}: icon dimensions do not match the catalog')
    copies.append((icon, catalog / name))
if not any(icon.name == 'AppIcon-512@2x.png' for icon, _ in copies):
    raise SystemExit('iOS catalog is missing the store icon')
for icon, target in copies:
    shutil.copyfile(icon, target)
print(f'Applied {len(copies)} opaque DMC iOS icons')
