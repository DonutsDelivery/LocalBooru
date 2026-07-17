# LocalBooru security patch

This directory contains the published `wayland-scanner` 0.31.10 crate source from
crates.io (MIT license).

LocalBooru applies the upstream dependency migration needed to address
RUSTSEC-2026-0194 and RUSTSEC-2026-0195:

- `quick-xml` is updated from 0.39 to 0.41.
- `ByteRef::xml_content()` is replaced with `ByteRef::xml10_content()`.

These changes match the relevant upstream changes in Smithay/wayland-rs commits
`ec2d932855593d48aa83c76820f3efbcfea86d39` and
`d07c4f91f28b42e5a485823ffd9d8d5a210b1053`, without including unrelated
unreleased wayland-rs API changes.
