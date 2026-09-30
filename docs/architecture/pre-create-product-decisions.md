# DonutMediaCenter: product decisions before Create

Recorded: 2026-09-30. Scope: this conversation up to the request to start the
ComfyUI creation add-on.

This is a durable record of requirements and design discussion, not a completion
report. A screenshot, a user report, or an earlier agent's claim does not establish
that the current build meets a requirement. Features need separate implementation
tasks and acceptance evidence against an identified build.

Part of the original feature request was truncated in the available conversation.
This record preserves the visible requirements and later clarifications; it does
not reconstruct unavailable text or assume every detail of that original plan.

## How to use this record

- **Requirement:** a requested outcome, including the user's later corrections.
- **Reported defect:** behavior the user observed; its cause and resolution still
  need evidence unless a separate task establishes them.
- **Proposal:** an approach discussed without a final implementation decision.
- **Open:** a choice still to make.

The detailed [shared libraries and mobile architecture](shared-libraries-and-mobile.md)
contains an earlier repository/device inventory and implementation proposals. The
later [creator commerce discussion](creator-commerce-and-offline-delivery.md)
records paid access, permissions, protection limits, and offline fulfillment.
This document is the index connecting those plans with the gallery requirements.

## 1. Product identity and library boundaries

**Requirements**

- Rebrand the application to **DonutMediaCenter**. Keep LocalBooru references where
  compatibility or historical identity requires them; decide deployment naming
  separately from the user-facing app name.
- Provide dedicated **Images**, **Videos**, and **Music** experiences. Retain the
  familiar masonry gallery and lightbox, adapted to each media type.
- Searches, filters, collections, counts, directory lists, and viewer controls
  belong to the active media section. Unrelated media and controls stay out.
- Each section remembers its search, filters, selected collection/source, and
  scroll position. Switching away and returning restores the user's place.
- Music **Albums** and **Songs** are views of the same library. They share
  organization and favorites, without duplicate collections.
- An album is a release with ordered tracks. A music collection is a user-created
  grouping of albums and/or individual tracks.
- Collections are accessible from each gallery's sidebar. They are scoped to that
  gallery; there is no separate global Collections navigation destination.
- Existing organization must survive the separation. Preserve membership and
  favorites; scope the presentation of existing mixed collections rather than
  silently discarding their other media.

### Relevant filters

| Gallery | Requested filters |
| --- | --- |
| Images | Tags, ratings, favorites, folders, dimensions, orientation |
| Videos | Relevant tags, ratings, favorites, folders, duration, resolution, watched status |
| Music | Artist, album, genre, year, favorites, folders |

Offer filters where the required metadata exists. Keep image-specific tag
categories and image processing tools out of Videos. Keep video-specific filters
out of Images. This does not remove useful video tags or ratings.

**Acceptance:** switching galleries changes both the media and the relevant
tools, and returning restores the previous browsing position.

## 2. Navigation and sidebar layout

**Requirements and corrections**

- The media switcher appears in the sidebar, with the same item spacing, padding,
  and visibility whether Images, Videos, or Music is selected. Remove the duplicate
  media switcher above the gallery.
- The latest navigation direction reduces six destinations to four:
  **Images | Videos | Music | Settings**.
- Directories becomes the default opening page within Settings because library
  setup is essential. It remains easy to reach.
- Music's Albums/Songs switch, search, and result count move into the sidebar.
  Keep gallery browsing controls together instead of splitting them across a
  top bar and sidebar.
- Show the version badge in the titlebar after the app title.
- Show Support in the sidebar footer only while the persistent media controls
  are not visible.
- Provide a GNOME desktop application-menu launcher for the installed app.

The earlier request for consistent five/six-item navigation was superseded by the
four-destination direction. The consistency requirement remains.

### Directory settings overhaul

**Requirement:** separate option types rather than presenting scanning, content
types, sharing flags, and destructive maintenance as one flat row of toggles.

The existing architecture proposes compact directory rows with grouped details:

| Group | Contents |
| --- | --- |
| Media | Images, Videos, Music, including clear selection when adding a directory |
| Scanning and metadata | Indexing, tagging/processing applicable to that media, progress |
| Access and sharing | Local/LAN availability, audience, explicit publication choices |
| Maintenance | Rescan, repair, prune, path changes, errors |
| Removal | Clearly described removal from the library |

Retain library selection and batch operations. SVG icons were suggested; the final
visual treatment remains a design choice. Family/content policy and access audience
are separate concepts. Reorganizing settings must not broaden existing sharing.

**Reported defect:** navigating into Settings and back feels poor, especially on
Android. The architecture proposes route-backed sections and Back behavior that
returns to the originating gallery with browsing state and music intact.

## 3. Directory scope and indexing

**Requirements**

- Directory media selection includes Music, not just Images and Videos.
- A music-only directory is absent from Images and Videos directory lists.
  Likewise, image-only and video-only directories appear only where applicable.
- Cover images in music-only directories serve as artwork; they must not be
  indexed into Images merely because they are image files.
- Generate video thumbnails when adding/scanning video-enabled directories.
- An unavailable or unmounted source must not contribute playable local results
  as if its files were available. Other available sources continue to work.

**Reported defect:** video thumbnails were not generated after adding directories.

**Reported defect:** many `GET /images/...` 404 error toasts appeared while visible
media still worked. The user suggested stale database records or unmounted devices
as possible causes; neither is established. Investigate availability filtering,
source/library-qualified IDs, and stale requests during navigation. Expected stale
or cancelled lookups should not produce repeated global error notifications; real
operation failures still need useful feedback.

## 4. Music listening and discovery

**Requirements**

- Starting an album plays its tracks in release order.
- Starting an individual song starts a related mix from the local library.
- Provide an album related-shuffle control and automatic related continuation
  after album playback.
- After any required initial setup, local playback and recommendations work
  offline without an online music service.
- Make upcoming recommendations visible. Explicitly queued tracks take priority
  over automatic additions.
- When there are no suitable further matches, explain that the related queue is
  exhausted. Do not label unrelated random tracks as similar.
- Closing the music lightbox keeps playback running. Switching galleries or
  Albums/Songs views does not stop music or replace the queue.
- Browsing another album does not change playback until an explicit play action.
- Starting a video pauses music to prevent overlapping audio. Merely opening
  Videos does not pause music.

The exact recommendation algorithm and metadata/analysis requirements remain
open. The product outcome is relevant continuation, with recommendation quality
verified on representative local material rather than assumed from a queue existing.

### Persistent player

**Requirement:** put the compact player at the bottom of the sidebar with visible
album artwork, current track and artist, seek/time display, previous/play/next,
and volume. Artwork should remain meaningful at this size; the reference was a
cover above compact controls, not a tiny floating player elsewhere in the window.

Selecting the player reopens the active album or mix. Opening an unrelated album
for browsing must not relabel or replace the active playback session.

**Icon correction:** the center play **and pause** icon on the green primary button
should be black. This request does not change neighboring controls to white. The
user repeatedly reported the center icon staying white, including after updates;
acceptance needs the rendered player in both states, not only a stylesheet edit.

### Album tiles and artwork

**Requirements**

- Put a small, well-presented Play button on the **right** of the album tile
  footer, using the space there. It starts the album without opening the lightbox.
  Its click must not also trigger the tile's open action.
- Resolve covers for albums, songs, and singles from available local artwork.
  Inspect embedded artwork and release-local image files as candidates.
- Use image shape as a useful disambiguation signal: when several plausible
  images exist and only one is square, it is probably the album cover.
- Do not reuse a generic ancestor-folder image across unrelated releases just
  because it is the nearest available image. Album/release association must take
  priority over a convenient fallback; shape alone does not establish identity.
- Represent an unresolved cover honestly rather than choosing unrelated art.

**Reported defects:** missing art on numerous releases and identical unrelated
art assigned to several releases. The user permitted inspecting the music library
for diagnosis. Keep that inventory and the media itself outside the repository;
use synthetic fixtures to document or test selection rules.

## 5. Gallery interaction and scrolling

**Requirements and reported defects**

- Music uses infinite scrolling like the other galleries, replacing Load more.
- Keep scrolling smooth, including while fetching and appending another page.
  The user reported visible jank at this boundary; measure the rendered interaction
  before declaring it fixed.
- Clicking another dropdown closes the open dropdown and opens the selected one
  in a single click. The two-click defect was observed in Music filters; other
  consumers of the same dropdown behavior should be considered during a fix.
- Clicking outside a lightbox panel closes it; Close is not the only exit action.
- A minus prefix excludes a tag from search, for example `-landscape`. Preserve
  positive tag search and explain mixed include/exclude behavior in its feature
  specification. Exact tokenization/quoting rules have not been decided here.

### Viewer close behavior

| Action | Required outcome |
| --- | --- |
| Close image lightbox | Return to the same gallery position |
| Close video lightbox | Save playback position and stop video |
| Close music lightbox | Continue through the persistent sidebar player |

## 6. Video and SVP reliability

**Reported symptoms**

- With SVP enabled, seeking or pausing during initial startup/buffering could
  freeze the **entire window**, including Escape/Close. The user associated this
  with SVP and confirmed trying playback with SVP disabled.
- SVP sometimes worked for several videos, then silently failed to connect on
  later opens and fell back to regular playback.
- Seeking could briefly play the SVP stream, switch to the regular stream at
  `00:00`, then jump back to the intended playhead with SVP.
- Seeking restrictions added as a workaround made seeking worse without
  eliminating the freezes.
- A thumbnail grid flashed briefly over a full-size pixelated thumbnail, and a
  poster could remain on top of video that was already playing behind it.

These are observations, not a proven diagnosis of the worker, transport, or decoder.

**Requirements**

- When SVP is already enabled at open, initialize that playback path directly.
  Avoid starting regular playback and switching to SVP several seconds later.
- Keep the UI responsive throughout initialization, seeking, cancellation, and
  failure. The user must be able to disable SVP or close the viewer while it starts.
- Show connection/startup failure and recovery actions. Do not silently report
  SVP playback while falling back to the regular stream.
- Keep seeking fast and preserve the requested position through stream changes.
  Do not restart regular playback at zero during an SVP seek.
- During loading, use a **grid of generated video thumbnails in place of** the
  large poster. It is not an overlay that briefly appears over the same poster.
- Do not begin hidden playback behind the loading artwork. Remove the loading
  grid/poster when the playable video frame is presented.
- Show the video title in the top-left while viewer UI is visible; hide it with
  the rest of the controls.

Acceptance must include repeated opens, early seek/pause, failed SVP startup,
disable/cancel while pending, repeated seeks, and close/reopen. A single successful
startup is insufficient evidence for the reported failures.

Related engineering records: [SVP roadmap](../svp-native-platform-roadmap.md),
[native video architecture](../NATIVE_VIDEO_ARCHITECTURE.md), and
[native video matrix](../NATIVE_VIDEO_TEST_MATRIX.md). This record does not replace
their platform findings or declare their work complete.

## 7. Mobile and desktop distribution

**Requirements**

- Verify the music library and listening experience in the Android APK.
- Verify iPhone feasibility and support, including whether an app/build path
  exists; do not infer accepted iPhone playback from desktop support.
- Use an Android VM/emulator to examine Settings, Back navigation, and music
  behavior. The user mentioned available server tooling and an SSH-accessible Mac.
- Provide desktop launch integration, including the requested GNOME menu entry.

The existing architecture records a read-only tooling inventory and proposes
storage/import bridges, native background audio, range/codec handling, playback
restoration, and source-scoped authorization. Those are engineering proposals and
acceptance work, not proof that the APK or iPhone experience works today.

Test exact artifacts using disposable profiles and synthetic media. Include local
and paired-source playback, album order, related queues, covers, seeking, Settings
round trips, Back, screen lock, interruptions, network loss, and restoration.
Real-device acceptance remains separate from simulator evidence.

## 8. Connected private libraries and creator collections

**Requirements and the final clarification**

- Existing QR-authorized connections primarily represent the user's own devices
  and private installations. Append their directories after local directories in
  the **private libraries** list, retaining source labels and media scoping.
- Keep multiple servers connected simultaneously. Browsing a remote directory
  must not require switching the whole app's backend in Settings.
- Public/global creator content has its **own collection list**, toggled from
  the gallery sidebar. Invite-only creator collections also belong to this sharing
  relationship, not the own-device administration relationship.
- The initial Local/Remote toggle idea was refined: physical remoteness does not
  define the boundary. Private paired libraries include remote devices.
- Creator collections form one unified list across added creators by default;
  a creator filter narrows that list. Preserve each creator's organization.
- Users can publish selected content directly from the app, with a QR code or
  link for connection. Installing a separate `donutbooru-node` must not be required
  merely to share a collection.
- Remote creator catalogs are read-only to viewers. Authors manage their content;
  viewer favorites, annotations, and personal collections remain viewer-owned.
- Make shared content and local copies clearly distinguishable. Adding a source
  does not silently import its content or transfer ownership.
- **Servers** manages remote installations. It is not a synonym for fediverse.
- Recover and reconcile the earlier DonutBooru integration with the media-library
  work. The user reported that its UI disappeared from the current build.

Public/global and invitation-only publication, streaming, downloading, and serving
copies are separate permissions. A private device-pairing grant must not become a
creator invitation with owner/admin access.

The [existing architecture](shared-libraries-and-mobile.md) details source-qualified
identities, independent clients/credentials, compatibility routes, collection
revisions, and migration away from the current global server switch.

## 9. What brings the network together

**Product direction:** creators publish collections; consumers discover or receive
invitations, stream according to the offer, and optionally buy/save permitted copies.
Authorized hosts can keep content available while its creator is offline. The app
and an optional public web/headless deployment use the same catalog semantics.

**Proposals and open choices**

- Use stable creator/publication identities, signed collection revisions, and
  exact representation hashes. A SHA-256 match verifies bytes, not authorship,
  authority, or the right to distribute them.
- Keep creator/catalog discovery distinct from finding a byte-serving provider.
  Peer-of-peer discovery through known nodes and multiple bootstrap options was
  requested; no custom transport or mandatory global directory was chosen.
- Find identical, authorized replicas when the author is offline. A replica may
  serve only under the creator's applicable permissions, not simply because it
  has the same hash.
- Friends' nodes and optional always-on hosts can retain permitted media. Broader
  discovery comes after source identity, authorization, and replication work.
- Evaluate zrok for reachability, alongside LAN/Tailscale/HTTPS and suitable P2P
  stacks. A tunnel, discovery index, and retained file host are different roles.
- A Donut application protocol for catalogs, invitations, permissions, and
  providers is plausible; build on established secure networking rather than
  assuming a new routing/cryptographic stack is necessary.
- ActivityPub/social interoperability is a possible adapter, not an already chosen
  solution for payments, full media delivery, or hash-based replica discovery.
- A public website/always-on deployment could serve as an alternative place for
  creator releases and video publishing. Settle naming and deployment roles before
  assuming LocalBooru, DonutBooru, and the node are interchangeable.

The current DonutBooru federation pilot is narrower: signed allowlisted HTTPS
reference events and explicitly permitted materialization, with open discovery
deferred. The new product direction requires migration/integration work; it is
not evidence that the pilot already implements the wider network.

## 10. Paid collections, copying, and offline purchases

The later [commerce record](creator-commerce-and-offline-delivery.md) preserves:

- creator-defined previews, full-access and full-collection offers;
- independent streaming, download, friend-sharing, and host permissions;
- buying from the creator while downloading a permitted copy from another host;
- crypto payment/receipt ideas with media kept off-chain;
- durable offers, entitlement proofs, and offline key/license fulfillment;
- protection limits in an editable open-source client and SVP compatibility.

The product permission boundary follows the creator's declared policy. The app
cannot establish authorship merely from an upload or eradicate fraudulent claims.
No blanket legal conclusion about all private friend sharing was adopted.

No payment chain, NFT scheme, DRM provider, or P2P stack was selected. Do not treat
these ideas as implemented features or prerequisites for the Create add-on.

## 11. Delivery and traceability

The original suggested milestones were:

1. Dedicated media browsing: scoped Images/Videos, collections, viewer controls.
2. Music library and player: Albums/Songs, artwork, track lists, persistent listening.
3. Related listening: song mixes, related shuffle, album continuation, visible queue.

Later work adds navigation/settings, attached private sources, mobile acceptance,
direct creator sharing, authorized replicas, wider discovery, and commerce. The
[architecture delivery sequence](shared-libraries-and-mobile.md#delivery-sequence-and-completion-evidence)
is a proposal for dependencies, not a statement that these milestones shipped.

Before implementing each slice, reconcile the relevant existing specs and tasks,
then give it observable acceptance criteria and exact build evidence. Keep reported
defects in dedicated tasks rather than marking this discussion record as a fix.

The user requested orchestrated work, `main` as integration branch, Mac pairing on
`main`, and task worktrees created from `main`. Product work is not finished solely
because it exists on an unmerged branch. Preserve unrelated and unmerged work;
publishing or pushing remains a separately authorized operation.

### Existing plans and repositories

- This repository: the shared-libraries/mobile architecture linked above, Online
  provider specs and earlier foundations plans, music/Android tasks, and video/SVP
  records. Audit current task state before deriving new work.
- `donutbooru`: coordination/reference checkout; its repository instructions point
  to `booru-node-clean` for the live website source.
- `donutbooru-node-fediverse`: federation candidate and
  `docs/architecture/federation-protocol-decision.md`, plus media visibility and
  deployment records. Its pilot scope must not be mistaken for open federation.

The unrelated progress-dashboard attachment was explicitly identified as belonging
to another session and is excluded. ComfyUI/DonutUI/Create installation, workflow,
model setup, and implementation status belong to the subsequent creation-add-on
task, outside this record.
