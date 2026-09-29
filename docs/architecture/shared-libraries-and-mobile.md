# Shared libraries, publishing, and mobile playback

Date: 2026-09-29

Status: architecture proposal based on repository and device inventory. The
navigation requirements below come from the user. Protocol selection and the
delivery sequence are recommendations. This document records planned acceptance;
it does not claim mobile playback or the sharing architecture has passed it.

## Product direction

DonutMediaCenter is a personal media library, player, and optional publisher.
People browse Images, Videos, and Music, connect to creators' shared libraries,
and explicitly choose what they publish from their own library.

DonutBooru is the proposed optional public website and always-on hosting role. Its
existing profiles, storage, and explicitly permitted peer copies provide foundations
for this role; DMC catalogs, audio, and mirror failover need integration and fresh
verification. Both products should expose the same catalog contract.
A creator should be able to share directly from the desktop app without installing
a separate node. Internet connections may still require discovery or relay
infrastructure, and staying available while the creator is offline requires a
host that retains the authorized media.

Hosting starts disabled. Adding a local directory does not publish it. Publication
exports only selected media and approved metadata, excluding local filesystem paths
and private library details. Mobile clients should support listening, browsing, and
offline copies; continuous hosting cannot depend on a suspended phone app.

### Navigation

There are four primary destinations: **Images | Videos | Music | Settings**.

- Settings opens **Directories** by default. Local directories remain immediately
  accessible; shared sources and hosting have their own settings sections.
- Each media sidebar has **Local** and **Shared** source groups. Only sources with
  the appropriate media capability appear. A connected source behaves like a
  browsable folder while retaining its remote identity and read-only controls.
- Shared items show their creator/source and availability. A combined view is an
  explicit choice; connecting a source does not silently add its media to Local.
- Servers continues to mean administration of remote app installations. Connected
  shared libraries, publishing destinations, and server administration are
  separate operations with separate credentials.
- Settings sections and directory details use routes. Back closes the top overlay
  first, then returns from detail to list, and from Settings to the originating
  gallery with its filters, scroll position, and playback intact.
- Existing `/directories` URLs redirect to their corresponding Settings route.
  Legacy `/online` Browse remains functional and reachable from Settings until
  equivalent Shared gallery browsing ships; only then redirect it. Existing
  connections, publication history, and settings remain reachable.

### Directory settings

Use a compact list showing name, path, enabled media types, availability, counts,
and a labeled Configure action. Open grouped details instead of showing every
toggle and maintenance action on every row.

| Group | Contents |
| --- | --- |
| Media | Images, Videos, Music; clear selection when adding a directory |
| Scanning and metadata | Scan behavior and only the processing tools applicable to its media |
| Access and sharing | Local/LAN access, content policy, and explicit publication choices |
| Maintenance | Rescan, repair, prune, path changes, and progress/errors |
| Removal | Remove from library, with the actual effect clearly stated |

Retain batch operations and library management. Use consistent SVG icons with
labels and accessible names. Family/content policy is distinct from who can access
media. Existing Public/LAN flags need a policy audit before migration; never widen
access as a side effect of reorganizing the screen. Publishing should preview the
selected catalog and permissions before enabling access.

## Existing foundations and gaps

The audit covered DMC `main` at `4b24ab1` and the available DonutBooru worktrees.
Repository records describe prior acceptance, not a new observation of live
production behavior.

| Area | Evidence and implication |
| --- | --- |
| DMC navigation | `frontend/src/components/SidebarNavigation.jsx` has three media entries plus Online, Directories, and Settings. Settings and directory Back buttons navigate to `/`; settings tabs use transient component state. |
| DMC remote content | `src-tauri/src/routes/online.rs` and `online/models.rs` provide connections, imports/provenance, capability flags, and publication records. Remote models still need reliable media classification and music album/track semantics. |
| Existing publishing UI | `frontend/src/pages/OnlinePage.jsx` prompts for a numeric image ID; publication handling assumes the primary library. Publishing needs complete library/media identity and selection from the galleries. |
| Older DMC plans | The August main-app online foundations plan and `online-content-providers` kspec deliberately keep Online separate and defer app-hosted federation. The July provider research correctly protects remote records from local mutation routes. |
| DonutBooru node | `docs/architecture/federation-protocol-decision.md` defines signed, allowlisted HTTPS events and explicitly permitted verified media copies. It defers ActivityPub and open discovery. |
| Node hosting | The August node execution record describes configurable instances, local/S3/Bunny storage, representation checksums, scoped publication credentials, revisions, receipts, and tombstones. These are reusable building blocks. |
| Node music | `api/upload_security.py` validates images and videos; a music catalog and audio validation are still required. Existing image collections cannot substitute for releases with ordered discs/tracks. |
| Repository ownership | `donutbooru/AGENTS.md` identifies that checkout as coordination/reference and `booru-node-clean` as the live website source. The federation worktree is a candidate with independent changes. Integration must reconcile these before deployment. |

The new direction changes the old Online product boundary. Preserve separate
identity, authorization, and explicit import internally while presenting shared
sources inside the media galleries. Existing specs must be reconciled when this
proposal becomes implementation scope.

## Common source and catalog model

Keep four concepts explicit:

1. **Source:** a local directory, an app-hosted shared library, or a web-hosted
   library. It advertises media capabilities and carries connection state.
2. **Publication:** stable publisher/creator identity plus a stable media ID and a
   signed revision. Hosts can change without changing the user's saved reference.
3. **Representation:** the actual original, transcode, artwork, thumbnail, or other
   asset, identified by MIME type, byte length, and exact-byte SHA-256. A transcode
   has its own hash. Albums contain ordered track references and disc/track numbers.
4. **Provider:** an endpoint currently serving an authorized representation. The
   publisher, catalog host, and byte-serving mirror can be different parties.

Sign manifests that bind publication identity, revision, metadata, representation
hashes, and publication policy or grant references. Keep recipient credentials and
private grants separate from public manifests; evaluate current authorization on
every request. A signature identifies the signing publisher; a
matching file hash proves byte identity, not authorship or permission. Plan device
delegation, key rotation/recovery, and host migration before multi-device publishing.

Local IDs must include library/directory identity. Remote references must retain
their publisher/catalog namespace. Never route a remote numeric ID to local delete,
move, or tagging endpoints. Existing generic providers can use capability-limited
adapters without inventing publisher signatures they do not support.

Favorites, collections, playback history, and personal annotations are viewer-owned
overlays. Visitors cannot edit the author's catalog. A music collection can reference
albums and tracks across sources without duplicating organization between Albums
and Songs modes. The queue stores stable media references and resolves expiring
URLs/hosts when needed; changing the visible source never replaces the session.

## Access, downloads, and offline state

Define independent decisions for:

- audience: private invitation, connected members/friends, or public;
- catalog/preview visibility and playback access;
- permission to save a copy;
- permission to serve copies to others, including allowed hosts and lifetime;
- optional public discovery and provider announcements.

Downloading does not imply permission to mirror. Private invitations and private
asset availability must stay out of public indexes. The initial mirroring milestone
should cover explicitly public, opted-in publications; private offline authorization
needs a separate, precise policy before it is offered.

Enforce access at search, counts, tags, catalog, artwork, original, transcode, range,
and download endpoints, across HTTP and any P2P transport. UI visibility alone is
insufficient. QR/link enrollment should confirm the source/publisher and requested
access, then exchange a scoped invitation for a revocable device credential.
Do not place owner credentials in share links.

Distinguish three storage operations:

- **Playback cache:** bounded temporary bytes, never advertised as a mirror.
- **Save offline:** retained, verified bytes associated with a Shared item, with
  progress, cancellation, quota, and an explicit Remove offline action.
- **Import into Local:** an explicit local library item with source provenance.

Removing an offline copy or disconnecting a source never deletes the author's
media. An unavailable or withdrawn reference remains understandable in a user's
collections. Permission withdrawal can stop future authorized access and replication;
it cannot guarantee erasure of already saved plaintext copies. Playback necessarily
delivers media bytes, so stream-only is an application permission, not copy protection.

## Connectivity and alternate hosts

Start with an authenticated catalog and HTTP range delivery that both app hosts and
web hosts can implement. Add a transport interface so reachability can evolve
without changing publication IDs or library behavior.

| Technology | Proposed role |
| --- | --- |
| HTTPS and byte ranges | Initial catalog, stream, download, and authorized mirror interoperability |
| Iroh | Candidate for direct app-to-app connectivity with NAT traversal and relay fallback; evaluate on desktop, Android, and iOS before committing |
| IPFS/libp2p | Alternative public content/provider routing worth evaluating if open distribution becomes the priority |
| ActivityPub | Later interoperability for creator profiles, follows, and publication announcements |

[Iroh](https://github.com/n0-computer/iroh) offers key-addressed QUIC connections,
hole punching, and relay fallback. Its blob protocol uses BLAKE3; the catalog can
retain SHA-256 and explicitly bind the transport identifier to it. Iroh is a
connectivity building block; the app still needs an authorized provider index.

[IPFS content addressing](https://docs.ipfs.tech/concepts/content-addressing/)
uses CIDs, which generally differ from a raw whole-file checksum. A SHA-256 lookup
therefore needs an explicit mapping. Its
[public provider announcements](https://docs.ipfs.tech/concepts/privacy-and-encryption/)
also require care with private shares. Public discovery is a deliberate opt-in.

[ActivityPub](https://www.w3.org/TR/activitypub/) specifies federated social
activities and delivery. Our design needs a separate availability lookup for
exact media copies. [PeerTube redundancy](https://docs.joinpeertube.org/admin/following-instances)
is an existing example of cooperating hosts serving video segments, with controls
over which hosts may mirror.

An offline-author fallback should work as follows:

1. Resolve the selected publication revision and exact representation hash.
2. Consult the author's manifest and known authorized hosts, then an optional
   public index for publications that allow discovery.
3. Validate the provider grant, revision/policy state, and current availability.
4. Serve verified chunks/ranges using a signed chunk manifest or Merkle proof;
   verify the full SHA-256 when a complete download finishes. A whole-file checksum
   alone cannot verify an arbitrary seek range before playback.
5. Keep the same media item/playhead while changing provider. Retain separate
   handling for different encodings and preserve the requested rendition identity.
6. Report unavailable when no eligible host has the bytes. An online relay cannot
   replace an offline file holder.

Start with selected friends' nodes and explicit retention/quota agreements. Add
broader provider discovery after the identity, permissions, and mirror behavior
work. Signed tombstones/revisions and bounded policy freshness are necessary for
compliant mirrors; withdrawal cannot recall independent downloaded copies.

## Mobile baseline

Read-only inventory found local Android API 35 emulator profiles with accessible
hardware acceleration, and a reachable Apple Silicon Mac with installed Xcode and
iPhone simulators. No emulator was started during this audit. Apple tooling emitted
a setup/authentication warning, so readiness still needs resolution before builds.

Android has a Tauri project and an APK build wrapper. iOS has an unsigned Tauri
workflow and older Capacitor scaffolding. This is build infrastructure; installed
and accepted iPhone music playback remains unproven.

Concrete work required:

- Replace forced navigation to Images with route-backed settings and meaningful
  Android Back/overlay behavior. Preserve the originating gallery and player.
- Bridge phone storage deliberately: Android SAF/MediaStore and durable URI
  access, and iOS document access/import. Current music indexing expects ordinary
  filesystem paths.
- Add a native playback adapter for reliable background/lock-screen behavior.
  Android's [MediaSessionService](https://developer.android.com/media/media3/session/background-playback)
  provides a suitable service model. iOS requires the appropriate
  [audio session and background mode](https://developer.apple.com/documentation/avfaudio/avaudiosession/category-swift.struct/playback).
- Persist queue, seed, album-order/shuffle policy, position, and paused state.
  Restoring an app must not unexpectedly start playback.
- Audit codec support, seek/range handling, and select a playable rendition;
  the current endpoint streams the original file without a compatibility fallback.
- Apply media-scoped authentication and directory access policy to music files,
  artwork, and queries. Current browser query-token handling is image-specific,
  and music routes do not visibly apply the directory sharing policy.

Use a dedicated disposable emulator profile and synthetic media. Record exact
source commit and APK/IPA identity. Cover local imports and paired remote playback,
album ordering, song-seeded queues, seeking, artwork, settings round trips, hardware
Back, screen lock, interruptions, network loss, reconnect, and process restoration.
Repeat on iOS simulator, then a signed build on a real iPhone for background/audio
acceptance. Simulators alone do not establish all hardware behavior.

## Delivery sequence and completion evidence

| Milestone | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1. Navigation and directories | Four primary destinations; Directories opens inside Settings; grouped directory details and correct Back behavior | Desktop plus a newly built APK from the exact milestone commit on a dedicated Android emulator; return to each gallery without losing state; all existing directory operations and legacy routes reachable |
| 2. Mobile music | Phone storage/import, playback from an existing paired personal server, native background player, persistent queue, and suitable formats | Android and iOS album/song playback matrix, screen-off and interruptions, exact artifact identity; physical-device limitations recorded |
| 3. Direct sharing | DMC hosts selected content; QR/link connects another client; shared-source gallery and explicit save/import; shared catalog contract | On a specified reachable LAN or HTTPS setup, first prove one image with private favorites, permission withdrawal, and verified offline copy; then an ordered album and seekable video on the same contract, including thumbnail and byte-range policy enforcement |
| 4. Public creator hosting and friends | Reuse DonutBooru profiles/storage/publication history; add audio catalogs and authorized mirrors | Creator shuts down DMC while another permitted host keeps the album/video available; hash mismatch rejected; source attribution, revisions, and permissions preserved |
| 5. Wider connectivity | Proven P2P transport, optional provider discovery, and ActivityPub adapter | Cross-network NAT/relay and mobile lifecycle evidence; discovery respects publication scope; actual interoperability before compatibility claims |

Milestones 1 and 2 can progress while the catalog contract is designed. Avoid
coupling the immediate navigation fixes to selection of a decentralized network.
For direct sharing over the internet in milestone 3, either an explicitly reachable
HTTP endpoint or an earlier successful P2P transport slice is necessary.

### First implementation boundary

The first code change should extract directory content from its standalone layout,
add a URL-backed Settings shell, update the shared navigation component, and keep
compatibility routes. Likely files: `frontend/src/App.jsx`,
`frontend/src/components/SidebarNavigation.jsx`,
`frontend/src/components/Sidebar/Sidebar.jsx`,
`frontend/src/pages/DirectoriesPage.jsx`, and the relevant sidebar/directory CSS.
Use the same navigation everywhere, including Music. Move source/account management
and publication history into connected-source settings. Keep legacy Online Browse
reachable through Settings until the remote media classification contract supports
equivalent browsing in the correct gallery. Personal-server playback in milestone 2
uses existing pairing; shared-source permissions and playback are milestone 3 work.

When formalizing this proposal, update the existing Online specs with explicit
migration requirements, then create dependent tasks for navigation, mobile storage
and playback, catalog/authorization, embedded hosting, web-host audio, and mirrors.
Integrate verified product commits into `main`. Track observable user outcomes and
exact tested artifacts for each milestone.
