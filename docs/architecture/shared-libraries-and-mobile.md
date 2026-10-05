# Shared libraries, publishing, and mobile playback

Date: 2026-09-29
Updated: 2026-10-05

Status: architecture proposal based on repository and device inventory, revised
with the user's distinction between QR-paired private libraries and creator
collections. The navigation requirements and initial Iroh direct/relayed scope
below are user decisions. Authentication requirements are recorded with proposed
mechanisms still to be designed; the delivery sequence is a recommendation.
This document records planned acceptance;
it does not claim mobile playback or the sharing architecture has passed it.

## Product direction

For the recorded pre-Create discussion, including gallery/playback requirements
and later corrections, see the [product decision record](pre-create-product-decisions.md).
The subsequent [creator commerce record](creator-commerce-and-offline-delivery.md)
adds paid collections, copying limits, and creator-offline purchases. These are
design records; this document's earlier inventory is not evidence of their completion.

DonutMediaCenter is a personal media library, player, and optional publisher.
People browse Images, Videos, and Music, connect to creators' shared libraries,
and explicitly choose what they publish from their own library.

DonutBooru is an optional community hub: a website, collection discovery/catalog
service, and relay or browser gateway. Retaining originals is a separate optional
hosting role; discovery and forwarding do not require permanent storage of all
media. Its existing profiles, storage, and explicitly permitted peer copies provide
foundations, but DMC catalogs, audio, and mirror failover need integration and fresh
verification. Both products should expose the same catalog contract. Other hubs
and known peers must remain usable without one mandatory DonutBooru directory.
A creator should be able to share directly from the desktop app without installing
a separate node. Internet connections may still require discovery or relay
infrastructure, and staying available while the creator is offline requires a
host that retains the authorized media.

The goal is creator-controlled distribution from home networks without requiring
each creator to rent a VPS. Community replicas retain permitted files and catalogs
while creators are offline. Storage and relay bandwidth still come from willing
participants or optional services; a relay cannot replace a retained copy.

Hosting starts disabled. Adding a local directory does not publish it. Publication
exports only selected media and approved metadata, excluding local filesystem paths
and private library details. Mobile clients should support listening, browsing, and
offline copies; continuous hosting cannot depend on a suspended phone app.

### Navigation

There are four primary destinations: **Images | Videos | Music | Settings**.

- Settings opens **Directories** by default. Local directories remain immediately
  accessible; shared sources and hosting have their own settings sections.
- Immediately above each gallery's directory/source list, offer two buttons. The
  proposed labels are **My libraries | Shared collections**. This is a trust and
  organization boundary: personal devices can be physically remote, and creator
  collections can be invite-only.
- **My libraries** appends directories from explicitly QR-paired personal servers
  after this device's directories. All these sources remain attached at the same
  time. Preserve server/library grouping or badges and offer a server filter when
  useful. Only directories enabled for the active media type appear.
- **Shared collections** shows a unified list of authorized published collections
  from creators the user has added. The default has no creator filter; filtering
  by creator narrows the same list. Public and granted invite-only collections
  belong here. Public network discovery does not automatically add creators.
- The two panes retain independent selection, filters, and scroll position. Their
  switch changes the visible source tools without replacing the application's
  backend or interrupting playback. Personal collections/favorites remain available.
- QR-paired installed servers retain their granted capabilities. Separate browsing
  and administration contexts during migration. Creator catalogs are read-only to visitors, with private viewer-owned
  favorites/collections. Connecting a source does not import media into a local
  directory or transfer ownership.
- Servers continues to mean administration of remote app installations. Treat
  browsing, publishing, and administration as separate roles. Preserve established
  device-pairing credentials during migration; creator invitations always use
  independently scoped grants. Current paired-server browsing and administration
  share a credential context.
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

### Platform requirements (user clarification, 2026-10-03)

DMC must work for ordinary installations, including a local library on Android;
the developer's folders, shell, Python installation, and paired PC are not prerequisites.

- Native app data, bundled resources, credentials, addon environments, and media
  helpers resolve from OS/application locations. Mac launches from Finder must
  work without shell startup files. Existing libraries keep their location.
- Desktop addon setup supplies its own pinned Python when a compatible existing
  interpreter is absent. It preserves system Python and user-selected ComfyUI
  installations. Python availability alone does not establish GPU/node support.
- Android's **This Device** indexes permitted local images, videos, and music in
  place. Android grants determine available media types. Selecting a paired
  server is optional, and restarting must retain an explicit local selection.
- Local media metadata and thumbnails use Android APIs instead of requiring
  desktop FFmpeg/Python. Paired desktop generation and server-side video
  processing remain separate from phone-local library access.
- Acceptance covers fresh and upgraded installs, denied/partial/revoked media
  permissions, spaces and Unicode, offline local use, and restart persistence.
  A desktop path test or emulator APK does not establish full Mac, Windows,
  or OEM Android support. Shipping claims require the exact package on hardware.

The current local Android implementation covers MediaStore shared storage;
cloud document providers/SAF and HEIC import are not implemented. Desktop helper
lookup includes packaged and standard Homebrew locations, but new Mac/Windows
installers still need bundled or installed FFmpeg. Occupied addon port recovery
currently uses Linux process APIs. These remain release acceptance gaps.


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

1. **Source:** a directory on this device, a directory on a paired personal server,
   or a creator's published collection hosted by an app or web node. It advertises
   media capabilities and carries connection state and relationship type.
2. **Publication:** stable publisher/creator identity plus a stable media ID and a
   signed revision. Hosts can change without changing the user's saved reference.
3. **Representation:** the actual original, transcode, artwork, thumbnail, or other
   asset, identified by MIME type, byte length, and exact-byte SHA-256. A transcode
   has its own hash. Albums contain ordered track references and disc/track numbers.
4. **Provider:** an endpoint currently serving an authorized representation. The
   publisher, catalog host, and byte-serving mirror can be different parties.

### Paired private libraries without switching the whole application

Saved servers already form a list, but `frontend/src/serverManager.js` stores one
active server. `frontend/src/api.js` uses one mutable URL/authentication context
and one Tauri remote proxy. Changing servers can reload the desktop app. Startup
also selects a remote backend automatically in some mobile/desktop cases. This
prevents simultaneous source browsing and can change how a media reference resolves.

Introduce a stable source ID and a per-source client or embedded gateway:

- Keep the embedded API anchored to this device. Gallery scope changes do not
  change the global active server, API base URL, or settings backend.
- Register existing QR-paired servers as personal sources. Append their directories
  to this device's list while preserving remote libraries, media flags, permissions,
  and a visible source label. LAN, Tailscale, and approved public URLs are connection
  methods for the same personal source.
- Resolve credentials, certificate pins, fallback addresses, tokens, and capabilities
  independently for each source. A keyed gateway resolves registered sources; it
  must not accept arbitrary proxy destinations. Forward ranges and artwork through
  immutable source-qualified media URLs. Verify the same remote identity and
  certificate/key expectations for every added or fallback address before sending
  credentials. The current health probe alone does not establish that identity.
- Qualify cards, directories, collections, lightboxes, queue items, request ownership,
  caches, and scroll state with source ID as well as library/media IDs. Current
  locators omit the server namespace. Paired server IDs derive from a library UUID,
  so cloned libraries and multiple endpoint aliases need explicit identity handling.
- Keep playing tracks tied to their original source while browsing another device
  or creator. Queue continuation resolves against the session's own sources.
- Migrate the saved active server into the selected personal source without losing
  its credentials or history. Startup must leave local and paired sources attached
  rather than silently rerouting the entire app to one server.

Directory list aggregation can happen independently of gallery result aggregation.
The combined directory list is the required default in My libraries. Clicking a
directory scopes results to that source. An All private directories view needs
independent cursors, supported filters, and failure state per source. During migration,
one selected source or explicitly grouped results can be delivered first; a fully
combined sorted masonry view needs a stable merge across all source cursors.
Concatenating first pages would misrepresent sorting and pagination. Offline devices
must not blank healthy sources, and incomplete totals need clear labeling.

This source-routing work applies to existing personal servers before introducing
any P2P network. Settings > Servers remains available for connection administration.

The existing Axum server can bind to LAN when local-network access is enabled, and
client records support primary/fallback URLs such as LAN and Tailscale. The public
access setting alone does not establish outside reachability: code currently does
not advertise a public URL or create a separate public-port listener. The UPnP
configuration must also target the actual listener port. A configured reverse proxy
or router mapping could expose the service; outside-network behavior has not been
verified in this audit.

### Creator collections as the publication unit

Collections provide the deliberate boundary between personal organization and a
creator's published catalog. Keep three states:

| State | Behavior |
| --- | --- |
| Unpublished | Personal organization, with no publication or network announcement |
| Invite only | Visible to authorized recipients with collection-scoped access |
| Public / Global | Published collection eligible for public network discovery |

A published collection has a stable creator/collection ID, title, description,
cover, ordered membership, media capabilities, and signed revision. Visibility
does not automatically grant downloading or mirroring. Public discovery can be
enabled for Public collections; invite-only metadata and availability stay private.

An added creator exposes their permitted collections as entries resembling
directories. The list combines collections from all added creators by default;
clearing the creator filter restores it. Discovering a node does not automatically
follow its creator or grant access to their invite-only collections.

An album remains an ordered release. A music collection can contain albums and
individual tracks without redefining album identity. Preserve current collections
through explicit adapters and scope contents/counts to Images, Videos, or Music.
Albums and Songs remain two views of the same organization.

Publishing records exact membership, metadata, and representations in a revision.
Local scans, dynamic searches, or new tracks discovered in a referenced album must
not silently enlarge it. Stage changes for an explicit Publish update; an eventual
automatic-update policy would need a deliberate choice. Pin the exact track and
disc order for each published album revision.

Bytes may be deduplicated across collections while grants remain collection-scoped.
Withdrawing one collection must not revoke another valid grant, delete storage
still referenced elsewhere, or erase an independent local copy. Republishing
another person's media requires applicable permission and retained provenance;
membership in an existing collection does not establish authorship.

Creator invitations need separate enrollment. Existing DMC device QR pairing issues
a write-capable session; it must not double as a read-only collection invitation.
Collection lists, counts, covers, membership, variants, and range requests all need
the same audience checks. Current music endpoints and some collection list/count
paths require policy enforcement work before they can serve this purpose.

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

Use one policy-enforcing collection protocol with stable publication identities
across transports. Initial app-to-app networking uses Iroh, supporting direct and
relayed connections. Preserve HTTPS and byte-range interoperability for web hosts
and browser gateways; these adapters must enforce the selected privacy policy.

### Agreed initial scope: Iroh direct and relayed

The user selected this initial scope on 2026-10-05. Anonymous routing is reserved
for a future implementation, not a dependency of the initial integration.
The [deferred anonymous networking design](anonymous-networking.md) describes its
full intended integration, compatibility rules, authentication boundaries, and
acceptance gates. Deferral applies to implementation, not to documenting the design.

| Mode | Initial behavior |
| --- | --- |
| Standard | Permit direct encrypted Iroh connections, with relay fallback. Connected peers can learn each other's IP addresses. |
| Relayed | Disable direct peer connections and address disclosure to peers. Carry discovery/catalog requests and media through approved relays or gateways; operators can still see connection metadata. |
| Anonymous (future) | Reserve the transport/policy interface and unsupported-capability result. Do not advertise or enable this mode until an implementation covers discovery and delivery. |

All initial desktop and Android nodes should implement both Standard and Relayed.
The setting controls permitted routes, not which networking library is installed.
A Standard node must accept a relayed connection from a Relayed node. Either
endpoint requiring Relayed rules out a direct connection; connection failure must
never silently weaken that requirement. Transport capability is separate from
volunteering to operate a relay or retain other people's files.

Put discovery, metadata, covers, provider lookup, transfer/range requests, and
redirect handling behind the route policy. Configuring an Iroh media endpoint
alone is insufficient if an HTTP catalog request exposes the same user's address.
Do not advertise a protected publisher's direct addresses in public records.
Authentication must report any external-provider/privacy boundary explicitly.
This protects the current connection path; it cannot undo historical IP disclosure.

Future anonymous support should fit the same provider resolver and collection
protocol. Stubs must return unsupported/unavailable without issuing direct network
requests; they must not pretend Iroh relaying is anonymity. Arti/Tor is a candidate
for later evaluation, not a selected or shipped dependency. Avoid binding the
collection protocol to the experimental Iroh Tor adapter or requiring a separately
installed anonymity client as part of the initial release.

The gallery/player resolves creator, collection revision, and representation
identity rather than storing a serving URL as the item's identity. Select an
available authorized provider and a route permitted by both endpoints. Preserve
the item, attribution, rendition, and playhead through provider changes; latency
or temporary unavailability may remain visible. Iroh does not supply DMC's catalog,
replication policy, or media-provider discovery automatically.

Use `iroh-blobs` verified ranges where suitable, binding its BLAKE3 identity to the
signed publication manifest. A future transport must preserve equivalent content
verification rather than assuming this protocol runs unchanged on every stream.
Community retention is an explicit opt-in with quotas, bandwidth limits, and
permitted-publication checks. Retain catalogs as well as bytes, and distinguish
recently confirmed replicas from unverified/stale availability announcements.

DonutBooru's directory locates eligible holders; an Iroh relay forwards encrypted
app-to-app traffic, while a browser gateway terminates connections and can access
the media it serves over HTTPS. The gateway may fetch from a permitted replica
when the creator is offline. If all eligible holders are offline, report the media
unavailable even when its retained catalog remains browsable. Relay/gateway
bandwidth remains an operational cost even without permanent original storage.

Initial acceptance must cover direct/direct, direct/relayed, and relayed/relayed
participants; cross-network reachability; no protected direct-address discovery
or request bypass; verified range failover with the creator offline; and mobile
lifecycle behavior. An anonymous-only peer must receive a clear unsupported result
until that transport is implemented. These are future gates, not completed tests.

References: [Iroh route configuration](https://docs.iroh.computer/configuring-networks),
[verified blob transfers](https://docs.iroh.computer/protocols/blobs),
[privacy boundaries](https://docs.iroh.computer/concepts/security-privacy), and
[current experimental Tor adapter](https://docs.iroh.computer/transports/tor).

| Technology | Proposed role |
| --- | --- |
| HTTPS and byte ranges | Initial catalog, stream, download, and authorized mirror interoperability |
| zrok / OpenZiti | Previously considered optional ingress; outside the selected initial Iroh integration |
| Iroh | Selected initial app-to-app transport: direct and enforced relayed modes; platform behavior still requires implementation and evidence |
| Arti / Tor | Future anonymous discovery/delivery candidate behind an unsupported transport stub; outside initial integration |
| libp2p | Alternative considered for discovery/DHT/gossip; not an additional initial networking dependency |
| IPFS | Optional public distribution through its existing content/provider ecosystem |
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

### zrok and our application protocol

[zrok public sharing](https://netfoundry.io/docs/zrok/concepts/public-shares/)
provides HTTP/HTTPS access to a running service, including stable names. Its
[private sharing](https://netfoundry.io/docs/zrok/concepts/private-shares/) uses
share tokens and requires clients on the same service instance. It is
[self-hostable](https://netfoundry.io/docs/zrok/category/host-your-own-zrok/).
[OpenZiti](https://netfoundry.io/docs/openziti/intro/) supplies controllers, a router
fabric, and authenticated endpoints underneath it.

If revisited later, evaluate zrok as an optional way to reach the DMC publication API through NAT.
Expose only the policy-enforcing publication service, excluding raw directories
and unrestricted management endpoints. Check ranges, seeking, large transfers,
reconnect, deployment overhead, and mobile integration with a disposable prototype.
Creator catalogs, hash-to-provider lookup, and persistent offline-author copies
remain application responsibilities.

Define a versioned DonutMediaCenter protocol for collection catalogs, revisions,
invitations, provider announcements, and verified media requests. Use established
encrypted transports and routing implementations beneath it. This keeps product
semantics under our control while reusing cryptography, NAT traversal, congestion
control, and connection recovery.

### Discovery through other connected nodes

The requested larger network is an overlay in which a node learns contacts through
nodes it already knows. Separate four jobs:

1. **Bootstrap:** start with an invitation, cached peer, local discovery, or one of
   several independent public entry points. A new node needs a reachable first
   contact; continued operation should not require one mandatory central directory.
2. **Peer/address lookup:** exchange bounded, signed public contact records and
   resolve a known peer's address. Learned contacts receive no implicit trust,
   private invitations, or access to QR-paired personal servers.
3. **Creator/catalog discovery:** use opt-in announcements and subscribable indexes
   for creator/collection metadata. Each peer need not store every public catalog.
4. **Provider lookup:** find online authorized hosts for a known representation
   hash, then connect directly or through a relay to request the media bytes.

[libp2p rendezvous](https://libp2p.io/docs/rendezvous/) provides peer registration
and discovery; its [DHT](https://libp2p.io/docs/dht/) provides
key-based routing. [Iroh address lookup](https://www.iroh.computer/blog/the-road-to-iroh-1-0)
can use signed records and a DHT for known endpoint identities; broader creator
and media discovery still needs our protocol. Freenet's
[small-world routing](https://freenet.org/build/manual/architecture/p2p-network/)
illustrates an overlay approach. Adopting its application/storage runtime would
require a separate suitability evaluation.

Evaluate the selected Iroh integration against these criteria: bootstrap loss, address
changes, NAT/relay paths, Android/iOS lifecycle, playback startup/seeking, stale or
hostile advertisements, bandwidth, and operation without one mandatory directory.
Use verification, expiration, resource limits, and user block lists for discovery.
Avoid unbounded network-wide query flooding. A key-based lookup does not implement
keyword search or guarantee that every peer is reachable.

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

## Hub enrollment and authentication direction

Recorded 2026-10-05 from the user's next design discussion. Hub-controlled access,
optional external identity providers, independent signups, DMC submissions, and
click-to-add enrollment are product requirements. The mechanisms below are design
proposals, not implemented endpoints or a finalized authentication protocol.

Adding DonutBooru should expose its authorized community collection/catalog in
DMC's Shared collections, alongside other added hubs and creators. Hub aggregation
must preserve original creator/publication identities. Browse collections in the
normal galleries; keep source/account configuration in Settings.

### Website links that open DMC

Provide an **Open in DMC / Add community** link, with the convenient experience of
a magnet link. Prefer an HTTPS app link with a website/install fallback and a
registered DMC deep link where needed; the exact URI format remains undecided.
Literal BitTorrent `magnet:` compatibility is not required or claimed.

The link identifies a hub/collection, its identity anchor, and permitted bootstrap
locators. Fetch and validate its descriptor under the user's current route policy,
then show its name and requested access for confirmation. Names from links are
untrusted display hints. Clicking a link must not upload, publish, share personal
directories, enable local-network access, or grant device administration. Keep
private invitations scoped, expiring/revocable, and separate from public links;
never place reusable account/admin credentials in them.

### Accounts belong to the hub

Each hub chooses whether browsing is public or requires membership, whether signup
is open/invited/approved, and which login methods it offers. Browsing, uploading,
publishing, moderation, and replication are separate grants. Public catalog access
does not imply upload permission; authentication does not bypass approval policy.
For private hubs, keep collection names, counts, covers, and provider records behind
authorization; expose only the minimal enrollment/authentication descriptor.

DonutBooru should accept authenticated DMC submissions without requiring the
Discord-upload/bot-sync pipeline. Discord may remain an optional login/import
integration. Distinguish submitting bytes for hub retention from submitting an
existing publication for discovery: the chosen hub policy must specify retention
and moderation, rather than imply every submission is permanently stored there.

Source inspection on 2026-10-05 found that the local DonutBooru node checkout's
`api/models.py` already has a hub-owned numeric `User.id`, uploads, roles, and
moderation records, but requires a unique non-null `discord_id`. Its
`api/routers/auth.py` creates/updates that account after Discord OAuth and requires
a configured Discord Level 2 role. Avatar construction and privilege synchronization
also assume Discord. This is code evidence, not a live deployment verification.

Proposed migration: retain existing user IDs and ownership; move external identities
to account-linked records keyed by provider/issuer and subject. Support independent
credentials without manufacturing a Discord ID. Preserve existing access until an
operator deliberately changes its policy, and explicitly choose which privileges
remain externally synchronized. Account linking requires proof of control; matching
names or email addresses alone must not merge accounts.

Candidate login methods are hub-local signup (passwords and/or passkeys), optional
Discord/Google/Facebook/X integrations, and invitation/key-based pseudonymous
membership. Provider availability and setup remain hub-specific; OAuth providers
are not interchangeable and not all implement OpenID Connect. DMC should consume
the hub's advertised login methods and obtain hub-scoped sessions/grants, rather
than ship an OAuth secret for every provider or send one hub's credentials to another.

For native browser-based authorization, use the system browser and a short-lived
authorization code with PKCE, request-bound state, and validated callbacks. The
hub handles external provider credentials and exchanges; DMC receives scoped hub
credentials. Provider login must not silently open an ordinary browser and claim
it stayed within a future anonymous routing mode. See
[OAuth for native apps](https://www.rfc-editor.org/rfc/rfc8252) and
[OpenID Connect](https://openid.net/specs/openid-connect-core-1_0.html).

For privacy-oriented communities, consider a distinct local key identity per hub:
the client proves control by signing a challenge and the hub issues scoped grants
after invitation/approval as required. This is pseudonymous authorization, not
untraceable activity: a hub can link actions under that account, and initial Iroh
relaying does not hide the user's IP from relay operators. Keep account keys distinct
from public creator-signing and transport keys. Recovery, device enrollment,
revocation, abuse limits, and an optional stronger unlinkable-credential design
remain open decisions; do not invent a new cryptographic protocol for the first slice.

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
| 1a. Navigation and directories | Four primary destinations; Directories opens inside Settings; grouped directory details and correct Back behavior | Desktop plus a newly built APK from the exact milestone commit on a dedicated Android emulator; return to each gallery without losing state; all existing directory operations and legacy routes reachable |
| 1b. Attached private sources | My libraries appends directories from QR-paired personal servers; per-source requests/media; no whole-backend switch | Two servers with colliding numeric/library IDs remain distinct; browsing another source leaves playback attached to its origin; offline sources do not blank healthy ones; credentials, pagination, filters, scroll state, and startup migration remain scoped correctly |
| 2. Mobile music | Phone storage/import, playback from an existing paired personal server, native background player, persistent queue, and suitable formats | Android and iOS album/song playback matrix, screen-off and interruptions, exact artifact identity; physical-device limitations recorded |
| 3. Direct collection sharing | DMC publishes Public or Invite-only collections over Iroh direct/relayed routes; a scoped invitation connects another client; Shared collections pane unifies added creators; explicit save/import | First gate on one image collection with invitation, both initial privacy modes, private favorites, withdrawal, and verified offline copy. Follow-on gates cover ordered albums and seekable video, creator filters, exact membership, cover/count policy, and ranges. QR creator invites never grant device-admin access |
| 4. Community hubs and authorized replicas | DonutBooru as optional discovery/gateway; click-to-add enrollment, hub-controlled accounts and DMC submission; audio catalogs and permitted retention | Creator shuts down DMC while another permitted host keeps the catalog and album/video available; hash mismatch rejected; source attribution, revisions, privacy routes, and permissions preserved. Public/private hub access and independent account migration require their own evidence |
| 5. Wider discovery and future transports | Broader provider discovery, optional ActivityPub adapter, and separately scoped anonymous transport implementation | No mandatory central directory; discovery respects publication scope; anonymous discovery and delivery require desktop/mobile evidence before enabling the reserved capability |

Milestones 1a, 1b, and 2 can progress while the catalog contract is designed. Avoid
coupling the immediate navigation fixes to selection of a decentralized network.
Milestone 3 includes the initial Iroh connectivity slice; a LAN-only or HTTP-only
demo does not establish completion of the selected direct/relayed networking scope.

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

In parallel, introduce source-qualified clients and media resolution in
`frontend/src/api.js`, `frontend/src/serverManager.js`, and the Rust remote-proxy
state/routes. Reuse existing paired credentials for the authorized personal-device
relationship. Migrate image/video locators and music queues before attaching a
second backend to a gallery. The sidebar toggle must not call the current global
server switch or reload the app. Establish the combined private directory list
before adding creator discovery; no networking-stack decision is required for it.

When formalizing this proposal, update the existing Online specs with explicit
migration requirements, then create dependent tasks for navigation, attached private
sources, mobile storage/playback, catalog/authorization, embedded hosting, web-host
audio, discovery, and mirrors.
Integrate verified product commits into `main`. Track observable user outcomes and
exact tested artifacts for each milestone.
