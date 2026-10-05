# Deferred anonymous discovery and delivery

Recorded: 2026-10-05. Status: reference design; implementation deferred.

Read with [shared libraries, publishing, and mobile playback](shared-libraries-and-mobile.md)
and [creator commerce and offline delivery](creator-commerce-and-offline-delivery.md).
The initial networking implementation is Iroh Standard and Relayed only. This
document explains the future anonymous integration so the initial interfaces do
not prevent it. It does not claim that Tor, anonymous authentication, or anonymous
mobile hosting has been implemented or verified in DMC.

## Purpose and privacy boundary

Anonymous mode must cover the entire shared-collection session: bootstrap into a
community, discovery/search, authentication where supported, catalog/cover reads,
provider lookup, media transfers, and publication/replica announcements. A private
media stream is insufficient if the hub sees the client's IP during discovery.

The intended network property is that community hubs and media providers do not
receive the user's real network address through DMC requests, and viewers do not
receive an anonymous publisher's real address. Use an established anonymity network
with independently operated routing nodes. A DMC-owned proxy immediately in front
of a DMC-owned hub does not hide the client's IP from that operator.

This does not conceal everything. The entry/guard can see the connecting IP;
the local network can observe a connection to the anonymity network. Accounts,
cookies, public creator identities, identical content, request timing, and traffic
correlation can link activity. Logging in with a known Discord account identifies
that account even over Tor. Do not promise protection against a global observer,
compromised endpoints, or identification from the media itself. Switching modes
cannot erase addresses previously associated with a publication or account.

The setting governs DMC community sharing, not all traffic from the computer.
Explicitly paired personal LAN devices are a separate feature. Product wording
must state the scope, and community code must not use the personal-device path
to bypass the selected policy.

## Reference implementation: embedded Arti

Use embedded Arti as the reference design for the deferred implementation. It fits
the Rust core and supplies Tor client and onion-service functionality. Its exact
version, native dependencies, platform coverage, and runtime limits must be checked
at implementation time. No Arti dependency is required for the initial stubs.

Keep Iroh and Arti as separate transport adapters beneath the same DMC collection
contract. Do not assume an Arti byte stream is interchangeable with Iroh QUIC or
that `iroh-blobs` runs on it unchanged. The existing Iroh Tor adapter is experimental
and documents a separate daemon/control-port dependency and recovery limitations;
it is not the planned production dependency. Reusing a later improved adapter would
require demonstrating the same embedding, isolation, and recovery properties.

Each future supported DMC build should include the same direct, relayed, and
anonymous capabilities. Users install DMC without separately installing Tor.
Tor client/onion-service use does not enroll a device as a Tor relay. Operating
network relays and retaining community copies remain separate opt-ins.

## Endpoints and mixed privacy modes

A publishing or replica node advertises a signed binding between its service
identity, permitted publication scope, transport capabilities, endpoint locators,
and expiration. An onion address authenticates an onion endpoint; bind that
endpoint to the intended DMC identity as well. Preserve creator identity separately
from the identity of a host serving a copy. Do not trust an unsigned alternate
URL, a display name, or the possession of a content hash as publishing authority.

After anonymous support ships, publishers in Standard or Relayed mode should also
maintain an onion service while they advertise anonymous reachability. This lets
an Anonymous viewer fetch from them without an intermediary that terminates the
media connection. Bundle capability and active availability are different facts:
do not advertise a usable onion endpoint before it is ready, or after suspension.

| Participants | Permitted route under the default mode definitions |
| --- | --- |
| Standard / Standard | Direct Iroh when available, otherwise Iroh relay |
| Standard / Relayed | Iroh relay |
| Relayed / Relayed | Iroh relay |
| Standard / Anonymous | Arti connection to the serving node's onion service |
| Relayed / Anonymous | Arti connection to the serving node's onion service |
| Anonymous / Anonymous | Arti connection to the serving node's onion service |

Standard permits stronger routes. Relayed forbids direct peer exposure and permits
the anonymous route. Anonymous permits only its supported anonymity transport.
Additional restrictions, such as a specific allowed provider or route, still apply:
select the intersection of both endpoints' policies rather than treating every
setting as a simple numeric level. No intersection means unavailable, never a
privacy downgrade.

An Anonymous publisher must not keep making Iroh direct-address announcements or
probe peers through UDP for that protected identity. Changing an active session
to Anonymous must stop incompatible requests/connections before starting new ones.
Use separate transport keys/records as appropriate, but do not imply that key
rotation removes links to an already public creator or previously exposed IP.

## Anonymous discovery and DonutBooru

DonutBooru and other participating hubs expose an onion endpoint for their catalog,
provider directory, authentication, and publication APIs. The web-facing HTTPS
endpoint can coexist for other users. Bind both to a verified hub identity; a
website redirect alone is not proof of an onion endpoint's identity.

1. Load an invitation, verified saved descriptor, or bundled/community bootstrap
   descriptor locally. An Anonymous session must not first contact the hub directly
   to ask how to use its anonymous endpoint.
2. Bootstrap Arti using Tor's supported directory/guard mechanisms. Those network
   connections are necessary and can reveal the client's IP to the entry side;
   they must not include DMC account, collection, or search information.
3. Resolve the hub's onion endpoint through Tor. Never send `.onion` lookups to the
   system DNS resolver or use ordinary Iroh discovery for a protected identity.
4. Fetch and validate the hub descriptor, and perform allowed catalog searches
   through that connection. Private community records require the hub's grants.
5. Fetch signed provider records through eligible hub/peer indexes, then contact
   a compatible holder's onion service for the requested representation.

Allow multiple bootstrap/index sources and locally retained signed catalogs, so
DonutBooru is optional. Bound lookup work and cache records with expiration. Do
not broadcast private memberships or media holdings to a public discovery index.

For the first anonymous implementation, require onion-capable hub/media endpoints.
A clearnet-only hub or old replica is unsupported unless another eligible holder
can serve the same publication anonymously. An optional future Tor-to-HTTPS exit
path would need its own TLS, routing, authentication, and disclosure design; it
must not appear as an accidental generic HTTP fallback.

## Media delivery and replicas

Expose transport-neutral operations for catalog/revision fetch, provider lookup,
authorized range reads, cancellation, and availability. Preserve stable publication
and rendition IDs in the gallery and player; never make a transient onion address,
Iroh endpoint, or HTTP URL the identity of a saved media item.

Use a signed manifest binding exact media hashes and representations. Iroh may use
its verified blob ranges; the Arti adapter must supply equivalent verifiable chunks
or range proofs using an established format. Bind all hash/proof schemes explicitly
to the same representation. Do not assume a whole-file hash verifies an arbitrary
seek range before playback, or invent custom encryption to bridge the adapters.

Both adapters enforce the same authorization, range bounds, revision rules, and
cancellation behavior. Decrypted bytes reach the normal local media pipeline; the
webview/player should not independently fetch remote URLs, subtitles, covers, or
playlist entries outside the resolver. Redirects and nested resource URLs must
return to the route-policy checks. The file provider still learns which bytes it
serves; IP privacy is not private information retrieval.

When a creator is offline, resolve a retained signed catalog and eligible replicas.
Try a reachable compatible holder and verify the requested chunks. On failure,
resume from another permitted holder at the needed offset without changing media
identity. Use bounded concurrency/timeouts; do not flood providers or launch a
direct speed probe. If no compatible holder is online, show unavailable while
preserving the item and playhead.

Creator permission to replicate, node consent to retain, and node connection
privacy remain independent. An Anonymous replica advertises only its permitted
anonymous endpoints. Offline purchases additionally require the entitlement/key
fulfillment path described in the commerce record, reached through allowed routes.
A relay or discovery index does not supply missing files or missing grants.

## Authentication and account isolation

Use the same hub permissions in every mode. Support public browsing without an
account where the hub permits it; private browsing and submissions require the
appropriate hub grant. Local signup or invitation/key-based accounts can avoid
requiring a third-party social identity, but are still linkable pseudonymous
accounts unless a separate unlinkable-credential system is deliberately designed.

Separate hub credentials and browsing sessions. Use Arti's supported stream/circuit
isolation for unrelated hub/account contexts, while recognizing that shared content
or identifiers can still link them. Account keys, creator-signing keys, and transport
keys have different roles; do not broadcast a universal private-user identifier.
Plan explicit device enrollment, recovery, expiration, and credential revocation.

An ordinary external-browser OAuth flow is outside an embedded Arti session.
Anonymous mode must not silently launch it and claim privacy was preserved. Until
a suitable browser authorization path is implemented, offer compatible hub-local
methods or report the provider flow unsupported under the current policy. A user
may deliberately change policy, but OAuth completion must never make that decision
implicitly. Even a correctly routed OAuth flow identifies the external account.

Do not reuse unrestricted owner credentials for a replica, expose tokens in public
links, or forward one hub's credentials to another endpoint after a redirect.
Application authorization must accompany transport authentication on both engines.

## Lifecycle, resource use, and user experience

Keep networking in the shared native core on Linux, Windows, macOS, and Android.
The Android integration must handle foreground-service/background restrictions,
network changes, process death, and secure native key storage. iOS support requires
its own lifecycle evidence before it can advertise equivalent hosting capability.
An APK or library compiling successfully does not establish background availability.

Start anonymous connectivity when needed, with explicit bootstrap/readiness/error
states. Publishers claiming anonymous availability must keep their service active;
mobile suspension withdraws or expires availability announcements. Reconnect after
resume under the same policy. Arti failure must stop affected requests rather than
route them through a surviving Iroh client.

Persist required Tor state/guards and onion keys appropriately; do not rotate
everything on each launch. Use OS-protected storage for secrets, keep private
catalogs/invites protected at rest, and define key rotation/recovery. Release
packages need the corresponding native libraries, dependency notices, and an update
strategy. Normal user logs must exclude credentials and private lookup histories.

The gallery layout stays unchanged. Show bootstrap, buffering, and unavailability
honestly, and expose the current route in optional connection details. Anonymous
mode cannot promise the same startup, seeking, or throughput as direct Iroh. Bound
disk cache, memory, concurrent streams, and retries; support cancellation and
backpressure. Never persist full transfers merely because a relay was involved.

## What initial stubs must reserve

- A versioned transport capability and explicit unsupported result for Anonymous.
- A route-policy decision before any network request or endpoint announcement.
- Stable media references plus typed endpoint records, rather than hard-coded HTTP
  URLs or Iroh IDs throughout gallery/player state.
- An adapter boundary for discovery/catalog operations and verified range delivery,
  including cancellation, authorization context, readiness, and failure reasons.
- Separate hub/account, creator, provider, and transport identities.

No Arti daemon/library startup, onion advertisements, experimental custom transport,
or selectable functioning Anonymous mode ships with these stubs. A link requesting
Anonymous returns unsupported before contacting its destination. The stubs are
not a promise of an available protection level.

## Deferred implementation sequence and acceptance

1. Pin and embed Arti; demonstrate client and onion-service operation in disposable
   desktop and Android instances. Establish resource/lifecycle limits and recovery.
2. Implement the common request/range adapter and signed endpoint bindings; verify
   identical publication and authorization behavior over Iroh and Arti.
3. Add onion bootstrap, hub discovery, provider lookup, and compatible authentication.
   Ensure covers, redirects, retries, and announcements obey the same policy.
4. Exercise all six mode pairings above, with both publisher and viewer roles where
   relevant. Include mixed-version peers: unsupported must not cause downgrade.
5. Turn off the creator, retrieve from permitted replicas, then interrupt a replica
   mid-transfer and seek through a verified alternate. Check policy/rendition fidelity.
6. Observe network traffic and instrument hub/provider endpoints with synthetic
   data: no direct community connections, system-DNS onion lookups, UDP probes, or
   real-address advertisements for Anonymous sessions. Tor entry/bootstrap traffic
   remains expected. Check failures during bootstrap, redirect, auth, and resume.
7. Check stale/malicious provider records, bad signatures/chunks, expired/revoked
   grants, browser handoff, mode changes, and mobile process death. Private community
   catalogs/holdings must remain absent from public indexes.
8. Record real desktop/Android behavior and obtain a security review before enabling
   the mode or making anonymity claims. Publish measured latency/throughput/resource
   limits; do not extrapolate anonymity or performance from a successful download.

Outstanding implementation decisions include the exact Arti/dependency versions,
adapter wire/proof format, supported platform matrix, service bootstrap timing,
identity recovery, and privacy-preserving external-browser authentication. These
do not block the initial direct/relayed integration, but must be resolved for the
anonymous release. Authentication mechanisms are further discussed in the hub
section of the parent plan.

## Primary references

- [Arti status and embedding](https://arti.torproject.org/FAQs/)
- [Arti client, service, and isolation APIs](https://docs.rs/arti-client/latest/arti_client/struct.TorClient.html)
- [Arti Android integration](https://arti.torproject.org/integrating-arti/custom-wrappers/android/)
- [Onion services](https://support.torproject.org/about-tor/onion-services/what-is-a-dot-onion/)
- [Remaining attacks against onion routing](https://support.torproject.org/about-tor/security/attacks-on-onion-routing/)
- [Current experimental Iroh Tor adapter](https://docs.iroh.computer/transports/tor)
