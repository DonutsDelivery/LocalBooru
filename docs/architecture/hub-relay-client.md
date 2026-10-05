# DMC community connection — initial Iroh gateway slice

This source implementation resumes the hub account/catalog foundation. It is not
a deployment or a claim of verified desktop/mobile interoperability. No app tests,
builds, running-app changes, or production configuration changes were authorized
for this pass.

Add a community under Settings → Fediverse & Nodes using its Open in DMC link.
The link includes its canonical HTTPS origin, pinned gateway public EndpointId,
and Iroh relay HTTPS origin. Those are public bootstrap locators, not credentials.
DMC fetches the minimal descriptor through that pinned peer, shows the community
name and access policy, and asks the user to add it. A link cannot enable local
network access, publish files, connect an account, or administer a personal server.
The published bootstrap and descriptor must match; obtain links from a source you
trust. The original node UUID remains an account/catalog identity, while Iroh's
public key authenticates the gateway connection. Key changes require deliberate
reenrollment rather than accepting a descriptor-supplied replacement automatically.

Connected communities appear under Shared collections in Images and Videos →
Collections. Community settings stay in Settings. Browse uses source-qualified
item identities and the existing backend media proxy. Import is an explicit local
copy; publication is an explicit authenticated submission for hub moderation.
Music catalogs and community replicas are outside this first image/video slice.

Relayed is the default for added hubs. DMC removes Iroh's direct IP transports and
address lookup, supplies only the pinned peer and approved relay, and routes the
descriptor, catalog, artwork, media ranges, token exchange, and publication requests
through that connection. Failure reports unavailable; it does not fall back to the
legacy direct HTTP adapter. Standard permits Iroh direct upgrades with relay
fallback. Anonymous returns unsupported before creating network endpoints. Legacy
Danbooru/DonutBooru HTTP sources remain separately configured legacy connections.
Relays can observe connection metadata; this is not anonymous networking.

Public browsing requires no account if the hub permits it. Private browsing and
submissions use hub-issued scoped credentials, kept in the existing backend-owned
credential store. Native login uses the hub's system-browser consent page, a
short-lived backend-owned PKCE verifier/state, the exact registered callback, and
a one-use code exchange over the selected Iroh route. The ordinary system browser
can expose the user's IP to the hub and external login provider; DMC requires an
explicit acknowledgment of this separate browser boundary in Relayed mode. It
does not change the saved relay policy for app requests. No refresh-token protocol
is advertised; expired access requires sign-in again. Manual scoped credentials
remain a collapsed advanced option for communities that need them.

The Rust gateway is a separate service in booru-node's tools/dmc-hub-gateway.
It forwards only an allowlisted subset of /api/client/v1 to one configured upstream
origin, does not accept arbitrary targets or follow HTTP redirects, and retains
HTTP status, byte-range headers, and the hub's authorization checks. It does not
hold a moderator credential or bypass private catalog access. It is a gateway,
not a retained replica, full decentralized collection protocol, or Iroh relay
server. An operator must provision its persistent key, existing relay, upstream,
and matching public bootstrap before publishing the website enrollment link.
DonutBooru's live service has not been changed by this implementation.

Dependency resolution changes the Rust baseline to 1.89 for pinned Iroh 0.97.
Actual compilation, relay reachability, public/private authorization, moderation
receipts, byte-range playback, peer IP observations, Android lifecycle, and
cross-network acceptance remain gates before releasing this slice.

References: [Iroh network configuration](https://docs.iroh.computer/configuring-networks),
[Tauri native deep links](https://v2.tauri.app/plugin/deep-linking/),
[shared library plan](shared-libraries-and-mobile.md), and the matching node
`docs/architecture/dmc-hub-relay.md` wire/operating contract.
