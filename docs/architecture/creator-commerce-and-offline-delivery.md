# Creator collections, purchases, and offline delivery

Recorded: 2026-09-30. Status: design discussion before the Create integration.

Read with the [product decision record](pre-create-product-decisions.md) and
[shared-source architecture](shared-libraries-and-mobile.md). This document adds
the later commerce ideas; it does not claim an existing store, payment system,
DRM implementation, or decentralized fulfillment service.

## Desired experience

A collection can be a creator's release or shop. The creator can offer previews,
streaming, individual purchases, or access to the whole collection. A listener
can buy from that creator and obtain the authorized files from another available
host when the creator's own computer is offline.

The friend-sharing idea is creator-approved circulation: a listener buys a copy,
and the license may let that listener stream it to friends or serve permitted
copies to other purchasers. These are separate permissions, not consequences of
merely possessing a file.

The user wants discovery that promotes creators and offers, without turning the
network into a global unrestricted file-download index. For a paid-only full
representation, broad provider retrieval is unlocked by the required purchase
entitlement. Free streams and previews use their own permitted representations.
Visible catalog metadata does not itself authorize fetching a full file.

## Policy belongs to the collection offer

The product's permission boundary follows the creator's declared policy. Record
that policy in a stable offer/license version that a purchase can reference.
Separate these decisions:

| Decision | Examples discussed |
| --- | --- |
| Visibility | Public/global, invitation only, unpublished |
| Before purchase | Metadata only, preview images, reduced-resolution video, lower-bitrate audio, permitted full streams |
| Purchased scope | One item, one release, full collection; version/membership policy to decide |
| Playback | Private streaming, permitted friend streaming, full rendition access |
| Download | Personal retained copy allowed or withheld |
| Rehosting | Serve only to entitled recipients, public/free serving if allowed, named hosts or wider permitted peers |

Buying access does not automatically buy copyright, exclusive ownership, or the
right to republish under another identity. The application needs explicit records
for the rights actually offered. Technical policy enforcement is not proof that
the uploader is the legitimate author.

False authorship and copyright fraud cannot be prevented universally. Creator
identity, publication provenance, and ways to report disputed offers need design.
Do not infer authorship from SHA-256, a signature alone, a wallet address, or an
uploaded file. A signature can establish which key made a claim.

No blanket legal conclusion about private friend streaming was agreed. The user
asked to focus the product around declared creator permissions and payment to the
creator rather than promise universal fraud prevention.

## Collection offers and receipts

**Proposed records, not a finalized wire format:**

1. A creator-signed offer identifies the creator and collection/item revision,
   included representations, terms/license version, price, payment recipient,
   expiry, and fulfillment policy.
2. A purchase request binds a buyer identity or receiving key to that offer.
   Define how it maps a payment to the correct order without exposing private
   relationships in public discovery records.
3. A verified payment produces a durable entitlement/receipt under the chosen
   scheme. It references the offer and granted scope, rather than trusting a
   client's assertion that it paid.
4. Hosts verify entitlement and their own permission to serve the requested
   representation. Catalog host, payment verifier, key issuer, and file provider
   may be separate roles.

The NFT analogy was an off-chain media copy with a purchase/license record. It
does not require a scarce transferable NFT or storing the media on-chain. A
normal signed receipt or another entitlement scheme may fit the outcome.
No blockchain, token standard, or wallet integration was chosen.

## When the creator is offline

Sending money to an address can be independently verifiable on a chosen payment
network. It does not by itself issue an entitlement or reveal a decryption key.
Offline purchases require the creator to prepare the offer and fulfillment path
before going offline.

**Proposed flow**

1. The creator publishes a durable signed offer and collection revision.
2. Authorized hosts retain the catalog and required representations. For encrypted
   replication, bind ciphertext identities and decoded representation identities
   explicitly; provider lookup must not conflate them.
3. An available verifier checks the payment against the exact offer/order and
   its settlement rules. The entitlement mechanism must work without a live
   response from the creator's desktop.
4. An available authorized fulfillment service releases a buyer-bound license or
   key. This could be a delegated issuer or another evaluated mechanism. Merely
   putting a decryption key in a public manifest would remove the access boundary.
5. The buyer discovers eligible hosts, verifies their service grants, downloads
   or streams the authorized representation, and verifies integrity.
6. A failed provider can be replaced without changing the purchased publication
   or playhead. If no eligible data/key host is reachable, show pending/unavailable
   delivery and provide the agreed recovery path.

Creator-offline availability needs retained data **and** a working entitlement/key
path. A relay cannot supply files it does not retain. A blockchain cannot retrieve
off-chain media that nobody is serving. Offline purchase is not a guarantee of
instant delivery if all authorized fulfillment hosts are unavailable.

The system must not rely on a purchaser manually trusting an arbitrary host's
payment claim or granting owner access to a replica provider. Paid retrieval still
uses collection-scoped policy and stable source identities.

## Protection levels and open-source limits

The user explored preventing copying, keeping media in encrypted memory, and
charging for full quality. The design must separate enforceable delivery gates
from promises about an entitled recipient's machine.

| Approach | Honest product promise |
| --- | --- |
| Preview rendition | An unauthorized viewer receives only the actual lower-quality file |
| Authenticated streaming | Only currently authorized requests receive the stream; UI can omit Download |
| Encrypted retained replicas | Storage/transport hosts need not receive plaintext; permitted recipients require key/license fulfillment |
| Download purchase | An entitled buyer can retain the licensed copy |
| Strong platform DRM | Candidate requiring trusted platform/decoder integration and separate feasibility work |

Previews must be distinct server-delivered lower-resolution/lower-bitrate assets,
not full files scaled down in CSS or hidden behind a client button. Each rendition
has its own identity and access policy.

Stream-only controls, short-lived URLs, and encrypted storage do not make decoded
media impossible to copy. An editable open-source client cannot be the sole trust
anchor for withholding keys, preventing export, or accepting payment claims.
Even a strong DRM arrangement cannot promise absolute capture prevention.
Encryption can protect media from an unentitled recipient or an untrusted replica
host; it does not permanently confine media after an entitled endpoint can decode it.

Strong DRM would require trust outside the editable application and introduce
platform support, licensing, and usability tradeoffs. No DRM provider or protected
playback path was selected. Start with clear permission and rendition controls;
do not advertise them as copy-proof.

### SVP and protected playback

The discussion distinguished ordinary video streams from strong DRM. SVP needs
access to decoded frames for interpolation. A secure protected decoder path may
prevent that access, so strong protection and unrestricted SVP processing cannot
be promised together. Unprotected or creator-permitted processing can use the
ordinary compatible path, subject to actual format/runtime support.

The user accepted that the earlier YouTube/VLC example did not establish a strong
DRM requirement. Any future protection option must state its playback and SVP
compatibility instead of silently changing the chosen player behavior.

## Open decisions before implementation

- Payment method, settlement network, currency/price handling, confirmations, and
  transaction-to-order binding; crypto is an idea, not a mandated implementation.
- Buyer identity/key recovery, proof privacy, replay/duplicate purchase handling.
- Entitlement issuance: delegated trust, signer/key rotation, compromise recovery,
  expiration, and revocation policy.
- Fulfillment availability, quotas, incentives/fees for data and key hosts, and
  what happens when payment succeeds but delivery fails.
- Refund/dispute handling and counterfeit creator/offer reports.
- Whether a collection purchase covers a pinned revision or later additions.
- Downloadable friend-sharing terms versus streaming-only grants; who can mirror,
  to whom, and for how long.
- Which catalogs/providers are publicly discoverable and when broad paid-content
  provider lookup is allowed. Avoid leaking invite-only membership or holdings.
- Protection tiers supported across desktop, browser, Android, and iOS; their
  actual interaction with codecs, caching, offline access, and SVP.

Revocation can deny future compliant delivery or key issuance. It cannot promise
erasure of already downloaded plaintext. Policy and buyer-facing terms need to
reflect that limit.

## Acceptance gates for a future slice

These are proposed gates, not current test results:

1. A free/preview collection and a paid full rendition enforce distinct delivery
   permissions at catalog, artwork, stream/range, and download endpoints.
2. A valid purchase receives the promised rights; invalid, wrong-offer, replayed,
   or insufficient payment claims do not grant access.
3. A creator goes offline after preparing an offer and authorized replicas. A
   fresh buyer can complete a purchase and retrieve the exact permitted content.
4. A malicious or stale provider cannot substitute content, upgrade permissions,
   or change the recipient/payment destination.
5. Loss of the last data or key host produces truthful unavailable/pending status
   and the chosen purchase-recovery behavior.
6. Personal downloads, friend streams, and rehosting follow their independently
   defined grants. Private relationships do not enter public provider discovery.

Commerce remains a separately scoped future feature. It must not block the
immediate gallery/settings/private-source improvements or be silently added to
the creation-add-on implementation.
