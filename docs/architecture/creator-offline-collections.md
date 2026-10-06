# Creator collections and voluntary offline hosting

Approved product direction: connecting to a hub provides creator discovery,
not a requirement to download the hub. A creator collection is publisher-owned;
artist attribution remains separate from the authenticated publishing account.
The account has a stable internal identity linked to Discord's stable user ID.

The community stays in Collections and uses the image/video gallery, filters,
tile sizing and lightbox. Its sidebar has searchable creators, Available,
Following, Available offline, and Show all collections. Directory selection does
not silently filter a gallery; selecting a creator does. Following is distinct
from keeping that creator's content offline. Followed creators remain visible
when nobody is currently serving their content.

## Offline copies

An explicit download keeps media, thumbnails and metadata in managed local
storage. Users review a positive storage budget and whether to help host before
starting. They may sync one creator or all current and future creators. Workers
are bounded, resumable across restarts, and show progress, errors, pauses and
budget exhaustion. Unknown size or incomplete coverage is not reported as a
complete offline collection. Disabling sync stops updates and sharing while
retaining the personal files unless the user separately removes them.

Personal use of an explicitly downloaded copy is retained, as with Import; it
does not require a continuously valid network login. Offline browsing is explicit
and respects the local family lock and source rating policy. Network gallery
requests do not fall back to protected cached results after authorization errors.
Local copy access is distinct from permission to advertise or serve that copy.

## Hosting and availability

Hosting shares only managed, explicitly authorized replicas while DMC is running.
Source origin and realm, publication, representation and content hash qualify all
replicas. Mere possession of a file does not establish its publishing owner.
Transfers verify the exact content hash. Ordinary community credentials are
never handed to another user's device; serving uses narrowly scoped, short-lived
hub authorization. Withdrawal, ban or changed publication removes eligibility
for community sharing after current authorization is checked. A personal copy
may remain, but is not an active community provider.

Creator availability derives from reachable media providers, not the creator's
login status. Available, partially available, offline and unknown are distinct.
Temporary lack of providers hides entries only in the default available view;
Show all collections preserves exploration, and Following retains chosen entries.
Provider claims expire. Peer lookup and content delivery preserve the selected
transport mode, source realm and access restrictions.

## Migration

Bunny-backed media remains a real provider during rollout. Discord login and
existing ownership remain intact. This implementation does not turn off the
Discord ingest bridge or delete existing storage; those cutovers require verified
replacement publishing and replication coverage. DonutBooru retains catalog,
search, moderation, identity and discovery services. Peer replication reduces
central media hosting but does not promise availability when every copy is offline.
New login methods link to existing internal accounts rather than transferring
ownership by matching display names.
