use std::collections::HashSet;

use rusqlite::Connection;

use crate::server::error::AppError;
use crate::server::middleware::AccessTier;

/// Detect the primary local (non-loopback) IPv4 address.
///
/// Connects a UDP socket to a public IP (no data sent) to determine which
/// local interface the OS would route through.
pub fn get_local_ip() -> Option<String> {
    use std::net::UdpSocket;
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let addr = socket.local_addr().ok()?;
    Some(addr.ip().to_string())
}

/// Return the set of directory IDs visible to the given access tier and
/// family-mode lock state.
///
/// Returns `None` when no filtering is needed (localhost + family mode
/// unlocked), meaning ALL directories are visible. Otherwise returns
/// `Some(HashSet<i64>)` containing the visible directory IDs.
pub fn get_visible_directory_ids(
    main_conn: &Connection,
    tier: AccessTier,
    family_locked: bool,
) -> Result<Option<HashSet<i64>>, AppError> {
    // Localhost with family mode unlocked → no filtering needed
    if tier == AccessTier::Localhost && !family_locked {
        return Ok(None);
    }

    let mut conditions: Vec<&str> = Vec::new();

    // Family mode: only show family-safe directories
    if family_locked {
        conditions.push("family_safe = 1");
    }

    // Network visibility based on access tier
    match tier {
        AccessTier::Localhost => {
            // Localhost sees all (only family_safe filter applies if locked)
        }
        AccessTier::LocalNetwork => {
            conditions.push("lan_visible = 1");
        }
        AccessTier::Public => {
            conditions.push("public_access = 1");
        }
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    let sql = format!("SELECT id FROM watch_directories{}", where_clause);
    let mut stmt = main_conn.prepare(&sql).map_err(|e| {
        AppError::Internal(format!("Failed to query visible directories: {}", e))
    })?;

    let ids: HashSet<i64> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| AppError::Internal(format!("Failed to read directory IDs: {}", e)))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(Some(ids))
}
