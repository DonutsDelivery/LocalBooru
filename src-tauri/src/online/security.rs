use crate::server::error::AppError;
use reqwest::Url;
use std::net::IpAddr;

pub fn normalize_base_url(raw: &str, allow_local_network: bool) -> Result<String, AppError> {
    let mut url = Url::parse(raw.trim())
        .map_err(|_| AppError::BadRequest("Source URL must be an absolute http(s) URL".into()))?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err(AppError::BadRequest(
            "Source URLs cannot contain credentials".into(),
        ));
    }
    match url.scheme() {
        "https" => {}
        "http" if allow_local_network => {}
        "http" => {
            return Err(AppError::BadRequest(
                "Internet sources must use HTTPS".into(),
            ))
        }
        _ => {
            return Err(AppError::BadRequest(
                "Only http and https sources are supported".into(),
            ))
        }
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(AppError::BadRequest(
            "Source base URL cannot contain query or fragment".into(),
        ));
    }
    let normalized_path = url.path().trim_end_matches('/').to_string();
    url.set_path(&normalized_path);
    Ok(url.to_string().trim_end_matches('/').to_string())
}
pub fn address_is_private(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || ip.octets()[0] == 0
                || ip.octets()[0] >= 224
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || (ip.segments()[0] & 0xfe00) == 0xfc00
                || (ip.segments()[0] & 0xffc0) == 0xfe80
                || ip
                    .to_ipv4_mapped()
                    .is_some_and(|v| address_is_private(IpAddr::V4(v)))
        }
    }
}
pub async fn validate_resolved_url(url: &Url, allow_local_network: bool) -> Result<(), AppError> {
    let host = url
        .host_str()
        .ok_or_else(|| AppError::BadRequest("Source URL has no host".into()))?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| AppError::BadRequest("Source URL has no usable port".into()))?;
    let addresses: Vec<_> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|_| AppError::ServiceUnavailable("Could not resolve source host".into()))?
        .map(|e| e.ip())
        .collect();
    if addresses.is_empty() {
        return Err(AppError::ServiceUnavailable(
            "Source host resolved to no addresses".into(),
        ));
    }
    if !allow_local_network && addresses.iter().any(|a| address_is_private(*a)) {
        return Err(AppError::Forbidden(
            "Internet source resolved to a private or reserved address".into(),
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocks_private() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "169.254.1.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
        ] {
            assert!(address_is_private(ip.parse().unwrap()));
        }
        assert!(!address_is_private("8.8.8.8".parse().unwrap()));
    }
    #[test]
    fn validates_base() {
        assert!(normalize_base_url("http://example.com", false).is_err());
        assert!(normalize_base_url("https://u:p@example.com", false).is_err());
        assert_eq!(
            normalize_base_url("https://example.com/", false).unwrap(),
            "https://example.com"
        );
    }
}
