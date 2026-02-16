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
