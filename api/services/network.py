"""
Network service for LocalBooru - IP classification, UPnP management, port testing
"""
import socket
import ipaddress
from typing import Optional, Literal


def classify_ip(ip: str) -> Literal["localhost", "local_network", "public"]:
    """
    Classify an IP address into access level categories.

    Returns:
        'localhost' - 127.0.0.1, ::1
        'local_network' - Private IP ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
        'public' - Everything else (internet)
    """
    try:
        addr = ipaddress.ip_address(ip)

        # Check localhost
        if addr.is_loopback:
            return "localhost"

        # Check private/local network ranges
        if addr.is_private:
            return "local_network"

        # Everything else is public
        return "public"

    except ValueError:
        # Invalid IP, treat as public for safety
        return "public"


def get_local_ip() -> Optional[str]:
    """
    Get the machine's primary LAN IP address.

    Returns the IP that would be used to connect to an external host.
    """
    try:
        # Create a dummy socket to determine our outgoing IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # Doesn't actually connect, just determines the route
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip
    except Exception:
        return None


def get_all_local_ips() -> list[str]:
    """Get all local IP addresses on this machine."""
    ips = []
    try:
        hostname = socket.gethostname()
        # Get all IPs associated with hostname
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and ip != '127.0.0.1':
                ips.append(ip)
    except Exception:
        pass

    # Also try the routing method
    primary = get_local_ip()
    if primary and primary not in ips and primary != '127.0.0.1':
        ips.insert(0, primary)

    return ips


def test_port_local(port: int) -> dict:
    """
    Test if a port is available for binding locally.

    Returns:
        {
            "available": bool,
            "error": str or None
        }
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.bind(('0.0.0.0', port))
        s.close()
        return {"available": True, "error": None}
    except OSError as e:
        if e.errno == 98 or e.errno == 10048:  # Address already in use (Linux/Windows)
            return {"available": False, "error": "Port is already in use"}
        elif e.errno == 13 or e.errno == 10013:  # Permission denied
            return {"available": False, "error": "Permission denied (port < 1024 requires root)"}
        return {"available": False, "error": str(e)}
    except Exception as e:
        return {"available": False, "error": str(e)}


class UPnPManager:
    """Manage UPnP port forwarding for router access."""

    def __init__(self):
        self._upnp = None
        self._external_ip = None
        self._gateway_found = False

    def _get_upnp(self):
        """Lazy-load UPnP client."""
        if self._upnp is None:
            try:
                import miniupnpc
                self._upnp = miniupnpc.UPnP()
                self._upnp.discoverdelay = 200  # 200ms timeout
            except ImportError:
                raise RuntimeError("miniupnpc not installed. Run: pip install miniupnpc")
        return self._upnp

    def discover(self) -> dict:
        """
        Discover UPnP gateway on the network.

        Returns:
            {
                "found": bool,
                "gateway": str or None,
                "external_ip": str or None,
                "error": str or None
            }
        """
        try:
            upnp = self._get_upnp()
            devices = upnp.discover()

            if devices == 0:
                self._gateway_found = False
                return {
                    "found": False,
                    "gateway": None,
                    "external_ip": None,
                    "error": "No UPnP devices found"
                }

            upnp.selectigd()  # Select the Internet Gateway Device
            self._gateway_found = True
            self._external_ip = upnp.externalipaddress()

            return {
                "found": True,
                "gateway": upnp.lanaddr,
                "external_ip": self._external_ip,
                "error": None
            }

        except Exception as e:
            self._gateway_found = False
            return {
                "found": False,
                "gateway": None,
                "external_ip": None,
                "error": str(e)
            }

    def add_port_mapping(
        self,
        external_port: int,
        internal_port: int,
        protocol: str = "TCP",
        description: str = "LocalBooru"
    ) -> dict:
        """
        Add a port mapping (open a port) via UPnP.

        Args:
            external_port: Port on router (public side)
            internal_port: Port on local machine
            protocol: "TCP" or "UDP"
            description: Human-readable name for the mapping

        Returns:
            {
                "success": bool,
                "external_port": int,
                "internal_port": int,
                "error": str or None
            }
        """
        try:
            upnp = self._get_upnp()

            if not self._gateway_found:
                # Try to discover first
                result = self.discover()
                if not result["found"]:
                    return {
                        "success": False,
                        "external_port": external_port,
                        "internal_port": internal_port,
                        "error": "No UPnP gateway found. Run discover first."
                    }

            local_ip = get_local_ip()
            if not local_ip:
                return {
                    "success": False,
                    "external_port": external_port,
                    "internal_port": internal_port,
                    "error": "Could not determine local IP address"
                }

            # Add the port mapping
            result = upnp.addportmapping(
                external_port,
                protocol,
                local_ip,
                internal_port,
                description,
                ''  # Remote host (empty = any)
            )

            if result:
                return {
                    "success": True,
                    "external_port": external_port,
                    "internal_port": internal_port,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "external_port": external_port,
                    "internal_port": internal_port,
                    "error": "Failed to add port mapping (router may have rejected it)"
                }

        except Exception as e:
            return {
                "success": False,
                "external_port": external_port,
                "internal_port": internal_port,
                "error": str(e)
            }

    def remove_port_mapping(
        self,
        external_port: int,
        protocol: str = "TCP"
    ) -> dict:
        """
        Remove a port mapping (close a port) via UPnP.

        Returns:
            {
                "success": bool,
                "external_port": int,
                "error": str or None
            }
        """
        try:
            upnp = self._get_upnp()

            if not self._gateway_found:
                result = self.discover()
                if not result["found"]:
                    return {
                        "success": False,
                        "external_port": external_port,
                        "error": "No UPnP gateway found"
                    }

            result = upnp.deleteportmapping(external_port, protocol)

            return {
                "success": True,
                "external_port": external_port,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "external_port": external_port,
                "error": str(e)
            }

    def get_port_mappings(self) -> list[dict]:
        """
        Get all current port mappings.

        Returns:
            List of port mapping dicts
        """
        mappings = []
        try:
            upnp = self._get_upnp()

            if not self._gateway_found:
                result = self.discover()
                if not result["found"]:
                    return mappings

            i = 0
            while True:
                try:
                    mapping = upnp.getgenericportmapping(i)
                    if mapping is None:
                        break
                    mappings.append({
                        "external_port": mapping[0],
                        "protocol": mapping[1],
                        "internal_host": mapping[2],
                        "internal_port": mapping[3],
                        "description": mapping[4],
                        "enabled": mapping[5],
                        "remote_host": mapping[6],
                        "lease_duration": mapping[7]
                    })
                    i += 1
                except Exception:
                    break

        except Exception:
            pass

        return mappings

    def get_external_ip(self) -> Optional[str]:
        """Get the external (public) IP address via UPnP."""
        if self._external_ip:
            return self._external_ip

        result = self.discover()
        return result.get("external_ip")


# Global UPnP manager instance
upnp_manager = UPnPManager()
