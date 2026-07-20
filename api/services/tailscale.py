"""
Tailscale detection for share stream feature.

Detects if Tailscale is installed and running, and provides URLs for sharing.
Uses `tailscale serve` to set up HTTPS reverse proxy for share links.
"""

import asyncio
import json
import logging
import platform
import time
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Cache detection result for 60 seconds
_cache: Optional[Dict] = None
_cache_time: float = 0
CACHE_TTL = 60

# Cache serve status per port
_serve_cache: Dict[int, bool] = {}
_serve_cache_time: float = 0

# Whether serve setup failed due to permissions (resets with cache TTL)
_needs_operator: bool = False
_needs_operator_time: float = 0


async def _run_cmd(cmd: list, timeout: float = 5) -> tuple:
    """Run a command asynchronously, return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')
    except FileNotFoundError:
        return -1, '', 'command not found'
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, '', 'timeout'
    except Exception as e:
        return -1, '', str(e)


async def detect_tailscale() -> Optional[Dict]:
    """
    Detect if Tailscale is installed and running.

    Returns dict with:
        - ip: Tailscale IP address
        - hostname: Machine's Tailscale hostname
        - dns_name: Full DNS name (e.g., my-pc.tailnet-name.ts.net)
    Or None if Tailscale is not available.
    """
    global _cache, _cache_time

    if _cache is not None and (time.time() - _cache_time) < CACHE_TTL:
        return _cache

    returncode, stdout, stderr = await _run_cmd(['tailscale', 'status', '--json'])

    if returncode != 0:
        if 'command not found' in stderr:
            logger.debug("[Tailscale] Not installed (tailscale command not found)")
        else:
            logger.debug(f"[Tailscale] Detection failed: {stderr}")
        _cache = None
        _cache_time = time.time()
        return None

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        _cache = None
        _cache_time = time.time()
        return None

    # Get this machine's info from the self node
    self_node = data.get('Self', {})
    tailscale_ips = self_node.get('TailscaleIPs', [])
    dns_name = self_node.get('DNSName', '').rstrip('.')
    hostname = self_node.get('HostName', '')

    if not tailscale_ips:
        _cache = None
        _cache_time = time.time()
        return None

    # Use first IPv4 address
    ip = None
    for addr in tailscale_ips:
        if '.' in addr:  # IPv4
            ip = addr
            break
    if not ip:
        ip = tailscale_ips[0]

    _cache = {
        'ip': ip,
        'hostname': hostname,
        'dns_name': dns_name,
    }
    _cache_time = time.time()
    logger.info(f"[Tailscale] Detected: IP={ip}, DNS={dns_name}")
    return _cache


async def check_tailscale_serve(port: int) -> bool:
    """Check if tailscale serve is already configured for this port."""
    global _serve_cache, _serve_cache_time

    if port in _serve_cache and (time.time() - _serve_cache_time) < CACHE_TTL:
        return _serve_cache[port]

    returncode, stdout, stderr = await _run_cmd(['tailscale', 'serve', 'status', '--json'])

    if returncode == 0 and stdout.strip():
        try:
            data = json.loads(stdout)
            # Check if our port is configured in the serve config
            status_str = json.dumps(data)
            is_configured = str(port) in status_str
            _serve_cache[port] = is_configured
            _serve_cache_time = time.time()
            return is_configured
        except json.JSONDecodeError:
            pass

    _serve_cache[port] = False
    _serve_cache_time = time.time()
    return False


async def setup_tailscale_funnel(https_port: int, local_port: int) -> bool:
    """
    Set up tailscale funnel to proxy HTTPS to local HTTP.

    Uses funnel (not just serve) so it works from the public internet.
    Funnel ports are limited to 443, 8443, 10000.
    e.g. https://<hostname>:8443 → http://127.0.0.1:8790
    """
    global _serve_cache, _serve_cache_time, _needs_operator, _needs_operator_time

    if await check_tailscale_serve(https_port):
        return True  # Already configured

    returncode, stdout, stderr = await _run_cmd(
        ['tailscale', 'funnel', '--bg', f'--https={https_port}', f'http://127.0.0.1:{local_port}'],
        timeout=15,
    )

    if returncode == 0:
        logger.info(f"[Tailscale] Set up HTTPS funnel: :{https_port} → :{local_port}")
        _serve_cache[https_port] = True
        _serve_cache_time = time.time()
        _needs_operator = False
        return True
    else:
        if 'Access denied' in stderr or 'operator' in stderr.lower():
            _needs_operator = True
            _needs_operator_time = time.time()
            logger.warning(f"[Tailscale] Funnel needs operator access: {stderr.strip()}")
        else:
            logger.warning(f"[Tailscale] Failed to set up funnel: {stderr.strip()}")
        return False


def needs_operator_setup() -> bool:
    """Check if tailscale serve failed due to missing operator permissions."""
    # Reset after cache TTL so it retries after user fixes permissions
    if _needs_operator and (time.time() - _needs_operator_time) > CACHE_TTL:
        return False
    return _needs_operator


# HTTPS port used by tailscale serve (separate from the local server port)
TAILSCALE_HTTPS_PORT = 8443


async def get_tailscale_url(local_port: int) -> Optional[str]:
    """Get the Tailscale URL for this machine, if available.

    Attempts to set up HTTPS via `tailscale serve` on a separate port (8443)
    that proxies to the local server. Falls back to HTTP if that fails.
    """
    info = await detect_tailscale()
    if not info:
        return None

    if info['dns_name']:
        # Try to set up HTTPS funnel on a separate port
        if await setup_tailscale_funnel(TAILSCALE_HTTPS_PORT, local_port):
            return f"https://{info['dns_name']}:{TAILSCALE_HTTPS_PORT}"
        # Fall back to HTTP via Tailscale IP
        return f"http://{info['dns_name']}:{local_port}"

    return f"http://{info['ip']}:{local_port}"


async def is_tailscale_https(local_port: int) -> bool:
    """Check if Tailscale HTTPS is active."""
    return await check_tailscale_serve(TAILSCALE_HTTPS_PORT)


def get_os_name() -> str:
    """Get OS name for Tailscale setup link."""
    system = platform.system().lower()
    if system == 'linux':
        return 'linux'
    elif system == 'darwin':
        return 'macos'
    elif system == 'windows':
        return 'windows'
    return system
