"""
TLS Certificate management for HTTPS with self-signed certificates.

Provides Syncthing-style security: server generates self-signed TLS certificate,
includes fingerprint in QR code, mobile app pins to that fingerprint.
"""
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from ..config import get_data_dir
from .network import get_all_local_ips


# Certificate settings
CERT_KEY_SIZE = 4096  # RSA key size in bits
CERT_VALIDITY_YEARS = 10


def _get_cert_paths() -> Tuple[Path, Path]:
    """Get paths to the TLS certificate and key files."""
    data_dir = get_data_dir()
    cert_path = data_dir / "tls_cert.pem"
    key_path = data_dir / "tls_key.pem"
    return cert_path, key_path


def _generate_certificate() -> Tuple[bytes, bytes]:
    """
    Generate a self-signed TLS certificate with local IPs as SANs.

    Returns:
        Tuple of (certificate_pem, private_key_pem)
    """
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=CERT_KEY_SIZE,
    )

    # Get all local IPs for Subject Alternative Names
    local_ips = get_all_local_ips()

    # Build list of SANs (Subject Alternative Names)
    san_entries = [
        x509.DNSName("localhost"),
    ]

    # Add IP addresses as SANs
    from ipaddress import ip_address
    san_entries.append(x509.IPAddress(ip_address("127.0.0.1")))
    san_entries.append(x509.IPAddress(ip_address("::1")))

    for ip in local_ips:
        try:
            san_entries.append(x509.IPAddress(ip_address(ip)))
        except ValueError:
            # Skip invalid IPs
            pass

    # Build certificate subject
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "LocalBooru"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LocalBooru"),
    ])

    # Build certificate
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=CERT_VALIDITY_YEARS * 365))
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Serialize to PEM format
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return cert_pem, key_pem


def get_or_create_certificate() -> Tuple[Path, Path]:
    """
    Get existing certificate paths or create a new self-signed certificate.

    Returns:
        Tuple of (cert_path, key_path)
    """
    cert_path, key_path = _get_cert_paths()

    # Check if both files exist
    if cert_path.exists() and key_path.exists():
        print(f"[Certificate] Using existing certificate: {cert_path}")
        return cert_path, key_path

    # Generate new certificate
    print("[Certificate] Generating new self-signed TLS certificate...")
    cert_pem, key_pem = _generate_certificate()

    # Write files with restrictive permissions
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)

    # Set restrictive permissions on key file (owner read only)
    try:
        key_path.chmod(0o600)
    except Exception:
        pass  # Windows doesn't support chmod

    print(f"[Certificate] Created certificate: {cert_path}")
    print(f"[Certificate] Created private key: {key_path}")

    # Log the fingerprint
    fingerprint = get_certificate_fingerprint()
    if fingerprint:
        print(f"[Certificate] Fingerprint: {fingerprint}")

    return cert_path, key_path


def get_certificate_fingerprint() -> Optional[str]:
    """
    Get the SHA-256 fingerprint of the TLS certificate.

    Returns the fingerprint as a colon-separated hex string (e.g., "AA:BB:CC:...")
    or None if no certificate exists.
    """
    cert_path, _ = _get_cert_paths()

    if not cert_path.exists():
        return None

    try:
        # Load the certificate
        cert_pem = cert_path.read_bytes()
        cert = x509.load_pem_x509_certificate(cert_pem)

        # Get the DER-encoded certificate and hash it
        cert_der = cert.public_bytes(serialization.Encoding.DER)
        fingerprint_bytes = hashlib.sha256(cert_der).digest()

        # Format as colon-separated hex string
        fingerprint = ":".join(f"{b:02X}" for b in fingerprint_bytes)

        return fingerprint
    except Exception as e:
        print(f"[Certificate] Error reading certificate: {e}")
        return None


def certificate_exists() -> bool:
    """Check if a TLS certificate already exists."""
    cert_path, key_path = _get_cert_paths()
    return cert_path.exists() and key_path.exists()


def get_certificate_info() -> Optional[dict]:
    """
    Get information about the current TLS certificate.

    Returns dict with:
        - fingerprint: SHA-256 fingerprint
        - not_before: Certificate validity start
        - not_after: Certificate validity end
        - sans: List of Subject Alternative Names
    """
    cert_path, _ = _get_cert_paths()

    if not cert_path.exists():
        return None

    try:
        cert_pem = cert_path.read_bytes()
        cert = x509.load_pem_x509_certificate(cert_pem)

        # Extract SANs
        sans = []
        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            for name in san_ext.value:
                if isinstance(name, x509.DNSName):
                    sans.append(f"DNS:{name.value}")
                elif isinstance(name, x509.IPAddress):
                    sans.append(f"IP:{name.value}")
        except x509.ExtensionNotFound:
            pass

        return {
            "fingerprint": get_certificate_fingerprint(),
            "not_before": cert.not_valid_before_utc.isoformat(),
            "not_after": cert.not_valid_after_utc.isoformat(),
            "sans": sans,
        }
    except Exception as e:
        print(f"[Certificate] Error getting certificate info: {e}")
        return None
