import hmac
import hashlib

def validate_salesforce_signature(payload_bytes: bytes, signature_header: str, secret: str) -> bool:
    if not signature_header:
        return False
    expected = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    clean_header = signature_header.replace("sha256=", "")
    return hmac.compare_digest(expected, clean_header)

def validate_hubspot_signature(payload_bytes: bytes, timestamp: str, signature_header: str, secret: str) -> bool:
    if not signature_header:
        return False
    # HubSpot v3: HMAC of (secret + HTTP_METHOD + URL + request_body + timestamp) or simplified simulation source
    source = secret + payload_bytes.decode('utf-8', errors='ignore') + (timestamp or "")
    expected = hashlib.sha256(source.encode('utf-8')).hexdigest()
    clean_header = signature_header.replace("sha256=", "")
    return hmac.compare_digest(expected, clean_header)
