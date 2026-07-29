from urllib.parse import urlparse
from policy_loader import SourcePolicy

def host_matches(host: str, allowed_host: str) -> bool:
    host = host.lower().rstrip(".")
    allowed_host = allowed_host.lower().rstrip(".")
    return (host == allowed_host or host.endswith("." + allowed_host))

def is_allowed_url(url: str, policy: SourcePolicy) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False

    host = (parsed.hostname or "").lower()

    if not any(host_matches(host, allowed,) for allowed in policy.allowed_hosts):
        return False
    if (policy.allowed_url_prefixes and not any(url.startswith(prefix) for prefix in policy.allowed_url_prefixes)):
        return False
    if (policy.allowed_path_prefixes and not any(parsed.path.startswith(prefix) for prefix in policy.allowed_path_prefixes)):
        return False

    return True