from ipaddress import ip_address
from urllib.parse import urlparse

import httpx

from ..config import get_settings


def search_web(query: str) -> str:
    endpoint = get_settings().web_search_url
    if not endpoint:
        raise RuntimeError("Web search is not configured.")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError("Web search endpoint must use HTTPS.")
    try:
        address = ip_address(parsed.hostname.strip("[]"))
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local):
        raise RuntimeError("Web search endpoint cannot target a private address.")
    response = httpx.get(endpoint, params={"q": query}, timeout=15, follow_redirects=False)
    response.raise_for_status()
    return response.text[:12000]
