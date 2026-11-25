import os
import ssl
import time
import urllib.request

_CTX = {"ctx": None, "insecure": False}


def _verified_ctx():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def get_ctx():
    if _CTX["ctx"] is None:
        _CTX["ctx"] = ssl._create_unverified_context() if os.environ.get("CRS_INSECURE") == "1" else _verified_ctx()
    return _CTX["ctx"]


def http_get(url, tries=4, timeout=60):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=get_ctx()) as r:
                return r.read()
        except Exception as e:
            last = e
            if "CERTIFICATE_VERIFY_FAILED" in str(e) and not _CTX["insecure"]:
                print("SSL certificate check failed; retrying without verification (public files only).")
                _CTX["insecure"] = True
                _CTX["ctx"] = ssl._create_unverified_context()
                continue
            time.sleep(2 * (k + 1))
    raise last
