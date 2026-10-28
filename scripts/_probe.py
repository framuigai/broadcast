import re
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

PAT = re.compile(
    r"https?://[^\s\"'<>\\)]+?(?:\.mp3|\.aac|\.m3u8|\.pls|/stream[\w/]*|:\d{4,5}/[\w\-./]*)",
    re.IGNORECASE,
)
NOISE = ("jquery", ".css", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".woff",
         "google", "facebook", "gstatic", "fonts", "gravatar", "wp-content/themes")


def probe(url: str) -> None:
    print(f"\n=== {url} ===")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception as exc:                        # noqa: BLE001 — probing, report and move on
        print(f"  FETCH FAILED: {type(exc).__name__}: {exc}")
        return

    seen: list[str] = []
    for m in PAT.findall(html):
        low = m.lower()
        if any(n in low for n in NOISE):
            continue
        if m not in seen:
            seen.append(m)

    for h in seen[:20]:
        print("  ", h)
    if not seen:
        print("   (no stream-like URL in the HTML — probably injected by JavaScript)")
        # Fall back to looking for common player config keys.
        for key in ("streamUrl", "stream_url", "audioUrl", "src:", "data-url", "mp3"):
            for m in re.findall(r'.{0,70}' + re.escape(key) + r'.{0,90}', html, re.I)[:3]:
                print("   hint:", " ".join(m.split())[:160])


if __name__ == "__main__":
    for u in sys.argv[1:]:
        probe(u)
