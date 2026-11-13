"""fetch_pubdata.py — locate & download a small REAL AIS sample from public sources.

Usage (network needs approval):
    python fetch_pubdata.py --hf <term>        search HF datasets by term
    python fetch_pubdata.py --gh <query>       search GitHub repos
    python fetch_pubdata.py --ghfiles <repo> [path]
    python fetch_pubdata.py --download <url> <outname>
"""
import json, os, sys, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
os.makedirs(RAW, exist_ok=True)
HF_API = "https://huggingface.co/api/datasets"
GH_API = "https://api.github.com"


def get(url, headers=None, retries=2):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "codex-research"})
    last = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 503):
                import time; time.sleep(4); continue
            raise
        except Exception as e:
            last = e
            import time; time.sleep(2)
    raise last


def download(url, outname):
    out = os.path.join(RAW, outname)
    data = get(url)
    with open(out, "wb") as f:
        f.write(data)
    print("saved", out, len(data), "bytes")


def hf_search(term):
    url = HF_API + f"?search={urllib.parse.quote(term)}&limit=50"
    try:
        data = json.loads(get(url))
    except Exception as e:
        print("HF search failed:", e); return
    for d in data:
        sid = d.get("id", "")
        blob = (sid + " " + " ".join(d.get("tags", []))).lower()
        if any(k in blob for k in ("ais", "vessel", "ship", "maritime", "trajectory", "marine", "navigation", "navigation")):
            print("CANDIDATE", sid, "|", d.get("downloads", 0), "dl")
            try:
                sib = json.loads(get(f"{HF_API}/{sid}?blobs=true"))
                for s in sib.get("siblings", []):
                    fn = s.get("rfilename", "")
                    if fn.lower().endswith((".csv", ".parquet", ".jsonl", ".zip", ".gzip", ".gz")):
                        print("    file:", fn, s.get("size", "?"))
            except Exception as e:
                print("    (siblings failed:", e, ")")


def gh_search(q):
    url = f"{GH_API}/search/repositories?q={urllib.parse.quote(q)}&sort=updated&per_page=20"
    try:
        data = json.loads(get(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "codex-research"}))
    except Exception as e:
        print("GH search failed:", e); return
    for it in data.get("items", []):
        print("REPO", it.get("full_name"), "| size", it.get("size"), "KB |",
              (it.get("description") or "")[:80])


def gh_files(repo, path=""):
    url = f"{GH_API}/repos/{repo}/contents/{path}"
    try:
        data = json.loads(get(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "codex-research"}))
    except Exception as e:
        print("contents failed:", e); return
    for it in data:
        if it["type"] == "dir":
            print("  DIR ", it["path"])
        else:
            print("  FILE", it["path"], it.get("size", "?"))


if __name__ == "__main__":
    import urllib.parse
    mode = sys.argv[1]
    if mode == "--hf":
        hf_search(sys.argv[2])
    elif mode == "--gh":
        gh_search(sys.argv[2])
    elif mode == "--ghfiles":
        gh_files(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    elif mode == "--download":
        download(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)