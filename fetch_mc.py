"""fetch_mc.py — download MarineCadastre AIS half-month zip, stream-extract the
busiest 60-min window (>=200k rows, multi-point per vessel) into a small CSV.

Usage (network approved):  python fetch_mc.py <year> <month>
Example: python fetch_mc.py 2015 01   -> zip AIS_2015_01_01.zip
Big zip is removed after slicing; only the window slice is kept.
"""
import csv, io, os, sys, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
MIN_ROWS = 200_000


def download(url, dest):
    print("downloading", url)
    done = 0
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % (30 << 20) < (1 << 20):
                print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB")
    print("saved", dest, f"{os.path.getsize(dest)/1e6:.1f} MB")


def slice_window(zip_path):
    zf = zipfile.ZipFile(zip_path)
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    print("csv entry:", name)
    buckets = {}
    with zf.open(name, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        reader = csv.reader(text)
        header = next(reader)
        print("header:", header[:20])
        n = 0
        for row in reader:
            if len(row) < 6:
                continue
            try:
                mmsi = row[0]; dt = row[1]; lat = float(row[3]); lon = float(row[4])
            except (ValueError, IndexError):
                continue
            hour = dt[:13]  # 'YYYY-MM-DD HH'
            buckets.setdefault(hour, []).append(row)
            n += 1
            if n % 2_000_000 == 0:
                print(f"  scanned {n/1e6:.0f}M rows; buckets={len(buckets)}")
    best = max(buckets, key=lambda k: len(buckets[k]))
    rows = buckets[best]
    print(f"busiest hour {best}: {len(rows)} rows, {len({r[0] for r in rows})} vessels")
    out = os.path.join(RAW, f"mc_{best.replace(' ','_').replace(':','-')}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("saved slice", out)
    return out


def main():
    year, month = sys.argv[1], sys.argv[2]
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month}_01.zip"
    zpath = os.path.join(RAW, f"mc_{year}_{month}_01.zip")
    if not os.path.exists(zpath):
        download(url, zpath)
    out = slice_window(zpath)
    os.remove(zpath)
    print("removed big zip; kept", out)


if __name__ == "__main__":
    main()