"""DriveGuard dataset step 1: obtain UTA-RLDD subject videos (real data only).

Downloads selected members directly out of the public UTA-RLDD fold archives
(Google Drive, shared by the dataset authors) using HTTP Range requests, so a
multi-gigabyte archive never has to be fully downloaded. Nothing is ever
synthesized: files are stored exactly as distributed.

Source: Ghoddoosian, Galib, Athitsos (2019), "A Realistic Dataset and Baseline
Temporal Model for Early Drowsiness Detection", arXiv:1904.07312
https://sites.google.com/view/utarldd/home

Usage:
  python 01_download_dataset.py --plan 00_plan.json --raw ../data/dataset/raw
"""
import argparse
import concurrent.futures
import io
import json
import os
import struct
import sys
import time
import zlib
import urllib.request

FOLD_IDS = {
    "Fold1_part1": "16x8afSOwfbdGAjUbfRFvwGj1ie8Lf5Do",
    "Fold1_part2": "1qPoJ06jeznytP_BGJ25hxdQy7eYTTjyk",
    "Fold2_part1": "12uVMVLv_kRXzQb4QDdru9RK3G9hzfpPQ",
    "Fold2_part2": "1nYGUpqE3x5ZeRLgDeMlUN4K_ccPjM2pe",
    "Fold3_part1": "1-JhE5TfkNyr4UbQk6PSkkpI7z-SChkWS",
    "Fold3_part2": "1LZU5KfJkFMj2pIkGwb3RjHDazoUfxSYQ",
    "Fold4_part1": "1IYKdrjvawf90bG_cfeoWA4rZSymllFbD",
    "Fold4_part2": "1zJlASBwABBgVSULQLiia0R_Eqar8_B3D",
    "Fold5_part1": "16w30wEgcAbaUr4gx9o4sl_2TbIunD5yA",
    "Fold5_part2": "1loOQE1-hoNM5dIOjXHI2Ss5NjjoPZcX6",
}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset", "utarldd_source_manifest.json")


def url_for(file_id):
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def fetch_range(url, start, end, retries=3):
    """Ranged GET that fails fast on non-206 responses (quota interstitials)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status != 206 or not r.headers.get("Content-Range"):
                    raise RuntimeError("non-206 response (likely download quota interstitial)")
                data = r.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"short read {len(data)} != {end-start+1}")
            return data
        except RuntimeError:
            if attempt == retries - 1:
                raise
            time.sleep(4 * (attempt + 1))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(4 * (attempt + 1))


def remote_size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])


def read_central_directory(url):
    """List zip members via ranged reads (no full download)."""
    size = remote_size(url)
    tail = fetch_range(url, max(0, size - 3_000_000), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    z64 = tail.rfind(b"PK\x06\x07")
    if z64 >= 0:
        off = struct.unpack("<Q", tail[z64 + 8:z64 + 16])[0]
        rec = fetch_range(url, off, off + 55)
        assert rec[:4] == b"PK\x06\x06"
        cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    else:
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    cd = fetch_range(url, cd_off, cd_off + cd_size - 1)
    members, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (c_size, u_size, nlen, elen, clen) = struct.unpack("<IIHHH", cd[p + 20:p + 34])
        lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals = struct.unpack("<" + "Q" * (hsz // 8), extra[q + 4:q + 4 + hsz])
                vi = 0
                if u_size == 0xFFFFFFFF and vi < len(vals):
                    u_size = vals[vi]; vi += 1
                if c_size == 0xFFFFFFFF and vi < len(vals):
                    c_size = vals[vi]; vi += 1
                if lho == 0xFFFFFFFF and vi < len(vals):
                    lho = vals[vi]
            q += 4 + hsz
        members.append({"name": name, "header_offset": lho, "compress_size": c_size,
                        "file_size": u_size, "compress_type": struct.unpack("<H", cd[p + 10:p + 12])[0]})
        p += 46 + nlen + elen + clen
    return members


def fetch_member(url, member, out_path, chunk=8 << 20, workers=1, progress=True):
    """Stream one zip member (stored or deflated) to out_path via ranged GETs.

    Compressed chunks are fetched in parallel (per-connection throughput to the
    archive host is the bottleneck) into a sparse staging file, then inflated
    sequentially -- deflate output must be produced in stream order.
    """
    lh = fetch_range(url, member["header_offset"], member["header_offset"] + 512)
    assert lh[:4] == b"PK\x03\x04", "bad local header"
    nlen, elen = struct.unpack("<HH", lh[26:30])
    data_off = member["header_offset"] + 30 + nlen + elen
    total = member["compress_size"]

    staging = out_path + ".rawpart"
    fd = os.open(staging, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(fd, total)

    def work(pos):
        end = min(pos + chunk, total) - 1
        blob = fetch_range(url, data_off + pos, data_off + end)  # retries inside
        os.pwrite(fd, blob, pos)
        time.sleep(0.12)  # pacing: rapid-fire parallel ranges trip host quotas
        return len(blob)

    done = 0
    last_print = [0.0]
    import time as _t
    t_start = _t.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for n in concurrent.futures.as_completed(
                    [ex.submit(work, p) for p in range(0, total, chunk)]):
                done += n.result()
                now = _t.time()
                if progress and (now - last_print[0] > 15 or done >= total):
                    last_print[0] = now
                    mbps = done / 1e6 / max(now - t_start, 1e-9)
                    print(f"    {os.path.basename(out_path)}: {done/total*100:.0f}% "
                          f"({mbps:.1f}MB/s)", flush=True)
    finally:
        os.close(fd)

    dec = zlib.decompressobj(-15) if member["compress_type"] == 8 else None
    tmp = out_path + ".part"
    with open(staging, "rb") as r, open(tmp, "wb") as w:
        while True:
            blob = r.read(1 << 22)
            if not blob:
                break
            w.write(dec and dec.decompress(blob) or blob)
        if dec:
            rest = dec.flush()
            if rest:
                w.write(rest)
    os.remove(staging)
    os.replace(tmp, out_path)
    if progress:
        print(f"\r    saved {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB)      ")


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    # rebuild from Drive if the cached manifest is absent
    manifest = {}
    for fold, fid in FOLD_IDS.items():
        print(f"listing {fold}...")
        manifest[fold] = {"file_id": fid, "videos": read_central_directory(url_for(fid))}
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def find_member(manifest, fold, subject, state):
    vids = [v for v in manifest[fold]["videos"] if f"/{subject}/" in v["name"]
            and v["name"].rsplit("/", 1)[-1].split(".")[0] == str(state)]
    return vids[0] if vids else None


def video_path_for(raw_dir, fold, subject, state, member):
    ext = member["name"].rsplit(".", 1)[-1].lower()
    return os.path.join(raw_dir, f"utarldd_{fold}_s{subject}_{state}.{ext}")


def download_item(manifest, item, raw_dir):
    member = find_member(manifest, item["fold"], str(item["subject"]), item["state"])
    if member is None:
        raise FileNotFoundError(f"no member for {item}")
    out = video_path_for(raw_dir, item["fold"], str(item["subject"]), item["state"], member)
    if not os.path.exists(out):
        print(f"  downloading {member['name']} ({member['file_size']/1e6:.0f}MB)")
        fetch_member(url_for(manifest[item["fold"]]["file_id"]), member, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(os.path.dirname(__file__), "00_plan.json"))
    ap.add_argument("--raw", default=os.path.join(os.path.dirname(__file__), "..", "data", "dataset", "raw"))
    ap.add_argument("--max-seconds", type=float, default=1e9)
    args = ap.parse_args()
    os.makedirs(args.raw, exist_ok=True)
    manifest = load_manifest()
    plan = json.load(open(args.plan))
    t0 = time.time()
    for item in plan["items"]:
        if time.time() - t0 > args.max_seconds:
            print("time budget reached"); break
        try:
            download_item(manifest, item, args.raw)
        except Exception as e:
            print(f"  [error] {item}: {e}")


if __name__ == "__main__":
    main()
