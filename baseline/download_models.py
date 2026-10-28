"""Download the approved, pinned model files directly into this project.

curl uses a process-local DoH resolver because this host has no active system
nameserver. All redirects bypass proxies. No system/network settings are changed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="artifacts/local_gpu/download_plan.json")
    parser.add_argument("--output", default="models")
    args = parser.parse_args(argv)
    plans = json.loads(Path(args.plan).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    jobs = []
    for plan in plans:
        folder = output / plan["repository"].split("/")[-1]
        folder.mkdir(exist_ok=True)
        for entry in plan["files"]:
            jobs.append((plan, entry, folder / entry["file"]))

    def download(job):
        plan, entry, path = job
        expected = entry["bytes"]
        if not expected:
            raise ValueError(f"Unknown size: {path}")
        if path.exists() and path.stat().st_size != expected:
            raise ValueError(f"Existing file has a different size: {path}")
        if not path.exists():
            part = path.with_name(path.name + ".part")
            print(f"Downloading {path} ({expected:,} bytes)", flush=True)
            url = (f"https://huggingface.co/{plan['repository']}/resolve/"
                   f"{plan['revision']}/{entry['file']}?download=true")
            subprocess.run([
                "curl", "--noproxy", "*", "--doh-url", "https://1.1.1.1/dns-query",
                "--location", "--fail", "--silent", "--show-error",
                "--connect-timeout", "20", "--retry", "3", "--retry-delay", "2",
                "--retry-all-errors", "--continue-at", "-", "--output", str(part), url,
            ], env=env, check=True)
            if part.stat().st_size != expected:
                raise ValueError(f"Incomplete {path}: {part.stat().st_size} != {expected}")
            part.rename(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        print(f"Ready {path}", flush=True)
        return {"path": str(path), "bytes": expected, "sha256": digest.hexdigest(),
                "repository": plan["repository"], "revision": plan["revision"]}

    done = []
    total = sum(entry["bytes"] for _, entry, _ in jobs)
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {executor.submit(download, job) for job in jobs}
        while pending:
            completed, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in completed:
                done.append(future.result())
            current = 0
            for _, _, path in jobs:
                part = path.with_name(path.name + ".part")
                if path.exists():
                    current += path.stat().st_size
                elif part.exists():
                    current += part.stat().st_size
            print(f"Progress {current / 1e9:.2f}/{total / 1e9:.2f} GB; "
                  f"verified files {len(done)}/{len(jobs)}", flush=True)
    (output / "download_manifest.json").write_text(json.dumps(done, indent=2) + "\n")


if __name__ == "__main__":
    main()
