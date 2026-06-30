import json
import hashlib
import tarfile
import io
import requests

# To get ALL reports ever, we need the last dump before the Oct 2019 schema change, 
# AND the latest dump (which contains everything from Oct 2019 onwards).
DUMP_URLS = [
    "https://raw.githubusercontent.com/bdefore/protondb-data/master/reports/reports_nov1_2019.tar.gz",
    "https://raw.githubusercontent.com/bdefore/protondb-data/master/reports/reports_apr1_2026.tar.gz"
]

def parse_v1_schema(report):
    """Legacy schema (August 2018 - October 2019)"""
    return {
        "app_id": str(report.get("appId", "")),
        "title": report.get("title", "Unknown"),
        "proton_version": report.get("protonVersion", "unknown"),
        "os": report.get("os", "unknown"),
        "gpu_driver": report.get("gpuDriver", "unknown"),
        "specs": report.get("specs", "unknown"),
        "rating": report.get("rating", "unknown"),
        "notes": report.get("notes", "").strip(),
        "timestamp": report.get("timestamp", 0)
    }

def parse_v2_schema(report):
    """Modern schema (October 2019 - Present, updated Feb 2022)"""
    app_id = report.get("app", {}).get("steam", {}).get("appId", "")
    title = report.get("app", {}).get("title", "Unknown")
    
    responses = report.get("responses", {})
    sys_info = report.get("systemInfo", {})
    
    # Combine user notes with any explicit launch options provided in the Feb 2022+ schema
    notes_extra = responses.get("notes", {}).get("extra", "")
    launch_options = responses.get("launchOptions", "")
    full_notes = f"{notes_extra} {launch_options}".strip()

    return {
        "app_id": str(app_id),
        "title": title,
        "proton_version": responses.get("protonVersion", responses.get("customProtonVersion", "unknown")),
        "os": sys_info.get("os", "unknown"),
        "gpu_driver": sys_info.get("gpuDriver", "unknown"),
        "specs": f"{sys_info.get('cpu', '')} / {sys_info.get('gpu', '')} / {sys_info.get('ram', '')}",
        "rating": responses.get("verdict", "unknown"),
        "notes": full_notes,
        "timestamp": report.get("timestamp", 0)
    }

def main():
    try:
        with open("seen_hashes.json", "r") as f:
            seen_hashes = set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        seen_hashes = set()

    raw_reports = []  
    new_count = 0
    skipped_empty = 0
    skipped_duplicate = 0

    for url in DUMP_URLS:
        print(f"📦 Downloading: {url.split('/')[-1]}...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"❌ Failed to download: HTTP {response.status_code}")
            continue

        print(f"📂 Extracting and filtering...")
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".json"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue

                try:
                    reports = json.loads(f.read().decode("utf-8"))
                    if not isinstance(reports, list):
                        reports = [reports]

                    for report in reports:
                        # Detect schema
                        if "appId" in report:
                            parsed = parse_v1_schema(report)
                        else:
                            parsed = parse_v2_schema(report)

                        notes = parsed["notes"]

                        # Only skip completely empty notes
                        if not notes:
                            skipped_empty += 1
                            continue

                        # Hash Deduplication (only filter we apply)
                        text_hash = hashlib.sha256(notes.encode("utf-8")).hexdigest()
                        if text_hash in seen_hashes:
                            skipped_duplicate += 1
                            continue

                        seen_hashes.add(text_hash)
                        raw_reports.append(parsed)
                        new_count += 1

                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    print(f"\n📊 Results:")
    print(f"   ✅ New reports to process: {new_count}")
    print(f"   ⏭️  Skipped (empty):         {skipped_empty}")
    print(f"   ⏭️  Skipped (duplicate):     {skipped_duplicate}")
    print(f"   📁 Total hashes tracked:    {len(seen_hashes)}")

    with open("seen_hashes.json", "w") as f:
        json.dump(list(seen_hashes), f)

    with open("raw_reports.json", "w") as f:
        json.dump(raw_reports, f)

if __name__ == "__main__":
    main()
