#!/usr/bin/env python3
"""
Ingest script for Maldives Knowledge Base.
Scans raw/ folder, cleans text, extracts metadata, updates manifest.json.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

KB_ROOT = Path(__file__).parent
RAW_DIR = KB_ROOT / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"
LOG_DIR = KB_ROOT / "logs"

# Category keywords for auto-categorization
CATEGORIES = {
    "planning": ["best time", "season", "when to visit", "weather", "monsoon", "budget", "cost", "price", "booking"],
    "atolls": ["atoll", "island", "north malé", "south malé", "baa", "addu", "laamu", "dhaalu", "meemu"],
    "resorts": ["resort", "hotel", "villa", "guesthouse", "accommodation", "stay", "hospitality"],
    "ecology": ["coral", "reef", "bleach", "whale shark", "manta", "marine", "fish", "whale", "dolphin", "turtle", "climate", "sea level", "environment"],
    "diving": ["dive", "diving", "snorkel", "scuba", "liveaboard", "water sports"],
    "practical": ["dress", "pack", "culture", "tradition", "cuisine", "food", "visa", "connectivity", "transfer", "airport"],
}

def load_manifest():
    """Load existing manifest or create empty one."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return {"sources": []}

def save_manifest(manifest):
    """Save manifest to file."""
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

def get_file_hash(filepath):
    """Compute SHA256 hash of file for deduplication."""
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        sha.update(f.read())
    return sha.hexdigest()

def infer_category(filename, content):
    """Infer topic category from filename and content."""
    text = (filename + " " + content[:500]).lower()
    scores = {}
    for cat, keywords in CATEGORIES.items():
        scores[cat] = sum(1 for kw in keywords if kw in text)
    if max(scores.values()) == 0:
        return "other"
    return max(scores, key=scores.get)

def clean_text(filepath):
    """Read and clean file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()

    # Basic cleaning: strip extra whitespace
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    return '\n'.join(lines)

def ingest_file(filepath, manifest):
    """Process a single file and add to manifest."""
    rel_path = str(filepath.relative_to(RAW_DIR))

    # Compute hash first for dedup (catches renames)
    file_hash = get_file_hash(filepath)

    # Check if already ingested (by path or by hash)
    for s in manifest["sources"]:
        if s["path"] == rel_path or s["hash"] == file_hash:
            return f"SKIP (already ingested): {rel_path}"

    content = clean_text(filepath)
    category = infer_category(filepath.name, content)

    # Add to manifest
    source_entry = {
        "path": rel_path,
        "filename": filepath.name,
        "category": category,
        "hash": file_hash,
        "size_bytes": filepath.stat().st_size,
        "word_count": len(content.split()),
        "ingested_at": datetime.now().isoformat(),
    }
    manifest["sources"].append(source_entry)

    return f"INGEST: {rel_path} → category: {category}"

def main():
    """Scan raw/ and ingest all new files."""
    LOG_DIR.mkdir(exist_ok=True)
    manifest = load_manifest()

    # Scan all raw files
    source_files = []
    for subdir in ["articles", "guides", "images"]:
        folder = RAW_DIR / subdir
        if folder.exists():
            source_files.extend(folder.glob("*"))

    source_files = [f for f in source_files if f.is_file()]

    if not source_files:
        print("No source files found in raw/")
        return

    print(f"\n📦 Ingesting {len(source_files)} file(s)...\n")

    log_entries = []
    for filepath in sorted(source_files):
        msg = ingest_file(filepath, manifest)
        print(f"  {msg}")
        log_entries.append(msg)

    # Save updated manifest
    save_manifest(manifest)

    # Log to file
    log_file = LOG_DIR / f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_entries))

    # Summary
    print(f"\n✅ Manifest updated: {len(manifest['sources'])} total sources")
    print(f"📝 Log saved to: {log_file}\n")

if __name__ == "__main__":
    main()
