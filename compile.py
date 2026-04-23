#!/usr/bin/env python3
"""
Compilation engine for Maldives Knowledge Base.
Uses Claude to read raw sources and generate wiki pages.
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic

KB_ROOT = Path(__file__).parent
RAW_DIR = KB_ROOT / "raw"
WIKI_DIR = KB_ROOT / "wiki"
MANIFEST_PATH = RAW_DIR / "manifest.json"
COMPILE_STATE_PATH = KB_ROOT / ".compile_state.json"
LOG_DIR = KB_ROOT / "logs"

MODEL = "claude-haiku-4-5-20251001"
MAX_SOURCE_CHARS = 4000

SYSTEM_PROMPT = """You are building a personal knowledge base about sustainable travel to the Maldives.

For each source document I give you, extract its content as one or more wiki pages. A wiki page represents a single distinct topic (an atoll, a resort, a species, a planning concept, etc.).

For each page, produce:
- title: short, canonical name (e.g., "Baa Atoll", "Whale Shark Conservation", "Monsoon Seasons")
- summary: 2-3 sentence overview
- key_facts: 3-8 concrete bullet points extracted from the source
- connections: 2-5 related wiki page titles (for backlinks)

Respond ONLY with valid JSON in this exact format — no markdown, no code fences, no commentary:
{"pages": [{"title": "...", "summary": "...", "key_facts": ["..."], "connections": ["..."]}]}"""

# Keyword → wiki subfolder for page categorization
WIKI_CATEGORIES = [
    (["atoll", "island", "malé", "baa", "addu", "laamu", "ari"], "atolls"),
    (["resort", "hotel", "villa", "guesthouse", "six senses", "soneva", "banyan", "four seasons"], "resorts"),
    (["coral", "reef", "bleach", "whale", "manta", "fish", "turtle", "climate", "sea level", "marine", "sustainability", "biosphere", "ecolog"], "ecology"),
    (["season", "monsoon", "budget", "visa", "pack", "dress", "cuisine", "culture", "transfer", "airport", "dive", "snorkel"], "planning"),
]

client = Anthropic()


def load_manifest():
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"sources": []}


def load_compile_state():
    if COMPILE_STATE_PATH.exists():
        with open(COMPILE_STATE_PATH) as f:
            return json.load(f)
    return {"compiled_hashes": []}


def save_compile_state(state):
    with open(COMPILE_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def read_source_file(filepath):
    try:
        with open(RAW_DIR / filepath, encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(RAW_DIR / filepath, encoding='latin-1') as f:
            return f.read()


def get_sources_to_compile(full_recompile=False):
    manifest = load_manifest()
    state = load_compile_state()
    if full_recompile:
        return manifest["sources"]
    return [s for s in manifest["sources"] if s["hash"] not in state["compiled_hashes"]]


def extract_json(text):
    """Robustly extract JSON from Claude response, handling markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    # Fallback: find first `{` to last `}`
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def compile_source(source_info):
    """Call Claude to extract wiki pages from a source."""
    content = read_source_file(source_info["path"])
    user_msg = f"Source filename: {source_info['filename']}\n\nSource content:\n\n{content[:MAX_SOURCE_CHARS]}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user_msg}],
    )

    try:
        data = extract_json(response.content[0].text)
        return data.get("pages", [])
    except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
        print(f"⚠️  Parse error for {source_info['filename']}: {e}")
        return []


def categorize_title(title):
    """Return the wiki subfolder name for a given page title."""
    lower = title.lower()
    for keywords, folder in WIKI_CATEGORIES:
        if any(kw in lower for kw in keywords):
            return folder
    return "planning"  # default


def safe_filename(title):
    """Convert title to safe filename."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return f"{slug}.md"


def parse_existing_page(filepath):
    """Parse an existing wiki page into sections for merging."""
    if not filepath.exists():
        return None
    text = filepath.read_text()
    sections = {"summary": "", "key_facts": [], "connections": [], "sources": []}
    current = None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## Summary"):
            current = "summary"
        elif stripped.startswith("## Key Facts"):
            current = "key_facts"
        elif stripped.startswith("## Connections"):
            current = "connections"
        elif stripped.startswith("## Sources"):
            current = "sources"
        elif stripped.startswith("## "):
            current = None
        elif current == "summary" and stripped:
            sections["summary"] += (" " if sections["summary"] else "") + stripped
        elif current in ("key_facts", "connections", "sources") and stripped.startswith("- "):
            item = stripped[2:].strip()
            # Strip [[ ]] from connections
            if current == "connections":
                item = item.replace("[[", "").replace("]]", "")
            sections[current].append(item)
    return sections


def upsert_page(title, summary, key_facts, connections, source_filename):
    """Create or merge a wiki page, preserving content from prior sources."""
    folder = categorize_title(title)
    filepath = WIKI_DIR / folder / safe_filename(title)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    existing = parse_existing_page(filepath)
    if existing:
        # Merge: keep original summary, dedupe-append new facts/connections/sources
        final_summary = existing["summary"] or summary
        final_facts = existing["key_facts"] + [f for f in key_facts if f not in existing["key_facts"]]
        final_connections = existing["connections"] + [c for c in connections if c not in existing["connections"]]
        final_sources = existing["sources"] + ([source_filename] if source_filename not in existing["sources"] else [])
        action = "MERGE"
    else:
        final_summary = summary
        final_facts = key_facts
        final_connections = connections
        final_sources = [source_filename]
        action = "CREATE"

    content = f"# {title}\n\n## Summary\n{final_summary}\n\n## Key Facts\n"
    for fact in final_facts:
        content += f"- {fact}\n"
    if final_connections:
        content += "\n## Connections\n"
        for conn in final_connections:
            content += f"- [[{conn}]]\n"
    if final_sources:
        content += "\n## Sources\n"
        for src in final_sources:
            content += f"- {src}\n"

    filepath.write_text(content)
    return action, str(filepath.relative_to(KB_ROOT))


def rebuild_index():
    """Regenerate wiki/index.md from current wiki pages."""
    index_path = WIKI_DIR / "index.md"
    lines = ["# Maldives Sustainable Travel Knowledge Base\n",
             "> LLM-compiled wiki. Do not edit manually — managed by compile.py.\n"]

    for folder in ["atolls", "resorts", "ecology", "planning"]:
        folder_path = WIKI_DIR / folder
        if not folder_path.exists():
            continue
        pages = sorted(folder_path.glob("*.md"))
        if not pages:
            continue
        lines.append(f"\n## {folder.capitalize()}\n")
        for page in pages:
            # Extract title from first heading
            first_line = page.read_text().split("\n")[0]
            title = first_line.replace("# ", "").strip()
            rel = page.relative_to(WIKI_DIR)
            lines.append(f"- [{title}]({rel})")

    index_path.write_text("\n".join(lines) + "\n")


def compile_all(full_recompile=False):
    sources = get_sources_to_compile(full_recompile)
    state = load_compile_state()

    if full_recompile:
        state["compiled_hashes"] = []  # reset so pages can be re-merged cleanly
        # Clear wiki pages to start fresh
        for folder in ["atolls", "resorts", "ecology", "planning"]:
            for p in (WIKI_DIR / folder).glob("*.md"):
                p.unlink()

    if not sources:
        print("✅ No new sources to compile.\n")
        return

    mode = "FULL RECOMPILE" if full_recompile else "INCREMENTAL"
    print(f"\n🔄 {mode}: {len(sources)} source(s)\n")

    pages_touched = 0
    log_entries = []
    LOG_DIR.mkdir(exist_ok=True)

    for i, source in enumerate(sources, 1):
        print(f"[{i}/{len(sources)}] {source['filename']} ({source['category']})...", end=" ", flush=True)

        pages = compile_source(source)
        if not pages:
            print("⚠️  no pages")
            log_entries.append(f"SKIP: {source['filename']}")
            state["compiled_hashes"].append(source["hash"])
            save_compile_state(state)  # incremental save
            continue

        for page in pages:
            try:
                action, page_path = upsert_page(
                    title=page["title"],
                    summary=page.get("summary", ""),
                    key_facts=page.get("key_facts", []),
                    connections=page.get("connections", []),
                    source_filename=source["filename"],
                )
                pages_touched += 1
                log_entries.append(f"{action}: {page_path} ← {source['filename']}")
            except KeyError:
                log_entries.append(f"MALFORMED PAGE from {source['filename']}")

        state["compiled_hashes"].append(source["hash"])
        save_compile_state(state)  # incremental save — safe to interrupt
        print(f"✅ {len(pages)} page(s)")

    # Rebuild index after all pages are written
    rebuild_index()

    log_file = LOG_DIR / f"compile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.write_text("\n".join(log_entries))

    print(f"\n✅ {len(sources)} source(s) → {pages_touched} page(s) created/merged")
    print(f"📑 Index rebuilt: wiki/index.md")
    print(f"📝 Log: {log_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Compile Maldives KB sources into wiki pages")
    parser.add_argument("--full", action="store_true", help="Full recompile (rebuild entire wiki)")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        exit(1)

    compile_all(full_recompile=args.full)


if __name__ == "__main__":
    main()
