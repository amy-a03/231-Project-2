#!/usr/bin/env python3
"""
Linter / health checker for Maldives Knowledge Base.

Runs two kinds of checks:
  1. Deterministic (fast, no LLM): dead links, thin pages, orphan pages.
  2. LLM-powered (slow, costs tokens): contradictions, missing connections, coverage gaps.

Outputs a health report to logs/lint_report.md.

Usage:
    python3 lint.py            # deterministic checks only (fast, free)
    python3 lint.py --llm      # add LLM-powered semantic checks
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from anthropic import Anthropic, RateLimitError

KB_ROOT = Path(__file__).parent
WIKI_DIR = KB_ROOT / "wiki"
LOG_DIR = KB_ROOT / "logs"

MODEL = "claude-sonnet-4-6"
THIN_PAGE_FACT_THRESHOLD = 3

client = Anthropic()


def load_all_pages():
    """Return {title: {path, content, facts, connections, sources}} for every wiki page."""
    pages = {}
    for folder in ["atolls", "resorts", "ecology", "planning"]:
        folder_path = WIKI_DIR / folder
        if not folder_path.exists():
            continue
        for page_file in sorted(folder_path.glob("*.md")):
            content = page_file.read_text()
            title = content.split("\n", 1)[0].replace("# ", "").strip()
            pages[title] = {
                "path": page_file.relative_to(KB_ROOT),
                "content": content,
                "folder": folder,
                "facts": extract_section_items(content, "Key Facts"),
                "connections": extract_section_items(content, "Connections"),
                "sources": extract_section_items(content, "Sources"),
            }
    return pages


def extract_section_items(content, section_name):
    """Extract bullet items from a ## section."""
    items = []
    in_section = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith(f"## {section_name}"):
            in_section = True
        elif stripped.startswith("## "):
            in_section = False
        elif in_section and stripped.startswith("- "):
            item = stripped[2:].strip()
            # Strip wiki-link brackets
            item = item.replace("[[", "").replace("]]", "")
            items.append(item)
    return items


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

def check_dead_links(pages):
    """Find [[Page]] references to pages that don't exist."""
    titles = set(pages.keys())
    titles_lower = {t.lower() for t in titles}
    dead = []
    for title, info in pages.items():
        for conn in info["connections"]:
            if conn not in titles and conn.lower() not in titles_lower:
                dead.append((title, conn))
    return dead


def check_thin_pages(pages):
    """Find pages with too few facts."""
    return [
        (title, len(info["facts"]))
        for title, info in pages.items()
        if len(info["facts"]) < THIN_PAGE_FACT_THRESHOLD
    ]


def check_orphan_pages(pages):
    """Pages that are never referenced by any other page's Connections."""
    referenced = set()
    titles_lower = {t.lower(): t for t in pages.keys()}
    for info in pages.values():
        for conn in info["connections"]:
            canonical = titles_lower.get(conn.lower())
            if canonical:
                referenced.add(canonical)
    return [title for title in pages if title not in referenced]


def check_source_coverage(pages):
    """Count unique sources and pages per folder."""
    folder_counts = defaultdict(int)
    all_sources = set()
    for info in pages.values():
        folder_counts[info["folder"]] += 1
        for s in info["sources"]:
            all_sources.add(s)
    return dict(folder_counts), len(all_sources)


# ---------------------------------------------------------------------------
# LLM-powered checks
# ---------------------------------------------------------------------------

def build_wiki_dump(pages):
    """Build a compact text dump of the wiki for LLM context."""
    parts = []
    for title, info in pages.items():
        parts.append(f"### {title} ({info['folder']})")
        if info["facts"]:
            parts.append("Facts:")
            for f in info["facts"]:
                parts.append(f"- {f}")
        parts.append("")
    return "\n".join(parts)


def llm_check(wiki_dump, check_name, instruction, max_retries=5):
    """Run one LLM-powered check against the wiki dump, retrying on rate limit."""
    system = f"""You are a knowledge-base linter. You audit a wiki about sustainable travel to the Maldives and flag issues.

Below is the current wiki content. Your task: {instruction}

Respond in markdown with a short list of findings. Be specific — cite page titles. If you find no issues, say "No issues found."

--- WIKI ---
{wiki_dump}"""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1200,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"Run the {check_name} audit now."}],
            )
            return response.content[0].text
        except RateLimitError:
            wait = 30 * (attempt + 1)  # 30s, 60s, 90s...
            print(f"  ⏳ Rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
    return "_Rate limit exceeded after retries._"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(pages, use_llm=False):
    """Build the full lint report as markdown."""
    lines = [
        f"# KB Health Report",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"**Total pages:** {len(pages)}",
    ]

    folder_counts, source_count = check_source_coverage(pages)
    lines.append(f"**Unique sources cited:** {source_count}")
    lines.append(f"**Pages per folder:** {dict(folder_counts)}")
    lines.append("")

    # --- Deterministic checks ---
    lines.append("## 1. Dead Links")
    dead = check_dead_links(pages)
    if dead:
        for page, missing in dead:
            lines.append(f"- **{page}** → missing `[[{missing}]]`")
    else:
        lines.append("_No dead links._")
    lines.append("")

    lines.append(f"## 2. Thin Pages (< {THIN_PAGE_FACT_THRESHOLD} facts)")
    thin = check_thin_pages(pages)
    if thin:
        for title, count in thin:
            lines.append(f"- **{title}** — only {count} fact(s)")
    else:
        lines.append("_All pages meet the fact threshold._")
    lines.append("")

    lines.append("## 3. Orphan Pages (not referenced by any other page)")
    orphans = check_orphan_pages(pages)
    if orphans:
        for title in orphans:
            lines.append(f"- {title}")
    else:
        lines.append("_All pages are connected._")
    lines.append("")

    # --- LLM checks ---
    if use_llm:
        print("🧠 Running LLM-powered checks (this takes a minute)...")
        wiki_dump = build_wiki_dump(pages)

        checks = [
            ("contradictions", "Find factual contradictions between pages (e.g., two pages claiming different 'best months'). List each contradiction with the pages involved."),
            ("missing connections", "Find pairs of pages that are clearly related but don't link to each other. E.g., a species page and the atoll where it's found. List 5–10 of the most valuable missing links."),
            ("coverage gaps", "Identify topics that should be in the KB but are underrepresented or missing. For example: 'no coverage of southern atolls' or 'no info on visa requirements'. Suggest 3–7 specific sources to add."),
        ]

        for name, instruction in checks:
            lines.append(f"## LLM Audit: {name.capitalize()}")
            try:
                result = llm_check(wiki_dump, name, instruction)
                lines.append(result)
            except Exception as e:
                lines.append(f"_Error running check: {e}_")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Lint the Maldives KB wiki")
    parser.add_argument("--llm", action="store_true", help="Enable LLM-powered semantic checks")
    args = parser.parse_args()

    if args.llm and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY required for --llm mode")
        sys.exit(1)

    pages = load_all_pages()
    if not pages:
        print("❌ No wiki pages found. Run compile.py first.")
        sys.exit(1)

    print(f"🔎 Linting {len(pages)} wiki pages...\n")
    report = build_report(pages, use_llm=args.llm)

    LOG_DIR.mkdir(exist_ok=True)
    report_path = LOG_DIR / "lint_report.md"
    report_path.write_text(report)

    # Print summary to terminal
    print(report)
    print(f"\n📝 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
