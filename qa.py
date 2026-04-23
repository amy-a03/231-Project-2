#!/usr/bin/env python3
"""
Q&A interface for Maldives Knowledge Base.
Loads all wiki pages into context and answers questions using Claude.

Usage:
    python3 qa.py "What atolls are best for diving?"   # single-shot
    python3 qa.py                                       # interactive REPL
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic, RateLimitError, APIError

KB_ROOT = Path(__file__).parent
WIKI_DIR = KB_ROOT / "wiki"
LOG_DIR = KB_ROOT / "logs"

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1500

SYSTEM_PROMPT_HEADER = """You are a knowledgeable assistant answering questions about sustainable travel to the Maldives. You have access to a curated wiki compiled from ~80 sources covering atolls, resorts, ecology, and travel planning.

RULES:
1. Answer ONLY using information from the wiki pages provided below. Do NOT use outside knowledge.
2. If the wiki doesn't contain the answer, say "The knowledge base doesn't cover this" and suggest what sources could be added.
3. ALWAYS cite the wiki pages you used in a "Sources" section at the end (format: `- [[Page Title]]`).
4. Be concrete and specific. Quote numbers, names, and facts directly from the pages when relevant.
5. When a question has trade-offs (e.g., luxury vs. budget), structure the answer around those trade-offs.

--- WIKI KNOWLEDGE BASE ---
"""

client = Anthropic()


def load_wiki(question=None, max_pages=60):
    """Load wiki pages. If question given, keyword-filter to most relevant pages
    (reduces tokens to stay under rate limits on large KBs)."""
    page_data = []
    for folder in ["atolls", "resorts", "ecology", "planning"]:
        folder_path = WIKI_DIR / folder
        if not folder_path.exists():
            continue
        for page_file in sorted(folder_path.glob("*.md")):
            content = page_file.read_text()
            page_data.append((folder, page_file.name, content))

    if not page_data:
        return None

    if question:
        # Keyword-score each page against the question, take top max_pages
        q_words = {w.lower().strip(".,?!") for w in question.split() if len(w) > 3}
        scored = []
        for folder, fname, content in page_data:
            lower = content.lower()
            score = sum(lower.count(w) for w in q_words)
            scored.append((score, folder, fname, content))
        scored.sort(reverse=True)
        selected = scored[:max_pages]
        selected.sort(key=lambda x: (x[1], x[2]))  # re-sort by folder/name for readability
        return "\n".join(f"\n=== {f}/{n} ===\n{c}" for _, f, n, c in selected)

    return "\n".join(f"\n=== {f}/{n} ===\n{c}" for f, n, c in page_data)


def ask(question, wiki_content, max_retries=4):
    """Send a question to Claude with the wiki as context (prompt-cached)."""
    full_system = SYSTEM_PROMPT_HEADER + wiki_content

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {"type": "text", "text": full_system, "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{"role": "user", "content": question}],
            )
            return response.content[0].text, response.usage
        except RateLimitError:
            wait = 2 ** (attempt + 2)  # 4, 8, 16, 32 sec
            print(f"⏳ Rate limited. Waiting {wait}s before retry ({attempt+1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded on rate limit")


def log_qa(question, answer):
    """Append Q&A to log file."""
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "qa_history.log"
    with open(log_file, "a") as f:
        f.write(f"\n--- {datetime.now().isoformat()} ---\n")
        f.write(f"Q: {question}\n\n")
        f.write(f"A: {answer}\n")


def print_usage_stats(usage):
    """Print token usage including cache hits."""
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    regular = usage.input_tokens
    print(f"\n[tokens: {regular} input + {cache_read} cached read + {cache_create} cache write + {usage.output_tokens} output]")


def interactive_mode(wiki_content):
    """REPL loop for asking multiple questions."""
    print("\n🏝️  Maldives KB — Interactive Q&A")
    print("Type your question (or 'exit' to quit)\n")

    while True:
        try:
            question = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print()
        try:
            answer, usage = ask(question, wiki_content)
            print(answer)
            print_usage_stats(usage)
            log_qa(question, answer)
        except Exception as e:
            print(f"❌ Error: {e}")
        print()


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    if len(sys.argv) > 1:
        # Single-shot mode — keyword-filter wiki by question for rate-limit safety
        question = " ".join(sys.argv[1:])
        wiki_content = load_wiki(question=question)
        if not wiki_content:
            print("❌ No wiki pages found. Run compile.py first.")
            sys.exit(1)
        page_count = wiki_content.count("=== ")
        word_count = len(wiki_content.split())
        print(f"📚 Loaded {page_count} most-relevant wiki pages ({word_count:,} words)")
        print(f"\n❯ {question}\n")
        answer, usage = ask(question, wiki_content)
        print(answer)
        print_usage_stats(usage)
        log_qa(question, answer)
    else:
        # Interactive mode — also filter per-question
        print("📚 Wiki loaded (keyword-filtered per question for rate-limit safety)")
        interactive_mode_dynamic()


def interactive_mode_dynamic():
    """REPL where wiki is filtered per question."""
    print("\n🏝️  Maldives KB — Interactive Q&A")
    print("Type your question (or 'exit' to quit)\n")
    while True:
        try:
            question = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        wiki_content = load_wiki(question=question)
        print()
        try:
            answer, usage = ask(question, wiki_content)
            print(answer)
            print_usage_stats(usage)
            log_qa(question, answer)
        except Exception as e:
            print(f"❌ Error: {e}")
        print()


if __name__ == "__main__":
    main()
