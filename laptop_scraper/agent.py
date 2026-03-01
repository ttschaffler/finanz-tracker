#!/usr/bin/env python3
"""
Laptop Configuration Scraper Agent

An AI-powered agent that scans internet pages for laptop configurations,
extracting and structuring hardware specs, pricing, and availability.

Usage:
    python agent.py <url> [<url> ...] [--verbose] [--json]

Requires:
    ANTHROPIC_API_KEY environment variable
"""

import json
import os
import re
import sys
import argparse
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import anthropic


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

# Truncate page text to avoid overflowing Claude's context
_MAX_CHARS = 80_000
_REQUEST_TIMEOUT = 15  # seconds
_MAX_LINKS = 25
_MAX_TURNS = 25  # agentic loop safety limit

SYSTEM_PROMPT = """\
You are an expert laptop configuration extraction agent.

Your job is to scan web pages and extract detailed hardware specifications for
every laptop found. Use the provided tools to fetch page content and discover
product links.

For each laptop, extract as many of these fields as possible:
  brand       – manufacturer (e.g. "Lenovo", "Dell", "Apple")
  model       – product name / model number
  cpu         – processor name and generation
  ram         – amount + type (e.g. "16 GB DDR5")
  storage     – capacity + type (e.g. "512 GB NVMe SSD")
  display     – size, resolution, panel type
  gpu         – graphics (e.g. "Intel Iris Xe" or "NVIDIA RTX 4060")
  battery     – capacity (Wh) and/or rated life
  os          – operating system
  price       – price with currency symbol (e.g. "€ 1.299")
  weight      – weight in kg/lbs
  url         – source URL where this laptop was found

When you are done collecting information, output ONLY a JSON object in this
exact shape — no extra text before or after it:

{
  "laptops": [
    { "brand": "...", "model": "...", ... }
  ]
}

If a field is not available, set it to null. Do not fabricate values.
"""


# ---------------------------------------------------------------------------
# Web tools (called by the agent)
# ---------------------------------------------------------------------------

def _clean_html(html: str) -> str:
    """Strip tags/scripts and return readable text, truncated to _MAX_CHARS."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "iframe",
                     "noscript", "svg", "img"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)[:_MAX_CHARS]


def tool_fetch_page(url: str) -> dict:
    """Fetch a URL and return cleaned text content."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT,
                            allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        content = _clean_html(resp.text)
        return {"success": True, "url": url, "title": title, "content": content}
    except Exception as exc:
        return {"success": False, "url": url, "error": str(exc)}


def tool_find_laptop_links(url: str, keyword: str = "laptop") -> dict:
    """Scan a page and return links whose text or href contains *keyword*."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT,
                            allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        base_parts = urlparse(url)
        base = f"{base_parts.scheme}://{base_parts.netloc}"

        seen: set[str] = set()
        links: list[dict] = []
        kw = keyword.lower()

        for a in soup.find_all("a", href=True):
            href: str = a["href"].strip()
            text: str = a.get_text(strip=True)
            if kw in href.lower() or kw in text.lower():
                full = urljoin(base, href)
                if full.startswith("http") and full not in seen:
                    seen.add(full)
                    links.append({"url": full, "text": text[:120]})
                    if len(links) >= _MAX_LINKS:
                        break

        return {"success": True, "links": links}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool schema (passed to the Anthropic API)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "fetch_page",
        "description": (
            "Fetch the full text content of any web page. Use this to read "
            "product specs, review pages, or search result pages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute URL of the page to fetch.",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "find_laptop_links",
        "description": (
            "Scan a page for links that likely point to laptop product pages "
            "or articles. Returns up to 25 relevant links."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Page URL to scan for laptop links.",
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "Keyword used to filter relevant links. "
                        "Default: 'laptop'."
                    ),
                },
            },
            "required": ["url"],
        },
    },
]

_TOOL_HANDLERS = {
    "fetch_page": tool_fetch_page,
    "find_laptop_links": tool_find_laptop_links,
}


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agent(urls: list[str], verbose: bool = False) -> list[dict]:
    """
    Run the scraping agent.

    The agent iterates, calling web tools as needed, until it outputs a final
    JSON result or the turn limit is reached.
    """
    client = anthropic.Anthropic()

    url_list = "\n".join(f"- {u}" for u in urls)
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Please extract all laptop configurations from the following "
                f"page(s):\n{url_list}\n\n"
                "Explore links where useful, then output the final JSON."
            ),
        }
    ]

    for turn in range(_MAX_TURNS):
        if verbose:
            print(f"\n[agent turn {turn + 1}]", file=sys.stderr)

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            print(f"  stop_reason: {response.stop_reason}", file=sys.stderr)

        tool_uses: list = []
        final_text = ""

        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append(block)
                if verbose:
                    print(
                        f"  tool_use: {block.name}({json.dumps(block.input)})",
                        file=sys.stderr,
                    )
            elif block.type == "text":
                final_text = block.text

        # If the model stopped without calling tools, we have the answer
        if response.stop_reason == "end_turn" or not tool_uses:
            return _parse_laptops(final_text)

        # Execute tool calls and feed results back
        messages.append({"role": "assistant", "content": response.content})
        tool_results: list[dict] = []

        for tu in tool_uses:
            handler = _TOOL_HANDLERS.get(tu.name)
            if handler:
                result = handler(**tu.input)
            else:
                result = {"error": f"Unknown tool: {tu.name}"}

            if verbose:
                ok = result.get("success", True)
                print(
                    f"  result [{tu.name}]: {'OK' if ok else 'FAIL'} "
                    f"{result.get('error', '')}",
                    file=sys.stderr,
                )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    print("[agent] Turn limit reached.", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_laptops(text: str) -> list[dict]:
    """
    Extract the 'laptops' array from the agent's final text response.
    Handles both fenced code blocks and raw JSON.
    """
    # Remove markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")

    # Find the outermost JSON object
    match = re.search(r'\{[\s\S]*"laptops"[\s\S]*\}', text)
    if match:
        try:
            data = json.loads(match.group(0))
            return data.get("laptops", [])
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

def _print_results(laptops: list[dict]) -> None:
    if not laptops:
        print("No laptop configurations found.")
        return

    sep = "─" * 62
    print(f"\n{sep}")
    print(f" Found {len(laptops)} laptop configuration(s)")
    print(sep)

    fields = [
        ("cpu",      "CPU"),
        ("ram",      "RAM"),
        ("storage",  "Storage"),
        ("display",  "Display"),
        ("gpu",      "GPU"),
        ("os",       "OS"),
        ("battery",  "Battery"),
        ("price",    "Price"),
        ("weight",   "Weight"),
        ("url",      "Source"),
    ]

    for i, laptop in enumerate(laptops, 1):
        brand = laptop.get("brand") or ""
        model = laptop.get("model") or ""
        print(f"\n[{i}]  {brand} {model}".rstrip())
        for key, label in fields:
            val = laptop.get(key)
            if val:
                print(f"     {label:<10} {val}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "AI agent that scans internet pages for laptop configurations "
            "and extracts hardware specs."
        )
    )
    p.add_argument(
        "urls",
        nargs="+",
        metavar="URL",
        help="One or more URLs to scan (product pages, review sites, …)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print agent reasoning steps to stderr",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON instead of human-readable text",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before running:  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    print(f"Scanning {len(args.urls)} URL(s) …", file=sys.stderr)
    laptops = run_agent(args.urls, verbose=args.verbose)

    if args.output_json:
        print(json.dumps({"laptops": laptops}, indent=2, ensure_ascii=False))
    else:
        _print_results(laptops)


if __name__ == "__main__":
    main()
