#!/usr/bin/env python3
"""
Simplified RUPP FE web scraper — single-file, easy-to-use.
Crawl → Clean → Chunk → Save
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import aiohttp
from bs4 import BeautifulSoup
from markdownify import markdownify
from langdetect import detect

# ============================================================================
# CONFIG
# ============================================================================

BASE_URL = "https://fe.rupp.edu.kh/"
MAX_DEPTH = 4
CONCURRENT_WORKERS = 3
OUTPUT_DIR = Path(__file__).parent / "outputs"
CHUNK_SIZE = 500  # words per chunk
OVERLAP = 50  # words of overlap between chunks

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CRAWLER
# ============================================================================

class SimpleCrawler:
    """Async web crawler with deduplication."""

    def __init__(self, base_url: str, max_depth: int, workers: int):
        self.base_url = base_url
        self.max_depth = max_depth
        self.workers = workers
        self.visited = set()
        self.failed = []
        self.pages = []

    async def crawl(self):
        """Crawl and return (url, html) pairs."""
        queue = asyncio.Queue()
        queue.put_nowait((self.base_url, 0))

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(self._worker(session, queue))
                for _ in range(self.workers)
            ]

            await asyncio.gather(*tasks)

    async def _worker(self, session: aiohttp.ClientSession, queue: asyncio.Queue):
        """Worker coroutine — fetch and parse pages."""
        while True:
            try:
                url, depth = queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.2)
                # Check if queue is truly empty (all workers idle)
                if queue.empty():
                    break
                continue

            if url in self.visited or depth > self.max_depth:
                continue

            self.visited.add(url)

            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        self.failed.append(url)
                        continue

                    html = await resp.text()
                    self.pages.append((url, html))
                    logger.info(f"✓ {url}")

                    # Extract links for next depth
                    soup = BeautifulSoup(html, "html.parser")
                    for link in soup.find_all("a", href=True):
                        next_url = self._resolve_url(link["href"])
                        if next_url and next_url not in self.visited:
                            try:
                                queue.put_nowait((next_url, depth + 1))
                            except asyncio.QueueFull:
                                pass

            except Exception as e:
                logger.warning(f"✗ {url}: {e}")
                self.failed.append(url)

    def _resolve_url(self, href: str) -> str:
        """Convert relative URLs to absolute."""
        if not href:
            return None
        if href.startswith("http"):
            return href if href.startswith(self.base_url) else None
        if href.startswith("/"):
            return self.base_url.rstrip("/") + href
        return None


# ============================================================================
# TEXT PROCESSING
# ============================================================================

class TextProcessor:
    """Clean and normalize text."""

    @staticmethod
    def clean_html(html: str) -> str:
        """Remove script, style, and junk tags."""
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()

        return str(soup)

    @staticmethod
    def html_to_text(html: str) -> str:
        """Convert HTML to clean markdown."""
        markdown = markdownify(html).strip()
        lines = [line.strip() for line in markdown.split("\n")]
        text = "\n".join(lines)

        # Remove excessive blank lines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text

    @staticmethod
    def is_english(text: str) -> bool:
        """Check if text is mostly English."""
        if len(text) < 100:
            return True
        try:
            return detect(text) == "en"
        except:
            return True


# ============================================================================
# CHUNKING
# ============================================================================

class TextChunker:
    """Split text into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into chunks with metadata."""
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            if len(chunk_text.strip()) < 50:
                i += self.chunk_size
                continue

            chunk_id = hashlib.md5(
                f"{metadata['url']}_{i}".encode()
            ).hexdigest()[:8]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "url": metadata["url"],
                    "title": metadata["title"],
                    "position": i // self.chunk_size,
                },
            })

            i += self.chunk_size - self.overlap

        return chunks if chunks else [
            {
                "chunk_id": hashlib.md5(metadata["url"].encode()).hexdigest()[:8],
                "text": text[:500],
                "metadata": metadata,
            }
        ]


# ============================================================================
# MAIN PIPELINE
# ============================================================================

async def scrape_and_save():
    """Main scraping pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "chunks").mkdir(exist_ok=True)

    logger.info(f"Starting crawl of {BASE_URL}")

    crawler = SimpleCrawler(BASE_URL, MAX_DEPTH, CONCURRENT_WORKERS)
    processor = TextProcessor()
    chunker = TextChunker(CHUNK_SIZE, OVERLAP)

    all_chunks = []
    doc_count = 0
    dedup_hashes = set()

    # Crawl all pages
    await crawler.crawl()

    # Process each page
    for url, html in crawler.pages:
        # Clean
        clean_html = processor.clean_html(html)
        text = processor.html_to_text(clean_html)

        if not text or len(text) < 100:
            continue

        if not processor.is_english(text):
            continue

        # Deduplicate
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in dedup_hashes:
            continue
        dedup_hashes.add(text_hash)

        # Extract metadata
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title")
        title = title.get_text() if title else url.split("/")[-1]

        metadata = {
            "url": url,
            "title": title,
            "scraped_at": datetime.now().isoformat(),
        }

        # Chunk
        chunks = chunker.chunk(text, metadata)
        all_chunks.extend(chunks)
        doc_count += 1

        logger.info(f"  → {len(chunks)} chunks from {title}")

    # Save
    output_file = OUTPUT_DIR / "chunks" / "all_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved {len(all_chunks)} chunks from {doc_count} docs to {output_file}")
    logger.info(f"✗ Failed: {len(crawler.failed)} URLs")

    return len(all_chunks)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Run scraper."""
    try:
        chunk_count = asyncio.run(scrape_and_save())
        logger.info(f"\n✓ Complete! {chunk_count} chunks ready for indexing.")
    except KeyboardInterrupt:
        logger.info("\nStopped by user.")
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)


if __name__ == "__main__":
    main()
