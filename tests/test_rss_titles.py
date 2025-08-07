import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from web_utils import fetch_rss_entries


class DummyResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    async def text(self) -> str:
        return self._body

    def raise_for_status(self) -> None:
        return None

    async def __aenter__(self) -> "DummyResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class DummyClientSession:
    def __init__(self, body: str) -> None:
        self._body = body

    async def __aenter__(self) -> "DummyClientSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, timeout: int) -> DummyResponse:
        return DummyResponse(self._body)


class FetchRSSTitleTest(unittest.TestCase):
    def test_fetch_rss_entries_strips_nbsp(self) -> None:
        recent_date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        rss_body = (
            f"<rss><channel><item><title><![CDATA[Test&nbsp;Title]]></title><link>http://example.com</link>"
            f"<guid>1</guid><pubDate>{recent_date}</pubDate><description>desc</description>"
            f"</item></channel></rss>"
        )
        with patch("aiohttp.ClientSession", lambda *args, **kwargs: DummyClientSession(rss_body)):
            entries = asyncio.run(fetch_rss_entries("http://example.com/rss"))
        self.assertTrue(entries)
        self.assertEqual(entries[0]["title"], "Test Title")


if __name__ == "__main__":
    unittest.main()
