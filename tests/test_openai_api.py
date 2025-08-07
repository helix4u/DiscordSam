"""Tests for openai_api module."""
import os
import sys
import asyncio
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from openai_api import create_chat_completion


class DummyResponses:
    def __init__(self):
        self.called_with = None

    async def create(self, **kwargs):
        self.called_with = kwargs
        return SimpleNamespace()


class DummyClient:
    def __init__(self):
        self.responses = DummyResponses()


def test_logit_bias_passed_in_responses(monkeypatch):
    """logit_bias should be sent via extra_body when using Responses API."""
    from config import config

    monkeypatch.setattr(config, "USE_RESPONSES_API", True)
    client = DummyClient()
    logit_bias = {"123": -1}
    asyncio.run(
        create_chat_completion(client, [], model="gpt-4o", logit_bias=logit_bias)
    )
    assert client.responses.called_with["extra_body"]["logit_bias"] == logit_bias
