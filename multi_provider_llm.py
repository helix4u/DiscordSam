"""Multi-Provider LLM API Integration.

This module provides a unified interface for multiple LLM providers:
- Local (LM Studio, Ollama)
- OpenAI (Chat Completions and Responses APIs)
- Claude (Anthropic)
- Google (Gemini)
- Mistral
- OpenRouter

Includes pricing information and cost tracking capabilities.
"""

import asyncio
import logging
import os
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from openai import AsyncOpenAI

from config import config

logger = logging.getLogger(__name__)


# ============================================================================
# Provider Definitions
# ============================================================================

class ProviderType(Enum):
    """Supported LLM provider types."""
    LOCAL = "local"           # LM Studio, Ollama
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OPENAI = "openai"         # OpenAI Chat Completions
    OPENAI_RESPONSES = "openai_responses"  # OpenAI Responses API
    ANTHROPIC = "anthropic"   # Claude
    GOOGLE = "google"         # Gemini
    MISTRAL = "mistral"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a model (per million tokens)."""
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    cached_input_cost_per_million: Optional[float] = None
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate total cost for a request."""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        
        cached_cost = 0.0
        if cached_tokens > 0 and self.cached_input_cost_per_million is not None:
            cached_cost = (cached_tokens / 1_000_000) * self.cached_input_cost_per_million
        
        return input_cost + output_cost + cached_cost


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider_type: ProviderType
    name: str
    api_base_url: str
    api_key: Optional[str] = None
    default_model: str = ""
    available_models: List[str] = field(default_factory=list)
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_logit_bias: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False
    max_context_length: int = 4096
    default_temperature: float = 0.7
    rate_limit_rpm: float = 60.0  # Requests per minute
    rate_limit_tpm: int = 100000  # Tokens per minute
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provider_type"] = self.provider_type.value
        return d


# ============================================================================
# Model Pricing Database
# ============================================================================

# Pricing as of late 2024/early 2025 (per million tokens)
MODEL_PRICING: Dict[str, ModelPricing] = {
    # OpenAI Models
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.60, 0.075),
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-32k": ModelPricing(60.00, 120.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
    "o1": ModelPricing(15.00, 60.00, 7.50),
    "o1-preview": ModelPricing(15.00, 60.00),
    "o1-mini": ModelPricing(3.00, 12.00, 1.50),
    "o3-mini": ModelPricing(1.10, 4.40, 0.55),
    "gpt-5": ModelPricing(30.00, 60.00),  # Estimated
    
    # Claude Models (Anthropic)
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00, 0.08),
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, 1.50),
    "claude-3-sonnet-20240229": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, 0.025),
    "claude-sonnet-4-20250514": ModelPricing(3.00, 15.00, 0.30),
    "claude-opus-4-20250514": ModelPricing(15.00, 75.00, 1.50),
    
    # Google Models (Gemini)
    "gemini-1.5-pro": ModelPricing(1.25, 5.00, 0.3125),
    "gemini-1.5-pro-latest": ModelPricing(1.25, 5.00, 0.3125),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30, 0.01875),
    "gemini-1.5-flash-latest": ModelPricing(0.075, 0.30, 0.01875),
    "gemini-2.0-flash-exp": ModelPricing(0.10, 0.40),
    "gemini-pro": ModelPricing(0.50, 1.50),
    "gemini-pro-vision": ModelPricing(0.50, 1.50),
    
    # Mistral Models
    "mistral-large-latest": ModelPricing(2.00, 6.00),
    "mistral-large-2411": ModelPricing(2.00, 6.00),
    "mistral-medium-latest": ModelPricing(2.70, 8.10),
    "mistral-small-latest": ModelPricing(0.20, 0.60),
    "codestral-latest": ModelPricing(0.20, 0.60),
    "ministral-8b-latest": ModelPricing(0.10, 0.10),
    "ministral-3b-latest": ModelPricing(0.04, 0.04),
    "open-mistral-7b": ModelPricing(0.25, 0.25),
    "open-mixtral-8x7b": ModelPricing(0.70, 0.70),
    "open-mixtral-8x22b": ModelPricing(2.00, 6.00),
    
    # OpenRouter Pricing (varies by model, these are common ones)
    "openrouter/auto": ModelPricing(1.00, 3.00),  # Estimated average
    "meta-llama/llama-3.1-405b-instruct": ModelPricing(2.70, 2.70),
    "meta-llama/llama-3.1-70b-instruct": ModelPricing(0.52, 0.75),
    "meta-llama/llama-3.1-8b-instruct": ModelPricing(0.055, 0.055),
    "anthropic/claude-3.5-sonnet": ModelPricing(3.00, 15.00),
    "google/gemini-pro-1.5": ModelPricing(1.25, 5.00),
    
    # Local models (no cost)
    "local-model": ModelPricing(0.0, 0.0),
}


def get_model_pricing(model: str) -> ModelPricing:
    """Get pricing for a model, with fallback for unknown models."""
    # Direct match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    
    # Try prefix matching
    model_lower = model.lower()
    for known_model, pricing in MODEL_PRICING.items():
        if model_lower.startswith(known_model.lower()):
            return pricing
    
    # Check for common patterns
    if "gpt-4" in model_lower:
        return MODEL_PRICING.get("gpt-4", ModelPricing(30.00, 60.00))
    if "gpt-3.5" in model_lower:
        return MODEL_PRICING.get("gpt-3.5-turbo", ModelPricing(0.50, 1.50))
    if "claude" in model_lower:
        if "opus" in model_lower:
            return ModelPricing(15.00, 75.00)
        if "haiku" in model_lower:
            return ModelPricing(0.25, 1.25)
        return ModelPricing(3.00, 15.00)  # Sonnet-like
    if "gemini" in model_lower:
        if "flash" in model_lower:
            return ModelPricing(0.075, 0.30)
        return ModelPricing(1.25, 5.00)
    if "mistral" in model_lower:
        if "large" in model_lower:
            return ModelPricing(2.00, 6.00)
        return ModelPricing(0.25, 0.75)
    if "llama" in model_lower:
        if "405b" in model_lower:
            return ModelPricing(2.70, 2.70)
        if "70b" in model_lower:
            return ModelPricing(0.52, 0.75)
        return ModelPricing(0.055, 0.055)
    
    # Unknown model - assume local/free
    logger.debug(f"Unknown model pricing for: {model}, assuming free/local")
    return ModelPricing(0.0, 0.0)


# ============================================================================
# Preset Provider Configurations
# ============================================================================

def get_preset_providers() -> Dict[str, ProviderConfig]:
    """Get preset provider configurations."""
    return {
        "lm_studio": ProviderConfig(
            provider_type=ProviderType.LM_STUDIO,
            name="LM Studio",
            api_base_url="http://localhost:1234/v1",
            default_model="local-model",
            supports_streaming=True,
            supports_json_mode=True,
            supports_logit_bias=False,
            supports_function_calling=True,
            max_context_length=32768,
            rate_limit_rpm=1000,  # Effectively unlimited locally
            rate_limit_tpm=1000000,
        ),
        "ollama": ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            name="Ollama",
            api_base_url="http://localhost:11434/v1",
            default_model="llama2",
            supports_streaming=True,
            supports_json_mode=True,
            supports_logit_bias=False,
            max_context_length=32768,
            rate_limit_rpm=1000,
            rate_limit_tpm=1000000,
        ),
        "openai": ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="OpenAI",
            api_base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            default_model="gpt-4o-mini",
            available_models=[
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
                "gpt-3.5-turbo", "o1", "o1-mini", "o3-mini"
            ],
            supports_streaming=True,
            supports_json_mode=True,
            supports_logit_bias=True,
            supports_function_calling=True,
            supports_vision=True,
            max_context_length=128000,
            rate_limit_rpm=60,
            rate_limit_tpm=90000,
        ),
        "openai_responses": ProviderConfig(
            provider_type=ProviderType.OPENAI_RESPONSES,
            name="OpenAI Responses API",
            api_base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            default_model="gpt-4o",
            available_models=["gpt-4o", "gpt-4o-mini", "o1", "o1-mini"],
            supports_streaming=False,  # Different streaming mechanism
            supports_json_mode=False,
            supports_logit_bias=False,
            supports_function_calling=True,
            supports_vision=True,
            max_context_length=128000,
            rate_limit_rpm=60,
            rate_limit_tpm=90000,
        ),
        "anthropic": ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            name="Anthropic (Claude)",
            api_base_url="https://api.anthropic.com/v1",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            default_model="claude-3-5-sonnet-latest",
            available_models=[
                "claude-3-5-sonnet-latest", "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229", "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            supports_streaming=True,
            supports_json_mode=False,
            supports_logit_bias=False,
            supports_function_calling=True,
            supports_vision=True,
            max_context_length=200000,
            rate_limit_rpm=50,
            rate_limit_tpm=100000,
            headers={"anthropic-version": "2023-06-01"},
        ),
        "google": ProviderConfig(
            provider_type=ProviderType.GOOGLE,
            name="Google (Gemini)",
            api_base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key=os.getenv("GOOGLE_API_KEY"),
            default_model="gemini-1.5-flash",
            available_models=[
                "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp",
                "gemini-pro", "gemini-pro-vision"
            ],
            supports_streaming=True,
            supports_json_mode=True,
            supports_logit_bias=False,
            supports_function_calling=True,
            supports_vision=True,
            max_context_length=1000000,  # 1M context for Gemini 1.5
            rate_limit_rpm=60,
            rate_limit_tpm=1000000,
        ),
        "mistral": ProviderConfig(
            provider_type=ProviderType.MISTRAL,
            name="Mistral AI",
            api_base_url="https://api.mistral.ai/v1",
            api_key=os.getenv("MISTRAL_API_KEY"),
            default_model="mistral-small-latest",
            available_models=[
                "mistral-large-latest", "mistral-medium-latest",
                "mistral-small-latest", "codestral-latest",
                "ministral-8b-latest", "ministral-3b-latest"
            ],
            supports_streaming=True,
            supports_json_mode=True,
            supports_logit_bias=False,
            supports_function_calling=True,
            supports_vision=False,
            max_context_length=32768,
            rate_limit_rpm=100,
            rate_limit_tpm=100000,
        ),
        "openrouter": ProviderConfig(
            provider_type=ProviderType.OPENROUTER,
            name="OpenRouter",
            api_base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            default_model="openrouter/auto",
            available_models=[
                "openrouter/auto",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro-1.5",
                "meta-llama/llama-3.1-405b-instruct",
                "meta-llama/llama-3.1-70b-instruct",
                "mistral/mistral-large",
            ],
            supports_streaming=True,
            supports_json_mode=True,
            supports_logit_bias=False,
            supports_function_calling=True,
            supports_vision=True,
            max_context_length=200000,
            rate_limit_rpm=20,  # Conservative for free tier
            rate_limit_tpm=50000,
            headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", ""),
                "X-Title": os.getenv("OPENROUTER_TITLE", "DiscordSam"),
            },
        ),
    }


# ============================================================================
# Multi-Provider Client Manager
# ============================================================================

@dataclass
class ProviderClient:
    """A client instance for a specific provider."""
    config: ProviderConfig
    client: AsyncOpenAI
    last_used: float = 0.0
    request_count: int = 0
    token_count: int = 0
    error_count: int = 0


class MultiProviderManager:
    """Manages multiple LLM provider clients."""

    def __init__(self):
        self._clients: Dict[str, ProviderClient] = {}
        self._active_provider: str = "local"
        self._lock = asyncio.Lock()
        self._presets = get_preset_providers()
        
        # Load saved provider preferences
        self._preferences_file = os.path.join(
            os.path.dirname(__file__), "provider_preferences.json"
        )
        self._load_preferences()

    def _load_preferences(self) -> None:
        """Load saved provider preferences."""
        try:
            if os.path.exists(self._preferences_file):
                with open(self._preferences_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._active_provider = data.get("active_provider", "local")
        except Exception as e:
            logger.error(f"Failed to load provider preferences: {e}")

    def _save_preferences(self) -> None:
        """Save provider preferences."""
        try:
            data = {"active_provider": self._active_provider}
            tmp_path = self._preferences_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self._preferences_file)
        except Exception as e:
            logger.error(f"Failed to save provider preferences: {e}")

    def _create_client(self, provider_config: ProviderConfig) -> AsyncOpenAI:
        """Create an AsyncOpenAI client for a provider."""
        timeout = getattr(config, "LLM_REQUEST_TIMEOUT_SECONDS", 900.0)
        max_retries = getattr(config, "OPENAI_CLIENT_MAX_RETRIES", 0)
        
        # Build default headers
        default_headers: Dict[str, str] = {}
        if provider_config.headers:
            default_headers.update(provider_config.headers)
        
        return AsyncOpenAI(
            base_url=provider_config.api_base_url,
            api_key=provider_config.api_key or "lm-studio",
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers if default_headers else None,
        )

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available providers with their configurations."""
        result = []
        for key, preset in self._presets.items():
            result.append({
                "key": key,
                "name": preset.name,
                "type": preset.provider_type.value,
                "default_model": preset.default_model,
                "available_models": preset.available_models,
                "has_api_key": bool(preset.api_key),
                "supports_vision": preset.supports_vision,
                "supports_streaming": preset.supports_streaming,
                "rate_limit_rpm": preset.rate_limit_rpm,
            })
        return result

    async def set_active_provider(
        self,
        provider_key: str,
        api_key: Optional[str] = None,
        custom_base_url: Optional[str] = None,
    ) -> bool:
        """Set the active provider."""
        async with self._lock:
            if provider_key not in self._presets:
                logger.error(f"Unknown provider: {provider_key}")
                return False
            
            preset = self._presets[provider_key]
            
            # Update API key if provided
            if api_key:
                preset = ProviderConfig(
                    provider_type=preset.provider_type,
                    name=preset.name,
                    api_base_url=custom_base_url or preset.api_base_url,
                    api_key=api_key,
                    default_model=preset.default_model,
                    available_models=preset.available_models,
                    supports_streaming=preset.supports_streaming,
                    supports_json_mode=preset.supports_json_mode,
                    supports_logit_bias=preset.supports_logit_bias,
                    supports_function_calling=preset.supports_function_calling,
                    supports_vision=preset.supports_vision,
                    max_context_length=preset.max_context_length,
                    default_temperature=preset.default_temperature,
                    rate_limit_rpm=preset.rate_limit_rpm,
                    rate_limit_tpm=preset.rate_limit_tpm,
                    headers=preset.headers,
                    metadata=preset.metadata,
                )
                self._presets[provider_key] = preset
            
            # Create or update client
            client = self._create_client(preset)
            self._clients[provider_key] = ProviderClient(
                config=preset,
                client=client,
            )
            
            self._active_provider = provider_key
            self._save_preferences()
            
            logger.info(f"Active provider set to: {provider_key} ({preset.name})")
            return True

    async def get_client(
        self,
        provider_key: Optional[str] = None,
    ) -> Tuple[AsyncOpenAI, ProviderConfig]:
        """Get a client for the specified or active provider."""
        key = provider_key or self._active_provider
        
        async with self._lock:
            if key in self._clients:
                pc = self._clients[key]
                pc.last_used = time.time()
                pc.request_count += 1
                return pc.client, pc.config
            
            # Create new client
            if key not in self._presets:
                key = "lm_studio"  # Fallback to local
            
            preset = self._presets[key]
            client = self._create_client(preset)
            
            self._clients[key] = ProviderClient(
                config=preset,
                client=client,
                last_used=time.time(),
                request_count=1,
            )
            
            return client, preset

    def get_active_provider_info(self) -> Dict[str, Any]:
        """Get information about the active provider."""
        if self._active_provider in self._presets:
            preset = self._presets[self._active_provider]
            return {
                "key": self._active_provider,
                "name": preset.name,
                "type": preset.provider_type.value,
                "default_model": preset.default_model,
                "api_base_url": preset.api_base_url,
                "has_api_key": bool(preset.api_key),
                "rate_limit_rpm": preset.rate_limit_rpm,
            }
        return {"key": self._active_provider, "name": "Unknown"}

    async def test_provider(self, provider_key: str) -> Dict[str, Any]:
        """Test connectivity to a provider."""
        try:
            client, config = await self.get_client(provider_key)
            
            # Simple test call
            start_time = time.time()
            response = await client.chat.completions.create(
                model=config.default_model,
                messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
                max_tokens=10,
            )
            latency = time.time() - start_time
            
            return {
                "success": True,
                "provider": provider_key,
                "model": config.default_model,
                "latency_ms": round(latency * 1000, 2),
                "response": response.choices[0].message.content if response.choices else "",
            }
        except Exception as e:
            return {
                "success": False,
                "provider": provider_key,
                "error": str(e),
            }

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all providers."""
        stats = {}
        for key, pc in self._clients.items():
            stats[key] = {
                "name": pc.config.name,
                "request_count": pc.request_count,
                "token_count": pc.token_count,
                "error_count": pc.error_count,
                "last_used": datetime.fromtimestamp(pc.last_used).isoformat() if pc.last_used else None,
            }
        return stats

    def update_token_count(self, provider_key: str, tokens: int) -> None:
        """Update token count for a provider."""
        if provider_key in self._clients:
            self._clients[provider_key].token_count += tokens

    def record_error(self, provider_key: str) -> None:
        """Record an error for a provider."""
        if provider_key in self._clients:
            self._clients[provider_key].error_count += 1


# ============================================================================
# Global Instance
# ============================================================================

_provider_manager: Optional[MultiProviderManager] = None


def get_provider_manager() -> MultiProviderManager:
    """Get or create the global provider manager."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = MultiProviderManager()
    return _provider_manager


async def get_llm_client_for_provider(
    provider_key: Optional[str] = None,
) -> Tuple[AsyncOpenAI, ProviderConfig]:
    """Get an LLM client for the specified or active provider."""
    manager = get_provider_manager()
    return await manager.get_client(provider_key)


async def set_llm_provider(
    provider_key: str,
    api_key: Optional[str] = None,
) -> bool:
    """Set the active LLM provider."""
    manager = get_provider_manager()
    return await manager.set_active_provider(provider_key, api_key)


def list_available_providers() -> List[Dict[str, Any]]:
    """List all available LLM providers."""
    manager = get_provider_manager()
    return manager.get_available_providers()


def get_active_provider() -> Dict[str, Any]:
    """Get information about the active provider."""
    manager = get_provider_manager()
    return manager.get_active_provider_info()


async def test_provider_connection(provider_key: str) -> Dict[str, Any]:
    """Test connection to a specific provider."""
    manager = get_provider_manager()
    return await manager.test_provider(provider_key)


# ============================================================================
# Pricing Utilities
# ============================================================================

def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Estimate cost for a request."""
    pricing = get_model_pricing(model)
    return pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def get_pricing_info(model: str) -> Dict[str, Any]:
    """Get pricing information for a model."""
    pricing = get_model_pricing(model)
    return {
        "model": model,
        "input_cost_per_million": pricing.input_cost_per_million,
        "output_cost_per_million": pricing.output_cost_per_million,
        "cached_input_cost_per_million": pricing.cached_input_cost_per_million,
        "example_1k_input_cost": format_cost(pricing.input_cost_per_million / 1000),
        "example_1k_output_cost": format_cost(pricing.output_cost_per_million / 1000),
    }


def list_model_pricing() -> List[Dict[str, Any]]:
    """List pricing for all known models."""
    result = []
    for model, pricing in sorted(MODEL_PRICING.items()):
        result.append({
            "model": model,
            "input_per_million": f"${pricing.input_cost_per_million:.2f}",
            "output_per_million": f"${pricing.output_cost_per_million:.2f}",
            "cached_per_million": (
                f"${pricing.cached_input_cost_per_million:.2f}"
                if pricing.cached_input_cost_per_million is not None
                else "N/A"
            ),
        })
    return result
