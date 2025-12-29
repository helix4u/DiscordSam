"""Multi-Provider API Manager for LLM providers.

Supports: Local, LM Studio, Ollama, OpenAI (completions/responses), Claude, Google, Mistral, OpenRouter
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
from datetime import datetime

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""
    LOCAL = "local"
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OPENAI_COMPLETIONS = "openai_completions"
    OPENAI_RESPONSES = "openai_responses"
    OPENAI_PROVIDER = "openai_provider"
    CLAUDE = "claude"
    GOOGLE = "google"
    MISTRAL = "mistral"
    OPENROUTER = "openrouter"


@dataclass
class ProviderPricing:
    """Pricing information for a provider/model."""
    input_price_per_1k_tokens: float = 0.0
    output_price_per_1k_tokens: float = 0.0
    currency: str = "USD"
    last_updated: Optional[str] = None


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider_type: ProviderType
    name: str
    api_base_url: str
    api_key: Optional[str] = None
    model: str = ""
    temperature: float = 0.7
    supports_logit_bias: bool = True
    supports_json_mode: bool = False
    use_responses_api: bool = False
    is_google_model: bool = False
    pricing: Optional[ProviderPricing] = None
    timeout: float = 900.0
    max_retries: int = 0


# Default pricing information (can be updated from API or config)
DEFAULT_PRICING: Dict[str, ProviderPricing] = {
    # OpenAI
    "gpt-4": ProviderPricing(0.03, 0.06, "USD"),
    "gpt-4-turbo": ProviderPricing(0.01, 0.03, "USD"),
    "gpt-3.5-turbo": ProviderPricing(0.0005, 0.0015, "USD"),
    "gpt-4o": ProviderPricing(0.005, 0.015, "USD"),
    "gpt-4o-mini": ProviderPricing(0.00015, 0.0006, "USD"),
    
    # Claude (Anthropic)
    "claude-3-opus": ProviderPricing(0.015, 0.075, "USD"),
    "claude-3-sonnet": ProviderPricing(0.003, 0.015, "USD"),
    "claude-3-haiku": ProviderPricing(0.00025, 0.00125, "USD"),
    "claude-3-5-sonnet": ProviderPricing(0.003, 0.015, "USD"),
    
    # Google
    "gemini-pro": ProviderPricing(0.0005, 0.0015, "USD"),
    "gemini-1.5-pro": ProviderPricing(0.00125, 0.005, "USD"),
    
    # Mistral
    "mistral-large": ProviderPricing(0.002, 0.006, "USD"),
    "mistral-medium": ProviderPricing(0.0027, 0.0081, "USD"),
    "mistral-small": ProviderPricing(0.002, 0.006, "USD"),
    
    # OpenRouter (approximate, varies by model)
    "openrouter-default": ProviderPricing(0.002, 0.002, "USD"),
}


class ProviderManager:
    """Manages multiple LLM providers and their configurations."""
    
    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._clients: Dict[str, AsyncOpenAI] = {}
        self._pricing_cache: Dict[str, ProviderPricing] = DEFAULT_PRICING.copy()
    
    def register_provider(
        self,
        name: str,
        provider_type: ProviderType,
        api_base_url: str,
        api_key: Optional[str] = None,
        model: str = "",
        **kwargs,
    ) -> ProviderConfig:
        """Register a new provider configuration."""
        config = ProviderConfig(
            provider_type=provider_type,
            name=name,
            api_base_url=api_base_url,
            api_key=api_key,
            model=model,
            **kwargs,
        )
        self._providers[name] = config
        
        # Create client
        client = AsyncOpenAI(
            base_url=api_base_url,
            api_key=api_key or "lm-studio",  # Default for local providers
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self._clients[name] = client
        
        logger.info(f"Registered provider '{name}' ({provider_type.value})")
        return config
    
    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get a provider configuration by name."""
        return self._providers.get(name)
    
    def get_client(self, name: str) -> Optional[AsyncOpenAI]:
        """Get an AsyncOpenAI client for a provider."""
        return self._clients.get(name)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())
    
    def update_pricing(self, model: str, pricing: ProviderPricing) -> None:
        """Update pricing information for a model."""
        self._pricing_cache[model] = pricing
        pricing.last_updated = datetime.now().isoformat()
    
    def get_pricing(self, model: str) -> Optional[ProviderPricing]:
        """Get pricing information for a model."""
        # Try exact match first
        if model in self._pricing_cache:
            return self._pricing_cache[model]
        
        # Try partial matches
        for cached_model, pricing in self._pricing_cache.items():
            if cached_model.lower() in model.lower() or model.lower() in cached_model.lower():
                return pricing
        
        return None
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Optional[float]:
        """Calculate the cost for a request."""
        pricing = self.get_pricing(model)
        if not pricing:
            return None
        
        input_cost = (input_tokens / 1000.0) * pricing.input_price_per_1k_tokens
        output_cost = (output_tokens / 1000.0) * pricing.output_price_per_1k_tokens
        return input_cost + output_cost


# Global provider manager instance
_provider_manager = ProviderManager()


def initialize_providers_from_config() -> None:
    """Initialize providers from environment variables."""
    from config import config as app_config
    
    # Local/LM Studio/Ollama (default)
    local_url = os.getenv("LOCAL_SERVER_URL", "http://localhost:1234/v1")
    local_key = os.getenv("LLM_API_KEY", "")
    
    _provider_manager.register_provider(
        "local",
        ProviderType.LOCAL,
        local_url,
        api_key=local_key or None,
        model=app_config.LLM_MODEL,
        temperature=app_config.LLM_TEMPERATURE,
        supports_logit_bias=app_config.LLM_SUPPORTS_LOGIT_BIAS,
        supports_json_mode=app_config.LLM_SUPPORTS_JSON_MODE,
        use_responses_api=app_config.LLM_USE_RESPONSES_API,
        is_google_model=app_config.IS_GOOGLE_MODEL,
    )
    
    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        _provider_manager.register_provider(
            "openai",
            ProviderType.OPENAI_COMPLETIONS,
            "https://api.openai.com/v1",
            api_key=openai_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            supports_logit_bias=True,
            supports_json_mode=True,
        )
    
    # Claude (Anthropic)
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    if claude_key:
        _provider_manager.register_provider(
            "claude",
            ProviderType.CLAUDE,
            "https://api.anthropic.com/v1",
            api_key=claude_key,
            model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            supports_logit_bias=False,
            supports_json_mode=False,
        )
    
    # Google
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        _provider_manager.register_provider(
            "google",
            ProviderType.GOOGLE,
            "https://generativelanguage.googleapis.com/v1",
            api_key=google_key,
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-pro"),
            supports_logit_bias=False,
            supports_json_mode=False,
            is_google_model=True,
        )
    
    # Mistral
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key:
        _provider_manager.register_provider(
            "mistral",
            ProviderType.MISTRAL,
            "https://api.mistral.ai/v1",
            api_key=mistral_key,
            model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
            supports_logit_bias=False,
            supports_json_mode=True,
        )
    
    # OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        _provider_manager.register_provider(
            "openrouter",
            ProviderType.OPENROUTER,
            "https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"),
            supports_logit_bias=True,
            supports_json_mode=True,
        )
    
    logger.info(f"Initialized {len(_provider_manager.list_providers())} providers")


def get_provider_manager() -> ProviderManager:
    """Get the global provider manager instance."""
    return _provider_manager
