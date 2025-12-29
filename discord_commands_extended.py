"""Extended Discord Commands for Knowledge Graph, Provider Selection, and Spending Reports.

This module contains slash commands for:
- Knowledge graph management and retrieval
- LLM provider selection and testing
- Usage tracking and spending reports
- Rate limiting status
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands

from config import config

# Import new modules
from knowledge_graph import (
    get_kg_manager,
    build_daily_knowledge_graph,
    retrieve_knowledge_for_query,
    maintain_knowledge_graphs,
)
from multi_provider_llm import (
    get_provider_manager,
    list_available_providers,
    get_active_provider,
    set_llm_provider,
    test_provider_connection,
    get_pricing_info,
    list_model_pricing,
    format_cost,
)
from usage_tracking import (
    get_usage_tracker,
    get_spending_report,
    get_cost_estimate,
    get_daily_breakdown,
    TimeFrame,
    format_summary_for_discord,
    format_projection_for_discord,
)
from rate_limiter import get_rate_limiter, get_channel_output_limiter
from state import BotState, OperationType

logger = logging.getLogger(__name__)

# Global references (set during setup)
bot_instance: Optional[commands.Bot] = None
bot_state_instance: Optional[BotState] = None


def is_admin():
    """Check if user is an admin."""
    async def predicate(interaction: discord.Interaction) -> bool:
        if not config.ADMIN_USER_IDS:
            return True
        return interaction.user.id in config.ADMIN_USER_IDS
    return app_commands.check(predicate)


def setup_extended_commands(bot: commands.Bot, bot_state: BotState) -> None:
    """Set up extended commands for the bot."""
    global bot_instance, bot_state_instance
    bot_instance = bot
    bot_state_instance = bot_state

    # ========================================================================
    # Knowledge Graph Commands
    # ========================================================================

    @bot.tree.command(
        name="kg_build",
        description="Build or rebuild the knowledge graph for a specific date."
    )
    @app_commands.describe(
        date="Date in YYYY-MM-DD format (defaults to today)",
        force_rebuild="Force rebuild even if graph exists"
    )
    @is_admin()
    async def kg_build_command(
        interaction: discord.Interaction,
        date: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        await interaction.response.defer(ephemeral=True)
        
        try:
            if date:
                target_date = datetime.strptime(date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            else:
                target_date = datetime.now(timezone.utc)
            
            snapshot = await build_daily_knowledge_graph(target_date, force_rebuild)
            
            stats = f"""**Knowledge Graph Built**
Date: {snapshot.period_start[:10]}
Entities: {len(snapshot.entities)}
Relations: {len(snapshot.relations)}
Observations: {len(snapshot.observations)}
Source Documents: {snapshot.source_doc_count}
Last Updated: {snapshot.last_updated[:19]}"""
            
            await interaction.followup.send(stats, ephemeral=True)
            
        except Exception as e:
            logger.error(f"KG build failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Failed to build knowledge graph: {e}",
                ephemeral=True
            )

    @bot.tree.command(
        name="kg_query",
        description="Query the knowledge graph for relevant information."
    )
    @app_commands.describe(
        query="What to search for in the knowledge graph",
        days_back="How many days of history to search (default: 30)"
    )
    async def kg_query_command(
        interaction: discord.Interaction,
        query: str,
        days_back: int = 30,
    ):
        await interaction.response.defer()
        
        try:
            result = await retrieve_knowledge_for_query(query, days_back=days_back)
            
            embed = discord.Embed(
                title=f"Knowledge Graph Results",
                description=result.get("summary", "No summary available."),
                color=config.EMBED_COLOR.get("complete", discord.Color.green().value),
            )
            
            # Add entities
            if result.get("entities"):
                entities_text = "\n".join([
                    f"• **{e['name']}** ({e['entity_type']})"
                    for e in result["entities"][:5]
                ])
                embed.add_field(
                    name=f"Key Entities ({len(result['entities'])})",
                    value=entities_text[:1024],
                    inline=False,
                )
            
            # Add relations
            if result.get("relations"):
                relations_text = "\n".join([
                    f"• {r['subject']} → {r['predicate']} → {r['object']}"
                    for r in result["relations"][:5]
                ])
                embed.add_field(
                    name=f"Key Relations ({len(result['relations'])})",
                    value=relations_text[:1024],
                    inline=False,
                )
            
            # Add observations
            if result.get("observations"):
                obs_text = "\n".join([
                    f"• {o['statement'][:100]}..."
                    if len(o['statement']) > 100 else f"• {o['statement']}"
                    for o in result["observations"][:5]
                ])
                embed.add_field(
                    name=f"Key Observations ({len(result['observations'])})",
                    value=obs_text[:1024],
                    inline=False,
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error(f"KG query failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Knowledge graph query failed: {e}",
                ephemeral=True
            )

    @bot.tree.command(
        name="kg_maintain",
        description="Run knowledge graph maintenance tasks."
    )
    @is_admin()
    async def kg_maintain_command(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            result = await maintain_knowledge_graphs()
            
            stats = result.get("statistics", {})
            message = f"""**Knowledge Graph Maintenance Complete**
Status: {result.get('status', 'unknown')}
Removed Snapshots: {result.get('removed_snapshots', 0)}

**Statistics:**
Total Snapshots: {stats.get('total_snapshots', 0)}
Total Entities: {stats.get('total_entities', 0)}
Total Relations: {stats.get('total_relations', 0)}
Total Observations: {stats.get('total_observations', 0)}"""
            
            await interaction.followup.send(message, ephemeral=True)
            
        except Exception as e:
            logger.error(f"KG maintenance failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Maintenance failed: {e}",
                ephemeral=True
            )

    @bot.tree.command(
        name="kg_stats",
        description="Show knowledge graph statistics."
    )
    async def kg_stats_command(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            manager = get_kg_manager()
            stats = manager.get_statistics()
            
            by_type = stats.get("snapshots_by_type", {})
            type_text = "\n".join([
                f"• {k}: {v}" for k, v in by_type.items()
            ]) or "None"
            
            message = f"""**Knowledge Graph Statistics**
Total Snapshots: {stats.get('total_snapshots', 0)}
Total Entities: {stats.get('total_entities', 0)}
Total Relations: {stats.get('total_relations', 0)}
Total Observations: {stats.get('total_observations', 0)}

**By Type:**
{type_text}

Storage Path: `{stats.get('storage_path', 'N/A')}`"""
            
            await interaction.followup.send(message, ephemeral=True)
            
        except Exception as e:
            logger.error(f"KG stats failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Failed to get stats: {e}",
                ephemeral=True
            )

    # ========================================================================
    # Provider Selection Commands
    # ========================================================================

    @bot.tree.command(
        name="provider_list",
        description="List available LLM providers."
    )
    async def provider_list_command(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        providers = list_available_providers()
        active = get_active_provider()
        
        lines = ["**Available LLM Providers**\n"]
        for p in providers:
            status = "✅" if p["key"] == active.get("key") else "⬜"
            key_status = "🔑" if p["has_api_key"] else "❌"
            lines.append(
                f"{status} **{p['name']}** (`{p['key']}`)\n"
                f"   Default Model: `{p['default_model']}`\n"
                f"   API Key: {key_status} | Vision: {'✓' if p['supports_vision'] else '✗'} | "
                f"RPM: {p['rate_limit_rpm']}"
            )
        
        await interaction.followup.send("\n".join(lines), ephemeral=True)

    @bot.tree.command(
        name="provider_set",
        description="Set the active LLM provider."
    )
    @app_commands.describe(
        provider="Provider key (e.g., openai, anthropic, mistral, openrouter)",
        api_key="Optional API key for the provider"
    )
    @app_commands.choices(provider=[
        app_commands.Choice(name="LM Studio (Local)", value="lm_studio"),
        app_commands.Choice(name="Ollama (Local)", value="ollama"),
        app_commands.Choice(name="OpenAI", value="openai"),
        app_commands.Choice(name="OpenAI Responses API", value="openai_responses"),
        app_commands.Choice(name="Anthropic (Claude)", value="anthropic"),
        app_commands.Choice(name="Google (Gemini)", value="google"),
        app_commands.Choice(name="Mistral", value="mistral"),
        app_commands.Choice(name="OpenRouter", value="openrouter"),
    ])
    @is_admin()
    async def provider_set_command(
        interaction: discord.Interaction,
        provider: str,
        api_key: Optional[str] = None,
    ):
        await interaction.response.defer(ephemeral=True)
        
        try:
            success = await set_llm_provider(provider, api_key)
            
            if success:
                info = get_active_provider()
                await interaction.followup.send(
                    f"✅ Provider set to **{info['name']}**\n"
                    f"Default model: `{info['default_model']}`\n"
                    f"API Key: {'Set' if info['has_api_key'] else 'Not set'}",
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    f"❌ Failed to set provider: {provider}",
                    ephemeral=True
                )
                
        except Exception as e:
            logger.error(f"Provider set failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Error setting provider: {e}",
                ephemeral=True
            )

    @bot.tree.command(
        name="provider_test",
        description="Test connectivity to a provider."
    )
    @app_commands.describe(provider="Provider to test")
    @app_commands.choices(provider=[
        app_commands.Choice(name="LM Studio", value="lm_studio"),
        app_commands.Choice(name="Ollama", value="ollama"),
        app_commands.Choice(name="OpenAI", value="openai"),
        app_commands.Choice(name="Anthropic", value="anthropic"),
        app_commands.Choice(name="Google", value="google"),
        app_commands.Choice(name="Mistral", value="mistral"),
        app_commands.Choice(name="OpenRouter", value="openrouter"),
    ])
    @is_admin()
    async def provider_test_command(
        interaction: discord.Interaction,
        provider: str,
    ):
        await interaction.response.defer(ephemeral=True)
        
        try:
            result = await test_provider_connection(provider)
            
            if result["success"]:
                await interaction.followup.send(
                    f"✅ **{provider}** connection successful!\n"
                    f"Model: `{result['model']}`\n"
                    f"Latency: {result['latency_ms']}ms\n"
                    f"Response: {result['response'][:100]}",
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    f"❌ **{provider}** connection failed!\n"
                    f"Error: {result['error']}",
                    ephemeral=True
                )
                
        except Exception as e:
            logger.error(f"Provider test failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Test error: {e}",
                ephemeral=True
            )

    @bot.tree.command(
        name="provider_info",
        description="Show current provider information."
    )
    async def provider_info_command(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        info = get_active_provider()
        
        message = f"""**Active LLM Provider**
Name: **{info.get('name', 'Unknown')}**
Key: `{info.get('key', 'unknown')}`
Type: `{info.get('type', 'unknown')}`
Default Model: `{info.get('default_model', 'N/A')}`
API Base: `{info.get('api_base_url', 'N/A')}`
API Key: {'✅ Set' if info.get('has_api_key') else '❌ Not set'}
Rate Limit: {info.get('rate_limit_rpm', 'N/A')} RPM"""
        
        await interaction.followup.send(message, ephemeral=True)

    # ========================================================================
    # Pricing Commands
    # ========================================================================

    @bot.tree.command(
        name="pricing",
        description="Show pricing information for a model."
    )
    @app_commands.describe(model="Model name to get pricing for")
    async def pricing_command(
        interaction: discord.Interaction,
        model: str,
    ):
        await interaction.response.defer(ephemeral=True)
        
        info = get_pricing_info(model)
        
        message = f"""**Pricing for `{info['model']}`**
Input: ${info['input_cost_per_million']:.2f} per million tokens
Output: ${info['output_cost_per_million']:.2f} per million tokens
Cached: {f"${info['cached_input_cost_per_million']:.2f}" if info['cached_input_cost_per_million'] else 'N/A'} per million tokens

**Example Costs (1K tokens):**
Input: {info['example_1k_input_cost']}
Output: {info['example_1k_output_cost']}"""
        
        await interaction.followup.send(message, ephemeral=True)

    @bot.tree.command(
        name="pricing_list",
        description="List pricing for all known models."
    )
    async def pricing_list_command(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        pricing = list_model_pricing()
        
        lines = ["**Model Pricing (per million tokens)**\n"]
        lines.append("```")
        lines.append(f"{'Model':<35} {'Input':>10} {'Output':>10} {'Cached':>10}")
        lines.append("-" * 67)
        
        for p in pricing[:25]:  # Limit to fit in Discord message
            lines.append(
                f"{p['model']:<35} {p['input_per_million']:>10} "
                f"{p['output_per_million']:>10} {p['cached_per_million']:>10}"
            )
        
        lines.append("```")
        
        if len(pricing) > 25:
            lines.append(f"_(Showing 25 of {len(pricing)} models)_")
        
        await interaction.followup.send("\n".join(lines), ephemeral=True)

    # ========================================================================
    # Spending Report Commands
    # ========================================================================

    @bot.tree.command(
        name="spending",
        description="Show spending report for a time period."
    )
    @app_commands.describe(period="Time period for the report")
    @app_commands.choices(period=[
        app_commands.Choice(name="Last Hour", value="hourly"),
        app_commands.Choice(name="Today", value="daily"),
        app_commands.Choice(name="This Week", value="weekly"),
        app_commands.Choice(name="This Month", value="monthly"),
        app_commands.Choice(name="This Year", value="yearly"),
        app_commands.Choice(name="All Time", value="all_time"),
    ])
    async def spending_command(
        interaction: discord.Interaction,
        period: str = "daily",
    ):
        await interaction.response.defer(ephemeral=True)
        
        try:
            time_frame = TimeFrame(period)
            summary = await get_spending_report(time_frame)
            
            message = format_summary_for_discord(summary)
            await interaction.followup.send(message, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Spending report failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Failed to generate report: {e}",
                ephemeral=True
            )

    @bot.tree.command(
        name="spending_projection",
        description="Show projected costs for the upcoming period."
    )
    @app_commands.describe(days="Number of days to project (default: 30)")
    async def spending_projection_command(
        interaction: discord.Interaction,
        days: int = 30,
    ):
        await interaction.response.defer(ephemeral=True)
        
        try:
            projection = await get_cost_estimate(days)
            message = format_projection_for_discord(projection)
            await interaction.followup.send(message, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Projection failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Failed to calculate projection: {e}",
                ephemeral=True
            )

    @bot.tree.command(
        name="spending_daily",
        description="Show daily spending breakdown."
    )
    @app_commands.describe(days="Number of days to show (default: 7)")
    async def spending_daily_command(
        interaction: discord.Interaction,
        days: int = 7,
    ):
        await interaction.response.defer(ephemeral=True)
        
        try:
            daily = await get_daily_breakdown(min(days, 30))
            
            if not daily:
                await interaction.followup.send(
                    "No spending data available.",
                    ephemeral=True
                )
                return
            
            lines = ["**Daily Spending Breakdown**\n"]
            lines.append("```")
            lines.append(f"{'Date':<12} {'Requests':>10} {'Tokens':>12} {'Cost':>12}")
            lines.append("-" * 48)
            
            total_cost = 0.0
            for day in daily:
                lines.append(
                    f"{day['date']:<12} {day['requests']:>10,} "
                    f"{day['input_tokens'] + day['output_tokens']:>12,} "
                    f"{day['formatted_cost']:>12}"
                )
                total_cost += day['cost']
            
            lines.append("-" * 48)
            lines.append(f"{'TOTAL':<12} {'':<10} {'':<12} {format_cost(total_cost):>12}")
            lines.append("```")
            
            await interaction.followup.send("\n".join(lines), ephemeral=True)
            
        except Exception as e:
            logger.error(f"Daily breakdown failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Failed to get breakdown: {e}",
                ephemeral=True
            )

    # ========================================================================
    # Rate Limiting Commands
    # ========================================================================

    @bot.tree.command(
        name="ratelimit_status",
        description="Show rate limiter status and statistics."
    )
    @is_admin()
    async def ratelimit_status_command(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            rate_limiter = get_rate_limiter()
            output_limiter = get_channel_output_limiter()
            
            rl_stats = rate_limiter.get_statistics()
            ol_stats = output_limiter.get_statistics()
            
            lines = ["**Rate Limiter Status**\n"]
            
            # API rate limiter stats
            lines.append("**API Rate Limiting:**")
            lines.append(f"- Requests Allowed: {rl_stats.get('requests_allowed', 0):,}")
            lines.append(f"- Requests Blocked: {rl_stats.get('requests_blocked', 0):,}")
            lines.append(f"- Rate Limit Hits: {rl_stats.get('rate_limit_hits', 0):,}")
            lines.append(f"- Cooldowns Triggered: {rl_stats.get('cooldowns_triggered', 0):,}")
            
            # Active requests
            active = rl_stats.get('active_requests', {})
            if active:
                lines.append("\n**Active Requests:**")
                for key, count in active.items():
                    lines.append(f"- {key}: {count}")
            
            # Channel output stats
            lines.append("\n**Channel Output Limiting:**")
            lines.append(f"- Edits Throttled: {ol_stats.get('edits_throttled', 0):,}")
            lines.append(f"- Operations Serialized: {ol_stats.get('operations_serialized', 0):,}")
            lines.append(f"- Active Channels: {ol_stats.get('active_channels', 0)}")
            
            await interaction.followup.send("\n".join(lines), ephemeral=True)
            
        except Exception as e:
            logger.error(f"Rate limit status failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Failed to get status: {e}",
                ephemeral=True
            )

    # ========================================================================
    # Bot Statistics Command
    # ========================================================================

    @bot.tree.command(
        name="bot_stats",
        description="Show comprehensive bot statistics."
    )
    @is_admin()
    async def bot_stats_command(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        try:
            if not bot_state_instance:
                await interaction.followup.send(
                    "Bot state not available.",
                    ephemeral=True
                )
                return
            
            stats = await bot_state_instance.get_bot_statistics()
            
            lines = ["**Bot Statistics**\n"]
            
            lines.append("**Memory:**")
            lines.append(f"- Channels with History: {stats.get('channels_with_history', 0)}")
            lines.append(f"- Total Messages Cached: {stats.get('total_messages_cached', 0)}")
            lines.append(f"- Pending Reminders: {stats.get('pending_reminders', 0)}")
            
            lines.append("\n**Scheduling:**")
            lines.append(f"- Scheduled Jobs: {stats.get('scheduled_jobs', 0)}")
            lines.append(f"- Paused: {'Yes' if stats.get('schedules_paused') else 'No'}")
            
            active_ops = stats.get('active_operations', {})
            if active_ops:
                lines.append("\n**Active Operations:**")
                for ch_id, op in active_ops.items():
                    lines.append(
                        f"- Channel {ch_id}: {op['type']} "
                        f"({op['duration_seconds']:.0f}s, {op['progress']*100:.0f}%)"
                    )
            
            queue_sizes = stats.get('output_queue_sizes', {})
            if queue_sizes:
                lines.append("\n**Output Queues:**")
                for ch_id, size in queue_sizes.items():
                    lines.append(f"- Channel {ch_id}: {size} items")
            
            if stats.get('playwright_last_usage'):
                lines.append(f"\n**Playwright:** Last used {stats['playwright_last_usage']}")
            
            await interaction.followup.send("\n".join(lines), ephemeral=True)
            
        except Exception as e:
            logger.error(f"Bot stats failed: {e}", exc_info=True)
            await interaction.followup.send(
                f"Failed to get stats: {e}",
                ephemeral=True
            )

    logger.info("Extended commands registered successfully.")
