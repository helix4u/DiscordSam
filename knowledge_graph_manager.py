"""Knowledge Graph Manager for per-day knowledge graph creation and maintenance.

This module handles:
- Creating per-day knowledge graphs from ChromaDB collections
- Deduplicating and merging knowledge entries
- Maintaining knowledge graphs for efficient retrieval
- Building knowledge graphs incrementally during memory retrieval
"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import uuid4
import json

import chromadb
from chromadb.errors import InternalError

from config import config
from llm_clients import get_llm_runtime
from openai_api import create_chat_completion, extract_text
from logit_biases import LOGIT_BIAS_UNWANTED_TOKENS_STR

logger = logging.getLogger(__name__)

# Global ChromaDB collections for knowledge graphs
kg_collection: Optional[chromadb.Collection] = None
kg_metadata_collection: Optional[chromadb.Collection] = None


def initialize_knowledge_graph_collections(chroma_client: chromadb.ClientAPI) -> bool:
    """Initialize ChromaDB collections for knowledge graphs."""
    global kg_collection, kg_metadata_collection
    
    try:
        kg_collection_name = getattr(config, "CHROMA_KNOWLEDGE_GRAPH_COLLECTION_NAME", "knowledge_graphs")
        kg_metadata_collection_name = getattr(config, "CHROMA_KNOWLEDGE_GRAPH_METADATA_COLLECTION_NAME", "kg_metadata")
        
        kg_collection = chroma_client.get_or_create_collection(name=kg_collection_name)
        kg_metadata_collection = chroma_client.get_or_create_collection(name=kg_metadata_collection_name)
        
        logger.info(
            f"Initialized knowledge graph collections: '{kg_collection_name}', '{kg_metadata_collection_name}'"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize knowledge graph collections: {e}", exc_info=True)
        kg_collection = None
        kg_metadata_collection = None
        return False


def _date_to_kg_key(date: datetime) -> str:
    """Convert a datetime to a knowledge graph key (YYYY-MM-DD format)."""
    return date.date().isoformat()


def _get_day_start_end(date: datetime) -> Tuple[datetime, datetime]:
    """Get the start and end of a day for a given datetime."""
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1) - timedelta(microseconds=1)
    return day_start, day_end


async def _extract_knowledge_from_text(
    text: str,
    date: datetime,
    source_collection: str,
    source_id: str,
) -> Optional[Dict[str, Any]]:
    """Extract structured knowledge from text using LLM."""
    if not text.strip():
        return None
    
    fast_runtime = get_llm_runtime("fast")
    fast_client = fast_runtime.client
    fast_provider = fast_runtime.provider
    fast_logit_bias = (
        LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
    )
    
    max_input_length = 120000
    truncated_text = text[:max_input_length]
    
    system_prompt = (
        "You are a knowledge extraction system. Extract structured knowledge from the provided text "
        "including entities, relationships, facts, and key insights. Format as JSON with keys: "
        "'entities' (list of entity names), 'relationships' (list of {subject, predicate, object}), "
        "'facts' (list of factual statements), and 'insights' (list of key insights or observations)."
    )
    
    user_prompt = f"""Extract structured knowledge from the following text. Focus on concrete facts, 
relationships between entities, and key insights. Be concise but comprehensive.

TEXT:
---
{truncated_text}
---

OUTPUT JSON STRUCTURE:
{{
  "entities": ["entity1", "entity2", ...],
  "relationships": [
    {{"subject": "entity1", "predicate": "relationship_type", "object": "entity2"}}
  ],
  "facts": ["factual statement 1", "factual statement 2", ...],
  "insights": ["insight 1", "insight 2", ...]
}}

Return only valid JSON, no explanations."""
    
    response_format_arg: Dict[str, Any] = {}
    if fast_provider.supports_json_mode:
        response_format_arg = {"response_format": {"type": "json_object"}}
    
    try:
        response = await create_chat_completion(
            fast_client,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=fast_provider.model,
            max_tokens=16384,
            temperature=fast_provider.temperature,
            logit_bias=fast_logit_bias,
            use_responses_api=fast_provider.use_responses_api,
            **response_format_arg,
        )
        
        raw_content = extract_text(response, fast_provider.use_responses_api)
        if raw_content:
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
                if raw_content.endswith("```"):
                    raw_content = raw_content[:-3]
                raw_content = raw_content.strip()
            
            try:
                extracted = json.loads(raw_content)
                if isinstance(extracted, dict):
                    extracted["source_collection"] = source_collection
                    extracted["source_id"] = source_id
                    extracted["extraction_date"] = date.isoformat()
                    return extracted
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from knowledge extraction for {source_id}")
        
        return None
    except Exception as e:
        logger.error(f"Failed to extract knowledge from text: {e}", exc_info=True)
        return None


async def _merge_knowledge_graphs(
    existing_kg: Dict[str, Any],
    new_knowledge: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge new knowledge into an existing knowledge graph with deduplication."""
    merged = {
        "entities": list(set(existing_kg.get("entities", []) + new_knowledge.get("entities", []))),
        "relationships": existing_kg.get("relationships", []) + new_knowledge.get("relationships", []),
        "facts": existing_kg.get("facts", []) + new_knowledge.get("facts", []),
        "insights": existing_kg.get("insights", []) + new_knowledge.get("insights", []),
        "sources": existing_kg.get("sources", []) + [{
            "collection": new_knowledge.get("source_collection"),
            "id": new_knowledge.get("source_id"),
            "date": new_knowledge.get("extraction_date"),
        }],
    }
    
    # Deduplicate relationships (same subject-predicate-object triple)
    seen_rels = set()
    deduplicated_rels = []
    for rel in merged["relationships"]:
        if isinstance(rel, dict):
            key = (
                rel.get("subject", ""),
                rel.get("predicate", ""),
                rel.get("object", "")
            )
            if key not in seen_rels and all(key):
                seen_rels.add(key)
                deduplicated_rels.append(rel)
    merged["relationships"] = deduplicated_rels
    
    # Deduplicate facts and insights (simple string matching)
    merged["facts"] = list(dict.fromkeys(merged["facts"]))  # Preserves order
    merged["insights"] = list(dict.fromkeys(merged["insights"]))
    
    return merged


async def build_knowledge_graph_for_day(
    date: datetime,
    chroma_client: chromadb.ClientAPI,
    *,
    collections_to_process: Optional[List[str]] = None,
) -> Optional[str]:
    """Build or update a knowledge graph for a specific day.
    
    Args:
        date: The date to build the knowledge graph for
        chroma_client: ChromaDB client instance
        collections_to_process: Optional list of collection names to process.
            If None, processes all relevant collections.
    
    Returns:
        The knowledge graph document ID if successful, None otherwise.
    """
    if not kg_collection or not kg_metadata_collection:
        logger.warning("Knowledge graph collections not initialized")
        return None
    
    day_key = _date_to_kg_key(date)
    day_start, day_end = _get_day_start_end(date)
    
    # Import collections dynamically to avoid circular imports
    from rag_chroma_manager import (
        chat_history_collection,
        distilled_chat_summary_collection,
        rss_summary_collection,
        tweets_collection,
        relation_collection,
        observation_collection,
        timeline_summary_collection,
    )
    
    collection_map = {
        "chat_history": chat_history_collection,
        "distilled": distilled_chat_summary_collection,
        "rss": rss_summary_collection,
        "tweets": tweets_collection,
        "relations": relation_collection,
        "observations": observation_collection,
        "timeline": timeline_summary_collection,
    }
    
    if collections_to_process:
        collection_map = {k: v for k, v in collection_map.items() if k in collections_to_process}
    
    # Check if KG already exists for this day
    existing_kg_id = None
    existing_kg_doc = None
    try:
        metadata_results = await asyncio.to_thread(
            kg_metadata_collection.get,
            where={"day_key": day_key},
            include=["documents", "metadatas"],
        )
        if metadata_results.get("ids"):
            existing_kg_id = metadata_results["ids"][0]
            # Fetch the actual KG document
            kg_results = await asyncio.to_thread(
                kg_collection.get,
                ids=[existing_kg_id],
                include=["documents"],
            )
            if kg_results.get("documents"):
                existing_kg_doc = json.loads(kg_results["documents"][0])
    except Exception as e:
        logger.debug(f"No existing KG found for {day_key} or error fetching: {e}")
    
    # Collect all documents from the day
    all_knowledge: List[Dict[str, Any]] = []
    
    for coll_name, collection in collection_map.items():
        if not collection:
            continue
        
        try:
            # Get all documents from the collection
            res = await asyncio.to_thread(collection.get, include=["documents", "metadatas"])
            docs = res.get("documents", [])
            metadatas = res.get("metadatas", [])
            
            for doc_text, meta in zip(docs, metadatas):
                if not isinstance(doc_text, str) or not isinstance(meta, dict):
                    continue
                
                # Check if document is from the target day
                ts_str = meta.get("timestamp")
                if not ts_str:
                    continue
                
                try:
                    doc_date = datetime.fromisoformat(ts_str)
                    if doc_date.tzinfo:
                        doc_date = doc_date.astimezone().replace(tzinfo=None)
                    
                    if day_start <= doc_date <= day_end:
                        extracted = await _extract_knowledge_from_text(
                            doc_text,
                            doc_date,
                            coll_name,
                            meta.get("id", "unknown"),
                        )
                        if extracted:
                            all_knowledge.append(extracted)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse timestamp {ts_str}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error processing collection {coll_name} for KG: {e}", exc_info=True)
            continue
    
    if not all_knowledge:
        logger.info(f"No knowledge extracted for day {day_key}")
        return existing_kg_id  # Return existing if no new knowledge
    
    # Merge all knowledge
    if existing_kg_doc:
        merged_kg = existing_kg_doc
        for new_knowledge in all_knowledge:
            merged_kg = await _merge_knowledge_graphs(merged_kg, new_knowledge)
    else:
        # Start with first knowledge entry
        merged_kg = {
            "entities": all_knowledge[0].get("entities", []),
            "relationships": all_knowledge[0].get("relationships", []),
            "facts": all_knowledge[0].get("facts", []),
            "insights": all_knowledge[0].get("insights", []),
            "sources": [{
                "collection": all_knowledge[0].get("source_collection"),
                "id": all_knowledge[0].get("source_id"),
                "date": all_knowledge[0].get("extraction_date"),
            }],
        }
        # Merge remaining knowledge
        for new_knowledge in all_knowledge[1:]:
            merged_kg = await _merge_knowledge_graphs(merged_kg, new_knowledge)
    
    # Store the merged knowledge graph
    kg_doc_id = existing_kg_id or f"kg_{day_key}_{uuid4().hex[:8]}"
    kg_doc_text = json.dumps(merged_kg, ensure_ascii=False)
    
    try:
        if existing_kg_id:
            # Update existing
            await asyncio.to_thread(
                kg_collection.update,
                ids=[kg_doc_id],
                documents=[kg_doc_text],
            )
            await asyncio.to_thread(
                kg_metadata_collection.update,
                ids=[kg_doc_id],
                metadatas=[{
                    "day_key": day_key,
                    "updated_at": datetime.now().isoformat(),
                    "entity_count": len(merged_kg.get("entities", [])),
                    "relationship_count": len(merged_kg.get("relationships", [])),
                    "fact_count": len(merged_kg.get("facts", [])),
                    "insight_count": len(merged_kg.get("insights", [])),
                }],
            )
        else:
            # Create new
            await asyncio.to_thread(
                kg_collection.add,
                documents=[kg_doc_text],
                metadatas=[{
                    "day_key": day_key,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "entity_count": len(merged_kg.get("entities", [])),
                    "relationship_count": len(merged_kg.get("relationships", [])),
                    "fact_count": len(merged_kg.get("facts", [])),
                    "insight_count": len(merged_kg.get("insights", [])),
                }],
                ids=[kg_doc_id],
            )
            await asyncio.to_thread(
                kg_metadata_collection.add,
                documents=[f"Metadata for knowledge graph {day_key}"],
                metadatas=[{
                    "day_key": day_key,
                    "kg_id": kg_doc_id,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }],
                ids=[kg_doc_id],
            )
        
        logger.info(f"Built/updated knowledge graph for {day_key}: {kg_doc_id}")
        return kg_doc_id
    
    except Exception as e:
        logger.error(f"Failed to store knowledge graph for {day_key}: {e}", exc_info=True)
        return None


async def retrieve_knowledge_graph_for_day(
    date: datetime,
) -> Optional[Dict[str, Any]]:
    """Retrieve a knowledge graph for a specific day."""
    if not kg_collection:
        return None
    
    day_key = _date_to_kg_key(date)
    
    try:
        metadata_results = await asyncio.to_thread(
            kg_metadata_collection.get,
            where={"day_key": day_key},
            include=["metadatas"],
        )
        
        if not metadata_results.get("ids"):
            return None
        
        kg_id = metadata_results["ids"][0]
        kg_results = await asyncio.to_thread(
            kg_collection.get,
            ids=[kg_id],
            include=["documents"],
        )
        
        if kg_results.get("documents"):
            return json.loads(kg_results["documents"][0])
        
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve knowledge graph for {day_key}: {e}", exc_info=True)
        return None


async def build_knowledge_graphs_for_date_range(
    start_date: datetime,
    end_date: datetime,
    chroma_client: chromadb.ClientAPI,
    *,
    collections_to_process: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    """Build knowledge graphs for a range of dates.
    
    Returns:
        Dictionary mapping date keys to knowledge graph IDs (or None if failed).
    """
    results = {}
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    while current_date <= end_date:
        day_key = _date_to_kg_key(current_date)
        kg_id = await build_knowledge_graph_for_day(
            current_date,
            chroma_client,
            collections_to_process=collections_to_process,
        )
        results[day_key] = kg_id
        current_date += timedelta(days=1)
    
    return results


async def get_knowledge_graph_summary(
    date: datetime,
) -> Optional[str]:
    """Get a text summary of a knowledge graph for a specific day."""
    kg = await retrieve_knowledge_graph_for_day(date)
    if not kg:
        return None
    
    summary_parts = [
        f"Knowledge Graph for {_date_to_kg_key(date)}",
        f"Entities: {len(kg.get('entities', []))}",
        f"Relationships: {len(kg.get('relationships', []))}",
        f"Facts: {len(kg.get('facts', []))}",
        f"Insights: {len(kg.get('insights', []))}",
    ]
    
    if kg.get("entities"):
        summary_parts.append(f"\nKey Entities: {', '.join(kg['entities'][:10])}")
    
    return "\n".join(summary_parts)
