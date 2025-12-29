"""Knowledge Graph Management System.

This module provides per-timeframe knowledge graph creation, deduplication,
merging, and maintenance capabilities. It consolidates raw database content
into structured knowledge graphs that improve retrieval efficiency.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import hashlib
import re

import chromadb
from chromadb.errors import InternalError

from config import config
from openai_api import create_chat_completion, extract_text
from llm_clients import get_llm_runtime
from logit_biases import LOGIT_BIAS_UNWANTED_TOKENS_STR

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class KGEntity:
    """Represents an entity in the knowledge graph."""
    name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    first_seen: str = ""
    last_seen: str = ""
    mention_count: int = 1
    source_doc_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGEntity":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KGRelation:
    """Represents a relation between entities."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    first_seen: str = ""
    last_seen: str = ""
    mention_count: int = 1
    source_doc_ids: List[str] = field(default_factory=list)
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGRelation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def signature(self) -> str:
        """Generate unique signature for deduplication."""
        return f"{self.subject.lower()}|{self.predicate.lower()}|{self.object.lower()}"


@dataclass
class KGObservation:
    """Represents an observation/fact in the knowledge graph."""
    statement: str
    observation_type: str
    entities_involved: List[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: str = ""
    source_doc_ids: List[str] = field(default_factory=list)
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGObservation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def content_hash(self) -> str:
        """Generate content hash for similarity detection."""
        normalized = re.sub(r'\s+', ' ', self.statement.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]


@dataclass
class KnowledgeGraphSnapshot:
    """A knowledge graph snapshot for a specific time period."""
    period_start: str
    period_end: str
    period_type: str  # 'daily', 'weekly', 'monthly', 'yearly'
    entities: Dict[str, KGEntity] = field(default_factory=dict)
    relations: List[KGRelation] = field(default_factory=list)
    observations: List[KGObservation] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = ""
    source_doc_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "period_type": self.period_type,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": [r.to_dict() for r in self.relations],
            "observations": [o.to_dict() for o in self.observations],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "source_doc_count": self.source_doc_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraphSnapshot":
        entities = {k: KGEntity.from_dict(v) for k, v in data.get("entities", {}).items()}
        relations = [KGRelation.from_dict(r) for r in data.get("relations", [])]
        observations = [KGObservation.from_dict(o) for o in data.get("observations", [])]
        return cls(
            period_start=data.get("period_start", ""),
            period_end=data.get("period_end", ""),
            period_type=data.get("period_type", "daily"),
            entities=entities,
            relations=relations,
            observations=observations,
            created_at=data.get("created_at", ""),
            last_updated=data.get("last_updated", ""),
            source_doc_count=data.get("source_doc_count", 0),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Knowledge Graph Manager
# ============================================================================

class KnowledgeGraphManager:
    """Manages knowledge graph creation, deduplication, merging, and retrieval."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        chroma_client: Optional[chromadb.ClientAPI] = None,
    ):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), "knowledge_graphs"
        )
        os.makedirs(self.storage_path, exist_ok=True)

        self._chroma_client = chroma_client
        self._kg_collection: Optional[chromadb.Collection] = None
        self._lock = asyncio.Lock()
        self._snapshots: Dict[str, KnowledgeGraphSnapshot] = {}

        # Load existing snapshots
        self._load_snapshots()

    def _load_snapshots(self) -> None:
        """Load existing knowledge graph snapshots from disk."""
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.storage_path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        snapshot = KnowledgeGraphSnapshot.from_dict(data)
                        key = f"{snapshot.period_type}_{snapshot.period_start}"
                        self._snapshots[key] = snapshot
            logger.info(f"Loaded {len(self._snapshots)} knowledge graph snapshots.")
        except Exception as e:
            logger.error(f"Error loading knowledge graph snapshots: {e}", exc_info=True)

    def _save_snapshot(self, snapshot: KnowledgeGraphSnapshot) -> None:
        """Save a knowledge graph snapshot to disk."""
        try:
            key = f"{snapshot.period_type}_{snapshot.period_start}"
            filename = f"kg_{key}.json"
            filepath = os.path.join(self.storage_path, filename)
            tmp_path = filepath + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, filepath)
            self._snapshots[key] = snapshot
            logger.info(f"Saved knowledge graph snapshot: {key}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph snapshot: {e}", exc_info=True)

    def _initialize_kg_collection(self) -> Optional[chromadb.Collection]:
        """Initialize the ChromaDB collection for knowledge graphs."""
        if self._kg_collection is not None:
            return self._kg_collection

        if self._chroma_client is None:
            # Try to get the global chroma client
            from rag_chroma_manager import chroma_client
            self._chroma_client = chroma_client

        if self._chroma_client is None:
            logger.warning("ChromaDB client not available for knowledge graph storage.")
            return None

        try:
            collection_name = os.getenv(
                "CHROMA_KG_COLLECTION_NAME", "knowledge_graph_snapshots"
            )
            self._kg_collection = self._chroma_client.get_or_create_collection(
                name=collection_name
            )
            logger.info(f"Initialized ChromaDB collection for knowledge graphs: {collection_name}")
            return self._kg_collection
        except Exception as e:
            logger.error(f"Failed to initialize KG collection: {e}", exc_info=True)
            return None

    # ========================================================================
    # Entity Extraction and Deduplication
    # ========================================================================

    async def extract_entities_from_text(
        self,
        text: str,
        source_doc_id: str,
    ) -> List[KGEntity]:
        """Extract entities from text using LLM."""
        if not text.strip():
            return []

        fast_runtime = get_llm_runtime("fast")
        fast_client = fast_runtime.client
        fast_provider = fast_runtime.provider
        fast_logit_bias = (
            LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
        )

        system_prompt = (
            "You are an expert entity extraction system. Extract all named entities "
            "from the text. Return a JSON array of objects with 'name', 'type', and "
            "'description' fields. Types include: PERSON, ORGANIZATION, LOCATION, "
            "CONCEPT, EVENT, PRODUCT, DATE, OTHER."
        )

        user_prompt = f"""Extract entities from this text:

{text[:30000]}

Return ONLY a JSON array like:
[{{"name": "Entity Name", "type": "ENTITY_TYPE", "description": "Brief description"}}]"""

        try:
            response_format_arg: Dict[str, Any] = {}
            if fast_provider.supports_json_mode:
                response_format_arg = {"response_format": {"type": "json_object"}}

            response = await create_chat_completion(
                fast_client,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=fast_provider.model,
                max_tokens=4096,
                temperature=fast_provider.temperature,
                logit_bias=fast_logit_bias,
                use_responses_api=fast_provider.use_responses_api,
                **response_format_arg,
            )

            content = extract_text(response, fast_provider.use_responses_api)
            if not content:
                return []

            # Parse JSON response
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Try to find array in response
            if content.startswith("{"):
                data = json.loads(content)
                entities_data = data.get("entities", [])
            else:
                entities_data = json.loads(content)

            if not isinstance(entities_data, list):
                return []

            entities = []
            now_iso = datetime.now(timezone.utc).isoformat()
            for item in entities_data:
                if isinstance(item, dict) and item.get("name"):
                    entity = KGEntity(
                        name=item["name"],
                        entity_type=item.get("type", "OTHER"),
                        description=item.get("description", ""),
                        first_seen=now_iso,
                        last_seen=now_iso,
                        source_doc_ids=[source_doc_id],
                    )
                    entities.append(entity)

            logger.debug(f"Extracted {len(entities)} entities from document {source_doc_id}")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            return []

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        return re.sub(r'\s+', ' ', name.lower().strip())

    def _are_entities_similar(self, e1: KGEntity, e2: KGEntity) -> bool:
        """Check if two entities are similar enough to merge."""
        norm1 = self._normalize_entity_name(e1.name)
        norm2 = self._normalize_entity_name(e2.name)

        # Exact match
        if norm1 == norm2:
            return True

        # One is substring of other (for abbreviations)
        if len(norm1) > 3 and len(norm2) > 3:
            if norm1 in norm2 or norm2 in norm1:
                return True

        # Check aliases
        for alias in e1.aliases:
            if self._normalize_entity_name(alias) == norm2:
                return True
        for alias in e2.aliases:
            if self._normalize_entity_name(alias) == norm1:
                return True

        return False

    def _merge_entities(self, existing: KGEntity, new: KGEntity) -> KGEntity:
        """Merge two entities into one."""
        # Use the longer/more descriptive name
        if len(new.name) > len(existing.name):
            existing.aliases.append(existing.name)
            existing.name = new.name
        elif new.name != existing.name:
            if new.name not in existing.aliases:
                existing.aliases.append(new.name)

        # Merge descriptions
        if new.description and not existing.description:
            existing.description = new.description
        elif new.description and existing.description:
            if len(new.description) > len(existing.description):
                existing.description = new.description

        # Update timestamps
        if new.first_seen and (not existing.first_seen or new.first_seen < existing.first_seen):
            existing.first_seen = new.first_seen
        if new.last_seen and (not existing.last_seen or new.last_seen > existing.last_seen):
            existing.last_seen = new.last_seen

        # Merge source docs
        for doc_id in new.source_doc_ids:
            if doc_id not in existing.source_doc_ids:
                existing.source_doc_ids.append(doc_id)

        existing.mention_count += new.mention_count

        return existing

    # ========================================================================
    # Relation Deduplication
    # ========================================================================

    def _are_relations_similar(self, r1: KGRelation, r2: KGRelation) -> bool:
        """Check if two relations are similar enough to merge."""
        return r1.signature() == r2.signature()

    def _merge_relations(self, existing: KGRelation, new: KGRelation) -> KGRelation:
        """Merge two relations into one."""
        # Update timestamps
        if new.first_seen and (not existing.first_seen or new.first_seen < existing.first_seen):
            existing.first_seen = new.first_seen
        if new.last_seen and (not existing.last_seen or new.last_seen > existing.last_seen):
            existing.last_seen = new.last_seen

        # Merge source docs
        for doc_id in new.source_doc_ids:
            if doc_id not in existing.source_doc_ids:
                existing.source_doc_ids.append(doc_id)

        existing.mention_count += new.mention_count

        # Average confidence
        existing.confidence = (existing.confidence + new.confidence) / 2

        # Keep longer context
        if len(new.context) > len(existing.context):
            existing.context = new.context

        return existing

    # ========================================================================
    # Observation Deduplication
    # ========================================================================

    async def _compute_observation_similarity(
        self, o1: KGObservation, o2: KGObservation
    ) -> float:
        """Compute semantic similarity between observations."""
        # Quick hash check first
        if o1.content_hash() == o2.content_hash():
            return 1.0

        # Simple word overlap for now (can be enhanced with embeddings)
        words1 = set(o1.statement.lower().split())
        words2 = set(o2.statement.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _merge_observations(
        self, existing: KGObservation, new: KGObservation
    ) -> KGObservation:
        """Merge two observations."""
        # Keep the more detailed statement
        if len(new.statement) > len(existing.statement):
            existing.statement = new.statement

        # Merge entities involved
        for entity in new.entities_involved:
            if entity not in existing.entities_involved:
                existing.entities_involved.append(entity)

        # Merge source docs
        for doc_id in new.source_doc_ids:
            if doc_id not in existing.source_doc_ids:
                existing.source_doc_ids.append(doc_id)

        # Average confidence
        existing.confidence = (existing.confidence + new.confidence) / 2

        return existing

    # ========================================================================
    # Knowledge Graph Building
    # ========================================================================

    async def build_daily_graph(
        self,
        date: datetime,
        force_rebuild: bool = False,
    ) -> KnowledgeGraphSnapshot:
        """Build or update a knowledge graph for a specific day."""
        date_str = date.strftime("%Y-%m-%d")
        key = f"daily_{date_str}"

        async with self._lock:
            # Check if we already have this snapshot
            if not force_rebuild and key in self._snapshots:
                logger.info(f"Knowledge graph for {date_str} already exists.")
                return self._snapshots[key]

            # Get start and end of day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)

            if start_of_day.tzinfo is None:
                start_of_day = start_of_day.replace(tzinfo=timezone.utc)
                end_of_day = end_of_day.replace(tzinfo=timezone.utc)

            snapshot = KnowledgeGraphSnapshot(
                period_start=start_of_day.isoformat(),
                period_end=end_of_day.isoformat(),
                period_type="daily",
            )

            # Fetch documents from collections for this date range
            await self._populate_snapshot_from_collections(
                snapshot, start_of_day, end_of_day
            )

            snapshot.last_updated = datetime.now(timezone.utc).isoformat()
            self._save_snapshot(snapshot)

            return snapshot

    async def _populate_snapshot_from_collections(
        self,
        snapshot: KnowledgeGraphSnapshot,
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        """Populate a snapshot with data from ChromaDB collections."""
        from rag_chroma_manager import (
            relation_collection,
            observation_collection,
            chat_history_collection,
            distilled_chat_summary_collection,
        )

        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()

        # Fetch relations
        if relation_collection:
            await self._fetch_relations_for_snapshot(
                snapshot, relation_collection, start_iso, end_iso
            )

        # Fetch observations
        if observation_collection:
            await self._fetch_observations_for_snapshot(
                snapshot, observation_collection, start_iso, end_iso
            )

        # Extract entities from distilled summaries
        if distilled_chat_summary_collection:
            await self._extract_entities_from_summaries(
                snapshot, distilled_chat_summary_collection, start_iso, end_iso
            )

    async def _fetch_relations_for_snapshot(
        self,
        snapshot: KnowledgeGraphSnapshot,
        collection: chromadb.Collection,
        start_iso: str,
        end_iso: str,
    ) -> None:
        """Fetch and deduplicate relations for a snapshot."""
        try:
            result = await asyncio.to_thread(
                collection.get,
                include=["documents", "metadatas"]
            )

            if not result or not result.get("documents"):
                return

            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            ids = result.get("ids", [])

            relation_map: Dict[str, KGRelation] = {}

            for doc, meta, doc_id in zip(docs, metas, ids):
                if not meta or not isinstance(meta, dict):
                    continue

                ts = meta.get("timestamp", "")
                if not ts or not (start_iso <= ts <= end_iso):
                    continue

                relation = KGRelation(
                    subject=meta.get("subject_name", ""),
                    predicate=meta.get("predicate", ""),
                    object=meta.get("object_name", ""),
                    context=meta.get("context_phrase", ""),
                    first_seen=ts,
                    last_seen=ts,
                    source_doc_ids=[meta.get("source_doc_id", doc_id)],
                )

                if not relation.subject or not relation.predicate or not relation.object:
                    continue

                sig = relation.signature()
                if sig in relation_map:
                    relation_map[sig] = self._merge_relations(relation_map[sig], relation)
                else:
                    relation_map[sig] = relation

                # Also extract entities from relations
                for entity_name in [relation.subject, relation.object]:
                    normalized = self._normalize_entity_name(entity_name)
                    if normalized not in snapshot.entities:
                        snapshot.entities[normalized] = KGEntity(
                            name=entity_name,
                            entity_type="EXTRACTED",
                            first_seen=ts,
                            last_seen=ts,
                            source_doc_ids=[doc_id],
                        )
                    else:
                        existing = snapshot.entities[normalized]
                        existing.mention_count += 1
                        if ts > existing.last_seen:
                            existing.last_seen = ts
                        if doc_id not in existing.source_doc_ids:
                            existing.source_doc_ids.append(doc_id)

            snapshot.relations = list(relation_map.values())
            snapshot.source_doc_count += len(docs)
            logger.info(f"Fetched {len(snapshot.relations)} relations for snapshot.")

        except Exception as e:
            logger.error(f"Error fetching relations for snapshot: {e}", exc_info=True)

    async def _fetch_observations_for_snapshot(
        self,
        snapshot: KnowledgeGraphSnapshot,
        collection: chromadb.Collection,
        start_iso: str,
        end_iso: str,
    ) -> None:
        """Fetch and deduplicate observations for a snapshot."""
        try:
            result = await asyncio.to_thread(
                collection.get,
                include=["documents", "metadatas"]
            )

            if not result or not result.get("documents"):
                return

            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            ids = result.get("ids", [])

            observation_map: Dict[str, KGObservation] = {}

            for doc, meta, doc_id in zip(docs, metas, ids):
                if not doc or not meta or not isinstance(meta, dict):
                    continue

                ts = meta.get("timestamp", "")
                if not ts or not (start_iso <= ts <= end_iso):
                    continue

                entities_str = meta.get("entities_involved", "[]")
                try:
                    entities = json.loads(entities_str) if entities_str else []
                except json.JSONDecodeError:
                    entities = []

                observation = KGObservation(
                    statement=doc,
                    observation_type=meta.get("observation_type", "GENERAL"),
                    entities_involved=entities if isinstance(entities, list) else [],
                    context=meta.get("context_phrase", ""),
                    timestamp=ts,
                    source_doc_ids=[meta.get("source_doc_id", doc_id)],
                )

                content_hash = observation.content_hash()
                if content_hash in observation_map:
                    observation_map[content_hash] = self._merge_observations(
                        observation_map[content_hash], observation
                    )
                else:
                    observation_map[content_hash] = observation

            snapshot.observations = list(observation_map.values())
            logger.info(f"Fetched {len(snapshot.observations)} observations for snapshot.")

        except Exception as e:
            logger.error(f"Error fetching observations for snapshot: {e}", exc_info=True)

    async def _extract_entities_from_summaries(
        self,
        snapshot: KnowledgeGraphSnapshot,
        collection: chromadb.Collection,
        start_iso: str,
        end_iso: str,
    ) -> None:
        """Extract entities from distilled summaries."""
        try:
            result = await asyncio.to_thread(
                collection.get,
                include=["documents", "metadatas"]
            )

            if not result or not result.get("documents"):
                return

            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            ids = result.get("ids", [])

            count = 0
            for doc, meta, doc_id in zip(docs, metas, ids):
                if not doc or not meta or not isinstance(meta, dict):
                    continue

                ts = meta.get("timestamp", "")
                if not ts or not (start_iso <= ts <= end_iso):
                    continue

                # Extract entities from this document
                entities = await self.extract_entities_from_text(doc, doc_id)

                for entity in entities:
                    normalized = self._normalize_entity_name(entity.name)

                    if normalized in snapshot.entities:
                        snapshot.entities[normalized] = self._merge_entities(
                            snapshot.entities[normalized], entity
                        )
                    else:
                        snapshot.entities[normalized] = entity

                count += 1
                if count % 10 == 0:
                    logger.debug(f"Processed {count} documents for entity extraction.")

            logger.info(
                f"Extracted entities from {count} summaries. "
                f"Total entities: {len(snapshot.entities)}"
            )

        except Exception as e:
            logger.error(f"Error extracting entities from summaries: {e}", exc_info=True)

    # ========================================================================
    # Retrieval from Knowledge Graphs
    # ========================================================================

    async def retrieve_relevant_knowledge(
        self,
        query: str,
        max_entities: int = 10,
        max_relations: int = 15,
        max_observations: int = 15,
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """Retrieve relevant knowledge for a query."""
        result = {
            "entities": [],
            "relations": [],
            "observations": [],
            "summary": "",
        }

        # Get recent snapshots
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        relevant_snapshots: List[KnowledgeGraphSnapshot] = []
        for key, snapshot in self._snapshots.items():
            try:
                period_end = datetime.fromisoformat(snapshot.period_end.replace("Z", "+00:00"))
                if period_end >= start_date:
                    relevant_snapshots.append(snapshot)
            except ValueError:
                continue

        if not relevant_snapshots:
            logger.info("No relevant knowledge graph snapshots found.")
            return result

        # Score entities, relations, and observations based on query relevance
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_entities: List[Tuple[float, KGEntity]] = []
        scored_relations: List[Tuple[float, KGRelation]] = []
        scored_observations: List[Tuple[float, KGObservation]] = []

        for snapshot in relevant_snapshots:
            # Score entities
            for entity in snapshot.entities.values():
                score = self._score_entity_relevance(entity, query_lower, query_words)
                if score > 0:
                    scored_entities.append((score, entity))

            # Score relations
            for relation in snapshot.relations:
                score = self._score_relation_relevance(relation, query_lower, query_words)
                if score > 0:
                    scored_relations.append((score, relation))

            # Score observations
            for observation in snapshot.observations:
                score = self._score_observation_relevance(
                    observation, query_lower, query_words
                )
                if score > 0:
                    scored_observations.append((score, observation))

        # Sort and limit results
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        scored_relations.sort(key=lambda x: x[0], reverse=True)
        scored_observations.sort(key=lambda x: x[0], reverse=True)

        result["entities"] = [e.to_dict() for _, e in scored_entities[:max_entities]]
        result["relations"] = [r.to_dict() for _, r in scored_relations[:max_relations]]
        result["observations"] = [
            o.to_dict() for _, o in scored_observations[:max_observations]
        ]

        # Generate summary if we have results
        if result["entities"] or result["relations"] or result["observations"]:
            result["summary"] = await self._generate_knowledge_summary(query, result)

        return result

    def _score_entity_relevance(
        self, entity: KGEntity, query_lower: str, query_words: Set[str]
    ) -> float:
        """Score entity relevance to query."""
        score = 0.0

        # Name match
        entity_name_lower = entity.name.lower()
        if entity_name_lower in query_lower:
            score += 3.0
        elif any(word in entity_name_lower for word in query_words):
            score += 1.5

        # Alias match
        for alias in entity.aliases:
            if alias.lower() in query_lower:
                score += 2.0
                break

        # Description match
        if entity.description:
            desc_lower = entity.description.lower()
            matches = sum(1 for word in query_words if word in desc_lower)
            score += matches * 0.3

        # Boost by mention count (popularity)
        score += min(entity.mention_count * 0.1, 1.0)

        return score

    def _score_relation_relevance(
        self, relation: KGRelation, query_lower: str, query_words: Set[str]
    ) -> float:
        """Score relation relevance to query."""
        score = 0.0

        # Subject/object match
        subject_lower = relation.subject.lower()
        object_lower = relation.object.lower()

        if subject_lower in query_lower or object_lower in query_lower:
            score += 2.5

        for word in query_words:
            if word in subject_lower or word in object_lower:
                score += 1.0

        # Predicate match
        if relation.predicate.lower() in query_lower:
            score += 1.5

        # Context match
        if relation.context:
            context_lower = relation.context.lower()
            matches = sum(1 for word in query_words if word in context_lower)
            score += matches * 0.2

        # Boost by mention count
        score += min(relation.mention_count * 0.1, 1.0)

        return score

    def _score_observation_relevance(
        self, observation: KGObservation, query_lower: str, query_words: Set[str]
    ) -> float:
        """Score observation relevance to query."""
        score = 0.0

        statement_lower = observation.statement.lower()

        # Direct query match
        if query_lower in statement_lower:
            score += 3.0

        # Word matches
        matches = sum(1 for word in query_words if word in statement_lower)
        score += matches * 0.5

        # Entity involvement match
        for entity in observation.entities_involved:
            if entity.lower() in query_lower:
                score += 1.5

        # Type-based boost
        if observation.observation_type in ("Fact", "Key_Insight"):
            score *= 1.2

        return score

    async def _generate_knowledge_summary(
        self, query: str, knowledge: Dict[str, Any]
    ) -> str:
        """Generate a summary of retrieved knowledge."""
        fast_runtime = get_llm_runtime("fast")
        fast_client = fast_runtime.client
        fast_provider = fast_runtime.provider
        fast_logit_bias = (
            LOGIT_BIAS_UNWANTED_TOKENS_STR if fast_provider.supports_logit_bias else None
        )

        # Build context from knowledge
        context_parts = []

        if knowledge["entities"]:
            entity_strs = [
                f"- {e['name']} ({e['entity_type']}): {e.get('description', 'No description')}"
                for e in knowledge["entities"][:5]
            ]
            context_parts.append("Key Entities:\n" + "\n".join(entity_strs))

        if knowledge["relations"]:
            relation_strs = [
                f"- {r['subject']} {r['predicate']} {r['object']}"
                for r in knowledge["relations"][:8]
            ]
            context_parts.append("Key Relations:\n" + "\n".join(relation_strs))

        if knowledge["observations"]:
            obs_strs = [f"- {o['statement']}" for o in knowledge["observations"][:8]]
            context_parts.append("Key Observations:\n" + "\n".join(obs_strs))

        if not context_parts:
            return ""

        context = "\n\n".join(context_parts)

        prompt = f"""Based on the following knowledge graph data, provide a brief summary relevant to this query: "{query}"

{context}

Provide a 2-4 sentence summary that synthesizes the most relevant information."""

        try:
            response = await create_chat_completion(
                fast_client,
                [
                    {
                        "role": "system",
                        "content": "You are a knowledge synthesis assistant. Provide concise, relevant summaries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=fast_provider.model,
                max_tokens=500,
                temperature=fast_provider.temperature,
                logit_bias=fast_logit_bias,
                use_responses_api=fast_provider.use_responses_api,
            )

            return extract_text(response, fast_provider.use_responses_api) or ""

        except Exception as e:
            logger.error(f"Failed to generate knowledge summary: {e}", exc_info=True)
            return ""

    # ========================================================================
    # Maintenance and Cleanup
    # ========================================================================

    async def merge_daily_to_weekly(self, week_start: datetime) -> KnowledgeGraphSnapshot:
        """Merge daily snapshots into a weekly snapshot."""
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        if week_start.tzinfo is None:
            week_start = week_start.replace(tzinfo=timezone.utc)
        week_end = week_start + timedelta(days=7) - timedelta(microseconds=1)

        key = f"weekly_{week_start.strftime('%Y-%m-%d')}"

        async with self._lock:
            weekly_snapshot = KnowledgeGraphSnapshot(
                period_start=week_start.isoformat(),
                period_end=week_end.isoformat(),
                period_type="weekly",
            )

            # Gather daily snapshots for this week
            for i in range(7):
                day = week_start + timedelta(days=i)
                day_key = f"daily_{day.strftime('%Y-%m-%d')}"
                if day_key in self._snapshots:
                    daily = self._snapshots[day_key]
                    self._merge_snapshot_into(weekly_snapshot, daily)

            weekly_snapshot.last_updated = datetime.now(timezone.utc).isoformat()
            self._save_snapshot(weekly_snapshot)

            return weekly_snapshot

    def _merge_snapshot_into(
        self,
        target: KnowledgeGraphSnapshot,
        source: KnowledgeGraphSnapshot,
    ) -> None:
        """Merge source snapshot data into target."""
        # Merge entities
        for key, entity in source.entities.items():
            if key in target.entities:
                target.entities[key] = self._merge_entities(target.entities[key], entity)
            else:
                target.entities[key] = entity

        # Merge relations
        existing_sigs = {r.signature(): i for i, r in enumerate(target.relations)}
        for relation in source.relations:
            sig = relation.signature()
            if sig in existing_sigs:
                idx = existing_sigs[sig]
                target.relations[idx] = self._merge_relations(
                    target.relations[idx], relation
                )
            else:
                target.relations.append(relation)
                existing_sigs[sig] = len(target.relations) - 1

        # Merge observations
        existing_hashes = {o.content_hash(): i for i, o in enumerate(target.observations)}
        for obs in source.observations:
            h = obs.content_hash()
            if h in existing_hashes:
                idx = existing_hashes[h]
                target.observations[idx] = self._merge_observations(
                    target.observations[idx], obs
                )
            else:
                target.observations.append(obs)
                existing_hashes[h] = len(target.observations) - 1

        target.source_doc_count += source.source_doc_count

    async def cleanup_old_snapshots(self, keep_days: int = 90) -> int:
        """Remove old daily snapshots that have been merged."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        removed = 0

        async with self._lock:
            keys_to_remove = []
            for key, snapshot in self._snapshots.items():
                if snapshot.period_type != "daily":
                    continue

                try:
                    period_end = datetime.fromisoformat(
                        snapshot.period_end.replace("Z", "+00:00")
                    )
                    if period_end < cutoff:
                        keys_to_remove.append(key)
                except ValueError:
                    continue

            for key in keys_to_remove:
                # Remove from memory
                del self._snapshots[key]

                # Remove file
                filename = f"kg_{key}.json"
                filepath = os.path.join(self.storage_path, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    removed += 1

        logger.info(f"Cleaned up {removed} old knowledge graph snapshots.")
        return removed

    # ========================================================================
    # Statistics and Reporting
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph system."""
        total_entities = 0
        total_relations = 0
        total_observations = 0
        snapshots_by_type: Dict[str, int] = defaultdict(int)

        for key, snapshot in self._snapshots.items():
            snapshots_by_type[snapshot.period_type] += 1
            total_entities += len(snapshot.entities)
            total_relations += len(snapshot.relations)
            total_observations += len(snapshot.observations)

        return {
            "total_snapshots": len(self._snapshots),
            "snapshots_by_type": dict(snapshots_by_type),
            "total_entities": total_entities,
            "total_relations": total_relations,
            "total_observations": total_observations,
            "storage_path": self.storage_path,
        }


# ============================================================================
# Global Instance
# ============================================================================

_kg_manager: Optional[KnowledgeGraphManager] = None


def get_kg_manager() -> KnowledgeGraphManager:
    """Get or create the global knowledge graph manager."""
    global _kg_manager
    if _kg_manager is None:
        _kg_manager = KnowledgeGraphManager()
    return _kg_manager


async def build_daily_knowledge_graph(
    date: Optional[datetime] = None,
    force_rebuild: bool = False,
) -> KnowledgeGraphSnapshot:
    """Build a daily knowledge graph."""
    manager = get_kg_manager()
    if date is None:
        date = datetime.now(timezone.utc)
    return await manager.build_daily_graph(date, force_rebuild)


async def retrieve_knowledge_for_query(
    query: str,
    days_back: int = 30,
) -> Dict[str, Any]:
    """Retrieve relevant knowledge for a query."""
    manager = get_kg_manager()
    return await manager.retrieve_relevant_knowledge(query, days_back=days_back)


async def maintain_knowledge_graphs() -> Dict[str, Any]:
    """Run maintenance tasks on knowledge graphs."""
    manager = get_kg_manager()

    # Build today's graph if not exists
    today = datetime.now(timezone.utc)
    await manager.build_daily_graph(today)

    # Build yesterday's graph if not exists
    yesterday = today - timedelta(days=1)
    await manager.build_daily_graph(yesterday)

    # Cleanup old snapshots
    removed = await manager.cleanup_old_snapshots()

    return {
        "status": "completed",
        "removed_snapshots": removed,
        "statistics": manager.get_statistics(),
    }
