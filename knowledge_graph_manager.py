import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

import chromadb
from config import config
from llm_clients import get_llm_client, get_llm_provider
from openai_api import create_chat_completion, extract_text
from rag_chroma_manager import relation_collection, observation_collection, daily_kg_collection

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    async def consolidate_daily_graph(self, target_date: Optional[datetime] = None):
        """
        Consolidate relations and observations for a specific day into a single Knowledge Graph.
        """
        if not target_date:
            target_date = datetime.now() - timedelta(days=1) # Default to yesterday

        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        logger.info(f"Consolidating KG for {start_of_day.date()}...")

        # 1. Fetch raw relations and observations
        raw_data = await self._fetch_raw_data(start_of_day, end_of_day)
        if not raw_data:
            logger.info("No data found for this day.")
            return

        # 2. Use LLM to deduplicate and merge
        consolidated_kg = await self._merge_with_llm(raw_data)
        if not consolidated_kg:
            logger.warning("Failed to consolidate KG.")
            return

        # 3. Store in daily_kg_collection
        await self._store_daily_kg(consolidated_kg, start_of_day)

    async def _fetch_raw_data(self, start_dt: datetime, end_dt: datetime) -> str:
        # Fetch from Chroma (this is tricky because Chroma doesn't support complex date filtering in `get` easily without metadata)
        # We rely on metadata timestamp.
        # Efficient way: fetch all IDs for the collection, filter by metadata timestamp in python. 
        # For production with millions of rows this is bad, but for a bot it's okay.
        
        combined_text = []
        
        for name, coll in [("relations", relation_collection), ("observations", observation_collection)]:
            if not coll: continue
            
            # Fetch generic batch
            try:
                # Assuming reasonable size. If huge, need pagination.
                result = await asyncio.to_thread(coll.get, include=["documents", "metadatas"])
                if result and result.get('documents') and result.get('metadatas'):
                    for doc, meta in zip(result['documents'], result['metadatas']):
                        if not meta: continue
                        ts_str = meta.get('timestamp')
                        if ts_str:
                            try:
                                ts = datetime.fromisoformat(ts_str)
                                if ts.tzinfo: ts = ts.replace(tzinfo=None) # naive comparison
                                if start_dt <= ts < end_dt:
                                    combined_text.append(f"[{name.upper()}] {doc}")
                            except: pass
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")

        return "\n".join(combined_text)

    async def _merge_with_llm(self, raw_text: str) -> Optional[str]:
        # Split if too long
        if len(raw_text) > 50000:
             raw_text = raw_text[:50000] + "\n[Truncated]"

        prompt = (
            "You are a Knowledge Graph Maintainer. Your task is to consolidate the following raw set of "
            "relations and observations extracted from a day's conversation into a clean, deduplicated, "
            "and merged Knowledge Graph Summary.\n\n"
            "Rules:\n"
            "1. Merge duplicate entities (e.g. 'Sam' and 'Sam the bot' -> 'Sam').\n"
            "2. Consolidate conflicting or redundant observations.\n"
            "3. Output a coherent narrative or a structured list of key facts and relations that represents "
            "the 'Daily Knowledge Update'.\n\n"
            "RAW DATA:\n"
            f"{raw_text}\n\n"
            "CONSOLIDATED DAILY KNOWLEDGE GRAPH:"
        )
        
        client = get_llm_client("fast") # Use fast model
        provider = get_llm_provider("fast")
        
        try:
            response = await create_chat_completion(
                client,
                [{"role": "user", "content": prompt}],
                model=provider.model,
                temperature=0.3
            )
            return extract_text(response, provider.use_responses_api)
        except Exception as e:
            logger.error(f"KG Merge failed: {e}")
            return None

    async def _store_daily_kg(self, content: str, date: datetime):
        if not daily_kg_collection: return
        
        doc_id = f"daily_kg_{date.date().isoformat()}"
        metadata = {"date": date.date().isoformat(), "timestamp": datetime.now().isoformat(), "type": "daily_kg"}
        
        try:
            await asyncio.to_thread(
                daily_kg_collection.upsert,
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.info(f"Stored Daily KG for {date.date()}")
        except Exception as e:
            logger.error(f"Failed to store Daily KG: {e}")

kg_manager = KnowledgeGraphManager()
