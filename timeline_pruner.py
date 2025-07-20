import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any, Dict, List

from openai import AsyncOpenAI

from config import config
import rag_chroma_manager as rcm

logger = logging.getLogger(__name__)

PRUNE_DAYS = getattr(config, "TIMELINE_PRUNE_DAYS", 30)


def _fetch_old_documents(prune_days: int) -> List[Dict[str, Any]]:
    if not rcm.chat_history_collection:
        logger.warning("Chat history collection unavailable")
        return []

    cutoff = datetime.now() - timedelta(days=prune_days)
    cutoff_iso = cutoff.isoformat()

    total = rcm.chat_history_collection.count()
    limit = 100
    old_docs: List[Dict[str, Any]] = []

    for offset in range(0, total, limit):
        res = rcm.chat_history_collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            ts_str = meta.get("timestamp") or meta.get("create_time")
            try:
                ts = datetime.fromisoformat(ts_str) if ts_str else None
            except Exception:
                ts = None
            if ts and ts.isoformat() <= cutoff_iso:
                old_docs.append({"id": doc_id, "document": docs[i], "metadata": meta, "timestamp": ts})
    return old_docs


# Removed _group_by_day as we are now grouping by 6-hour blocks directly in prune_and_summarize.

def _store_timeline_summary(start: datetime, end: datetime, summary: str, source_ids: List[str], block_key: str):
    if not rcm.timeline_summary_collection:
        logger.warning("Timeline summary collection unavailable")
        return
    from uuid import uuid4
    import json

    # Using block_key in the doc_id for better identification if needed
    doc_id = f"timeline_block_{block_key}_{uuid4().hex}"
    # Serialize source_ids so the metadata values are primitive types for ChromaDB.
    # To deserialize later, use json.loads(serialized_ids).
    serialized_ids = json.dumps(source_ids)
    metadata = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "source_ids": serialized_ids,
    }
    try:
        rcm.timeline_summary_collection.add(documents=[summary], metadatas=[metadata], ids=[doc_id])
        logger.info(f"Stored timeline summary {doc_id} spanning {metadata['start']} to {metadata['end']}")
    except Exception as e:
        logger.error(f"Failed to store timeline summary: {e}")


def _get_collection_timestamps(collection) -> List[datetime]:
    """Retrieve timestamp metadata from all documents in a collection."""
    if not collection:
        return []
    total = collection.count()
    if total == 0:
        return []

    limit = 100
    timestamps: List[datetime] = []
    for offset in range(0, total, limit):
        res = collection.get(limit=limit, offset=offset, include=["metadatas"])
        metas = res.get("metadatas", [])
        for meta in metas:
            ts_str = (meta or {}).get("timestamp") or (meta or {}).get("create_time")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                continue
            timestamps.append(ts)
    return timestamps


def print_collection_metrics() -> None:
    if not rcm.initialize_chromadb():
        logger.error("ChromaDB initialization failed")
        return

    # Ensure rcm (rag_chroma_manager) has its global variables populated if not already
    # This is usually done by initialize_chromadb(), which is called at the start of print_collection_metrics
    collections_to_check = [
        ("Chat History", rcm.chat_history_collection),
        ("Distilled Chat Summary", rcm.distilled_chat_summary_collection),
        ("News Summary", rcm.news_summary_collection),
        ("Timeline Summary", rcm.timeline_summary_collection),
        ("Entity", rcm.entity_collection),
        ("Relation", rcm.relation_collection),
        ("Observation", rcm.observation_collection),
    ]
    
    logger.info("--- ChromaDB Collection Metrics ---")

    for name, coll_instance in collections_to_check:  # Renamed 'coll' to 'coll_instance'
        if not coll_instance:
            logger.info(
                f"Collection '{name}' (variable: rcm.{name.lower().replace(' ', '_')}_collection) unavailable or not initialized."
            )
            continue

        count = coll_instance.count()
        timestamps = _get_collection_timestamps(coll_instance)

        if timestamps:
            earliest = min(timestamps)
            latest = max(timestamps)
            grouped = defaultdict(int)
            for ts in timestamps:
                grouped[ts.strftime("%Y-%m")] += 1
            group_str = ", ".join(f"{m}: {c}" for m, c in sorted(grouped.items()))
            logger.info(
                f"Collection '{name}' has {count} docs. Oldest: {earliest.date()}, latest: {latest.date()}"
            )
            logger.info(f"    Counts by month: {group_str}")
        else:
            logger.info(f"Collection '{name}' has {count} docs (no timestamp metadata found)")


async def prune_and_summarize(prune_days: int = PRUNE_DAYS):
    # The ChromaDB client is now expected to be initialized by main_bot.py.
    # We just check if the necessary collections are available.
    if not rcm.chat_history_collection or not rcm.timeline_summary_collection:
        logger.error("Pruner: One or more ChromaDB collections are not available. Aborting.")
        return

    logger.info(f"Starting timeline pruning for documents older than {prune_days} days.")
    llm_client = AsyncOpenAI(base_url=config.LOCAL_SERVER_URL, api_key=config.LLM_API_KEY or "lm-studio")

    try:
        docs = _fetch_old_documents(prune_days)
    except Exception as e:
        logger.error(f"Pruner: An error occurred while fetching old documents: {e}", exc_info=True)
        return

    if not docs:
        logger.info("Pruner: No documents found eligible for pruning.")
        return

    logger.info(f"Pruner: Found {len(docs)} documents to potentially prune and summarize.")
    # Sort all documents by timestamp once
    try:
        docs.sort(key=lambda x: x["timestamp"])
    except (KeyError, TypeError) as e:
        logger.error(f"Pruner: Could not sort documents due to missing or invalid timestamp data: {e}", exc_info=True)
        return

    current_block_items: List[Dict[str, Any]] = []
    current_block_start_time: Optional[datetime] = None

    for doc in docs:
        doc_ts = doc.get("timestamp")
        if not isinstance(doc_ts, datetime):
            logger.warning(f"Pruner: Skipping document with invalid or missing timestamp: ID {doc.get('id', 'N/A')}")
            continue

        if current_block_start_time is None:
            current_block_start_time = doc_ts
            current_block_items.append(doc)
        else:
            is_same_day = doc_ts.date() == current_block_start_time.date()
            block_index_start = current_block_start_time.hour // 6
            block_index_doc = doc_ts.hour // 6

            if not is_same_day or block_index_doc != block_index_start:
                if current_block_items:
                    logger.info(f"Pruner: Processing a block of {len(current_block_items)} documents.")
                    try:
                        block_start_dt = current_block_items[0]["timestamp"]
                        block_end_dt = current_block_items[-1]["timestamp"]
                        block_key_for_id = f"{block_start_dt.strftime('%Y%m%d%H%M%S')}_{block_end_dt.strftime('%Y%m%d%H%M%S')}"

                        texts = [item["document"] for item in current_block_items]
                        text_pairs = [(t, "timeline_block") for t in texts]
                        query = (
                            f"Summarize the included chat history that took place from "
                            f"{block_start_dt.strftime('%Y-%m-%d %H:%M:%S')} to "
                            f"{block_end_dt.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"as a keyword dense narrative focusing on content, date, novel learnings, etc. "
                            f"Do not state anything about or reference the word snippet, conversation, retrieved, etc. "
                            f"This is for a RAG db. Keep it clean, but keep all pertinant detail."
                        )
                        summary = await rcm.synthesize_retrieved_contexts_llm(
                            llm_client, text_pairs, query
                        )

                        if summary:
                            ids_to_prune = [item["id"] for item in current_block_items]
                            _store_timeline_summary(block_start_dt, block_end_dt, summary, ids_to_prune, block_key_for_id)

                            if rcm.chat_history_collection:
                                rcm.chat_history_collection.delete(ids=ids_to_prune)
                                logger.info(f"Pruner: Pruned {len(ids_to_prune)} documents for block {block_start_dt.strftime('%Y-%m-%d %H:%M')} - {block_end_dt.strftime('%Y-%m-%d %H:%M')}")
                                rcm.remove_full_conversation_references(ids_to_prune)
                            else:
                                logger.error("Pruner: chat_history_collection is None, cannot prune documents.")
                        else:
                            logger.warning(f"Pruner: No summary generated for block {block_start_dt.strftime('%Y-%m-%d %H:%M')} - {block_end_dt.strftime('%Y-%m-%d %H:%M')}; skipping deletion")
                    except Exception as e:
                        logger.error(f"Pruner: Failed to process a document block: {e}", exc_info=True)

                current_block_items = [doc]
                current_block_start_time = doc_ts
            else:
                current_block_items.append(doc)

    if current_block_items and current_block_start_time is not None:
        logger.info(f"Pruner: Processing the final block of {len(current_block_items)} documents.")
        try:
            block_start_dt = current_block_items[0]["timestamp"]
            block_end_dt = current_block_items[-1]["timestamp"]
            block_key_for_id = f"{block_start_dt.strftime('%Y%m%d%H%M%S')}_{block_end_dt.strftime('%Y%m%d%H%M%S')}"

            texts = [item["document"] for item in current_block_items]
            text_pairs = [(t, "timeline_block") for t in texts]
            query = (
                f"Summarize the included chat history that took place from "
                f"{block_start_dt.strftime('%Y-%m-%d %H:%M:%S')} to "
                f"{block_end_dt.strftime('%Y-%m-%d %H:%M:%S')} "
                f"as a keyword dense narrative focusing on content, date, novel learnings, etc. "
                f"Do not state anything about the reference the word snippet, conversation, retrieved, etc. "
                f"This is for a RAG db. Keep it clean."
            )
            summary = await rcm.synthesize_retrieved_contexts_llm(
                llm_client, text_pairs, query
            )

            if summary:
                ids_to_prune = [item["id"] for item in current_block_items]
                _store_timeline_summary(block_start_dt, block_end_dt, summary, ids_to_prune, block_key_for_id)
                if rcm.chat_history_collection:
                    rcm.chat_history_collection.delete(ids=ids_to_prune)
                    logger.info(f"Pruner: Pruned {len(ids_to_prune)} documents for final block {block_start_dt.strftime('%Y-%m-%d %H:%M')} - {block_end_dt.strftime('%Y-%m-%d %H:%M')}")
                    rcm.remove_full_conversation_references(ids_to_prune)
                else:
                    logger.error("Pruner: chat_history_collection is None, cannot prune documents for final block.")
            else:
                logger.warning(f"Pruner: No summary generated for final block {block_start_dt.strftime('%Y-%m-%d %H:%M')} - {block_end_dt.strftime('%Y-%m-%d %H:%M')}; skipping deletion")
        except Exception as e:
            logger.error(f"Pruner: Failed to process the final document block: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Summarize and prune old chat history")
    parser.add_argument("--days", type=int, default=PRUNE_DAYS, help="Prune items older than this many days")
    parser.add_argument("--stats", action="store_true", help="Print metrics about ChromaDB collections and exit")
    args = parser.parse_args()

    if args.stats:
        print_collection_metrics()
    else:
        asyncio.run(prune_and_summarize(args.days))
