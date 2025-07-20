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
    """
    Fetches all documents from the chat history collection and filters them by age in memory.
    This is a temporary workaround because ChromaDB's metadata filtering with ISO date strings is unreliable.
    """
    if not rcm.chat_history_collection:
        logger.warning("Chat history collection unavailable for fetching old documents.")
        return []

    cutoff_date = datetime.now() - timedelta(days=prune_days)
    # ChromaDB stores timestamps as ISO-8601 strings. Use an ISO string for the
    # comparison so we don't have to migrate existing data.
    cutoff_iso = cutoff_date.isoformat()

    logger.info(
        "Fetching documents with timestamps older than %s.", cutoff_iso
    )

    try:
        # Use the ISO formatted timestamp for the comparison. ChromaDB performs
        # lexicographical comparison for string fields, which works for
        # ISO-8601 timestamps.
        where_filter = {"timestamp": {"$lte": cutoff_iso}}

        # We get all results at once. If this dataset is enormous, pagination might be needed,
        # but it's still better than loading everything into memory.
        res = rcm.chat_history_collection.get(where=where_filter, include=["documents", "metadatas"])

        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])

        if not ids:
            logger.info("No documents found in the chat history collection.")
            return []

        logger.info(f"Fetched {len(ids)} total documents. Filtering in memory...")

        old_docs: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            doc_content = docs[i] if i < len(docs) else ""

            try:
                ts_val = meta.get("timestamp")
                if isinstance(ts_val, (int, float)):
                    ts = datetime.fromtimestamp(ts_val)
                elif isinstance(ts_val, str):
                    # Handle common ISO formats. If parsing fails, fall back to None.
                    ts = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
                else:
                    ts = None
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Could not convert timestamp '%s' for document ID %s. Error: %s. Skipping.",
                    ts_val,
                    doc_id,
                    e,
                )
                ts = None

            # The 'where' clause should prevent this, but as a safeguard:
            if ts and ts <= cutoff_date:
                old_docs.append({"id": doc_id, "document": doc_content, "metadata": meta, "timestamp": ts})

        logger.info(f"Found {len(old_docs)} documents meeting the prune criteria after in-memory filtering.")
        return old_docs

    except Exception as e:
        logger.error(f"An error occurred fetching or filtering documents from ChromaDB: {e}", exc_info=True)
        return []


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
        "start": start.timestamp(),
        "end": end.timestamp(),
        "source_ids": serialized_ids,
        "start_iso": start.isoformat(),
        "end_iso": end.isoformat(),
    }
    try:
        rcm.timeline_summary_collection.add(documents=[summary], metadatas=[metadata], ids=[doc_id])
        logger.info(f"Stored timeline summary {doc_id} spanning {metadata['start']} to {metadata['end']}")
    except Exception as e:
        logger.error(f"Failed to store timeline summary: {e}")


def _parse_timestamp(ts_val: Any) -> Optional[datetime]:
    """
    Parses a timestamp from various formats (ISO string, Unix timestamp) into a datetime object.
    """
    if ts_val is None:
        return None
    try:
        if isinstance(ts_val, (int, float)):
            return datetime.fromtimestamp(ts_val)
        elif isinstance(ts_val, str):
            # Handle ISO 8601 format, including 'Z' for UTC
            return datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        else:
            logger.warning(f"Unrecognized timestamp type: {type(ts_val)}")
            return None
    except (ValueError, TypeError) as e:
        logger.debug(f"Could not parse timestamp '{ts_val}': {e}")
        return None

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
            ts_val = (meta or {}).get("timestamp") or (meta or {}).get("create_time")
            ts = _parse_timestamp(ts_val)
            if ts:
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
    logger.info("Pruner task invoked.")
    try:
        # The ChromaDB client is now expected to be initialized by main_bot.py.
        # We just check if the necessary collections are available.
        if not rcm.chat_history_collection or not rcm.timeline_summary_collection:
            logger.error("Pruner: One or more ChromaDB collections are not available. Aborting.")
            return

        logger.info(f"Starting timeline pruning for documents older than {prune_days} days.")
        llm_client = AsyncOpenAI(base_url=config.LOCAL_SERVER_URL, api_key=config.LLM_API_KEY or "lm-studio")

        try:
            docs = _fetch_old_documents(prune_days)
            logger.info(f"Pruner: Fetched {len(docs)} documents from the database.")
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
            logger.debug("Pruner: Documents sorted by timestamp.")
        except (KeyError, TypeError) as e:
            logger.error(f"Pruner: Could not sort documents due to missing or invalid timestamp data: {e}", exc_info=True)
            return

        current_block_items: List[Dict[str, Any]] = []
        current_block_start_time: Optional[datetime] = None

        # Main processing loop for documents
        for i, doc in enumerate(docs):
            logger.debug(f"Pruner: Processing document {i+1}/{len(docs)} with ID {doc.get('id', 'N/A')}")
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

        # Process the final remaining block of documents
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

    except Exception as e:
        # This is a broad catch-all for any unexpected errors in the main function body
        logger.critical(f"Pruner: An unexpected critical error occurred in prune_and_summarize: {e}", exc_info=True)
    finally:
        # This will run whether the function succeeds or fails, helping to confirm if the function is exiting prematurely.
        logger.info("Pruner: prune_and_summarize function finished.")


def main():
    """Main function to run the pruner from the command line."""
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
        # This is a blocking call that runs the async function and manages the event loop.
        try:
            asyncio.run(prune_and_summarize(args.days))
        except KeyboardInterrupt:
            logger.info("Pruning process interrupted by user.")
        except Exception as e:
            logger.critical(f"An unhandled error occurred during pruning: {e}", exc_info=True)

if __name__ == "__main__":
    # This block is executed only when the script is run directly (e.g., `python timeline_pruner.py`)
    # It will not run when the script is imported by another module, such as discord_events.py.
    main()
