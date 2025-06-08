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


def _group_by_day(docs: List[Dict[str, Any]]):
    groups = defaultdict(list)
    for d in docs:
        day = d["timestamp"].strftime("%Y-%m-%d")
        groups[day].append(d)
    return groups


def _store_timeline_summary(start: datetime, end: datetime, summary: str, source_ids: List[str]):
    if not rcm.timeline_summary_collection:
        logger.warning("Timeline summary collection unavailable")
        return
    from uuid import uuid4

    doc_id = f"timeline_{int(start.timestamp())}_{int(end.timestamp())}_{uuid4().hex}"
    metadata = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "source_ids": source_ids,
    }
    try:
        rcm.timeline_summary_collection.add(documents=[summary], metadatas=[metadata], ids=[doc_id])
        logger.info(f"Stored timeline summary {doc_id} spanning {metadata['start']} to {metadata['end']}")
    except Exception as e:
        logger.error(f"Failed to store timeline summary: {e}")


def _get_collection_timestamps(collection) -> List[datetime]:
    """Retrieve timestamp metadata from all documents in a collection."""
    total = collection.count()
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

    collections = [
    ]

    for name, coll in collections:
        if not coll:
            logger.info(f"Collection '{name}' unavailable")
            continue

        count = coll.count()
        timestamps = _get_collection_timestamps(coll)

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
    if not rcm.initialize_chromadb():
        logger.error("ChromaDB initialization failed")
        return

    llm_client = AsyncOpenAI(base_url=config.LOCAL_SERVER_URL, api_key=config.LLM_API_KEY or "lm-studio")

    docs = _fetch_old_documents(prune_days)
    if not docs:
        logger.info("No documents eligible for pruning")
        return

    groups = _group_by_day(docs)
    for day, items in groups.items():
        items.sort(key=lambda x: x["timestamp"])
        texts = [i["document"] for i in items]
        start = items[0]["timestamp"]
        end = items[-1]["timestamp"]
        query = f"Summarize chat history from {start.date()} to {end.date()} as a narrative"
        summary = await rcm.synthesize_retrieved_contexts_llm(llm_client, texts, query)
        if summary:
            ids = [i["id"] for i in items]
            _store_timeline_summary(start, end, summary, ids)
            rcm.chat_history_collection.delete(ids=ids)
            logger.info(f"Pruned {len(ids)} documents from {day}")
        else:
            logger.warning(f"No summary generated for {day}; skipping deletion")


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
