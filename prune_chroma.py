import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Any, List

from openai import AsyncOpenAI

from config import config
from rag_chroma_manager import (
    initialize_chromadb,
    chat_history_collection,
    distilled_chat_summary_collection,
    synthesize_retrieved_contexts_llm,
)

logger = logging.getLogger(__name__)

async def prune_chroma_history(llm_client: Any, dry_run: bool = False) -> None:
    if not initialize_chromadb() or not chat_history_collection or not distilled_chat_summary_collection:
        logger.error("ChromaDB collections not available. Aborting prune.")
        return

    cutoff = datetime.now() - timedelta(days=config.PRUNE_DAYS)
    batch_size = config.PRUNE_BATCH_SIZE

    while True:
        try:
            results = chat_history_collection.get(
                where={
                    "$or": [
                        {"timestamp": {"$lte": cutoff.isoformat()}},
                        {"create_time": {"$lte": cutoff.isoformat()}},
                    ]
                },
                limit=batch_size,
                include=["ids", "documents", "metadatas"],
            )
        except Exception as e_get:
            logger.error(f"Failed to fetch old documents: {e_get}", exc_info=True)
            break

        ids: List[str] = results.get("ids", []) if results else []
        if not ids:
            break

        docs: List[str] = results.get("documents", []) if results else []
        metadatas: List[dict] = results.get("metadatas", []) if results else []

        summary = None
        if docs:
            try:
                summary = await synthesize_retrieved_contexts_llm(
                    llm_client, docs, "Summarize pruned chat history"
                )
            except Exception as e_sum:
                logger.error(f"Failed to synthesize summary: {e_sum}", exc_info=True)

        if summary and not dry_run:
            times = []
            for md in metadatas:
                t = md.get("timestamp") or md.get("create_time")
                if t:
                    try:
                        times.append(datetime.fromisoformat(str(t)))
                    except ValueError:
                        pass
            if times:
                start_time = min(times)
                end_time = max(times)
            else:
                start_time = end_time = cutoff

            doc_id = f"pruned_{int(start_time.timestamp())}_{uuid4().hex}"
            metadata = {
                "source_doc_ids": ids,
                "summary_start": start_time.isoformat(),
                "summary_end": end_time.isoformat(),
                "type": "pruned_summary",
            }
            try:
                distilled_chat_summary_collection.add(
                    documents=[summary],
                    metadatas=[metadata],
                    ids=[doc_id],
                )
                logger.info(f"Inserted summary {doc_id} covering {len(ids)} docs.")
            except Exception as e_add:
                logger.error(f"Failed to add summary doc: {e_add}", exc_info=True)

        if not dry_run:
            try:
                chat_history_collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} old documents from chat_history_collection.")
            except Exception as e_del:
                logger.error(f"Failed to delete old docs: {e_del}", exc_info=True)

        if len(ids) < batch_size:
            break

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Prune old chat history from ChromaDB.")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify collections.")
    args = parser.parse_args()

    llm_client = AsyncOpenAI(
        base_url=config.LOCAL_SERVER_URL,
        api_key=config.LLM_API_KEY or "lm-studio",
    )
    await prune_chroma_history(llm_client, dry_run=args.dry_run)

if __name__ == "__main__":
    asyncio.run(main())
