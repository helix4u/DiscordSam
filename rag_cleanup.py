#!/usr/bin/env python3
import logging
from typing import Dict, List, Tuple

import rag_chroma_manager as rcm

logger = logging.getLogger(__name__)


def cleanup_distilled_entries(batch_size: int = 100, dry_run: bool = False) -> int:
    """Remove references to pruned conversations from distilled summaries.

    Instead of deleting the distilled documents entirely, this function clears
    the ``full_conversation_document_id`` metadata field when the referenced
    conversation no longer exists. This preserves the distilled sentences while
    ensuring stale references do not linger in the database.
    """
    if not rcm.initialize_chromadb():
        logger.error("ChromaDB initialization failed")
        return 0

    if not rcm.chat_history_collection or not rcm.distilled_chat_summary_collection:
        logger.error("Required ChromaDB collections are unavailable")
        return 0

    total = rcm.distilled_chat_summary_collection.count()
    removed = 0

    for offset in range(0, total, batch_size):
        res = rcm.distilled_chat_summary_collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        ids = res.get("ids", [])
        metas = res.get("metadatas", [])

        ids_to_check: Dict[str, Tuple[str, Dict]] = {}
        invalid_docs: Dict[str, Dict] = {}

        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            full_id = meta.get("full_conversation_document_id")
            if not full_id:
                logger.info(
                    "Distilled doc %s missing full_conversation_document_id",
                    doc_id,
                )
                invalid_docs[doc_id] = meta
            else:
                ids_to_check[doc_id] = (str(full_id), meta)

        if ids_to_check:
            try:
                existence_res = rcm.chat_history_collection.get(
                    ids=[val[0] for val in ids_to_check.values()],
                    include=[],
                )
                existing = set(existence_res.get("ids", []))
            except Exception as e_get:
                logger.error("Failed checking existence of chat history docs: %s", e_get)
                existing = set()
            for doc_id, (full_id, meta) in ids_to_check.items():
                if full_id not in existing:
                    logger.info(
                        "Distilled doc %s references missing conversation %s",
                        doc_id,
                        full_id,
                    )
                    invalid_docs[doc_id] = meta

        if invalid_docs:
            if dry_run:
                logger.info(
                    "Would clear references for %d distilled docs: %s",
                    len(invalid_docs),
                    list(invalid_docs.keys()),
                )
            else:
                for doc_id, meta in invalid_docs.items():
                    clean_meta = dict(meta)
                    clean_meta.pop("full_conversation_document_id", None)
                    try:
                        rcm.distilled_chat_summary_collection.update(
                            ids=doc_id,
                            metadatas=clean_meta,
                        )
                        removed += 1
                    except Exception as e_update:
                        logger.error(
                            "Failed updating metadata for distilled doc %s: %s",
                            doc_id,
                            e_update,
                        )

    return removed


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Clear references to missing chat history documents in distilled summaries."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List bad entries without deleting them",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to process per batch",
    )
    args = parser.parse_args()

    removed_count = cleanup_distilled_entries(
        batch_size=args.batch_size, dry_run=args.dry_run
    )

    if args.dry_run:
        logger.info("Dry run complete")
    else:
        logger.info("Cleared references from %d distilled entries", removed_count)

