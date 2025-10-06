import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import chromadb
    from config import config  # expects CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, optional SYSTEM_PROMPT_FILE
except ImportError as e:
    logger.error("Chromadb or config import failed: %s", e)
    chromadb = None
    config = None

@dataclass
class ConversationRecord:
    user: str
    analysis: str
    final: str
    developer: Optional[str] = None

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

def _split_think_blocks(text: str) -> tuple[str, str]:
    analyses = [m.strip() for m in _THINK_RE.findall(text or "")]
    analysis = "\n\n".join(analyses).strip()
    final = _THINK_RE.sub("", text or "").strip()
    return analysis, final

def _parse_message_log(doc_text: str) -> List[Dict[str, str]]:
    """
    Parse DiscordSam's saved conversation text block into a list of messages.
    Lines look like:
    user (name: foo): hello
    assistant (name: Sam): <think>..</think>hi
    Multiline messages continue until the next 'role (name: ...): ' line.
    """
    lines = doc_text.splitlines()
    try:
        start = lines.index('--- Message Log ---') + 1
    except ValueError:
        # fallback... treat whole file as messages region
        start = 0
    lines = lines[start:]

    messages: List[Dict[str, str]] = []
    cur: Optional[Dict[str, str]] = None

    for line in lines:
        if line.startswith("user (name: ") or line.startswith("assistant (name: "):
            if cur:
                messages.append(cur)
            role = "user" if line.startswith("user ") else "assistant"
            prefix = f"{role} (name: "
            rest = line[len(prefix):]
            sep = "): "
            pos = rest.find(sep)
            if pos == -1:
                # malformed... skip
                cur = None
                continue
            name = rest[:pos]
            content = rest[pos + len(sep):]
            cur = {"role": role, "name": name, "content": content}
        else:
            if cur:
                cur["content"] += "\n" + line
            else:
                # stray line... ignore
                continue

    if cur:
        messages.append(cur)

    return messages

def load_chroma_records(channel_id: Optional[str] = None, user_id: Optional[str] = None) -> List[ConversationRecord]:
    records: List[ConversationRecord] = []

    if chromadb is None or config is None:
        logger.error("ChromaDB or config unavailable")
        return records

    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    except Exception as e:
        logger.error("Failed to open ChromaDB: %s", e, exc_info=True)
        return records

    where: Dict[str, Any] = {"type": "full_conversation_log"}
    if channel_id:
        where["channel_id"] = str(channel_id)
    if user_id:
        where["user_id"] = str(user_id)

    try:
        res = collection.get(where=where, include=["documents", "metadatas"])
    except Exception as e:
        logger.error("Chroma get failed: %s", e, exc_info=True)
        return records

    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    pairs = list(zip(docs, metas))

    # try to sort by timestamp to keep order stable
    try:
        pairs.sort(key=lambda p: p[1].get("timestamp", ""))
    except Exception:
        pass

    # load developer/system prompt once
    developer_text: Optional[str] = None
    try:
        prompt_path = getattr(config, "SYSTEM_PROMPT_FILE", "system_prompt.md")
        with open(prompt_path, "r", encoding="utf-8") as f:
            developer_text = f.read().strip()
    except Exception:
        developer_text = None

    total_pairs = 0
    kept = 0

    for doc, meta in pairs:
        if not isinstance(doc, str) or not doc.strip():
            continue

        msgs = _parse_message_log(doc)

        # stitch developer with conversation timestamp if present
        conv_developer = developer_text
        ts = (meta or {}).get("timestamp")
        if developer_text and ts:
            try:
                dt = datetime.fromisoformat(ts)
                conv_developer = f"{developer_text}\nCurrent Date: {dt.strftime('%B %d %Y %H:%M:%S.%f')}"
            except Exception:
                pass

        # create user...assistant records
        for i, m in enumerate(msgs):
            if m.get("role") != "user":
                continue
            user_text = (m.get("content") or "").strip()
            if not user_text:
                continue

            # find the next assistant message
            asst = None
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                asst = msgs[i + 1]
            else:
                for j in range(i + 1, len(msgs)):
                    if msgs[j].get("role") == "assistant":
                        asst = msgs[j]
                        break
            if not asst:
                continue

            asst_text = (asst.get("content") or "").strip()
            total_pairs += 1

            analysis, final = _split_think_blocks(asst_text)

            # hard filter... keep only when analysis present
            if not analysis:
                continue

            kept += 1
            records.append(
                ConversationRecord(
                    user=user_text,
                    analysis=analysis,
                    final=final,
                    developer=conv_developer,
                )
            )

    logger.info("Candidate pairs: %d... kept with analysis: %d... dropped: %d",
                total_pairs, kept, total_pairs - kept)
    return records

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="Export Harmony-style JSONL from ChromaDB chat history")
    parser.add_argument("--channel", "-c", help="Filter by channel_id")
    parser.add_argument("--user", "-u", help="Filter by user_id")
    parser.add_argument("-o", "--output", help="Output JSONL file... defaults to stdout")
    args = parser.parse_args()

    recs = load_chroma_records(channel_id=args.channel, user_id=args.user)

    out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    if out_f is sys.stdout and out_f.encoding and out_f.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)  # type: ignore[attr-defined]
            out_f = sys.stdout

    for r in recs:
        json.dump(asdict(r), out_f, ensure_ascii=False)
        out_f.write("\n")

    if out_f is not sys.stdout:
        out_f.close()

    logger.info("Wrote %d records", len(recs))
