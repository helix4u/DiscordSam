import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationRecord:
    """Represents a single user/assistant exchange used for dataset creation."""
    user: str
    analysis: str
    final: str
    developer: Optional[str] = None


def _split_think_blocks(text: str) -> tuple[str, str]:
    """Split out `<think>` blocks from the provided text.

    Parameters
    ----------
    text: str
        The assistant response potentially containing `<think>` blocks.

    Returns
    -------
    tuple[str, str]
        A tuple of (analysis, final) strings. `analysis` contains the joined
        contents of all `<think>` blocks, while `final` is the text with those
        blocks removed.
    """
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    analyses = [m.strip() for m in pattern.findall(text)]
    analysis = "\n\n".join(analyses).strip()
    final = pattern.sub("", text).strip()
    return analysis, final


def _parse_chatgpt_export(conversation: Dict[str, Any]) -> List[ConversationRecord]:
    """Parse a single ChatGPT export conversation into records.

    The ChatGPT export format stores the conversation as a mapping of nodes.
    Each user node may have one or more children; the first child authored by
    the assistant is paired with the user message to form a record.

    Parameters
    ----------
    conversation: Dict[str, Any]
        A conversation object from ChatGPT's `conversations.json` export.

    Returns
    -------
    List[ConversationRecord]
        A list of extracted conversation records.
    """
    mapping = conversation.get("mapping")
    if not isinstance(mapping, dict):
        return []

    developer: Optional[str] = None
    records: List[ConversationRecord] = []

    # Look for a root system/developer message to carry forward
    for node in mapping.values():
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        author = message.get("author", {})
        role = author.get("role")
        if role in {"system", "developer"}:
            parts = message.get("content", {}).get("parts", [])
            if isinstance(parts, list):
                developer = "".join(str(p) for p in parts).strip() or None
            break

    for node in mapping.values():
        message = node.get("message")
        if not isinstance(message, dict):
            continue

        author = message.get("author", {})
        if author.get("role") != "user":
            continue

        user_parts = message.get("content", {}).get("parts", [])
        if not isinstance(user_parts, list):
            continue
        user_text = "".join(str(p) for p in user_parts).strip()
        if not user_text:
            continue

        child_ids = node.get("children", [])
        if not isinstance(child_ids, list):
            continue

        assistant_text = ""
        for child_id in child_ids:
            child = mapping.get(child_id)
            if not isinstance(child, dict):
                continue
            child_msg = child.get("message")
            if not isinstance(child_msg, dict):
                continue
            child_author = child_msg.get("author", {}).get("role")
            if child_author == "assistant":
                parts = child_msg.get("content", {}).get("parts", [])
                if isinstance(parts, list):
                    assistant_text = "".join(str(p) for p in parts).strip()
                break

        if not assistant_text:
            continue

        analysis, final = _split_think_blocks(assistant_text)
        records.append(
            ConversationRecord(
                user=user_text,
                analysis=analysis,
                final=final,
                developer=developer,
            )
        )

    return records


def load_records(path: str) -> List[ConversationRecord]:
    """Load conversation records from a JSON file.

    The file may contain either a list of pre-formatted records or ChatGPT
    export conversations (each with a `mapping` key). Items with a `mapping`
    field are processed via :func:`_parse_chatgpt_export`.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
        return []

    records: List[ConversationRecord] = []

    if isinstance(data, dict) and "mapping" in data:
        records.extend(_parse_chatgpt_export(data))
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "mapping" in item:
                records.extend(_parse_chatgpt_export(item))
                continue
            user = str(item.get("user", "")).strip()
            analysis = str(item.get("analysis", "")).strip()
            final = str(item.get("final", "")).strip()
            developer = item.get("developer")
            if developer is not None:
                developer = str(developer).strip() or None
            if user and (analysis or final):
                records.append(
                    ConversationRecord(
                        user=user, analysis=analysis, final=final, developer=developer
                    )
                )
    else:
        logger.warning("Unsupported data format in %s", path)

    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset from conversations")
    parser.add_argument("input", help="Path to the JSON file containing conversations")
    parser.add_argument(
        "-o",
        "--output",
        help="File to write JSONL records (defaults to stdout)",
    )
    args = parser.parse_args()

    loaded = load_records(args.input)

    if args.output:
        out_f = open(args.output, "w", encoding="utf-8")
    else:
        out_f = sys.stdout
        # Ensure stdout can handle Unicode characters even on Windows consoles.
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except AttributeError:
                sys.stdout = open(
                    sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1
                )  # type: ignore[attr-defined]
                out_f = sys.stdout

    for rec in loaded:
        json.dump(asdict(rec), out_f, ensure_ascii=False)
        out_f.write("\n")

    if out_f is not sys.stdout:
        out_f.close()