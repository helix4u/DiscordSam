import logging
from uuid import uuid4
from datetime import datetime
from typing import List, Optional, Any, Dict, Union, Tuple
import json
import re
import random
import ast


import chromadb
from chromadb.errors import InternalError

from config import config
from common_models import MsgNode


logger = logging.getLogger(__name__)

chroma_client: Optional[chromadb.ClientAPI] = None
chat_history_collection: Optional[chromadb.Collection] = None
distilled_chat_summary_collection: Optional[chromadb.Collection] = None
news_summary_collection: Optional[chromadb.Collection] = None
rss_summary_collection: Optional[chromadb.Collection] = None # New collection for RSS summaries
timeline_summary_collection: Optional[chromadb.Collection] = None
entity_collection: Optional[chromadb.Collection] = None
relation_collection: Optional[chromadb.Collection] = None
observation_collection: Optional[chromadb.Collection] = None
tweets_collection: Optional[chromadb.Collection] = None # New collection for tweets


def _parse_json_with_recovery(content: str) -> Optional[Dict[str, Any]]:
    """Attempt to load JSON, trimming extraneous text and repairing minor issues."""

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("Initial JSON parse failed: %s; attempting recovery.", exc)

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None

    candidate = match.group(0)

    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    if open_braces > close_braces:
        candidate += "}" * (open_braces - close_braces)

    decoder = json.JSONDecoder()
    last_error: Optional[Exception] = None
    for i in range(len(candidate), 1, -1):
        try:
            return decoder.decode(candidate[:i])
        except Exception as exc2:  # json.JSONDecodeError or ValueError
            last_error = exc2

    try:
        # Fallback to Python literal parsing (handles single quotes, etc.)
        return ast.literal_eval(candidate)  # type: ignore[arg-type]
    except Exception as exc3:
        last_error = exc3

    logger.error("Trimmed JSON parse failed: %s", last_error)
    return None


def initialize_chromadb() -> bool:
    global chroma_client, chat_history_collection, distilled_chat_summary_collection, \
           news_summary_collection, rss_summary_collection, timeline_summary_collection, entity_collection, \
           relation_collection, observation_collection, tweets_collection # Added tweets_collection

    if chroma_client:
        logger.debug("ChromaDB already initialized.")
        return True
    try:
        logger.info(f"Initializing ChromaDB client with path: {config.CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_COLLECTION_NAME}")
        chat_history_collection = chroma_client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_DISTILLED_COLLECTION_NAME}")
        distilled_chat_summary_collection = chroma_client.get_or_create_collection(name=config.CHROMA_DISTILLED_COLLECTION_NAME)

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_NEWS_SUMMARY_COLLECTION_NAME}")
        news_summary_collection = chroma_client.get_or_create_collection(name=config.CHROMA_NEWS_SUMMARY_COLLECTION_NAME)

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_RSS_SUMMARY_COLLECTION_NAME}") # New
        rss_summary_collection = chroma_client.get_or_create_collection(name=config.CHROMA_RSS_SUMMARY_COLLECTION_NAME) # New

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME}")
        timeline_summary_collection = chroma_client.get_or_create_collection(name=config.CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME)

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_ENTITIES_COLLECTION_NAME}")
        entity_collection = chroma_client.get_or_create_collection(name=config.CHROMA_ENTITIES_COLLECTION_NAME)

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_RELATIONS_COLLECTION_NAME}")
        relation_collection = chroma_client.get_or_create_collection(name=config.CHROMA_RELATIONS_COLLECTION_NAME)

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_OBSERVATIONS_COLLECTION_NAME}")
        observation_collection = chroma_client.get_or_create_collection(name=config.CHROMA_OBSERVATIONS_COLLECTION_NAME)

        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_TWEETS_COLLECTION_NAME}") # New
        tweets_collection = chroma_client.get_or_create_collection(name=config.CHROMA_TWEETS_COLLECTION_NAME) # New

        logger.info(
            f"ChromaDB initialized successfully. Collections: "
            f"'{config.CHROMA_COLLECTION_NAME}', "
            f"'{config.CHROMA_DISTILLED_COLLECTION_NAME}', "
            f"'{config.CHROMA_NEWS_SUMMARY_COLLECTION_NAME}', "
            f"'{config.CHROMA_RSS_SUMMARY_COLLECTION_NAME}', "
            f"'{config.CHROMA_TIMELINE_SUMMARY_COLLECTION_NAME}', "
            f"'{config.CHROMA_ENTITIES_COLLECTION_NAME}', "
            f"'{config.CHROMA_RELATIONS_COLLECTION_NAME}', "
            f"'{config.CHROMA_OBSERVATIONS_COLLECTION_NAME}', "
            f"'{config.CHROMA_TWEETS_COLLECTION_NAME}'." # New
        )
        return True
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB collections: {e}", exc_info=True)
        chroma_client = None
        chat_history_collection = None
        distilled_chat_summary_collection = None
        news_summary_collection = None
        rss_summary_collection = None
        timeline_summary_collection = None
        entity_collection = None
        relation_collection = None
        observation_collection = None
        tweets_collection = None # New
        return False

async def extract_structured_data_llm(
    llm_client: Any,
    text_content: str,
    source_doc_id: str # To link extracted data back to its origin
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Uses an LLM to extract entities, relations, and observations from text_content.
    Links them using source_doc_id.
    """
    if not text_content.strip():
        logger.debug("extract_structured_data_llm: text_content is empty, skipping extraction.")
        return None

    max_input_length = 120000
    truncated_text = text_content[:max_input_length]

    system_prompt = (
        "You are an expert data extraction system. Your task is to analyze the provided text "
        "and extract entities, relations between those entities, and key observations or facts. "
        "Format your output as a single JSON object with three top-level keys: "
        "'entities', 'relations', and 'observations'."
    )

    user_prompt_template = """\
Please extract entities, relations, and observations from the following text.
Pay close attention to the most recent interactions (user query and assistant response),
typically found towards the end of the text, as they are of high importance for extraction.

TEXT TO ANALYZE:
---
{text_to_analyze}
---

OUTPUT JSON STRUCTURE:
Your output MUST be a single, valid JSON object.
{{
  "entities": [
    {{
      "name": "string (normalized name of the entity)",
      "type": "string (e.g., Person, Organization, Location, Concept, Technology, Product, Event, Other)",
      "text_span": "[integer, integer] (optional start and end character offset in the original text)"
    }}
  ],
  "relations": [
    {{
      "subject_name": "string (name of the subject entity, must match an entity in the 'entities' list)",
      "predicate": "string (verb phrase describing the relationship, e.g., 'is_a', 'works_for', 'located_in', 'uses', 'developed_by')",
      "object_name": "string (name of the object entity, must match an entity in the 'entities' list)",
      "context_phrase": "string (optional, the sentence or phrase from the text that supports this relation)"
    }}
  ],
  "observations": [
    {{
      "statement": "string (a concise statement of the observation or fact)",
      "type": "string (e.g., Fact, Event, User_Query_Topic, System_Action, Key_Insight)",
      "entities_involved": ["string", ...] (list of entity names involved in this observation, must match names in 'entities' list),
      "context_phrase": "string (optional, the sentence or phrase from the text that supports this observation)"
    }}
  ]
}}

Ensure all entity names in 'relations' and 'observations' correspond to entities defined in the 'entities' list.
If no specific data can be extracted for a category (e.g. no relations found), provide an empty list for that category.
Do not include any explanations or conversational text outside the JSON object.
"""

    user_prompt = user_prompt_template.format(text_to_analyze=truncated_text)

    try:
        logger.debug(f"Requesting structured data extraction from model {config.FAST_LLM_MODEL} for source_doc_id: {source_doc_id}.")

        response_format_arg = {}
        # Assuming LLM_SUPPORTS_JSON_MODE is a boolean attribute in your config
        if getattr(config, "LLM_SUPPORTS_JSON_MODE", False):
             response_format_arg = {"response_format": {"type": "json_object"}}

        response = await llm_client.chat.completions.create(
            model=config.FAST_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=8192, # Increased max_tokens for potentially larger JSON outputs
            temperature=0.2,
            stream=False,
            **response_format_arg
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            raw_content = response.choices[0].message.content.strip()

            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
                if raw_content.endswith("```"):
                    raw_content = raw_content[:-3]
                raw_content = raw_content.strip()

            extracted_data = _parse_json_with_recovery(raw_content)
            if extracted_data is None:
                logger.error(f"extract_structured_data_llm: Failed to decode JSON from LLM response for {source_doc_id}. Content: {raw_content[:500]}")
                return None
            if not isinstance(extracted_data, dict) or \
               not all(key in extracted_data for key in ["entities", "relations", "observations"]) or \
               not isinstance(extracted_data["entities"], list) or \
               not isinstance(extracted_data["relations"], list) or \
               not isinstance(extracted_data["observations"], list):
                logger.warning(f"extract_structured_data_llm: LLM response for {source_doc_id} was not the expected dict structure. Content: {raw_content[:500]}")
                return None

            logger.info(
                f"Successfully extracted structured data for {source_doc_id}: "
                f"{len(extracted_data['entities'])} entities, "
                f"{len(extracted_data['relations'])} relations, "
                f"{len(extracted_data['observations'])} observations."
            )
            return extracted_data
    except Exception as e:
        if "response_format" in str(e) and response_format_arg:
            logger.warning(f"extract_structured_data_llm: Failed with response_format, retrying without it for {source_doc_id}. Error: {e}")
            try:
                response = await llm_client.chat.completions.create(
                    model=config.FAST_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=2048,
                    temperature=0.2,
                    stream=False
                )
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    raw_content = response.choices[0].message.content.strip()
                    if raw_content.startswith("```json"):
                        raw_content = raw_content[7:]
                        if raw_content.endswith("```"):
                            raw_content = raw_content[:-3]
                        raw_content = raw_content.strip()
                    extracted_data = _parse_json_with_recovery(raw_content)
                    if extracted_data is None:
                        logger.error(f"extract_structured_data_llm (retry): Failed to decode JSON from LLM response for {source_doc_id}. Content: {raw_content[:500]}")
                        return None
                    if not isinstance(extracted_data, dict) or  \
                       not all(key in extracted_data for key in ["entities", "relations", "observations"]):
                        logger.warning(f"extract_structured_data_llm (retry): LLM response for {source_doc_id} was not the expected dict structure. Content: {raw_content[:500]}")
                        return None
                    logger.info(f"Successfully extracted structured data (on retry) for {source_doc_id}.")
                    return extracted_data
                else:
                    logger.warning(f"extract_structured_data_llm (retry): LLM returned no content for {source_doc_id}.")
                    return None

            except Exception as retry_e:
                logger.error(f"extract_structured_data_llm: Failed on retry for {source_doc_id}: {retry_e}", exc_info=True)
                return None
        else:
            logger.error(f"extract_structured_data_llm: Failed to extract data for {source_doc_id}: {e}", exc_info=True)
            return None

async def distill_conversation_to_sentence_llm(llm_client: Any, text_to_distill: str) -> Optional[str]:
    if not text_to_distill.strip():
        logger.debug("Distillation skipped: text_to_distill is empty.")
        return None

    truncated_text = text_to_distill[:30000]

    prompt = (
        "You are a text distillation expert. Read the following conversational exchange (User query and Assistant response) "
        "and summarize its absolute core essence into a few keyword-rich, data-dense sentences. These sentences "
        "will be used for semantic search to recall this specific exchange later. Focus on unique "
        "entities, key actions, insights, and primary topics directly discussed in this pair. The sentences should be concise and highly informative.\n\n"
        "CONVERSATIONAL EXCHANGE:\n---\n"
        f"{truncated_text}"
        "\n---\n\n"
        "DISTILLED SENTENCE(S) (focus on the exchange):"
    )
    try:
        logger.debug(f"Requesting distillation from model {config.FAST_LLM_MODEL} for focused exchange.")
        response = await llm_client.chat.completions.create(
            model=config.FAST_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert contextual knowledge distiller focusing on user-assistant turn pairs."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
            temperature=0.5,
            stream=False
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            distilled = response.choices[0].message.content.strip()
            logger.info(f"Distilled exchange to sentence(s): '{distilled[:100]}...'")
            return distilled
        logger.warning("LLM distillation (focused exchange) returned no content.")
        return None
    except Exception as e:
        logger.error(f"Failed to distill focused exchange: {e}", exc_info=True)
        return None

async def synthesize_retrieved_contexts_llm(llm_client: Any, retrieved_contexts: List[Tuple[str, str]], current_query: str) -> Optional[str]:
    if not retrieved_contexts:
        logger.debug("Context synthesis skipped: no retrieved_contexts provided.")
        return None

    formatted_snippets = ""
    for i, (text, source_name) in enumerate(retrieved_contexts):
        formatted_snippets += f"--- Memory {i+1} (from: {source_name}) ---\n{text[:120000]}\n\n"

    current_date_str = datetime.now().isoformat()

    prompt = (
        f"CURRENT DATE: {current_date_str}\n"
        "You are a pre-thought context synthesization mechanism. Below are several retrieved conversation memories that "
        "might be relevant to the user's current query. Your task is to read all of them and synthesize "
        "a single, detailed paragraph that captures the most relevant information from these memories "
        "as it pertains to the user's query. This synthesized paragraph will be used as pre-metacognition recall context "
        "for an artificial being named Sam. You are a mental mechanism of Sam's. "
        "Do not answer the user's query. Focus on extracting and combining relevant "
        "facts and discussion points from the memories.\n"
        "When you incorporate information from a specific memory, you can explicitly state the approximate date (e.g., 'On {Month Day, Year}...' or '{X days/weeks/months ago}...') "
        "derived from the timestamp data. "
        "This can provide crucial temporal and contextual grounding.\n"
        "If no memories are truly relevant, use them anyway. Create "
        "a generisized context that will assist Sam's other cognitive components in creating a better informed response for the user "
        "(unless doing timeline summaries, then ignore the above instruction and use the summary prompt). "
        "Be detailed and personal. Do not use <think> tags or metacognition for this.\n\n"
        f"USER'S CURRENT QUERY:\n---\n{current_query}\n---\n\n"
        f"RETRIEVED MEMORY FRAGMENTS:\n---\n{formatted_snippets}---\n\n"
        "SYNTHESIZED CONTEXT PARAGRAPH (3-8 sentences ideally. Do not use <think> tags or metacognition for this.):"
    )
    try:
        logger.debug(f"Requesting context synthesis from model {config.FAST_LLM_MODEL}.")
        response = await llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert context synthesizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3072,
            temperature=0.6,
            stream=False
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            synthesized_context = response.choices[0].message.content.strip()
            logger.info(f"Synthesized RAG context: '{synthesized_context[:150]}...'")
            return synthesized_context
        logger.warning("LLM context synthesis returned no content.")
        return None
    except Exception as e:
        logger.error(f"Failed to synthesize RAG context: {e}", exc_info=True)
        return None


async def merge_memory_snippet_with_summary_llm(
    llm_client: Any, old_memory: str, new_summary: str
) -> Optional[str]:
    """Merge an existing memory snippet with a summary of a new conversation."""
    if not old_memory.strip() or not new_summary.strip():
        logger.debug("merge_memory_snippet_with_summary_llm: Missing text to merge.")
        return None

    prompt = (
        "You maintain long-term memories for an AI assistant. "
        "Rewrite the existing memory with any new insights from the conversation summary. "
        "Return 3-8 concise sentences that preserve important older details and incorporate the new information." 
    )

    timestamp_match = re.search(r"Conversation recorded at:\s*(.+)", old_memory)
    old_date_str = timestamp_match.group(1).strip() if timestamp_match else "unknown"
    current_date_str = datetime.now().isoformat()

    user_text = (
        f"OLD MEMORY DATE: {old_date_str}\n"
        f"CURRENT DATE: {current_date_str}\n"
        f"OLD MEMORY:\n{old_memory}\n\nNEW CONVERSATION SUMMARY:\n{new_summary}\n\nUPDATED MEMORY:"
    )

    try:
        response = await llm_client.chat.completions.create(
            model=config.FAST_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a memory consolidation assistant."},
                {"role": "user", "content": user_text},
            ],
            max_tokens=2048,
            temperature=0.4,
            stream=False,
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        logger.warning("merge_memory_snippet_with_summary_llm: LLM returned no content.")
        return None
    except Exception as e:
        logger.error(f"merge_memory_snippet_with_summary_llm: Failed to merge memory: {e}", exc_info=True)
        return None


async def update_retrieved_memories(
    llm_client: Any, retrieved_snippets: Optional[List[Tuple[str, str]]], new_summary: str
) -> None:
    """Create updated memory snippets by merging retrieved ones with a new summary."""
    if not retrieved_snippets:
        return
    if not distilled_chat_summary_collection:
        logger.warning("update_retrieved_memories: distilled_chat_summary_collection unavailable.")
        return

    for snippet_text, source_name in retrieved_snippets:
        merged = await merge_memory_snippet_with_summary_llm(llm_client, snippet_text, new_summary)
        if not merged:
            continue
        doc_id = f"updated_{uuid4().hex}"
        metadata = {
            "source": source_name,
            "timestamp": datetime.now().isoformat(),
            "type": "merged_memory_update",
        }
        try:
            distilled_chat_summary_collection.add(documents=[merged], metadatas=[metadata], ids=[doc_id])
            logger.info(f"Stored merged memory update {doc_id} from source {source_name}.")
        except Exception as e_add:
            logger.error(f"Failed to store merged memory update {doc_id}: {e_add}", exc_info=True)

async def retrieve_and_prepare_rag_context(llm_client: Any, query: str, n_results_sentences: int = config.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH) -> Tuple[Optional[str], Optional[List[Tuple[str, str]]]]:
    if not chroma_client or not distilled_chat_summary_collection or not chat_history_collection:
        logger.warning("ChromaDB collections not available, skipping RAG context retrieval.")
        return None, None

    retrieved_contexts_raw: List[Tuple[str, str]] = [] # Stores tuples of (document_text, source_collection_name)

    try:
        logger.debug(f"RAG: Querying distilled_chat_summary_collection for query: '{query[:50]}...' (n_results={n_results_sentences})")
        results = distilled_chat_summary_collection.query(
            query_texts=[query] if isinstance(query, str) else query,
            n_results=n_results_sentences,
            include=["metadatas", "documents"]
        )

        retrieved_distilled_sentences_for_log: List[str] = []
        full_convo_ids_to_fetch = []

        if not results or not results.get('ids') or not results['ids'][0]:
            logger.info(f"RAG: No relevant distilled sentences found for query: '{str(query)[:50]}...'")
            # No return here, proceed to other collections
        else:
            query_result_ids = results['ids'][0]
            query_result_metadatas = results['metadatas'][0] if results['metadatas'] and results['metadatas'][0] else [{} for _ in query_result_ids]
            query_result_documents = results['documents'][0] if results['documents'] and results['documents'][0] else ["[Distilled sentence not found]" for _ in query_result_ids]

            for i in range(len(query_result_ids)):
                dist_metadata = query_result_metadatas[i]
                dist_sentence = query_result_documents[i]

                retrieved_distilled_sentences_for_log.append(dist_sentence or "[No sentence content]")

                full_convo_id = dist_metadata.get('full_conversation_document_id')
                if full_convo_id:
                    full_convo_ids_to_fetch.append(str(full_convo_id))
                else:
                    logger.warning(f"RAG: Distilled sentence (ID: {query_result_ids[i]}) missing 'full_conversation_document_id' in metadata: {dist_metadata}")

            if retrieved_distilled_sentences_for_log:
                log_sentences = "\n- ".join(retrieved_distilled_sentences_for_log)
                logger.info(f"RAG: Top {len(retrieved_distilled_sentences_for_log)} distilled sentences retrieved:\n- {log_sentences}")

        if not full_convo_ids_to_fetch:
            logger.info("RAG: No full conversation document IDs from distilled sentence metadata to fetch. Will proceed with other collections.")
            # Removed early return here: return None, None

        unique_full_convo_ids_to_fetch = list(set(full_convo_ids_to_fetch))
        if unique_full_convo_ids_to_fetch: # Check if there are IDs to fetch
            logger.debug(f"RAG: Fetching full conversation documents for IDs: {unique_full_convo_ids_to_fetch}")
            if chat_history_collection:
                try:
                    full_convo_docs_result = chat_history_collection.get(ids=unique_full_convo_ids_to_fetch, include=["documents"])
                    if full_convo_docs_result and full_convo_docs_result.get('documents'):
                        valid_docs = [doc for doc in full_convo_docs_result['documents'] if isinstance(doc, str)]
                        max_chars = getattr(config, 'RAG_MAX_FULL_CONVO_CHARS', 20000)
                        for doc_text in valid_docs:
                            truncated_doc = doc_text[-max_chars:] if len(doc_text) > max_chars else doc_text
                            if len(doc_text) > max_chars:
                                logger.debug(
                                    f"RAG: Truncated chat_history_via_distilled document from {len(doc_text)} to {max_chars} chars."
                                )
                            retrieved_contexts_raw.append((truncated_doc, "chat_history_via_distilled"))
                        logger.info(f"RAG: Retrieved {len(valid_docs)} full conversation texts via distilled sentences.")
                    else:
                        logger.warning(f"RAG: Could not retrieve some/all full conversation documents for IDs: {unique_full_convo_ids_to_fetch}. Result: {full_convo_docs_result}")
                except Exception as e_get_full:
                    logger.error(f"RAG: Error fetching full conversation docs for IDs {unique_full_convo_ids_to_fetch}: {e_get_full}", exc_info=True)
            else:
                logger.warning("RAG: chat_history_collection is None, cannot fetch full conversation documents.")

        n_results_collections = config.RAG_NUM_COLLECTION_DOCS_TO_FETCH
        additional_collections = [
            ("chat_history", chat_history_collection),
            ("timeline", timeline_summary_collection),
            ("news", news_summary_collection),
            ("rss", rss_summary_collection),
            ("tweets", tweets_collection), # New
            ("entity", entity_collection),
            ("relation", relation_collection),
            ("observation", observation_collection),
        ]

        for name, collection_obj in additional_collections:
            if not collection_obj:
                logger.debug(f"RAG: Collection '{name}' is not available, skipping.")
                continue
            try:
                logger.debug(
                    f"RAG: Querying {name} collection for query: '{str(query)[:50]}...' (n_results={n_results_collections})"
                )
                # Ensure query_texts is a list of strings
                query_texts_list = [query] if isinstance(query, str) else query

                res = collection_obj.query(
                    query_texts=query_texts_list,
                    n_results=n_results_collections,
                    include=["documents"]
                )

                # Check if 'documents' key exists and it's not None
                if res and res.get("documents") and res["documents"][0] is not None:
                    # Ensure that res["documents"][0] is a list of strings
                    docs_from_collection = [doc for doc in res["documents"][0] if isinstance(doc, str)]
                    if docs_from_collection:
                        max_chars = getattr(config, 'RAG_MAX_FULL_CONVO_CHARS', 20000)
                        for doc_text in docs_from_collection:
                            truncated_doc = doc_text
                            if name == 'chat_history' and len(doc_text) > max_chars:
                                truncated_doc = doc_text[-max_chars:]
                                logger.debug(
                                    f"RAG: Truncated chat_history document from {len(doc_text)} to {max_chars} chars."
                                )
                            retrieved_contexts_raw.append((truncated_doc, name))
                        logger.info(f"RAG: Retrieved {len(docs_from_collection)} documents from '{name}' collection.")
                    else:
                        logger.info(f"RAG: No documents found in '{name}' collection for the query.")
                else:
                    logger.info(f"RAG: No 'documents' found or documents list is empty/None in response from '{name}' collection for query: '{str(query)[:50]}...'")
            except InternalError as e_other: # Specific ChromaDB error for empty/missing on disk
                if "Cannot get collection" in str(e_other) or "doesn't exist" in str(e_other) or "Nothing found on disk" in str(e_other) :
                    logger.info(
                        f"RAG: '{name}' collection appears empty or not found on disk; skipping query. Error: {e_other}"
                    )
                    continue
                logger.error(
                    f"RAG: ChromaDB InternalError querying '{name}' collection: {e_other}",
                    exc_info=True, # Log stack trace for internal errors
                )
            except Exception as e_other: # Catch other potential errors during query
                logger.error(
                    f"RAG: Generic error querying '{name}' collection: {e_other}",
                    exc_info=True, # Log stack trace
                )

        if not retrieved_contexts_raw:
            logger.info("RAG: No context texts retrieved from any source for synthesis.")
            return None, None # Return None for both synthesized summary and raw snippets

        # Pass the list of tuples to the synthesis function
        synthesized_context = await synthesize_retrieved_contexts_llm(llm_client, retrieved_contexts_raw, query)

        # If synthesis fails but we have raw contexts, we should still return them.
        # The synthesized_context can be None if synthesis fails.
        return synthesized_context, retrieved_contexts_raw

    except Exception as e:
        logger.error(f"RAG: Failed during context retrieval/preparation: {e}", exc_info=True)
        return None, None # Return None for both in case of overarching errors

def _format_msg_content_for_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = [part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text" and "text" in part]
        if text_parts:
            return " ".join(text_parts)
        has_image = any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
        if has_image:
            return "[User sent an image with no accompanying text]" if not text_parts else "[User sent an image]"
    return "[Unsupported content format]"


async def ingest_conversation_to_chromadb(
    llm_client: Any,
    channel_id: int,
    user_id: Union[int, str],
    conversation_history_for_rag: List[MsgNode],
    retrieved_snippets: Optional[List[Tuple[str, str]]] = None
):
    # Check if essential collections are available
    if not all([chroma_client, chat_history_collection, distilled_chat_summary_collection,
                entity_collection, relation_collection, observation_collection]):
        logger.warning("One or more ChromaDB collections are not available. Skipping full ingestion pipeline.")
        return

    timestamp_now = datetime.now() # Define earlier
    str_user_id = str(user_id) # Define earlier

    can_do_focused_distillation = False
    if len(conversation_history_for_rag) >= 2:
        if conversation_history_for_rag[-2].role == 'user' and conversation_history_for_rag[-1].role == 'assistant':
            can_do_focused_distillation = True

    initiating_user_question = "N/A"
    if conversation_history_for_rag:
        first_user_msg = next((msg for msg in conversation_history_for_rag if msg.role == 'user'), None)
        if first_user_msg:
            initiating_user_question = _format_msg_content_for_text(first_user_msg.content)
        elif conversation_history_for_rag[0].role == 'user': # Fallback if no user role in later messages but first is user
            initiating_user_question = _format_msg_content_for_text(conversation_history_for_rag[0].content)


    header_text_parts = [
        f"Conversation recorded at: {timestamp_now.isoformat()}",
        f"Initiating User ID: {str_user_id}",
        f"Initiating User Question: {initiating_user_question}",
        "--- Message Log ---"
    ]

    full_conversation_text_parts = []
    for msg in conversation_history_for_rag:
        msg_name_str = str(msg.name) if msg.name else "N/A"
        formatted_content = _format_msg_content_for_text(msg.content)
        # Prepend timestamp to each message as well, for more granular context if needed later,
        # though the main request is about the conversation's start date.
        # For now, the main conversation timestamp is in the header.
        full_conversation_text_parts.append(f"{msg.role} (name: {msg_name_str}): {formatted_content}")

    original_full_text_for_storage = "\n".join(header_text_parts + full_conversation_text_parts)

    if not full_conversation_text_parts: # Check if there's any actual message content
        logger.info(f"Skipping ingestion of conversation with no actual messages. Channel: {channel_id}, User: {user_id}")
        return

    timestamp_now = datetime.now()
    str_user_id = str(user_id)
    full_convo_doc_id = f"full_channel_{channel_id}_user_{str_user_id}_{int(timestamp_now.timestamp())}_{uuid4().hex}"

    full_convo_metadata: Dict[str, Any] = {
        "channel_id": str(channel_id), "user_id": str_user_id, "timestamp": timestamp_now.isoformat(),
        "type": "full_conversation_log"
    }

    try:
        chat_history_collection.add(
            documents=[original_full_text_for_storage],
            metadatas=[full_convo_metadata],
            ids=[full_convo_doc_id]
        )
        logger.info(f"Ingested full conversation log (ID: {full_convo_doc_id}) into '{config.CHROMA_COLLECTION_NAME}'.")
    except Exception as e_add_full:
        logger.error(f"Failed to add full conversation log (ID: {full_convo_doc_id}) to ChromaDB: {e_add_full}", exc_info=True)
        return # If full log fails, probably stop

    # --- New: Structured Data Extraction and Storage ---
    extracted_data = await extract_structured_data_llm(llm_client, original_full_text_for_storage, full_convo_doc_id)

    if extracted_data:
        # Store Entities
        entities_to_add = extracted_data.get("entities", [])
        if entities_to_add and entity_collection:
            entity_docs, entity_metadatas, entity_ids = [], [], []
            for ent in entities_to_add:
                if not ent.get("name") or not ent.get("type"): # Basic validation
                    logger.warning(f"Skipping entity with missing name or type: {ent}")
                    continue
                entity_ids.append(f"entity_{full_convo_doc_id}_{uuid4().hex}")
                entity_docs.append(ent["name"]) # Document is the entity name
                meta = {
                    "source_doc_id": full_convo_doc_id,
                    "entity_type": ent["type"],
                    "timestamp": timestamp_now.isoformat(),
                    "raw_details": json.dumps(ent) # Store all other details from LLM
                }
                entity_metadatas.append(meta)
            if entity_ids:
                try:
                    entity_collection.add(documents=entity_docs, metadatas=entity_metadatas, ids=entity_ids)
                    logger.info(f"Ingested {len(entity_ids)} entities for conversation {full_convo_doc_id}.")
                except Exception as e_add_ent:
                    logger.error(f"Failed to add entities for {full_convo_doc_id} to ChromaDB: {e_add_ent}", exc_info=True)

        # Store Relations
        relations_to_add = extracted_data.get("relations", [])
        if relations_to_add and relation_collection:
            rel_docs, rel_metadatas, rel_ids = [], [], []
            for rel in relations_to_add:
                # Basic validation
                if not all(k in rel for k in ["subject_name", "predicate", "object_name"]):
                    logger.warning(f"Skipping relation with missing fields: {rel}")
                    continue
                rel_ids.append(f"relation_{full_convo_doc_id}_{uuid4().hex}")
                # Document could be a textual representation of the relation
                rel_docs.append(f"{rel['subject_name']} {rel['predicate']} {rel['object_name']}")
                meta = {
                    "source_doc_id": full_convo_doc_id,
                    "subject_name": rel["subject_name"],
                    "predicate": rel["predicate"],
                    "object_name": rel["object_name"],
                    "context_phrase": rel.get("context_phrase", ""),
                    "timestamp": timestamp_now.isoformat(),
                    "raw_details": json.dumps(rel)
                }
                rel_metadatas.append(meta)
            if rel_ids:
                try:
                    relation_collection.add(documents=rel_docs, metadatas=rel_metadatas, ids=rel_ids)
                    logger.info(f"Ingested {len(rel_ids)} relations for conversation {full_convo_doc_id}.")
                except Exception as e_add_rel:
                    logger.error(f"Failed to add relations for {full_convo_doc_id} to ChromaDB: {e_add_rel}", exc_info=True)

        # Store Observations
        observations_to_add = extracted_data.get("observations", [])
        if observations_to_add and observation_collection:
            obs_docs, obs_metadatas, obs_ids = [], [], []
            for obs in observations_to_add:
                if not obs.get("statement") or not obs.get("type"): # Basic validation
                    logger.warning(f"Skipping observation with missing statement or type: {obs}")
                    continue
                obs_ids.append(f"observation_{full_convo_doc_id}_{uuid4().hex}")
                obs_docs.append(obs["statement"]) # Document is the observation statement
                meta = {
                    "source_doc_id": full_convo_doc_id,
                    "observation_type": obs["type"],
                    "entities_involved": json.dumps(obs.get("entities_involved", [])), # Store as JSON string
                    "context_phrase": obs.get("context_phrase", ""),
                    "timestamp": timestamp_now.isoformat(),
                    "raw_details": json.dumps(obs)
                }
                obs_metadatas.append(meta)
            if obs_ids:
                try:
                    observation_collection.add(documents=obs_docs, metadatas=obs_metadatas, ids=obs_ids)
                    logger.info(f"Ingested {len(obs_ids)} observations for conversation {full_convo_doc_id}.")
                except Exception as e_add_obs:
                    logger.error(f"Failed to add observations for {full_convo_doc_id} to ChromaDB: {e_add_obs}", exc_info=True)
    else:
        logger.info(f"No structured data extracted for conversation {full_convo_doc_id}.")
    # --- End of New Structured Data Extraction ---


    # 2. Prepare text for focused distillation (last user query + assistant response)
    text_for_distillation = ""
    distillation_preview_text = ""

    if can_do_focused_distillation:
        last_user_msg_node = conversation_history_for_rag[-2]
        assistant_response_node = conversation_history_for_rag[-1]

        user_content_str = _format_msg_content_for_text(last_user_msg_node.content)
        assistant_content_str = _format_msg_content_for_text(assistant_response_node.content)

        user_name_str = str(last_user_msg_node.name) if last_user_msg_node.name else "User"
        assistant_name_str = str(assistant_response_node.name) if assistant_response_node.name else "Assistant"

        text_for_distillation = f"{user_name_str}: {user_content_str}\n{assistant_name_str}: {assistant_content_str}"
        distillation_preview_text = text_for_distillation[:200]
        logger.info(f"Preparing focused text for distillation (last user/assistant turn). Preview: {distillation_preview_text[:100]}...")
    else:
        logger.warning("Could not form a focused user-assistant pair from the end of conversation_history_for_rag. "
                       "Falling back to distilling the full conversation log for RAG summary.")
        text_for_distillation = original_full_text_for_storage
        distillation_preview_text = original_full_text_for_storage[:200]


    if not text_for_distillation.strip():
        logger.warning(f"Text for distillation is empty for convo ID {full_convo_doc_id}. Skipping distilled sentence storage.")
        return

    distilled_sentence = await distill_conversation_to_sentence_llm(llm_client, text_for_distillation)

    if not distilled_sentence or not distilled_sentence.strip():
        logger.warning(f"Distillation failed or returned empty for convo ID {full_convo_doc_id} (source text preview: {distillation_preview_text[:100]}...). Skipping distilled sentence storage.")
        return

    # Prefix the distilled summary with metadata headers so future merges retain the date
    snippet_text = (
        f"Conversation recorded at: {timestamp_now.isoformat()}\n"
        f"Initiating User Question: {initiating_user_question}\n"
        f"{distilled_sentence.strip()}"
    )

    distilled_doc_id = f"distilled_{full_convo_doc_id}"
    distilled_metadata: Dict[str, Any] = {
        "channel_id": str(channel_id),
        "user_id": str_user_id,
        "timestamp": timestamp_now.isoformat(),
        "full_conversation_document_id": full_convo_doc_id,
        "original_text_preview": distillation_preview_text
    }

    if distilled_chat_summary_collection:  # This check is good
        try:
            distilled_chat_summary_collection.add(
                documents=[snippet_text],
                metadatas=[distilled_metadata],
                ids=[distilled_doc_id]
            )
            logger.info(
                f"Ingested distilled sentence (ID: {distilled_doc_id}, linked to {full_convo_doc_id}) into '{config.CHROMA_DISTILLED_COLLECTION_NAME}'."
            )
            # Optionally merge retrieved snippets with the new summary for long-term consolidation
            if config.ENABLE_MEMORY_MERGE:
                await update_retrieved_memories(llm_client, retrieved_snippets, distilled_sentence)
        except Exception as e_add_distilled:
            logger.error(
                f"Failed to add distilled sentence (ID: {distilled_doc_id}) to ChromaDB: {e_add_distilled}",
                exc_info=True,
            )
    else:
        logger.error("distilled_chat_summary_collection is None, cannot add distilled document.")


# --- ChatGPT Export Processing (remains largely the same, but uses the updated distill_conversation_to_sentence_llm) ---
def parse_chatgpt_export(json_file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            conversations_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"ChatGPT export file not found: {json_file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in ChatGPT export file: {json_file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading ChatGPT export file {json_file_path}: {e}", exc_info=True)
        return []

    extracted_conversations: List[Dict[str, Any]] = []
    if not isinstance(conversations_data, list):
        logger.error(f"ChatGPT export file {json_file_path} does not contain a list of conversations at the top level.")
        return []

    for convo_idx, convo in enumerate(conversations_data):
        if not isinstance(convo, dict):
            logger.warning(f"Skipping item at index {convo_idx} in ChatGPT export: not a dictionary.")
            continue

        title = convo.get('title', f'Untitled_Export_{convo_idx}')
        create_time_ts = convo.get('create_time')
        create_time = datetime.fromtimestamp(create_time_ts) if isinstance(create_time_ts, (int, float)) else datetime.now()

        messages: List[Dict[str, str]] = []
        current_node_id = convo.get('current_node')
        mapping = convo.get('mapping', {})
        if not isinstance(mapping, dict):
            logger.warning(f"Skipping conversation '{title}': mapping is not a dictionary.")
            continue

        visited_node_ids = set()

        while current_node_id:
            if current_node_id in visited_node_ids:
                logger.error(f"Circular dependency detected in node mapping for conversation '{title}' at node ID '{current_node_id}'. Stopping parse for this conversation.")
                break
            visited_node_ids.add(current_node_id)

            node = mapping.get(current_node_id)
            if not isinstance(node, dict):
                logger.warning(f"Skipping node ID '{current_node_id}' in conversation '{title}': node is not a dictionary.")
                break

            message_data = node.get('message')
            if isinstance(message_data, dict) and \
               isinstance(message_data.get('content'), dict) and \
               message_data['content'].get('content_type') == 'text' and \
               isinstance(message_data.get('author'), dict):

                author_role = message_data['author'].get('role')
                text_parts = message_data['content'].get('parts', [])

                text_content = ""
                if isinstance(text_parts, list) and all(isinstance(p, str) for p in text_parts):
                    text_content = "".join(text_parts).strip()

                if text_content and author_role in ['user', 'assistant', 'system']:
                    messages.append({'role': author_role, 'content': text_content})

            current_node_id = node.get('parent')

        messages.reverse()
        if messages:
            extracted_conversations.append({'title': title, 'create_time': create_time, 'messages': messages})

    logger.info(f"Parsed {len(extracted_conversations)} valid conversations from ChatGPT export '{json_file_path}'.")
    return extracted_conversations

async def store_chatgpt_conversations_in_chromadb(llm_client: Any, conversations: List[Dict[str, Any]], source: str = "chatgpt_export") -> int:
    if not all([chroma_client, chat_history_collection, distilled_chat_summary_collection,
                entity_collection, relation_collection, observation_collection]):
        logger.error("ChromaDB collections not available for ChatGPT import. Skipping.")
        return 0

    added_count = 0
    if not conversations:
        logger.info("No conversations provided to store_chatgpt_conversations_in_chromadb.")
        return 0

    for i, convo_data in enumerate(conversations):
        # ... (rest of the function remains the same, but we'd also call extract_structured_data_llm here if desired for imported data)
        # For now, focusing on the main ingestion pipeline. This function might need similar logic if entities from imports are needed.
        # This is outside the scope of the current plan step which focuses on `ingest_conversation_to_chromadb`.
        # The following is the original logic for distilling and storing imported conversations.

        full_conversation_text_parts_import = [] # Renamed to avoid conflict
        last_user_content = ""
        last_assistant_content = ""
        temp_messages = convo_data.get('messages', [])
        if len(temp_messages) >= 2:
            if temp_messages[-2].get('role') == 'user' and temp_messages[-1].get('role') == 'assistant':
                last_user_content = str(temp_messages[-2].get('content', '')).strip()
                last_assistant_content = str(temp_messages[-1].get('content', '')).strip()

        text_for_distillation_import = ""
        if last_user_content and last_assistant_content:
            text_for_distillation_import = f"User: {last_user_content}\nAssistant: {last_assistant_content}"
        else:
            for msg in temp_messages:
                content_str = str(msg.get('content', '')).strip()
                if content_str:
                    full_conversation_text_parts_import.append(f"{msg.get('role', 'unknown_role')}: {content_str}")
            text_for_distillation_import = "\n".join(full_conversation_text_parts_import)

        if not text_for_distillation_import.strip():
            logger.debug(f"Skipping empty or non-distillable conversation from import: {convo_data.get('title', 'Untitled')}")
            continue

        full_exported_text_parts_for_storage = []
        for msg in temp_messages:
            content_str = str(msg.get('content', '')).strip()
            if content_str:
                full_exported_text_parts_for_storage.append(f"{msg.get('role', 'unknown_role')}: {content_str}")
        full_exported_text_to_store = "\n".join(full_exported_text_parts_for_storage)

        if not full_exported_text_to_store.strip():
            logger.debug(f"Skipping empty full conversation for storage from import: {convo_data.get('title', 'Untitled')}")
            continue

        timestamp_import = convo_data.get('create_time', datetime.now()) # Renamed
        safe_title = re.sub(r'\W+', '_', convo_data.get('title', 'untitled'))[:50]
        full_convo_doc_id_import = f"{source}_full_{safe_title}_{i}_{int(timestamp_import.timestamp())}_{uuid4().hex}" # Renamed

        full_convo_metadata_import: Dict[str, Any] = {  # Renamed
            "title": convo_data.get('title', 'Untitled'),
            "source": source,
            "create_time": timestamp_import.isoformat(),
            "type": "full_conversation_import"
        }
        try:
            if chat_history_collection:
                chat_history_collection.add(
                    documents=[full_exported_text_to_store],
                    metadatas=[full_convo_metadata_import],
                    ids=[full_convo_doc_id_import]
                )
                logger.info(f"Stored full ChatGPT import (ID: {full_convo_doc_id_import}) in '{config.CHROMA_COLLECTION_NAME}'.")
            else: # Should have been caught by the all() check above
                logger.error("chat_history_collection is None, cannot add imported document.")
                continue

            # Potentially call extract_structured_data_llm for imported conversations as well
            # extracted_data_import = await extract_structured_data_llm(llm_client, full_exported_text_to_store, full_convo_doc_id_import)
            # ... and then store entities, relations, observations from imported_data ...
            # This is an extension beyond the current immediate task.

            distilled_sentence = await distill_conversation_to_sentence_llm(llm_client, text_for_distillation_import)

            if not distilled_sentence or not distilled_sentence.strip():
                logger.warning(f"Distillation failed for imported convo: {convo_data.get('title', 'Untitled')} (ID: {full_convo_doc_id_import}). Skipping distilled sentence storage.")
                continue

            snippet_text_import = (
                f"Conversation recorded at: {timestamp_import.isoformat()}\n"
                f"{distilled_sentence.strip()}"
            )

            distilled_doc_id_import = f"distilled_{full_convo_doc_id_import}" # Renamed
            distilled_metadata_import: Dict[str, Any] = { # Renamed
                "title": convo_data.get('title', 'Untitled'),
                "source": source,
                "create_time": timestamp_import.isoformat(),
                "full_conversation_document_id": full_convo_doc_id_import,
                "original_text_preview": text_for_distillation_import[:200]
            }
            if distilled_chat_summary_collection:
                distilled_chat_summary_collection.add(
                    documents=[snippet_text_import],
                    metadatas=[distilled_metadata_import],
                    ids=[distilled_doc_id_import]
                )
                logger.info(
                    f"Stored distilled sentence for imported convo (ID: {distilled_doc_id_import}, linked to {full_convo_doc_id_import}) in '{config.CHROMA_DISTILLED_COLLECTION_NAME}'."
                )
                added_count += 1
            else: # Should have been caught by the all() check above
                logger.error("distilled_chat_summary_collection is None, cannot add distilled document for import.")

        except Exception as e_add:
            logger.error(f"Error processing/adding imported conversation '{convo_data.get('title', 'Untitled')}' (Full ID: {full_convo_doc_id_import}) to ChromaDB: {e_add}", exc_info=True)

    if added_count > 0:
        logger.info(f"Successfully processed and stored {added_count} imported conversations with distillations into ChromaDB.")
    elif conversations:
        logger.info("Processed ChatGPT export, but no conversations were successfully stored with distillation.")
    else:
        logger.info("No conversations found in the provided ChatGPT export to process.")

    return added_count


def store_news_summary(topic: str, url: str, summary_text: str, timestamp: Optional[datetime] = None) -> bool:
    if not chroma_client or not news_summary_collection: # Ensure news_summary_collection is checked
        logger.warning("ChromaDB news summary collection not available, skipping storage.")
        return False

    timestamp = timestamp or datetime.now()
    doc_id = f"news_{int(timestamp.timestamp())}_{random.randint(1000,9999)}"
    metadata = {
        "topic": topic,
        "url": url,
        "timestamp": timestamp.isoformat(),
    }

    try:
        news_summary_collection.add(
            documents=[summary_text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.info(
            f"Stored news summary (ID: {doc_id}) for topic '{topic}' and url '{url}' in '{config.CHROMA_NEWS_SUMMARY_COLLECTION_NAME}'."
        )
        return True
    except Exception as e:
        logger.error(f"Failed to store news summary for {url}: {e}", exc_info=True)
        return False

def store_rss_summary(feed_url: str, article_url: str, title: str, summary_text: str, timestamp: Optional[datetime] = None) -> bool:
    if not chroma_client or not rss_summary_collection:
        logger.warning("ChromaDB RSS summary collection not available, skipping storage.")
        return False

    timestamp = timestamp or datetime.now()
    # Create a unique ID that's still somewhat readable/relatable
    # Hashing parts of URLs can lead to very long IDs if not careful, so using timestamp and random int.
    # Using article_url to help ensure uniqueness if multiple feeds link to the same article around the same time.
    # A simple hash of article_url might be too long or complex for an ID.
    # Combining timestamp with a random int is a common pattern for unique enough IDs.
    url_hash_part = uuid4().hex[:8] # Short hash part from article_url for some uniqueness
    doc_id = f"rss_{int(timestamp.timestamp())}_{url_hash_part}_{random.randint(1000,9999)}"

    metadata = {
        "feed_url": feed_url,
        "article_url": article_url,
        "title": title,
        "timestamp": timestamp.isoformat(),
        "type": "rss_summary" # Explicitly type it
    }

    try:
        rss_summary_collection.add(
            documents=[summary_text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.info(
            f"Stored RSS summary (ID: {doc_id}) for article '{title}' from feed '{feed_url}' (article URL: '{article_url}') in '{config.CHROMA_RSS_SUMMARY_COLLECTION_NAME}'."
        )
        return True
    except Exception as e:
        logger.error(f"Failed to store RSS summary for article {article_url} from feed {feed_url}: {e}", exc_info=True)
        return False

def remove_full_conversation_references(pruned_doc_ids: List[str], batch_size: int = 100):
    """
    Finds distilled summaries linked to pruned full conversations and removes the linking metadata.
    """
    if not distilled_chat_summary_collection:
        logger.error("distilled_chat_summary_collection is unavailable for cleanup.")
        return

    if not pruned_doc_ids:
        logger.info("No pruned document IDs provided for reference cleanup.")
        return

    logger.info(f"Starting reference cleanup for {len(pruned_doc_ids)} pruned documents.")

    total = distilled_chat_summary_collection.count()
    updated_count = 0

    for offset in range(0, total, batch_size):
        try:
            res = distilled_chat_summary_collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )
            ids_to_update = []
            metadatas_to_update = []

            distilled_ids = res.get("ids", [])
            metadatas = res.get("metadatas", [])

            for i, meta in enumerate(metadatas):
                if not meta:
                    continue

                full_id = meta.get("full_conversation_document_id")
                if full_id and full_id in pruned_doc_ids:
                    logger.info(f"Found distilled doc {distilled_ids[i]} linked to pruned doc {full_id}. Removing reference.")
                    # Create a new metadata dict without the key
                    new_meta = {k: v for k, v in meta.items() if k != "full_conversation_document_id"}
                    ids_to_update.append(distilled_ids[i])
                    metadatas_to_update.append(new_meta)

            if ids_to_update:
                distilled_chat_summary_collection.update(
                    ids=ids_to_update,
                    metadatas=metadatas_to_update,
                )
                updated_count += len(ids_to_update)
                logger.info(f"Updated metadata for {len(ids_to_update)} distilled documents.")

        except Exception as e:
            logger.error(f"An error occurred during reference cleanup in batch starting at offset {offset}: {e}", exc_info=True)

    logger.info(f"Completed reference cleanup. Updated {updated_count} distilled document(s).")
