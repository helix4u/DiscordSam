import logging
from uuid import uuid4
from datetime import datetime
from typing import List, Optional, Any, Dict, Union
import json
import re


import chromadb 

from config import config
from common_models import MsgNode 


logger = logging.getLogger(__name__)

chroma_client: Optional[chromadb.ClientAPI] = None
chat_history_collection: Optional[chromadb.Collection] = None
distilled_chat_summary_collection: Optional[chromadb.Collection] = None
news_summary_collection: Optional[chromadb.Collection] = None

def initialize_chromadb() -> bool:
    global chroma_client, chat_history_collection, distilled_chat_summary_collection, news_summary_collection
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

        logger.info(
            f"ChromaDB initialized successfully. Main Collection: '{config.CHROMA_COLLECTION_NAME}', "
            f"Distilled Collection: '{config.CHROMA_DISTILLED_COLLECTION_NAME}', "
            f"News Collection: '{config.CHROMA_NEWS_SUMMARY_COLLECTION_NAME}'"
        )
        return True
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB collections: {e}", exc_info=True)
        chroma_client = None
        chat_history_collection = None
        distilled_chat_summary_collection = None
        news_summary_collection = None
        return False

async def distill_conversation_to_sentence_llm(llm_client: Any, text_to_distill: str) -> Optional[str]:
    if not text_to_distill.strip():
        logger.debug("Distillation skipped: text_to_distill is empty.")
        return None
    
    truncated_text = text_to_distill[:3000] 

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
            max_tokens=300, # Adjusted for potentially shorter input
            temperature=0.5, # Slightly lower for more focused distillation
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

async def synthesize_retrieved_contexts_llm(llm_client: Any, retrieved_full_texts: List[str], current_query: str) -> Optional[str]:
    if not retrieved_full_texts:
        logger.debug("Context synthesis skipped: no retrieved_full_texts provided.")
        return None

    formatted_snippets = ""
    for i, text in enumerate(retrieved_full_texts):
        formatted_snippets += f"--- Snippet {i+1} ---\n{text[:1500]}\n\n" 

    prompt = (
        "You are a master context synthesizer. Below are several retrieved conversation snippets that "
        "might be relevant to the user's current query. Your task is to read all of them and synthesize "
        "a single, concise paragraph that captures the most relevant information from these snippets "
        "as it pertains to the user's query. This synthesized paragraph will be used to give an AI "
        "assistant context. Do not answer the user's query. Focus on extracting and combining relevant "
        "facts and discussion points from the snippets. If no snippets are truly relevant, state that "
        "no specific relevant context was found in past conversations. Be objective.\n\n" # Removed "Use the conversation history" as it's not directly provided here
        f"USER'S CURRENT QUERY:\n---\n{current_query}\n---\n\n"
        f"RETRIEVED SNIPPETS:\n---\n{formatted_snippets}---\n\n"
        "SYNTHESIZED CONTEXT PARAGRAPH (2-5 sentences ideally):"
    )
    try:
        logger.debug(f"Requesting context synthesis from model {config.FAST_LLM_MODEL}.")
        response = await llm_client.chat.completions.create(
            model=config.FAST_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert context synthesizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300, 
            temperature=0.5, 
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

async def retrieve_and_prepare_rag_context(llm_client: Any, query: str, n_results_sentences: int = config.RAG_NUM_DISTILLED_SENTENCES_TO_FETCH) -> Optional[str]:
    if not chroma_client or not distilled_chat_summary_collection or not chat_history_collection:
        logger.warning("ChromaDB collections not available, skipping RAG context retrieval.")
        return None
    
    try:
        logger.debug(f"RAG: Querying distilled_chat_summary_collection for query: '{query[:50]}...' (n_results={n_results_sentences})")
        results = distilled_chat_summary_collection.query(
            query_texts=[query] if isinstance(query, str) else query, # type: ignore
            n_results=n_results_sentences,
            include=["metadatas", "documents"] 
        )
        
        if not results or not results.get('ids') or not results['ids'][0]: 
            logger.info(f"RAG: No relevant distilled sentences found for query: '{str(query)[:50]}...'")
            return None

        retrieved_full_conversation_texts: List[str] = []
        retrieved_distilled_sentences_for_log: List[str] = []
        full_convo_ids_to_fetch = []

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
            logger.info("RAG: No full conversation document IDs from distilled sentence metadata to fetch.")
            return None

        unique_full_convo_ids_to_fetch = list(set(full_convo_ids_to_fetch))
        logger.debug(f"RAG: Fetching full conversation documents for IDs: {unique_full_convo_ids_to_fetch}")
        if unique_full_convo_ids_to_fetch and chat_history_collection:
            try:
                full_convo_docs_result = chat_history_collection.get(ids=unique_full_convo_ids_to_fetch, include=["documents"])
                if full_convo_docs_result and full_convo_docs_result.get('documents'):
                    valid_docs = [doc for doc in full_convo_docs_result['documents'] if isinstance(doc, str)]
                    retrieved_full_conversation_texts.extend(valid_docs)
                    logger.info(f"RAG: Retrieved {len(valid_docs)} full conversation texts.")
                else:
                    logger.warning(f"RAG: Could not retrieve some/all full conversation documents for IDs: {unique_full_convo_ids_to_fetch}. Result: {full_convo_docs_result}")
            except Exception as e_get_full:
                logger.error(f"RAG: Error fetching full conversation docs for IDs {unique_full_convo_ids_to_fetch}: {e_get_full}", exc_info=True)

        if not retrieved_full_conversation_texts:
            logger.info("RAG: No full conversation texts could be retrieved for synthesis.")
            return None
            
        synthesized_context = await synthesize_retrieved_contexts_llm(llm_client, retrieved_full_conversation_texts, query)
        return synthesized_context

    except Exception as e:
        logger.error(f"RAG: Failed during context retrieval/preparation: {e}", exc_info=True)
        return None

def _format_msg_content_for_text(content: Any) -> str:
    """Helper to convert MsgNode content to a simple string for distillation/logging."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = [part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text" and "text" in part]
        if text_parts:
            return " ".join(text_parts)
        # Check if there are image parts to mention them
        has_image = any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
        if has_image:
            return "[User sent an image with no accompanying text]" if not text_parts else "[User sent an image]"
    return "[Unsupported content format]"


async def ingest_conversation_to_chromadb(
    llm_client: Any, 
    channel_id: int, 
    user_id: Union[int, str], 
    conversation_history_for_rag: List[MsgNode] # This is final_prompt (including last user message) + assistant_response
):
    if not chroma_client or not chat_history_collection or not distilled_chat_summary_collection:
        logger.warning("ChromaDB collections not available, skipping ingestion.")
        return

    # Ensure there are enough messages to form a user-assistant pair for focused distillation
    # and that the last two messages are indeed user and assistant.
    can_do_focused_distillation = False
    if len(conversation_history_for_rag) >= 2:
        if conversation_history_for_rag[-2].role == 'user' and conversation_history_for_rag[-1].role == 'assistant':
            can_do_focused_distillation = True

    # 1. Prepare and store the "full" conversation record (includes more context than just the last turn)
    # This text includes system prompts, RAG context, history, the last user query, and the assistant's response.
    full_conversation_text_parts = []
    for msg in conversation_history_for_rag: 
        msg_name_str = str(msg.name) if msg.name else "N/A"
        formatted_content = _format_msg_content_for_text(msg.content) # Use helper for consistent text extraction
        full_conversation_text_parts.append(f"{msg.role} (name: {msg_name_str}): {formatted_content}")
    original_full_text_for_storage = "\n".join(full_conversation_text_parts)

    if not original_full_text_for_storage.strip():
        logger.info(f"Skipping ingestion of empty full conversation text. Channel: {channel_id}, User: {user_id}")
        return

    timestamp_now = datetime.now()
    str_user_id = str(user_id) 
    full_convo_doc_id = f"full_channel_{channel_id}_user_{str_user_id}_{int(timestamp_now.timestamp())}_{uuid4().hex}"
    
    full_convo_metadata: Dict[str, Any] = {
        "channel_id": str(channel_id), "user_id": str_user_id, "timestamp": timestamp_now.isoformat(),
        "type": "full_conversation_log" # Changed type slightly for clarity
    }
    
    if chat_history_collection:
        try:
            chat_history_collection.add(
                documents=[original_full_text_for_storage],
                metadatas=[full_convo_metadata],
                ids=[full_convo_doc_id]
            )
            logger.info(f"Ingested full conversation log (ID: {full_convo_doc_id}) into '{config.CHROMA_COLLECTION_NAME}'.")
        except Exception as e_add_full:
            logger.error(f"Failed to add full conversation log (ID: {full_convo_doc_id}) to ChromaDB: {e_add_full}", exc_info=True)
            # If full storage fails, we might not want to proceed with distillation, or handle it gracefully.
            # For now, we'll proceed but this is a point of potential failure.
    else:
        logger.error("chat_history_collection is None, cannot add full conversation log.")
        return # Critical to have the main collection

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
        text_for_distillation = original_full_text_for_storage # Fallback to the full text
        distillation_preview_text = original_full_text_for_storage[:200]


    if not text_for_distillation.strip():
        logger.warning(f"Text for distillation is empty for convo ID {full_convo_doc_id}. Skipping distilled sentence storage.")
        return

    distilled_sentence = await distill_conversation_to_sentence_llm(llm_client, text_for_distillation)

    if not distilled_sentence or not distilled_sentence.strip():
        logger.warning(f"Distillation failed or returned empty for convo ID {full_convo_doc_id} (source text preview: {distillation_preview_text[:100]}...). Skipping distilled sentence storage.")
        return

    distilled_doc_id = f"distilled_{full_convo_doc_id}" 
    distilled_metadata: Dict[str, Any] = {
        "channel_id": str(channel_id), 
        "user_id": str_user_id, 
        "timestamp": timestamp_now.isoformat(),
        "full_conversation_document_id": full_convo_doc_id, 
        "original_text_preview": distillation_preview_text # Preview of what was distilled
    }
    
    if distilled_chat_summary_collection:
        try:
            distilled_chat_summary_collection.add(
                documents=[distilled_sentence],
                metadatas=[distilled_metadata],
                ids=[distilled_doc_id]
            )
            logger.info(f"Ingested distilled sentence (ID: {distilled_doc_id}, linked to {full_convo_doc_id}) into '{config.CHROMA_DISTILLED_COLLECTION_NAME}'.")
        except Exception as e_add_distilled:
            logger.error(f"Failed to add distilled sentence (ID: {distilled_doc_id}) to ChromaDB: {e_add_distilled}", exc_info=True)
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
    if not chroma_client or not chat_history_collection or not distilled_chat_summary_collection:
        logger.error("ChromaDB collections not available for ChatGPT import. Skipping.")
        return 0
    
    added_count = 0
    if not conversations:
        logger.info("No conversations provided to store_chatgpt_conversations_in_chromadb.")
        return 0

    for i, convo_data in enumerate(conversations):
        full_conversation_text_parts = []
        last_user_content = ""
        last_assistant_content = ""
        # Extract last user/assistant pair for focused distillation
        temp_messages = convo_data.get('messages', [])
        if len(temp_messages) >= 2:
            # Assuming a typical user -> assistant flow in the export
            if temp_messages[-2].get('role') == 'user' and temp_messages[-1].get('role') == 'assistant':
                last_user_content = str(temp_messages[-2].get('content', '')).strip()
                last_assistant_content = str(temp_messages[-1].get('content', '')).strip()
        
        text_for_distillation_import = ""
        if last_user_content and last_assistant_content:
            text_for_distillation_import = f"User: {last_user_content}\nAssistant: {last_assistant_content}"
        else: # Fallback: build full text and distill that
            for msg in temp_messages: 
                content_str = str(msg.get('content', '')).strip()
                if content_str:
                    full_conversation_text_parts.append(f"{msg.get('role', 'unknown_role')}: {content_str}")
            text_for_distillation_import = "\n".join(full_conversation_text_parts)


        if not text_for_distillation_import.strip(): 
            logger.debug(f"Skipping empty or non-distillable conversation from import: {convo_data.get('title', 'Untitled')}")
            continue 
        
        # For full storage, we still store the entire conversation from the export
        full_exported_text_parts_for_storage = []
        for msg in temp_messages:
            content_str = str(msg.get('content', '')).strip()
            if content_str:
                full_exported_text_parts_for_storage.append(f"{msg.get('role', 'unknown_role')}: {content_str}")
        full_exported_text_to_store = "\n".join(full_exported_text_parts_for_storage)
        if not full_exported_text_to_store.strip():
            # This case should be caught by the text_for_distillation_import check if it was also empty
            logger.debug(f"Skipping empty full conversation for storage from import: {convo_data.get('title', 'Untitled')}")
            continue


        timestamp = convo_data.get('create_time', datetime.now()) 
        safe_title = re.sub(r'\W+', '_', convo_data.get('title', 'untitled'))[:50] 
        full_convo_doc_id = f"{source}_full_{safe_title}_{i}_{int(timestamp.timestamp())}_{uuid4().hex}"
        
        full_convo_metadata: Dict[str, Any] = {  
            "title": convo_data.get('title', 'Untitled'), 
            "source": source, 
            "create_time": timestamp.isoformat(), 
            "type": "full_conversation_import" 
        }
        try:
            if chat_history_collection: 
                chat_history_collection.add(
                    documents=[full_exported_text_to_store], # Store the full exported text
                    metadatas=[full_convo_metadata],
                    ids=[full_convo_doc_id]
                )
                logger.info(f"Stored full ChatGPT import (ID: {full_convo_doc_id}) in '{config.CHROMA_COLLECTION_NAME}'.")
            else:
                logger.error("chat_history_collection is None, cannot add imported document.")
                continue


            distilled_sentence = await distill_conversation_to_sentence_llm(llm_client, text_for_distillation_import)

            if not distilled_sentence or not distilled_sentence.strip():
                logger.warning(f"Distillation failed for imported convo: {convo_data.get('title', 'Untitled')} (ID: {full_convo_doc_id}). Skipping distilled sentence storage.")
                continue 

            distilled_doc_id = f"distilled_{full_convo_doc_id}" 
            distilled_metadata: Dict[str, Any] = { 
                "title": convo_data.get('title', 'Untitled'), 
                "source": source, 
                "create_time": timestamp.isoformat(),
                "full_conversation_document_id": full_convo_doc_id, 
                "original_text_preview": text_for_distillation_import[:200] 
            }
            if distilled_chat_summary_collection: 
                distilled_chat_summary_collection.add(
                    documents=[distilled_sentence],
                    metadatas=[distilled_metadata],
                    ids=[distilled_doc_id]
                )
                logger.info(f"Stored distilled sentence for imported convo (ID: {distilled_doc_id}, linked to {full_convo_doc_id}) in '{config.CHROMA_DISTILLED_COLLECTION_NAME}'.")
                added_count += 1 
            else:
                logger.error("distilled_chat_summary_collection is None, cannot add distilled document.")

        except Exception as e_add:
            logger.error(f"Error processing/adding imported conversation '{convo_data.get('title', 'Untitled')}' (Full ID: {full_convo_doc_id}) to ChromaDB: {e_add}", exc_info=True)

    if added_count > 0:
        logger.info(f"Successfully processed and stored {added_count} imported conversations with distillations into ChromaDB.")
    elif conversations: 
        logger.info("Processed ChatGPT export, but no conversations were successfully stored with distillation.")
    else:
        logger.info("No conversations found in the provided ChatGPT export to process.")

    return added_count


def store_news_summary(topic: str, url: str, summary_text: str, timestamp: Optional[datetime] = None) -> bool:
    """Store a summarized news article in the dedicated ChromaDB collection."""
    if not chroma_client or not news_summary_collection:
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
