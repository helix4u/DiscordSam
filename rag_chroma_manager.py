import logging
import random
from datetime import datetime
from typing import List, Optional, Any, Dict, Union 
import json 
import re 
import hashlib # Was in original, keeping for now, though not directly used in this version's functions


import chromadb 

# Assuming config is imported from config.py
from config import config
# Import MsgNode from the new common_models.py
from common_models import MsgNode 
# llm_client will be passed as an argument to functions needing it.


logger = logging.getLogger(__name__)

# --- ChromaDB Client Initialization ---
# These are module-level globals, initialized by initialize_chromadb()
chroma_client: Optional[chromadb.ClientAPI] = None
chat_history_collection: Optional[chromadb.Collection] = None 
distilled_chat_summary_collection: Optional[chromadb.Collection] = None 

def initialize_chromadb() -> bool:
    global chroma_client, chat_history_collection, distilled_chat_summary_collection
    if chroma_client: # Already initialized
        logger.debug("ChromaDB already initialized.")
        return True
    try:
        logger.info(f"Initializing ChromaDB client with path: {config.CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        
        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_COLLECTION_NAME}")
        chat_history_collection = chroma_client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
        
        logger.info(f"Getting or creating ChromaDB collection: {config.CHROMA_DISTILLED_COLLECTION_NAME}")
        distilled_chat_summary_collection = chroma_client.get_or_create_collection(name=config.CHROMA_DISTILLED_COLLECTION_NAME)
        
        logger.info(f"ChromaDB initialized successfully. Main Collection: '{config.CHROMA_COLLECTION_NAME}', Distilled Collection: '{config.CHROMA_DISTILLED_COLLECTION_NAME}'")
        return True
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB collections: {e}", exc_info=True)
        chroma_client = None 
        chat_history_collection = None
        distilled_chat_summary_collection = None
        return False

async def distill_conversation_to_sentence_llm(llm_client: Any, full_conversation_text: str) -> Optional[str]:
    """Uses an LLM to distill a full conversation into a few keyword-rich sentences."""
    if not full_conversation_text.strip():
        logger.debug("Distillation skipped: full_conversation_text is empty.")
        return None
    
    truncated_conversation_text = full_conversation_text[:3000] 

    prompt = (
        "You are a text distillation expert. Read the following conversations and summarize their "
        "absolute core essence into a few keyword-rich, data-dense sentences. These sentences "
        "will be used for semantic search to recall these conversations later. Focus on unique "
        "entities, key actions, insights, and primary topics. The sentences should be concise and highly informative.\n\n"
        "CONVERSATION:\n---\n"
        f"{truncated_conversation_text}" 
        "\n---\n\n"
        "DISTILLED SENTENCE(S):" 
    )
    try:
        logger.debug(f"Requesting distillation from model {config.FAST_LLM_MODEL}.")
        response = await llm_client.chat.completions.create(
            model=config.FAST_LLM_MODEL, 
            messages=[
                {"role": "system", "content": "You are an expert contexual knowledge distiller."}, 
                {"role": "user", "content": prompt}
            ],
            max_tokens=900, 
            temperature=0.6, 
            stream=False
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            distilled = response.choices[0].message.content.strip()
            logger.info(f"Distilled conversation to sentence(s): '{distilled[:100]}...'")
            return distilled
        logger.warning("LLM distillation returned no content for sentence generation.")
        return None
    except Exception as e:
        logger.error(f"Failed to distill conversation to sentence(s): {e}", exc_info=True)
        return None

async def synthesize_retrieved_contexts_llm(llm_client: Any, retrieved_full_texts: List[str], current_query: str) -> Optional[str]:
    """
    Uses an LLM to synthesize multiple retrieved conversation texts into a single,
    concise paragraph relevant to the current_query.
    """
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
        "no specific relevant context was found in past conversations. Use the conversation history. Be objective.\n\n"
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
        logger.warning("ChromaDB collections not available (chroma_client or specific collections are None), skipping RAG context retrieval.")
        return None
    
    try:
        logger.debug(f"RAG: Querying distilled_chat_summary_collection for query: '{query[:50]}...' (n_results={n_results_sentences})")
        results = distilled_chat_summary_collection.query(
            query_texts=[query] if isinstance(query, str) else query, # type: ignore
            n_results=n_results_sentences,
            include=["metadatas", "documents"] 
        )
        
        if not results or not results.get('ids') or not results['ids'][0]: 
            logger.info(f"RAG: No relevant distilled sentences found in ChromaDB for query: '{str(query)[:50]}...'")
            return None

        retrieved_full_conversation_texts: List[str] = []
        retrieved_distilled_sentences_for_log: List[str] = []
        full_convo_ids_to_fetch = []

        query_result_ids = results['ids'][0]
        query_result_metadatas = results['metadatas'][0] if results['metadatas'] and results['metadatas'][0] else [{} for _ in query_result_ids] # Handle missing metadatas
        query_result_documents = results['documents'][0] if results['documents'] and results['documents'][0] else ["[Distilled sentence not found]" for _ in query_result_ids] # Handle missing documents

        for i in range(len(query_result_ids)):
            dist_metadata = query_result_metadatas[i]
            dist_sentence = query_result_documents[i]
            
            retrieved_distilled_sentences_for_log.append(dist_sentence or "[No sentence content]") # Ensure string
            
            full_convo_id = dist_metadata.get('full_conversation_document_id')
            if full_convo_id:
                full_convo_ids_to_fetch.append(str(full_convo_id)) 
            else:
                logger.warning(f"RAG: Retrieved distilled sentence (ID: {query_result_ids[i]}) but missing 'full_conversation_document_id' in metadata: {dist_metadata}")
        
        if retrieved_distilled_sentences_for_log:
            log_sentences = "\n- ".join(retrieved_distilled_sentences_for_log)
            logger.info(f"RAG: Top {len(retrieved_distilled_sentences_for_log)} distilled sentences retrieved:\n- {log_sentences}")

        if not full_convo_ids_to_fetch:
            logger.info("RAG: No full conversation document IDs found from distilled sentence metadata.")
            return None

        unique_full_convo_ids_to_fetch = list(set(full_convo_ids_to_fetch))
        logger.debug(f"RAG: Fetching full conversation documents for IDs: {unique_full_convo_ids_to_fetch}")
        if unique_full_convo_ids_to_fetch and chat_history_collection: # Ensure collection is not None
            try:
                full_convo_docs_result = chat_history_collection.get(ids=unique_full_convo_ids_to_fetch, include=["documents"])
                if full_convo_docs_result and full_convo_docs_result.get('documents'):
                    # Ensure documents are strings
                    valid_docs = [doc for doc in full_convo_docs_result['documents'] if isinstance(doc, str)]
                    retrieved_full_conversation_texts.extend(valid_docs)
                    logger.info(f"RAG: Retrieved {len(valid_docs)} full conversation texts.")
                else:
                    logger.warning(f"RAG: Could not retrieve some/all full conversation documents for IDs: {unique_full_convo_ids_to_fetch}. Result: {full_convo_docs_result}")
            except Exception as e_get_full:
                logger.error(f"RAG: Error fetching full conversation docs for IDs {unique_full_convo_ids_to_fetch}: {e_get_full}", exc_info=True)

        if not retrieved_full_conversation_texts:
            logger.info("RAG: No full conversation texts could be retrieved for synthesis from ChromaDB.")
            return None
            
        synthesized_context = await synthesize_retrieved_contexts_llm(llm_client, retrieved_full_conversation_texts, query)
        return synthesized_context

    except Exception as e:
        logger.error(f"RAG: Failed during context retrieval/preparation: {e}", exc_info=True)
        return None

async def ingest_conversation_to_chromadb(
    llm_client: Any, 
    channel_id: int, 
    user_id: Union[int, str], 
    conversation_history_for_rag: List[MsgNode]
):
    if not chroma_client or not chat_history_collection or not distilled_chat_summary_collection:
        logger.warning("ChromaDB collections not available, skipping ingestion.")
        return

    non_system_messages = [msg for msg in conversation_history_for_rag if msg.role in ['user', 'assistant']]
    if len(non_system_messages) < 2: 
        logger.debug(f"Skipping ChromaDB ingestion for short RAG history (non-system messages: {len(non_system_messages)}). Channel: {channel_id}, User: {user_id}")
        return

    try:
        full_conversation_text_parts = []
        for msg in conversation_history_for_rag: 
            msg_name_str = str(msg.name) if msg.name else "N/A"
            if isinstance(msg.content, str):
                full_conversation_text_parts.append(f"{msg.role} (name: {msg_name_str}): {msg.content}")
            elif isinstance(msg.content, list): 
                text_parts_for_chroma = [part["text"] for part in msg.content if isinstance(part, dict) and part.get("type") == "text" and "text" in part]
                if text_parts_for_chroma:
                    full_conversation_text_parts.append(f"{msg.role} (name: {msg_name_str}): {' '.join(text_parts_for_chroma)}")
                else: 
                    full_conversation_text_parts.append(f"{msg.role} (name: {msg_name_str}): [Media content, no text part for ChromaDB]")
        original_full_text = "\n".join(full_conversation_text_parts)

        if not original_full_text.strip():
            logger.info(f"Skipping ingestion of empty full conversation text to ChromaDB. Channel: {channel_id}, User: {user_id}")
            return

        timestamp_now = datetime.now()
        str_user_id = str(user_id) 
        full_convo_doc_id = f"full_channel_{channel_id}_user_{str_user_id}_{int(timestamp_now.timestamp())}_{random.randint(1000,9999)}"
        
        full_convo_metadata: Dict[str, Any] = { # Type hint for clarity
            "channel_id": str(channel_id), "user_id": str_user_id, "timestamp": timestamp_now.isoformat(),
            "type": "full_conversation" 
        }
        logger.debug(f"Adding full conversation to ChromaDB. ID: {full_convo_doc_id}, Metadata: {full_convo_metadata}")
        if chat_history_collection: # Ensure collection is not None
            chat_history_collection.add(
                documents=[original_full_text],
                metadatas=[full_convo_metadata],
                ids=[full_convo_doc_id]
            )
            logger.info(f"Ingested full conversation (ID: {full_convo_doc_id}) into main ChromaDB collection '{config.CHROMA_COLLECTION_NAME}'.")
        else:
            logger.error("chat_history_collection is None, cannot add document.")
            return


        distilled_sentence = await distill_conversation_to_sentence_llm(llm_client, original_full_text)

        if not distilled_sentence or not distilled_sentence.strip():
            logger.warning(f"Distillation failed for full_convo_id {full_convo_doc_id}. Skipping distilled sentence storage.")
            return

        distilled_doc_id = f"distilled_{full_convo_doc_id}" 
        distilled_metadata: Dict[str, Any] = { # Type hint for clarity
            "channel_id": str(channel_id), "user_id": str_user_id, "timestamp": timestamp_now.isoformat(),
            "full_conversation_document_id": full_convo_doc_id, 
            "original_text_preview": original_full_text[:200] 
        }
        logger.debug(f"Adding distilled sentence to ChromaDB. ID: {distilled_doc_id}, Metadata: {distilled_metadata}")
        if distilled_chat_summary_collection: # Ensure collection is not None
            distilled_chat_summary_collection.add(
                documents=[distilled_sentence],
                metadatas=[distilled_metadata],
                ids=[distilled_doc_id]
            )
            logger.info(f"Ingested distilled sentence (ID: {distilled_doc_id}, linked to {full_convo_doc_id}) into distilled ChromaDB collection '{config.CHROMA_DISTILLED_COLLECTION_NAME}'.")
        else:
            logger.error("distilled_chat_summary_collection is None, cannot add document.")


    except Exception as e:
        logger.error(f"Failed to ingest conversation into ChromaDB (dual collection) for Ch: {channel_id}, User: {user_id}: {e}", exc_info=True)

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
        for msg in convo_data.get('messages', []): 
            content_str = str(msg.get('content', '')).strip()
            if content_str:
                 full_conversation_text_parts.append(f"{msg.get('role', 'unknown_role')}: {content_str}")
        
        full_conversation_text = "\n".join(full_conversation_text_parts)
        if not full_conversation_text.strip(): 
            logger.debug(f"Skipping empty conversation from import: {convo_data.get('title', 'Untitled')}")
            continue 

        timestamp = convo_data.get('create_time', datetime.now()) 
        safe_title = re.sub(r'\W+', '_', convo_data.get('title', 'untitled'))[:50] 
        full_convo_doc_id = f"{source}_full_{safe_title}_{i}_{int(timestamp.timestamp())}_{random.randint(1000,9999)}"
        
        full_convo_metadata: Dict[str, Any] = {  
            "title": convo_data.get('title', 'Untitled'), 
            "source": source, 
            "create_time": timestamp.isoformat(), 
            "type": "full_conversation_import" 
        }
        try:
            logger.debug(f"Adding imported ChatGPT conversation to main collection. ID: {full_convo_doc_id}")
            if chat_history_collection: # Ensure collection is not None
                chat_history_collection.add(
                    documents=[full_conversation_text],
                    metadatas=[full_convo_metadata],
                    ids=[full_convo_doc_id]
                )
                logger.info(f"Stored full ChatGPT import (ID: {full_convo_doc_id}) in '{config.CHROMA_COLLECTION_NAME}'.")
            else:
                logger.error("chat_history_collection is None, cannot add imported document.")
                continue # Skip to next conversation if main collection is unavailable


            distilled_sentence = await distill_conversation_to_sentence_llm(llm_client, full_conversation_text)

            if not distilled_sentence or not distilled_sentence.strip():
                logger.warning(f"Distillation failed for imported convo: {convo_data.get('title', 'Untitled')} (ID: {full_convo_doc_id}). Skipping distilled sentence storage.")
                continue 

            distilled_doc_id = f"distilled_{full_convo_doc_id}" 
            distilled_metadata: Dict[str, Any] = { 
                "title": convo_data.get('title', 'Untitled'), 
                "source": source, 
                "create_time": timestamp.isoformat(),
                "full_conversation_document_id": full_convo_doc_id, 
                "original_text_preview": full_conversation_text[:200] 
            }
            logger.debug(f"Adding distilled sentence for imported convo to distilled collection. ID: {distilled_doc_id}")
            if distilled_chat_summary_collection: # Ensure collection is not None
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
