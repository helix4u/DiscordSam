import chromadb
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_entity_collection():
    """Connects to ChromaDB and deletes the entity collection."""
    try:
        logger.info(f"Initializing ChromaDB client with path: {config.CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

        collection_name = config.CHROMA_ENTITIES_COLLECTION_NAME
        logger.info(f"Attempting to delete collection: {collection_name}")

        # Check if the collection exists before attempting to delete
        existing_collections = [c.name for c in chroma_client.list_collections()]
        if collection_name in existing_collections:
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
        else:
            logger.info(f"Collection '{collection_name}' not found, no action taken.")

    except Exception as e:
        logger.critical(f"An error occurred while trying to delete the collection: {e}", exc_info=True)

if __name__ == "__main__":
    print("This script will permanently delete the entity collection from ChromaDB.")
    user_confirmation = input("Are you sure you want to continue? (yes/no): ")
    if user_confirmation.lower() == 'yes':
        delete_entity_collection()
    else:
        print("Operation cancelled by user.")
