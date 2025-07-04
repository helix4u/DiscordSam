# This script will build a knowledge graph from data stored in ChromaDB.
# It will fetch entities and relations and construct a graph using networkx.

import logging
import chromadb
from typing import Optional, Dict, Any, List
import networkx as nx # Import networkx
import argparse # Import argparse

# Attempt to import global config and specific collections from rag_chroma_manager
try:
    from config import config
    # We will define our own collection variables in this script
except ImportError:
    print("Error: Could not import 'config' or 'rag_chroma_manager'. "
          "Ensure these files are in the PYTHONPATH and accessible.")
    exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for ChromaDB client and collections
chroma_client: Optional[chromadb.ClientAPI] = None
entity_collection: Optional[chromadb.Collection] = None
relation_collection: Optional[chromadb.Collection] = None

def initialize_chromadb_for_graph() -> bool:
    """
    Initializes the ChromaDB client and specific collections needed for graph building.
    """
    global chroma_client, entity_collection, relation_collection

    if chroma_client:
        logger.debug("ChromaDB already initialized for graph.")
        return True
    try:
        logger.info(f"Initializing ChromaDB client with path: {config.CHROMA_DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

        logger.info(f"Getting ChromaDB entity collection: {config.CHROMA_ENTITIES_COLLECTION_NAME}")
        entity_collection = chroma_client.get_collection(name=config.CHROMA_ENTITIES_COLLECTION_NAME)

        logger.info(f"Getting ChromaDB relation collection: {config.CHROMA_RELATIONS_COLLECTION_NAME}")
        relation_collection = chroma_client.get_collection(name=config.CHROMA_RELATIONS_COLLECTION_NAME)

        logger.info(
            "ChromaDB initialized successfully for graph building. Required collections are accessible."
        )
        return True
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB or get required collections for graph: {e}", exc_info=True)
        chroma_client = None
        entity_collection = None
        relation_collection = None
        return False

if __name__ == "__main__":
    logger.info("Starting knowledge graph builder script.")
    if initialize_chromadb_for_graph():
        logger.info("ChromaDB initialized successfully.")

        if entity_collection:
            try:
                logger.info(f"Fetching all entities from collection: {config.CHROMA_ENTITIES_COLLECTION_NAME}")
                # Fetch all entries. ChromaDB's get() without IDs or where_document fetches all.
                # We need documents (entity names) and metadatas (type, etc.)
                entities_data = entity_collection.get(include=["documents", "metadatas"])

                num_entities = len(entities_data.get("ids", []))
                logger.info(f"Successfully fetched {num_entities} entities.")

                # Example: Print first few entities if any
                # for i in range(min(3, num_entities)):
                #    logger.info(f"Entity ID: {entities_data['ids'][i]}, "
                #                f"Name: {entities_data['documents'][i]}, "
                #                f"Metadata: {entities_data['metadatas'][i]}")

            except Exception as e:
                logger.error(f"Failed to fetch entities: {e}", exc_info=True)
                entities_data = None # Ensure it's None on failure
        else:
            logger.error("Entity collection is not available. Cannot fetch entities.")
            entities_data = None

        if relation_collection:
            try:
                logger.info(f"Fetching all relations from collection: {config.CHROMA_RELATIONS_COLLECTION_NAME}")
                # Fetch all entries. We need metadatas (subject, predicate, object).
                # Documents for relations are textual representations like "Subject Predicate Object"
                relations_data = relation_collection.get(include=["documents", "metadatas"])

                num_relations = len(relations_data.get("ids", []))
                logger.info(f"Successfully fetched {num_relations} relations.")

                # Example: Print first few relations if any
                # for i in range(min(3, num_relations)):
                #    logger.info(f"Relation ID: {relations_data['ids'][i]}, "
                #                f"Document: {relations_data['documents'][i]}, "
                #                f"Metadata: {relations_data['metadatas'][i]}")

            except Exception as e:
                logger.error(f"Failed to fetch relations: {e}", exc_info=True)
                relations_data = None # Ensure it's None on failure
        else:
            logger.error("Relation collection is not available. Cannot fetch relations.")
            relations_data = None

        # Further steps will go here, using entities_data and relations_data

        graph = nx.DiGraph() # Create a directed graph

        if entities_data and entities_data.get("documents") and entities_data.get("metadatas"):
            logger.info("Adding entities to the graph...")
            entity_names = entities_data["documents"]
            entity_metadatas = entities_data["metadatas"]
            for i, entity_name in enumerate(entity_names):
                if entity_name: # Ensure entity_name is not None or empty
                    metadata = entity_metadatas[i] if entity_metadatas and i < len(entity_metadatas) else {}
                    graph.add_node(
                        entity_name,
                        entity_type=metadata.get("entity_type", "Unknown"),
                        source_doc_id=metadata.get("source_doc_id", ""),
                        raw_details=metadata.get("raw_details", "{}")
                    )
            logger.info(f"Added {graph.number_of_nodes()} nodes to the graph.")
        else:
            logger.warning("No valid entity data to add to the graph.")

        if relations_data and relations_data.get("metadatas"):
            logger.info("Adding relations to the graph...")
            relation_metadatas = relations_data["metadatas"]
            # Documents for relations are also available if needed: relations_data.get("documents")

            edges_added = 0
            for i, rel_meta in enumerate(relation_metadatas):
                subject = rel_meta.get("subject_name")
                predicate = rel_meta.get("predicate")
                obj = rel_meta.get("object_name")

                if subject and predicate and obj:
                    # Ensure nodes exist before adding edge, or let networkx create them (default behavior)
                    # For cleaner graph, prefer nodes to be defined from entity_collection first.
                    if not graph.has_node(subject):
                        logger.warning(f"Subject node '{subject}' for relation not found in entities. Adding as a simple node.")
                        graph.add_node(subject, entity_type="Unknown_From_Relation")
                    if not graph.has_node(obj):
                        logger.warning(f"Object node '{obj}' for relation not found in entities. Adding as a simple node.")
                        graph.add_node(obj, entity_type="Unknown_From_Relation")

                    graph.add_edge(
                        subject,
                        obj,
                        predicate=predicate,
                        context_phrase=rel_meta.get("context_phrase", ""),
                        source_doc_id=rel_meta.get("source_doc_id", ""),
                        raw_details=rel_meta.get("raw_details", "{}")
                    )
                    edges_added += 1
                else:
                    logger.warning(f"Skipping relation due to missing subject, predicate, or object: {rel_meta}")
            logger.info(f"Added {edges_added} edges to the graph.")
            logger.info(f"Graph construction complete. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        else:
            logger.warning("No valid relation data to add to the graph.")

        if graph.number_of_nodes() > 0 or graph.number_of_edges() > 0:
            logger.info(f"Final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

            # Output: Save to GEXF file
            output_filename = "knowledge_graph.gexf" # Default filename
            try:
                nx.write_gexf(graph, output_filename)
                logger.info(f"Knowledge graph saved to {output_filename}")
            except Exception as e:
                logger.error(f"Failed to save graph to GEXF file: {e}", exc_info=True)
        else:
            logger.info("Graph is empty. Nothing to save.")

    else:
        logger.error("Failed to initialize ChromaDB. Exiting.")

def main():
    parser = argparse.ArgumentParser(description="Build a knowledge graph from ChromaDB collections.")
    parser.add_argument(
        "-o", "--output",
        default="knowledge_graph.gexf",
        help="Output file path for the GEXF graph file (default: knowledge_graph.gexf)"
    )
    args = parser.parse_args()

    logger.info("Starting knowledge graph builder script.")
    if initialize_chromadb_for_graph():
        logger.info("ChromaDB initialized successfully.")

        entities_data = None
        if entity_collection:
            try:
                logger.info(f"Fetching all entities from collection: {config.CHROMA_ENTITIES_COLLECTION_NAME}")
                entities_data = entity_collection.get(include=["documents", "metadatas"])
                num_entities = len(entities_data.get("ids", []))
                logger.info(f"Successfully fetched {num_entities} entities.")
            except Exception as e:
                logger.error(f"Failed to fetch entities: {e}", exc_info=True)
        else:
            logger.error("Entity collection is not available. Cannot fetch entities.")

        relations_data = None
        if relation_collection:
            try:
                logger.info(f"Fetching all relations from collection: {config.CHROMA_RELATIONS_COLLECTION_NAME}")
                relations_data = relation_collection.get(include=["documents", "metadatas"])
                num_relations = len(relations_data.get("ids", []))
                logger.info(f"Successfully fetched {num_relations} relations.")
            except Exception as e:
                logger.error(f"Failed to fetch relations: {e}", exc_info=True)
        else:
            logger.error("Relation collection is not available. Cannot fetch relations.")

        graph = nx.DiGraph()

        if entities_data and entities_data.get("documents") and entities_data.get("metadatas"):
            logger.info("Adding entities to the graph...")
            entity_names = entities_data["documents"]
            entity_metadatas = entities_data["metadatas"]
            for i, entity_name in enumerate(entity_names):
                if entity_name:
                    metadata = entity_metadatas[i] if entity_metadatas and i < len(entity_metadatas) else {}
                    graph.add_node(
                        entity_name,
                        entity_type=metadata.get("entity_type", "Unknown"),
                        source_doc_id=metadata.get("source_doc_id", ""),
                        raw_details=metadata.get("raw_details", "{}")
                    )
            logger.info(f"Added {graph.number_of_nodes()} nodes to the graph.")
        else:
            logger.warning("No valid entity data to add to the graph.")

        if relations_data and relations_data.get("metadatas"):
            logger.info("Adding relations to the graph...")
            relation_metadatas = relations_data["metadatas"]
            edges_added = 0
            for i, rel_meta in enumerate(relation_metadatas):
                subject = rel_meta.get("subject_name")
                predicate = rel_meta.get("predicate")
                obj = rel_meta.get("object_name")

                if subject and predicate and obj:
                    if not graph.has_node(subject):
                        logger.warning(f"Subject node '{subject}' for relation not found in entities. Adding as a simple node.")
                        graph.add_node(subject, entity_type="Unknown_From_Relation")
                    if not graph.has_node(obj):
                        logger.warning(f"Object node '{obj}' for relation not found in entities. Adding as a simple node.")
                        graph.add_node(obj, entity_type="Unknown_From_Relation")
                    graph.add_edge(
                        subject,
                        obj,
                        predicate=predicate,
                        context_phrase=rel_meta.get("context_phrase", ""),
                        source_doc_id=rel_meta.get("source_doc_id", ""),
                        raw_details=rel_meta.get("raw_details", "{}")
                    )
                    edges_added += 1
                else:
                    logger.warning(f"Skipping relation due to missing subject, predicate, or object: {rel_meta}")
            logger.info(f"Added {edges_added} edges to the graph.")
            logger.info(f"Graph construction complete. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        else:
            logger.warning("No valid relation data to add to the graph.")

        if graph.number_of_nodes() > 0 or graph.number_of_edges() > 0:
            logger.info(f"Final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
            output_filename = args.output # Use filename from command line arguments
            try:
                nx.write_gexf(graph, output_filename)
                logger.info(f"Knowledge graph saved to {output_filename}")
            except Exception as e:
                logger.error(f"Failed to save graph to GEXF file: {e}", exc_info=True)
        else:
            logger.info("Graph is empty. Nothing to save.")
    else:
        logger.error("Failed to initialize ChromaDB. Exiting.")

if __name__ == "__main__":
    main()
