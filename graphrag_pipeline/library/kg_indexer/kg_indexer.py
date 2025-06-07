import neo4j
from neo4j_graphrag.indexes import (
    create_vector_index, 
    retrieve_vector_index_info, 
    create_fulltext_index, 
    retrieve_fulltext_index_info,
    drop_index_if_exists
)

class KGIndexer:

    """Class for managing indexes in a Neo4j knowledge graph."""

    def __init__(self, driver: neo4j.Driver):
        self.driver = driver
    
    def create_vector_index(
        self,
        index_name: str,
        label: str,
        embedding_property: str,
        dimensions: int
    ):
        """
        Create a vector index in Neo4j.
        
        Args:
            index_name: Name of the index.
            label: Node label to index.
            embedding_property: Name of the property containing the embeddings.
            dimensions: Vector dimensions of the used embeddings to construct the knowledge graph.
        """
        create_vector_index(
            driver=self.driver,
            name=index_name,
            label=label,
            embedding_property=embedding_property,
            dimensions=dimensions,
            similarity_fn='cosine',  # Similarity function to use for the index (cosine for normalized proximity calculation)
            fail_if_exists=False   # Set to True if you want to fail if the index already exists. Setting to False enables running the script multiple times (with additional nodes) without errors.
        )

        # Print success message
        print(f"Vector index '{index_name}' created successfully.")
    
    def retrieve_vector_index_info(
        self,
        index_name: str,
        label_or_type: str,
        embedding_property: str
    ):
        """
        Retrieve information about a vector index in Neo4j.
        
        Args:
            index_name: Name of the index to retrieve information about.
            label_or_type: Node label or relationship type to check for the index.
            embedding_property: Name of the property containing the embeddings.
        
        Returns:
            Information about the first vector index if it exists, otherwise None.
        """
        first_index_info =  retrieve_vector_index_info(
            driver=self.driver,
            index_name=index_name,
            label_or_type=label_or_type,
            embedding_property=embedding_property
        )
    
        if first_index_info:
            print(f"Vector index '{index_name}' exists with the following details:")
            print(first_index_info)
        else:
            print(f"Vector index '{index_name}' does not exist or could not be retrieved. Index creation may have failed.")

    def create_fulltext_index(
        self,
        index_name: str,
        label: str,
        node_properties: list[str]
    ):
        """
        Create an index on the full text in Neo4j.
        
        Args:
            index_name: Name of the index.
            label: Node label to index.
            node_properties: List of properties containing the full text.
        """
        create_fulltext_index(
            driver=self.driver,
            name=index_name,
            label=label,
            node_properties=node_properties,
            fail_if_exists=False  # Set to True if you want to fail if the index already exists. Setting to False enables running the script multiple times (with additional nodes) without errors.
        )

        # Print success message
        print(f"Full text index '{index_name}' created successfully.")
    
    def retrieve_fulltext_index_info(
        self,
        index_name: str,
        label_or_type: str,
        text_properties: list[str]
    ):
        """
        Retrieve information about a full text index in Neo4j.
        
        Args:
            index_name: Name of the index to retrieve information about.
            label_or_type: Node label or relationship type to check for the index.
            text_properties: List of properties containing the full text.
        
        Returns:
            Information about the first full text index if it exists, otherwise None.
        """
        first_index_info =  retrieve_fulltext_index_info(
            driver=self.driver,
            index_name=index_name,
            label_or_type=label_or_type,
            text_properties=text_properties
        )

        if first_index_info:
            print(f"Full text index '{index_name}' exists with the following details:")
            print(first_index_info)
        else:
            print(f"Full text index '{index_name}' does not exist or could not be retrieved. Index creation may have failed.")
    
    def drop_index_if_exists(
        self,
        index_name: str
    ):
        """
        Drop an index if it exists in Neo4j.
        
        Args:
            index_name: Name of the index to drop.
        """
        drop_index_if_exists(
            driver=self.driver,
            name=index_name
        )
        print(f"Index '{index_name}' dropped if it existed.")
    
    def list_all_indexes(self):
        """
        List all indexes in the Neo4j database.
        
        Returns:
            List of dictionaries containing information about all indexes.
        """
        with self.driver.session() as session:

            # For Neo4j 4.0+
            result = session.run("SHOW INDEXES")
            indexes = [record.data() for record in result]
            
            if not indexes:
                # For older Neo4j versions
                result = session.run("CALL db.indexes()")
                indexes = [record.data() for record in result]
                
            if indexes:
                print(f"Found {len(indexes)} indexes in the database:")
                for idx, index in enumerate(indexes, 1):
                    print(f"\n{idx}. {index}")
            else:
                print("No indexes found in the database.")
                
            return indexes