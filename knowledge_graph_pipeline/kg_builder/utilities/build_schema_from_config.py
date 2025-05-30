from typing import Dict, List, Any, Optional, Tuple
import json
import os
from dotenv import load_dotenv
import pprint

from neo4j_graphrag.experimental.components.schema import (
    SchemaEntity, 
    SchemaRelation, 
    SchemaProperty, 
    SchemaConfig
)

# script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the directory where this script is located

# # Load environment variables from a .env file
# dotenv_path = os.path.join(script_dir, '.env')  # Path to the .env file
# load_dotenv(dotenv_path, override=True)

# # Open configuration file from JSON format
# config_path = os.path.join(script_dir, 'config.json')  # Path to the configuration file

# with open(config_path, 'r') as config_file:
#     config = json.load(config_file)

# schema_config = config["schema_config"]

def build_schema_from_config(schema_config: Dict[str, Any]) -> Optional[SchemaConfig]:
    """
    Builds a SchemaConfig object from the schema configuration in the config.json file.
    
    Args:
        schema_config: The schema_config section from the config.json file
        
    Returns:
        SchemaConfig object or None if schema creation is disabled
    """
    # Check if schema creation is enabled
    if not schema_config.get("create_schema", False):
        return None, None, None  # If not, return None for entities, relations, and patterns
    
    # Convert node types to SchemaEntity objects
    entities = []
    for node in schema_config.get("nodes", []):
        # Convert properties to SchemaProperty objects
        properties = []
        for prop in node.get("properties", []):
            properties.append(
                SchemaProperty(
                    name=prop["name"],
                    type=prop["type"],
                    description=prop.get("description", "")
                )
            )
        
        # Create SchemaEntity
        entities.append(
            SchemaEntity(
                label=node["label"],
                description=node.get("description", ""),
                properties=properties
            )
        )
    
    # Convert relationship types to SchemaRelation objects
    relations = []
    for edge in schema_config.get("edges", []):
        # Convert properties to SchemaProperty objects
        properties = []
        for prop in edge.get("properties", []):
            properties.append(
                SchemaProperty(
                    name=prop["name"],
                    type=prop["type"],
                    description=prop.get("description", "")
                )
            )
        
        # Create SchemaRelation
        relations.append(
            SchemaRelation(
                label=edge["label"],
                description=edge.get("description", ""),
                properties=properties
            )
        )
    
    # Get patterns if they should be used
    triplets = None
    if schema_config.get("suggest_pattern", False):
        triplets = schema_config.get("triplets", [])
        # Convert triplets to tuples of (source, relation, target)
        triplets = [
            (triplet[0], triplet[1], triplet[2])
            for triplet in triplets
        ]
    
    return entities, relations, triplets

# # Example usage of the build_schema_from_config function
# async def main():
#     entities, relations, triplets = await build_schema_from_config(schema_config)
#     if entities is not None:
#         print("Schema created successfully:")
#         print("\nEntities:\n")
#         pprint.pprint(entities)
#         print("\nRelations:\n")
#         pprint.pprint(relations)
#         print("\nTriplets:\n")
#         pprint.pprint(triplets)
#     else:
#         print("Schema creation is disabled in the configuration.")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())