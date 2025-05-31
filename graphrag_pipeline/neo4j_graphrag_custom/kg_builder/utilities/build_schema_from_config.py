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

def build_schema_from_config(schema_config: Dict[str, Any]) -> Optional[SchemaConfig]:
    """
    Builds a knowledge graph schema from a configuration dictionary.
    
    Args:
        schema_config: Dictionary containing node types, edge types, and relationship patterns
                       for the knowledge graph schema
    
    Returns:
        Tuple of (entities, relations, triplets) where:
        - entities: List of SchemaEntity objects representing node types
        - relations: List of SchemaRelation objects representing edge types
        - triplets: List of tuples (source, relation, target) representing valid relationship patterns
        Returns (None, None, None) if schema creation is disabled in config
    
    Example:
        Given a config with entities like 'Event', 'Actor', 'Country' and relations like 
        'OCCURRED_IN', 'PARTICIPATED_IN', this function will return:
        
        - entities: [SchemaEntity(label='Event', ...), SchemaEntity(label='Actor', ...), ...]
        - relations: [SchemaRelation(label='OCCURRED_IN', ...), ...]
        - triplets: [('Event', 'OCCURRED_IN', 'Country'), ('Actor', 'PARTICIPATED_IN', 'Event'), ...]
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