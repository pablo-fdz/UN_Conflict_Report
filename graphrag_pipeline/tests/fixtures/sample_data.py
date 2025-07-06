"""
Sample data fixtures for testing pipeline components.

This module provides sample data that mimics real data from ACLED, Factiva, 
and Google News sources for use in unit and integration tests.
"""

import polars as pl
from typing import Dict, List, Any

# Sample ACLED data
SAMPLE_ACLED_DATA = [
    {
        "event_id": "SSD12345",
        "event_date": "2024-01-15",
        "country": "Sudan",
        "admin1": "Khartoum",
        "admin2": "Khartoum",
        "location": "Khartoum",
        "latitude": 15.5007,
        "longitude": 32.5599,
        "event_type": "Violence against civilians", 
        "sub_event_type": "Attack",
        "actor1": "Military forces of Sudan (2019-)",
        "actor2": "Civilians (Sudan)",
        "fatalities": 12,
        "notes": "Government forces attacked civilian protesters in central Khartoum, resulting in multiple casualties.",
        "source": "Reuters; BBC",
        "iso3": "SDN"
    },
    {
        "event_id": "SSD12346",
        "event_date": "2024-01-16",
        "country": "Sudan",
        "admin1": "Darfur",
        "admin2": "North Darfur",
        "location": "El Fasher",
        "latitude": 13.6274,
        "longitude": 25.3459,
        "event_type": "Battles",
        "sub_event_type": "Armed clash",
        "actor1": "Sudan Liberation Movement/Army (SLM/A)",
        "actor2": "Rapid Support Forces (Sudan)",
        "fatalities": 8,
        "notes": "Clashes between SLM/A and RSF forces near El Fasher resulted in casualties on both sides.",
        "source": "Sudan Tribune",
        "iso3": "SDN"
    }
]

# Sample Factiva data
SAMPLE_FACTIVA_DATA = [
    {
        "title": "Sudan Peace Talks Resume in Jeddah",
        "content": "Peace negotiations between Sudanese military factions resumed in Jeddah today, with mediators from Saudi Arabia and the United States facilitating discussions. The talks aim to establish a ceasefire and humanitarian corridors.",
        "source": "Reuters",
        "publication_date": "2024-01-20",
        "country_codes": ["SD"],
        "doc_id": "REUT_20240120_001"
    },
    {
        "title": "Humanitarian Crisis Deepens in Sudan",
        "content": "The ongoing conflict in Sudan has displaced over 3 million people according to UN estimates. International organizations are struggling to provide aid amid the security challenges.",
        "source": "BBC World Service",
        "publication_date": "2024-01-18",
        "country_codes": ["SD"],
        "doc_id": "BBC_20240118_002"
    }
]

# Sample Google News data
SAMPLE_GOOGLE_NEWS_DATA = [
    {
        "title": "Sudan Military Leaders Meet International Mediators",
        "description": "Senior military officials from Sudan's warring factions met with international mediators in an effort to broker peace.",
        "url": "https://example.com/news/sudan-military-talks",
        "published_date": "2024-01-22T10:30:00Z",
        "source": "Al Jazeera",
        "country": "Sudan"
    },
    {
        "title": "Aid Organizations Call for Safe Passage in Sudan", 
        "description": "International aid organizations are calling for guaranteed safe passage to deliver humanitarian assistance to conflict-affected areas in Sudan.",
        "url": "https://example.com/news/sudan-aid-passage",
        "published_date": "2024-01-21T14:15:00Z",
        "source": "Reuters",
        "country": "Sudan"
    }
]

# Sample Knowledge Graph entities for testing
SAMPLE_KG_ENTITIES = [
    {
        "id": "actor_001",
        "name": "Military forces of Sudan (2019-)",
        "type": "Actor",
        "category": "Military",
        "description": "Government military forces of Sudan established after 2019"
    },
    {
        "id": "actor_002", 
        "name": "Rapid Support Forces (Sudan)",
        "type": "Actor",
        "category": "Paramilitary",
        "description": "Paramilitary rapid support forces operating in Sudan"
    },
    {
        "id": "location_001",
        "name": "Khartoum",
        "type": "Location",
        "latitude": 15.5007,
        "longitude": 32.5599,
        "admin_level": "Capital"
    },
    {
        "id": "event_001",
        "name": "Khartoum Civilian Attack",
        "type": "Event",
        "event_type": "Violence against civilians",
        "date": "2024-01-15",
        "fatalities": 12
    }
]

# Sample configuration for testing
SAMPLE_CONFIG = {
    "data_ingestion": {
        "acled": {
            "country": "Sudan",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "limit": 1000
        },
        "factiva": {
            "search_terms": ["Sudan", "conflict", "peace"],
            "max_results": 500
        },
        "google_news": {
            "query": "Sudan conflict",
            "max_results": 100
        }
    },
    "kg_building": {
        "llm_config": {
            "model": "gemini-1.5-flash",
            "temperature": 0.1
        },
        "entity_types": ["Actor", "Event", "Location", "Organization"],
        "relationship_types": ["INVOLVED_IN", "OCCURRED_AT", "PART_OF"]
    },
    "embedder_config": {
        "model_name": "all-MiniLM-L6-v2",
        "dimensions": 384
    }
}

def get_sample_acled_dataframe() -> pl.DataFrame:
    """Return sample ACLED data as a Polars DataFrame."""
    return pl.DataFrame(SAMPLE_ACLED_DATA)

def get_sample_factiva_dataframe() -> pl.DataFrame:
    """Return sample Factiva data as a Polars DataFrame."""
    return pl.DataFrame(SAMPLE_FACTIVA_DATA)

def get_sample_google_news_dataframe() -> pl.DataFrame:
    """Return sample Google News data as a Polars DataFrame."""
    return pl.DataFrame(SAMPLE_GOOGLE_NEWS_DATA)

def get_sample_kg_entities() -> List[Dict[str, Any]]:
    """Return sample knowledge graph entities."""
    return SAMPLE_KG_ENTITIES.copy()

def get_sample_config() -> Dict[str, Any]:
    """Return sample configuration for testing."""
    return SAMPLE_CONFIG.copy()

# Sample GraphRAG query and response for testing
SAMPLE_GRAPHRAG_QUERY = "What are the main actors involved in the conflict in Sudan?"

SAMPLE_GRAPHRAG_RESPONSE = {
    "answer": "The main actors in the Sudan conflict include the Military forces of Sudan (2019-) representing the government, and the Rapid Support Forces (Sudan), a paramilitary group. Other actors include various rebel groups like the Sudan Liberation Movement/Army (SLM/A) and civilian populations affected by the violence.",
    "context": [
        "Government forces attacked civilian protesters in central Khartoum",
        "Clashes between SLM/A and RSF forces near El Fasher", 
        "Peace negotiations between Sudanese military factions"
    ]
}

# Sample evaluation data
SAMPLE_EVALUATION_CLAIMS = [
    {
        "claim": "Government forces attacked civilians in Khartoum on January 15, 2024",
        "evidence_type": "event",
        "confidence": 0.95
    },
    {
        "claim": "Peace talks resumed in Jeddah in January 2024",
        "evidence_type": "diplomatic",
        "confidence": 0.90
    }
]

SAMPLE_EVALUATION_QUESTIONS = [
    {
        "question": "What happened in Khartoum on January 15, 2024?",
        "expected_answer": "Government forces attacked civilian protesters, resulting in 12 fatalities",
        "question_type": "factual"
    },
    {
        "question": "Who are the main parties in the Sudan conflict?",
        "expected_answer": "Military forces of Sudan and Rapid Support Forces are the main parties",
        "question_type": "analytical"
    }
]

class SampleKGData:
    """Sample data for KG building and testing."""
    
    @staticmethod
    def sample_dataframe() -> pl.DataFrame:
        """Generate a sample DataFrame for KG construction testing."""
        data = [
            {
                "id": "doc_001",
                "text": "Violence erupted in Khartoum between government forces and protesters, resulting in multiple casualties.",
                "country": "Sudan",
                "date": "2024-01-15",
                "source": "ACLED",
                "event_type": "Violence against civilians"
            },
            {
                "id": "doc_002", 
                "text": "Clashes between armed groups in Darfur region displaced hundreds of civilians from their homes.",
                "country": "Sudan",
                "date": "2024-01-16",
                "source": "Google News",
                "event_type": "Battle"
            },
            {
                "id": "doc_003",
                "text": "Humanitarian organizations struggle to deliver aid to conflict-affected areas due to security constraints.",
                "country": "Sudan", 
                "date": "2024-01-17",
                "source": "Factiva",
                "event_type": "Other"
            }
        ]
        return pl.DataFrame(data)
    
    @staticmethod
    def sample_acled_data() -> List[Dict]:
        """Return sample ACLED data."""
        return SAMPLE_ACLED_DATA
    
    @staticmethod
    def sample_google_news_data() -> List[Dict]:
        """Return sample Google News data."""
        return SAMPLE_GOOGLE_NEWS_DATA
    
    @staticmethod
    def sample_factiva_data() -> List[Dict]:
        """Return sample Factiva data."""
        return SAMPLE_FACTIVA_DATA
    
    @staticmethod
    def sample_kg_entities() -> List[Dict]:
        """Sample KG entities for testing."""
        return [
            {"name": "Sudan", "type": "Country", "properties": {"iso3": "SDN"}},
            {"name": "Khartoum", "type": "City", "properties": {"country": "Sudan"}},
            {"name": "Darfur", "type": "Region", "properties": {"country": "Sudan"}},
            {"name": "Military forces of Sudan", "type": "Actor", "properties": {"type": "state"}},
            {"name": "Protesters", "type": "Actor", "properties": {"type": "civilian"}}
        ]


class SampleConfigs:
    """Sample configuration data for testing."""
    
    @staticmethod
    def kg_building_config() -> Dict[str, Any]:
        """Sample KG building configuration."""
        return {
            "llm_config": {
                "model_name": "gemini-pro",
                "max_requests_per_minute": 20,
                "model_params": {"temperature": 0.1}
            },
            "embedder_config": {
                "model_name": "all-MiniLM-L6-v2"
            },
            "entity_resolution_config": {
                "use_resolver": True,
                "resolver": "FuzzyMatchResolver",
                "FuzzyMatchResolver_config": {
                    "filter_query": None,
                    "resolve_properties": ["name"],
                    "similarity_threshold": 0.8
                }
            },
            "schema_config": {},
            "prompt_template_config": {"use_default": True},
            "text_splitter_config": {},
            "examples_config": {},
            "dev_settings": {
                "on_error": "IGNORE",
                "batch_size": 1000,
                "max_concurrency": 5
            }
        }
    
    @staticmethod
    def graphrag_config() -> Dict[str, Any]:
        """Sample GraphRAG configuration."""
        return {
            "llm_config": {
                "model_name": "gemini-pro",
                "max_requests_per_minute": 20,
                "model_params": {"temperature": 0.0}
            },
            "rag_template_config": {
                "template": None,
                "system_instructions": None
            }
        }
    
    @staticmethod
    def evaluation_config() -> Dict[str, Any]:
        """Sample evaluation configuration."""
        return {
            "llm_config": {
                "model_name": "gemini-pro",
                "model_params": {"temperature": 0.0}
            },
            "evaluation_params": {
                "max_claims": 50,
                "max_questions": 100,
                "accuracy_threshold": 0.7
            }
        }
    
    @staticmethod
    def indexing_config() -> Dict[str, Any]:
        """Sample indexing configuration."""
        return {
            "vector_index": {
                "name": "embeddings_index",
                "dimensions": 384,
                "similarity_function": "cosine"
            },
            "fulltext_index": {
                "name": "fulltext_index",
                "properties": ["text", "title"]
            }
        }
