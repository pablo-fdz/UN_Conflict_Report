# Configuration Files Guide

This guide provides a detailed explanation of each configuration file used in the pipeline. These files allow you to customize the pipeline's behavior without modifying the source code.

> **Important**: All of the values from the JSON configuration files can be adjusted. Nevertheless, **the keys should not be modified under any circumstance**, as the pipeline code expects the existence of some keys with particular names.

## `data_ingestion_config.json`

This file controls which data sources are ingested and used for building the knowledge graph.

### acled

Configuration for the ACLED data source (structured, high-quality data but with limited perspectives).

#### *ingestion*

`true` to download and process data from this source, `false` to skip it. 

*Trade-offs*: 
- Advantages: Ingesting more sources provides a richer knowledge base.
- Disadvantages: Ingesting more sources increases processing time and storage requirements.

*Recommended value*: if storage constraints are not an issue, set to `true`.

#### *include_in_kg*

`true` to use the ingested data from this source during the knowledge graph building phase. 

*Trade-offs*: 

- Advantages: Including more data in the KG can improve the comprehensiveness of reports.
- Disadvantages: including more data increases the complexity and cost of KG construction and querying.

*Recommended value*: if Neo4j storage constraints are not an issue, set to `true` to improve the richness of the knowledge graph.

### factal

Configuration for the Factal data source (high-quality event compilation, but with limited perspectives).

#### *ingestion* 

Same as for ACLED data source.

#### *include_in_kg* 

Same as for ACLED data source.

### google_news

Configuration for the Google News data source (highly unstructured data, potentially less trustworthy, but rich in diversity).

#### *ingestion* 

Same as for ACLED data source.

#### *include_in_kg* 

Same as for ACLED data source.

### Most Impactful Parameters

The `ingestion` and `include_in_kg` flags for each data source are the key parameters here. They directly determine the scope and content of your knowledge graph.

## `kg_building_config.json`

This file configures every aspect of the knowledge graph construction process, including **text processing**, **named entity recognition** (NER) and **entity resolution**.

### text_splitter_config

Defines how input documents are divided into smaller chunks for embedding and doing named entity recognition on the resulting chunks.

#### *chunk_size*

The maximum number of characters (*not* tokens - tokens can be assimilated to a word) for each text chunk. 

*Trade-offs*: 
- Larger chunks provide more context to the LLM can be less precise (due to positional bias) for embedding, named entity recognition and retrieval. 
- Smaller chunks require doing more LLM requests (which could increase the bill if pricing is based on the number of requests) and more focused but may lose important context that spans across chunks. 
- Consider also the embedder and LLM (for NER) context window for setting this parameter (common embedders have between 256 to 512 tokens of context window, while modern LLMs have more than 1M tokens of context window). The number of characters of regular news articles can span between 5,000-20,000 characters.

*Recommended value*: set a value high enough so that the whole input text can be included in a text chunk (e.g., 1M characters). Since input texts are relatively short (at most, long news articles), positional bias is not really an issue here, and embeddings can capture most of the meaning of an article with just the first few hundred tokens. If you want to split articles into more than one chunk, set a relatively low value (at most 5,000 characters).

#### *chunk_overlap*

The number of characters from the previous chunk to overlap with each chunk. Must be less than `chunk_size`. 

*Trade-offs*: Higher overlap helps preserve context across chunk boundaries but increases the total amount of data to process and store.

*Recommended value*: if `chunk_size` is large enough, this parameter will not have any impact. On the other hand, if `chunk_size` is low, consider setting this parameter to 10% of the chunk size to preserve context.

### embedder_config

Specifies the model used to create numerical representations (embeddings) of text chunks.

#### *model_name*

The name of the SentenceTransformer model to use. 

*Trade-offs*: There's a balance between model quality, speed, context window size and the LLM's processing cost.
- Splitting articles increases the number of calls to the entity extraction LLM, which can be costly and hit rate limits.
- Embedding models like `all-MiniLM-L6-v2` have small context windows (e.g., 256 tokens), which may require splitting a single news article into multiple chunks to consider the whole input text for embedding.
- While the most important information in news is often at the beginning, larger context windows can capture more nuance. Consider these [`SentenceTransformer` models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models), all of which work out of the box within the pipeline:
    - `all-mpnet-base-v2`: Best quality, 384 token limit.
    - `all-distilroberta-v1`: Faster, 512 token limit.
    - `all-MiniLM-L6-v2`: Fastest, good quality, 256 token limit.
- To capture the whole input text for embedding with large text chunks, an alternative is Google's `text-embedding-004`, which is free (with rate limits) and supports up to 2,048 tokens. In that case, however, compatibility is not guaranteed.

*Recommended value*: use `all-MiniLM-L6-v2`. Performance is very fast, quality is comparable to those of the best embedding models and the 256 token context window is more than enough to capture the main meaning of a news article.

> Note that this same embedder will also be used (for consistency and ensuring comparability) at the graph retrieval step (see [`graphrag_config.json`](#graphrag_configjson)), for embedding the search query (`search_text`); but also at the accuracy evaluation step (see [`evaluation_config.json`](#evaluation_configjson)) for embedding the search query of fact-checking. Therefore, make sure that the context window of the chosen embedder covers the length of both of these search queries.

### llm_config

Configures the Large Language Model used for extracting entities and relationships (edges) from the text chunks.

#### *model_name*

The identifier for the LLM (e.g., `gemini-2.5-flash`). The pipeline works out-of-the-box with [Google Gemini models](https://ai.google.dev/gemini-api/docs/models) (as they are of high quality and their API has a [free tier](https://ai.google.dev/gemini-api/docs/rate-limits#free-tier)), but the code could be easily adjusted to work with other providers.

*Trade-offs*: 
- More powerful models may yield more accurate extractions but are slower and more expensive. 
- Lighter models are faster and cheaper but might miss nuances.

*Recommended value*: consider choosing a high-quality model for this step (e.g., `gemini-2.5-flash` on June 2025), as the quality of the entity extraction will have a large impact across the whole pipeline. Furthermore, structured output is needed at this step, which is more ensured with high-quality models.

#### *model_params*

Parameters to control the LLM's behavior, like `temperature`. A `temperature` of `0.0` makes the output more deterministic and is recommended for extraction tasks.

The parameters must be included in a dictionary-like structure:

```json
"model_params": {
    "temperature": 0.0,
    "response_mime_type": "application/json",
    "max_output_tokens": 1000
}
```

All of the available parameters can be found [here](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig), and an explanation of the most common parameters can be found [here](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters).

*Recommended values*: at least set the `temperature` to a low value for more deterministic results and set `response_mime_type` to `"application/json"`, as the outputs of the LLM for entity extraction will be needed to be in JSON format in order to populate the neo4j graph database. More information about structured outputs with Google Gemini can be found [here](https://ai.google.dev/gemini-api/docs/structured-output) (note that in this case it is not possible to pass a schema).

#### *max_requests_per_minute*

Rate limit to avoid exceeding API quotas. 

*Recommended value*: set to the real rate limit for the corresponding model (check this [link](https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits) for the Google Gemini rate limits), as the code already implements some safety checks to avoid exceeding the maximum requests per minute and includes retry logic when requests fail.

> Check the usage of Gemini models in [Google AI Studio](https://aistudio.google.com/usage) if generation fails.

### schema_config

Defines the structure (nodes, edges, properties) of the knowledge graph.

#### *create_schema*

If `true`, the pipeline will attempt to create the defined schema in the Neo4j database. If `false`, the LLM will structure the knowledge graph based on its own criteria.

*Trade-offs*:
- If set to `true`, the structure of the knowledge graph is more deterministic and organized.
- If set to `false`, flexibility increases but the graph can become highly unorganized, with the LLM potentially creating new node and edge types each time it is called to do NER.

*Recommended value*: set to `true` and suggest a schema (see below). 

#### *suggest_pattern*

If set to `true`, the triplets will also be suggested when creating the graph from the schema. If `create_schema` is set to `false`, setting `suggest_pattern` to `true` will not have any effect.

*Trade-offs*:
- If `true`, the graph will be even more deterministic, with the relationships between the pre-defined entities already established.
- If `false`, flexibility will improve, but the LLM may come up with triplets which do not make sense.

*Recommended value*: set to `true` and suggest triplets (see below). 

#### *nodes*, *edges*, *triplets*

These arrays define the allowed entity types, relationship types, and the valid connections between them. This is the blueprint for your graph. 

*Trade-offs*:
- A more detailed schema can capture more specific information but makes the extraction task harder for the LLM. 
- A simpler schema is easier to populate but may be less expressive.

`nodes`, `edges` and `triplets` need to have the following structure:

```json
{
    "schema_config": {
        "nodes": [
            {"label": "Event", "description": "...", "properties": [
                {"name": "name", "type": "STRING", "description": "..."},
                {"name": "date", "type": "DATE"}
            ]},
            ...
        ],
        "edges": [
            {"label": "OCCURRED_IN", "description": "...", "properties": [
                {"name": "start_date", "type": "DATE"}
            ]},
            ...
        ],
        "triplets": [
            ["Event", "OCCURRED_IN", "Country"],
            ...
        ]
    }
}
```
Possible property types are: `BOOLEAN`, `DATE`, `DURATION`, `FLOAT`, `INTEGER`, `LIST`, `LOCAL DATETIME`, `LOCAL TIME`, `POINT`, `STRING`, `ZONED DATETIME`, and `ZONED TIME` (source: [neo4j](https://neo4j.com/docs/cypher-manual/current/values-and-types/property-structural-constructed/)).

*Suggested schema*:

```json
"nodes": [
    {"label": "Event", 
    "description": "Significant occurrences of the input text, such as conflicts, elections, coups, attacks or any other relevant information.",
    "properties": [
        {"name": "name", "type": "STRING"},
        {"name": "start_date", "type": "DATE", "description": "Date when the event started or when the information was first reported."},
        {"name": "end_date", "type": "DATE", "description": "Date when the event ended or when the information was last updated."},
        {"name": "type", "type": "STRING", "description": "Type of event, e.g., Conflict, Attack, Election."}
    ]},
    
    {"label": "Actor", 
    "description": "All kinds of entities mentioned, such as terrorist groups, political parties, military, individuals, etc.",
    "properties": [
        {"name": "name", "type": "STRING"},
        {"name": "type", "type": "STRING", "description": "Type of actor, e.g., civilian, military, government, international organization, etc."}
    ]},
    
    {"label": "Country", 
    "description": "Nation states mentioned in the story, like the United States, Sudan, Afghanistan, etc. Do not include in this category territories within countries.",
    "properties": [
        {"name": "name", "type": "STRING"}
    ]},

    {"label": "ADM1", 
    "description": "First-level administrative division within countries, like US states (e.g., California), provinces in Iran (e.g., Semnan) or regions in Ghana (e.g., Ashanti).",
    "properties": [
        {"name": "name", "type": "STRING"}
    ]},

    {"label": "Location",
    "description": "Particular geographical location of higher granularity than national (country) or first-level administrative divisions (ADM1), such as cities, towns, or specific sites (e.g., streets, buildings, squares, etc.)..",
    "properties": [
        {"name": "name", "type": "STRING"}
    ]}
    
],
"edges": [
    {"label": "HAPPENED_IN", 
    "description": "Indicates where (geographically) an event took place."},
    
    {"label": "CONFRONTED_WITH", 
    "description": "Negative connection between actors/territories and other actors/territories, such as conflicts, attacks, simple criticism (without violence) or other forms of confrontation.",
    "properties": [
        {"name": "type", "type": "STRING", "description": "Type of confrontation, e.g., conflict, attack, criticism, trade war, etc."},
        {"name": "start_date", "type": "DATE", "description": "Date when the confrontation started."},
        {"name": "end_date", "type": "DATE", "description": "Date when the confrontation ended or was last reported."}
    ]},
    
    {"label": "COOPERATED_WITH", 
    "description": "Cooperative (positive) relationship between actors/territories and other actors/territories, such as alliances, partnerships, or other forms of cooperation.",
    "properties": [
        {"name": "type", "type": "STRING", "description": "Type of cooperation, e.g., alliance, partnership, trade agreement, etc."},
        {"name": "start_date", "type": "DATE", "description": "Date when the cooperation started."},
        {"name": "end_date", "type": "DATE", "description": "Date when the cooperation ended or was last reported."}
    ]},

    {"label": "PARTICIPATED_IN", 
    "description": "Actor or country's involvement in an event",
    "properties": [
        {"name": "role", "type": "STRING", "description": "Role of the actor in the event, e.g., victim, perpetrator, participant, etc."}
    ]},
    
    {"label": "IS_FROM", 
    "description": "Physical location of actors within countries, first-level territorial divisions or locations",
    "properties": [
        {"name": "since", "type": "DATE", "description": "Date when the actor was first reported to be in this location."},
        {"name": "until", "type": "DATE", "description": "Date when the actor was last reported to be in this location."}
    ]},
    
    {"label": "IS_WITHIN",
    "description": "Indicates that a geographical location is part of a larger geographical entity, such as a city being within a country, a first-level territorial division being within a country or a building being within a city.",
    "properties": [
        {"name": "since", "type": "DATE", "description": "Date when the location was first reported to be within the larger geographical entity."},
        {"name": "until", "type": "DATE", "description": "Date when the location was last reported to be within the larger geographical entity."}
    ]}
],
"triplets": [

    ["Event", "HAPPENED_IN", "Location"],
    ["Event", "HAPPENED_IN", "ADM1"],
    ["Event", "HAPPENED_IN", "Country"],

    ["Actor", "CONFRONTED_WITH", "Actor"],
    ["Actor", "CONFRONTED_WITH", "Country"],
    ["Actor", "CONFRONTED_WITH", "ADM1"],
    ["Actor", "CONFRONTED_WITH", "Location"],
    ["Country", "CONFRONTED_WITH", "Actor"],
    ["Country", "CONFRONTED_WITH", "Country"],
    ["Country", "CONFRONTED_WITH", "ADM1"],
    ["Country", "CONFRONTED_WITH", "Location"],
    ["ADM1", "CONFRONTED_WITH", "Actor"],
    ["ADM1", "CONFRONTED_WITH", "Country"],
    ["ADM1", "CONFRONTED_WITH", "ADM1"],
    ["ADM1", "CONFRONTED_WITH", "Location"],
    ["Location", "CONFRONTED_WITH", "Actor"],
    ["Location", "CONFRONTED_WITH", "Country"],
    ["Location", "CONFRONTED_WITH", "ADM1"],
    ["Location", "CONFRONTED_WITH", "Location"],

    ["Actor", "COOPERATED_WITH", "Actor"],
    ["Actor", "COOPERATED_WITH", "Country"],
    ["Actor", "COOPERATED_WITH", "ADM1"],
    ["Actor", "COOPERATED_WITH", "Location"],
    ["Country", "COOPERATED_WITH", "Actor"],
    ["Country", "COOPERATED_WITH", "Country"],
    ["Country", "COOPERATED_WITH", "ADM1"],
    ["Country", "COOPERATED_WITH", "Location"],
    ["ADM1", "COOPERATED_WITH", "Actor"],
    ["ADM1", "COOPERATED_WITH", "Country"],
    ["ADM1", "COOPERATED_WITH", "ADM1"],
    ["ADM1", "COOPERATED_WITH", "Location"],
    ["Location", "COOPERATED_WITH", "Actor"],
    ["Location", "COOPERATED_WITH", "Country"],
    ["Location", "COOPERATED_WITH", "ADM1"],
    ["Location", "COOPERATED_WITH", "Location"],
    
    ["Actor", "PARTICIPATED_IN", "Event"],
    ["Country", "PARTICIPATED_IN", "Event"],
    ["ADM1", "PARTICIPATED_IN", "Event"],

    ["Actor", "IS_FROM", "Country"],
    ["Actor", "IS_FROM", "ADM1"],
    ["Actor", "IS_FROM", "Location"],

    ["ADM1", "IS_WITHIN", "Country"],
    ["Location", "IS_WITHIN", "Country"],
    ["Location", "IS_WITHIN", "ADM1"],
    ["Location", "IS_WITHIN", "Location"]
]
```

#### *enforce_schema*

Whether to enforce and validate the resulting schema resulting from the LLM extraction (`"STRICT"`) or not (`"NONE"`).

*Recommended values*: set to `"NONE"` for improved robustness of the pipeline while consistently structuring the graph with a schema. 

### prompt_template_config

Configures the prompt used to instruct the LLM on how to extract information.

#### *use_default*

If `true`, it uses the default prompt from the `neo4j-graphrag` library. If `false`, it uses the custom `template` provided.

*Recommended value*: set to `false` and define a custom extraction `template` more optimized towards entity extraction for this use case (relevant security events in countries/regions).

#### *template*

A custom prompt template. This gives you fine-grained control over the LLM's extraction behavior.

*Suggested template*:
```
"You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph that will be used for creating security reports for different countries.\n\nExtract the entities (nodes) and specify their type from the following Input text.\nAlso extract the relationships between these nodes. The relationship direction goes from the start node to the end node.\n\nReturn result as JSON using the following format:\n{{\"nodes\": [ {{\"id\": \"0\", \"label\": \"the type of entity\", \"properties\": {{\"name\": \"name of entity\" }} }}],\n\"relationships\": [{{\"type\": \"TYPE_OF_RELATIONSHIP\", \"start_node_id\": \"0\", \"end_node_id\": \"1\", \"properties\": {{\"details\": \"Description of the relationship\"}} }}] }}\n\n- Use only the information from the Input text. Do not add any additional information.\n- If the input text is empty, return empty Json.\n- Make sure to create as many nodes and relationships as needed to offer rich context for generating a security-related knowledge graph.\n- An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions.\n- Multiple documents will be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general.\n- Do not create edges between nodes and chunks when the relationship is not clear enough.\n\nUse only the following nodes and relationships (if provided):\n{schema}\n\nAssign a unique ID (string) to each node, and reuse it to define relationships.\nDo respect the source and target node types for relationship and the relationship direction.\n\nDo not return any additional information other than the JSON in it.\n\nExamples:\n{examples}\n\nInput text:\n{text}"
```

> *Warning*: do NOT modify the information in the prompt that guides the LLM on how to structure the output. This information is essential for avoiding errors when populating the knowledge graph with the extracted entities and relationships.

### examples_config

#### *pass_examples*

Whether to do few-shot learning with the LLM for extracting the entities and structuring the output to populate the knowledge graph.

*Recommended value*: `false`, high-quality LLMs already do a good job even without examples. Passing examples will increase the input tokens and potentially increase LLM billing.

#### *examples*

Examples that will be passed to the LLM (as strings) as a model for extracting entities and configuring the JSON documents that should be used to populate the neo4j knowledge graph. These examples will only be passed if `pass_examples` is set to `true`.

*Suggested examples*: 
```json
"examples": [
    {
        "input_text": "Text: On January 1, 2023, a significant conflict erupted in the Middle East involving multiple countries and organizations. The conflict, named 'Middle East Conflict 2023', lasted until March 15, 2023. Key actors included the 'Middle East Coalition' and the 'Opposing Forces'. The conflict resulted in a high level of destruction and instability in the region.",
        "schema": {
            "nodes": [
                {"id": "0", "label": "Event", "properties": {"name": "Middle East Conflict 2023", "date": "2023-01-01", "end_date": "2023-03-15", "type": "Conflict", "severity": 5, "description": "A significant conflict in the Middle East."}},
                {"id": "1", "label": "Actor", "properties": {"name": "Middle East Coalition", "type": "Organization"}},
                {"id": "2", "label": "Actor", "properties": {"name": "Opposing Forces", "type": "Organization"}},
                {"id": "3", "label": "Region", "properties": {"name": "Middle East", "stability": 0.2}}
            ],
            "relationships": [
                {"type": "OCCURRED_IN", "start_node_id": "0", "end_node_id": "3", "properties": {"start_date": null, "end_date": null, "certainty": 1.0}},
                {"type": "PARTICIPATED_IN", "start_node_id": "1", "end_node_id": "0", "properties": {"role": null, "significance": 1.0, "start_date": null, "end_date": null}},
                {"type": "PARTICIPATED_IN", "start_node_id": "2", "end_node_id": "0", "properties": {"role": null, "significance": 1.0, "start_date": null, "end_date": null}}
            ]
        }
    },
    {
        "input_text": "Text: On February 14, 2023, ...",
        "schema": {
            "nodes": [
                {"id": "0", "label": "Event", "properties": {"name": "February 14 Incident", "date": "2023-02-14", "end_date": null, "type": "Attack", "severity": 4, "description": "An attack occurred on February 14."}}
            ]
        }
    }
]
```

### entity_resolution_config

Configures how the pipeline merges duplicate or similar entities. Nodes from the lexical graph (text chunks and document metadata) are *not* merged.

#### *use_resolver*

If `true`, enables the entity resolution step (if `resolver` is set to a valid resolver) and is performed automatically each time after the neo4j knowledge graph is populated. Relationships and properties of the merged nodes are integrated.

*Trade-offs*:
- Depending on the resolver that is used, entities with similar naming (but semantically different) may be merged.
- Simplifies and reduces the size of the knowledge graph.

*Recommended value*: set to `true`.

> Note that the time needed for the resolver to resolve nodes increases exponentially as the size of the knowledge graph increases. This cost is even higher for more extensive resolvers like `SpaCySemanticMatchResolver`.

#### *resolver* 

The name of the resolver algorithm to use. Either `SinglePropertyExactMatchResolver`, `FuzzyMatchResolver` or `SpaCySemanticMatchResolver` are valid options. 

*Recommended values*: `SpaCySemanticMatchResolver` is recommended for its ability to merge entities based on semantic meaning, which is more robust than exact string matching.

More information about the available resolvers can be found [here](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html#entity-resolver).

#### *SinglePropertyExactMatchResolver_config*

Configuration for the `SinglePropertyExactMatchResolver`. This is a simple resolver that merges nodes with the same label and identical `resolve_property` property.

*Parameters*:
- `filter_query`: query that is going to be used to limit the resolver's resolution scope. Must be a Cypher WHERE clause.
- `resolve_property`: the property that will be compared. If values match exactly, entities are merged.

*Recommended values*:
- `filter_query`: set to `null` (as this resolver is very fast and efficient, so the search can be in the whole KG).
- `resolve_property`: set to `name` (nodes with the same value in the `name` property will be merged).

#### *FuzzyMatchResolver_config*

Similarity-based resolver that resolves entities with the same label and similar set of textual properties using RapidFuzz for fuzzy matching.

*Parameters*:
- `filter_query`: query that is going to be used to limit the resolver's resolution scope. Must be a Cypher WHERE clause.
- `resolve_properties`: the list of properties that will be compared. The strings of all of these properties will be concatenated and then embedded. This embedding will be used to determine whether to merge nodes or not based on a `rapidfuzz`'s similarity method.
- `similarity_threshold`: similarity threshold above which nodes are merged (default is 0.8). Higher threshold will result in less false positives, but may miss some matches.

*Recommended values*:
- `filter_query`: if feasible due to the KG size, set to `null`. Alternative: `"WHERE (entity)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(doc:Document {id = 'docId'}"` to merge entities coming from the same document.
- `resolve_properties`: `["name"]` (merge nodes only based on its name).
- `similarity_threshold`: 0.95 (high value to reduce false positives, which are frequent).

#### *SpaCySemanticMatchResolver_config*

A semantic match resolver, which is based on spaCy embeddings and cosine similarities of embedding vectors. This resolver is ideal for higher quality KG resolution using static embeddings.

*Parameters*:
- `filter_query`: query that is going to be used to limit the resolver's resolution scope. Must be a Cypher WHERE clause.
- - `spacy_model`: SpaCy model used to embed the `resolve_properties` (see available *monolingual English* models [here](https://spacy.io/models/en)).
- `resolve_properties`: the list of properties that will be compared. The strings of all of these properties will be concatenated and then embedded. This embedding will be used to determine whether to merge nodes or not based on cosine similarity.
- `similarity_threshold`: similarity threshold above which nodes are merged (default is 0.8). Higher threshold will result in less false positives, but may miss some matches.

*Recommended values*:
- `filter_query`: if feasible due to the KG size, set to `null`. Alternative: `"WHERE (entity)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(doc:Document {id = 'docId'}"` to merge entities coming from the same document.
- `spacy_model`: set to `en_core_web_lg` (largest, best model - the bottleneck here is not the resolver performance).
- `resolve_properties`: `["name"]` (merge nodes only based on its name).
- `similarity_threshold`: 0.95 (high value to reduce false positives, which are frequent).

> Consider the language of the input documents for choosing the resolver method. SpaCy also has [multilingual embedding models](https://spacy.io/models/xx). In case of doubt, choose the most conservative option, `SinglePropertyExactMatchResolver`.

#### *ex_post_resolver*

The resolver to use in the optional ex-post resolution step. `SinglePropertyExactMatchResolver` is a fast and conservative choice for a final cleanup. Ex-post resolution can be called directly using `main.py` from the command line (see [`dev_guide.md`](dev_guide.md) for more information).

### dev_settings

Settings for development and debugging.

#### *build_with_sample_data*

If `true`, the pipeline will run with a small, predefined sample of data, which is useful for quick tests.

*Recommended value*: set to `true` if working to improve the pipeline, set to `false` for production.

#### *on_error*

Error handling strategy for entity extraction (`"IGNORE"` for ignoring the error or `"RAISE"` for raising the error and stopping the procedure).

*Recommended value*: set to `"RAISE"` if working to improve the pipeline, set to `"IGNORE"` for production.

#### *batch_size*

The number of nodes or relationships to write to the database in a batch.

*Trade-offs*: Larger batches can be more efficient but consume more memory.

*Recommended value*: 1000 (neo4j's default).

#### *max_concurrency*

Maximum number of concurrent LLM requests for entity extraction.

*Recommended value*: set to 5 (default). However, if running into LLM requests rate limit issues, consider decreasing this value.

### Most Impactful Parameters

`schema_config` is the most critical parameter as it defines the fundamental structure of your knowledge graph. The `llm_config` and `prompt_template_config` are also highly impactful, as they directly control the quality of the information extracted from your documents. Finally, the choice of `resolver` in `entity_resolution_config` significantly affects the cleanliness and connectivity of the final graph.

## `kg_retrieval_config.json`

This file configures the various "retrievers" that can be used to fetch information from the knowledge graph to answer a query. For each of the retrievers that are `enabled`, a report for the queried country will be generated (e.g., if a report is requested for Sudan and all 5 retrievers are enabled, 5 reports will be generated).

More information on each of the retrievers mentioned below can be found in the [`neo4j-graphrag` user guide](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html#retriever-configuration).

### VectorRetriever

Performs a simple similarity search on text chunk embeddings (regular RAG).

#### *enabled*

Set to `true` to use this retriever.

*Recommended value*: set to `false` (results are significantly better for retrievers that are enhanced with graph traversal, i.e., "Cypher" retrievers).

#### *return_properties*

List of properties of the text chunk nodes to return from the vector search results, apart from the similarity scores (cosine similarity scores by default). 

*Recommended value*: only include `"text"` in the list - together with the similarity score, only the text of the text chunk will be returned. 

#### *search_params*

Parameters that configure the context retrieval from the knowledge graph. For vector-based methods, this only includes `top_k` (integer), which denotes the top *k* properties and similarity scores to return (e.g., if set to 5, it will return the top 5 chunks with the highest cosine similarity between the search query and each of the text chunks in the knowledge graph).

*Trade-offs*:
- Setting a high `top_k` will retrieve more context, at the cost of more potential noise.
- Setting a low `top_k` may miss relevant context to generate the report.

*Recommended value*: set to a relatively high value, such as 20.

### VectorCypherRetriever 

Augments the vector search by running a Cypher query to fetch additional connected data from the graph, providing richer context.

#### *enabled*

Set to `true` to use this retriever.

*Recommended value*: set this retriever *or* the HybridCypherRetriever to `true`, as they are the retrievers that exhibit the highest-quality results.

#### *retrieval_query*

The Cypher query (as a string) to execute for each retrieved chunk to expand its context.

*Default query* (without JSON formatting):
```cypher
//1) Go out 2-3 hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
UNWIND relList AS rel

//2) collect relationships and text chunks
WITH collect(DISTINCT chunk) AS chunks,
    collect(DISTINCT rel) AS rels

//3) format and return context
RETURN '=== text ===\\n' + apoc.text.join([c in chunks | c.text], '\\n---\\n') + '\\n\\n=== kg_rels ===\\n' +
apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\\n---\\n') AS info
```

If you want to extend the number of hops done from each text chunk (e.g., 3-5 hops), the following change should be made: `[relList:!FROM_CHUNK]-{3,5}`.

*Suggested query* with the extraction of document metadata from the text chunks:

```cypher
// 1) Go out 2-3 hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
UNWIND relList AS rel

// 2) Collect chunks and KG relationships
WITH collect(DISTINCT chunk) AS chunks,
     collect(DISTINCT rel) AS kg_rels

// 3) Also pull in each chunk’s Document (optional match to avoid dropping the chunk if it has no FROM_DOCUMENT edge)
UNWIND chunks AS c
OPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(doc:Document)
WITH chunks, kg_rels,
     collect(DISTINCT {domain:doc.domain, url:doc.url, date:doc.date}) AS docs

// 4) Return formatted output
RETURN
  '=== text chunks ===\n'
  + apoc.text.join([c IN chunks | c.text], '\n---\n')
  + '\n\n=== text chunk document metadata ===\n'
  + apoc.text.join([d IN docs |
       d.domain + ' (domain): ' + d.url + ' (URL), ' d.date + '(date)' 
    ], '\n---\n')
  + '\n\n=== kg_rels ===\n'
  + apoc.text.join([r IN kg_rels |
       startNode(r).name
       + ' - ' + type(r)
       + '(' + coalesce(r.details,'') + ')'
       + ' -> ' + endNode(r).name
    ], '\n---\n')
  AS info
```

*Default query* (with proper JSON formatting):
```json
"//1) Go out 2-3 hops in the entity graph and get relationships\nWITH node AS chunk\nMATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()\nUNWIND relList AS rel\n\n//2) collect relationships and text chunks\nWITH collect(DISTINCT chunk) AS chunks,\n collect(DISTINCT rel) AS rels\n\n//3) format and return context\nRETURN '=== text ===\\n' + apoc.text.join([c in chunks | c.text], '\\n---\\n') + '\\n\\n=== kg_rels ===\\n' +\n apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\\n---\\n') AS info"
```

*Suggested query* (with proper JSON formatting):
```json
"// 1) Go out 2-3 hops in the entity graph and get relationships\nWITH node AS chunk\nMATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()\nUNWIND relList AS rel\n\n// 2) Collect chunks and KG relationships\nWITH collect(DISTINCT chunk) AS chunks,\n     collect(DISTINCT rel) AS kg_rels\n\n// 3) Also pull in each chunk’s Document (optional match to avoid dropping the chunk if it has no FROM_DOCUMENT edge)\nUNWIND chunks AS c\nOPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(doc:Document)\nWITH chunks, kg_rels,\n     collect(DISTINCT {domain:doc.domain, url:doc.url, date:doc.date}) AS docs\n\n// 4) Return formatted output\nRETURN\n  '=== text chunks ===\\n'\n  + apoc.text.join([c IN chunks | c.text], '\\n---\\n')\n  + '\\n\\n=== text chunk document metadata ===\\n'\n  + apoc.text.join([d IN docs |\n       d.domain + ' (domain): ' + d.url + ' (URL), ' + d.date + '(date)'\n    ], '\\n---\\n')\n  + '\\n\\n=== kg_rels ===\\n'\n  + apoc.text.join([r IN kg_rels |\n       startNode(r).name\n       + ' - ' + type(r)\n       + '(' + coalesce(r.details,'') + ')'\n       + ' -> ' + endNode(r).name\n    ], '\\n---\\n')\n  AS info"
```

*Recommended query*: set a query that is able to extract all of the necessary context surrounding an event (actors, countries, etc.) as well as the extraction of the sources for proper referencing. Consider the graph schema when setting this query.

#### *search_params*

Same as [VectorRetriever](#vectorretriever) (see above).


### HybridRetriever

Combines vector search (semantic) and full-text search (keyword).

#### *enabled*

Set to `true` to use this retriever.

*Recommended value*: set to `false` (results are significantly better for retrievers that are enhanced with graph traversal, i.e., "Cypher" retrievers).

#### *return_properties*

Same as [VectorRetriever](#vectorretriever) (see above).

#### *search_params*

Parameters that configure the context retrieval from the knowledge graph. 

*Parameters*:
- `top_k`: an integer which denotes the top *k* properties and similarity scores to return (e.g., if set to 5, it will return the top 5 chunks with the highest cosine similarity between the search query and each of the text chunks in the knowledge graph).
  - Setting a high `top_k` will retrieve more context, at the cost of more potential noise.
  - Setting a low `top_k` may miss relevant context to generate the report.
- `ranker`: ranker to use for ranking the results. `"linear"` is a simple linear combination of the vector and text scores, `"naive"` is default value and just combines the scores without weighting them.
- `alpha`: weight for the vector score when using the linear ranker. The fulltext index score is multiplied by (1 - alpha). Required when using the linear ranker; must be between 0 and 1. 

*Recommended values*:
- `top_k`: set to a relatively high value, such as 20.
- `ranker`: `linear`.
- `alpha`: set to 0.5 for equal weighting of vector and full text scores.

### HybridCypherRetriever

The same as `HybridRetriever`, but with an additional Cypher query to expand context, similar to `VectorCypherRetriever`.

#### *enabled*

Set to `true` to use this retriever.

*Recommended value*: set this retriever *or* the VectorCypherRetriever to `true`, as they are the retrievers that exhibit the highest-quality results.

#### *retrieval_query*

See the documentation in [VectorCypherRetriever](#vectorcypherretriever).

#### *search_params*

See the documentation in [HybridRetriever](#hybridretriever).

### Text2CypherRetriever

Translates a natural language question into a Cypher query using an LLM.

#### *enabled*

Set to `true` to use this retriever.

*Recommended value*: set to `false` (results are significantly better for retrievers that are enhanced with graph traversal, i.e., "Cypher" retrievers). This retriever does not use RAG (neither embeddings nor full text search). Its results are quite bad, so it should be avoided.

#### *llm_config*

Configuration for the LLM used for text-to-Cypher translation.

*Recommended values*:
- Fast model (highest quality is not needed).
- Low temperature.

#### *examples_config*

Configuration of the examples passed to the LLM to construct the Cypher queries from natural language. Examples are passed if `include_examples` is set to `true`.

*Suggested examples*:
```json
[
    "USER INPUT: 'What events happened in Sudan?'\nQUERY: MATCH (e:Event)-[:HAPPENED_IN]->(c:Country) WHERE c.name = 'Sudan' RETURN e.name, e.type",
    "USER INPUT: 'Which actors participated in attacks?'\nQUERY: MATCH (a:Actor)-[:PARTICIPATED_IN]->(e:Event) WHERE e.type = 'Attack' RETURN a.name, a.type"
]
```

### Most Impactful Parameters

The `enabled` flag for each retriever determines which strategies are available. The choice of retriever has a massive impact on the quality of the context provided to the final generation LLM. `HybridCypherRetriever` is often a powerful choice as it combines keyword search, semantic search, and graph traversal, but the `retrieval_query` must be well-crafted. Except for the `VectorCypherRetriever`, the rest should be avoided.

## `graphrag_config.json`

This file configures the final step of the pipeline: generating the security report using the context fetched by a retriever.

### llm_config

The LLM used to synthesize the final report from the retrieved context. The structure is the same as in [`kg_building_config.json`](#llm_config).

*Trade-offs*: 
- A more powerful and creative model (e.g., with a higher temperature) can generate more fluent and comprehensive reports, but may be more expensive and risk more hallucinations. 
- A less powerful model might produce more basic reports but will be faster and cheaper.

*Recommended value*: Use a high-quality model (like `gemini-2.5-flash` or better) for this final generation step, as it directly impacts the quality of the end product and it will not be called as many times as in other steps (e.g., entity extraction or evaluation). A low temperature (`0.0`) is recommended to ensure the report is based strictly on the provided context. Finally, consider setting as well the `max_output_tokens` parameter to limit the size of the report.

### rag_template_config

The prompt template for the final report generation.

#### *template*
The main template string that structures the final prompt to the LLM, including placeholders for `{query_text}`, `{context}`, and `{examples}`.

*Recommended value*: leave the template unchanged. The default template is generally sufficient, but can be modified if you need to fundamentally change how context and questions are presented to the LLM. Placeholders for `{query_text}`, `{context}`, and `{examples}` are needed.

```json
"# Question:\n{query_text}\n \n# Context:\n{context}\n \n# Examples:\n{examples}\n \n# Answer:\n"
```

#### *system_instructions*

High-level instructions for the LLM on its role and constraints (e.g., "Answer only using the context provided").

*Recommended value*: Keep instructions clear and concise. The default is a good starting point to prevent the LLM from hallucinating information not present in the retrieved context.

```json
"Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned. If no examples are provided, omit the Examples section in your answer."
```

### search_text

The initial, broad query used to kick off the retrieval process from the knowledge graph. This search text will be used to retrieve the most relevant context, through the cosine similarity of the search text's embedding and (with hybrid retrievers) with full text search. The `{country}` placeholder will be replaced with the target country at runtime. The `{hotspots_regions_list}` will be replaced with a comma-separated-list of ADM1 regions for which the number of violent events is expected to increase.

*Trade-offs*: A broader query retrieves more context, which can lead to a more comprehensive report but also introduces more noise. A narrower query is more focused but might miss relevant tangential information.

*Recommended value*: The default `"Security events, conflicts, and political stability in {country}. Focus on the following conflict hotspots: {hotspot_regions_list}."` is a good balance. Adjust it if you need to focus the report on a more specific topic (e.g., "Economic stability and trade agreements in {country}").

### query_text

The detailed prompt given to the LLM to generate the final report. It instructs the model on the desired structure, tone, and content. This query is *not* used to retrieve relevant context. It must contain the following placeholders: `{country}`, `{current_month_year}`, `{total_hotspots}` and `{hotspot_regions}` placeholders.

*Trade-offs*: 
- A very detailed prompt gives you more control over the output but can be restrictive. 
- A more open-ended prompt allows for more creativity from the LLM but may result in less structured reports.
- Consider as well the context window of the LLM that is used, as well as positional bias.

*Recommended value*: Be as specific as possible in your instructions. The suggested prompt is designed to produce a structured, markdown-formatted security report with citations, which is ideal for the project's goals.

*Suggested query* (structured report):

```json
"Generate a comprehensive security report for {country} based on the provided `Context` below. The report should cover recent events and offer a forward-looking perspective on the country's security situation. Structure the report with a clear focus on key events, their impact, and the actors involved, and ensure the text is coherent, objective and maintains a formal tone suitable for a security report. Format the entire output as a markdown document. Ensure you cite the sources of information used in the report as provided in the `Context`, with footnote-style citations (e.g., [1], [2]). Only cite the sources in the `Sources` section whenever you can cite with the following format (example): [<number>] <domain>: <url>, <date>. If the URL is not available, just cite with the domain and the date. Whenever available, integrate in the text of the report the dates when the events you mention have happened, either through the information provided in the text of the context or through the timestamp of the source for a claim. The markdown report MUST have the following structure, with the heading levels and names (whenever they are not in brackets - []) denoted below: \n# [Title for the security report]\n## 1. Overview\n## 2. Key Security Events\n## 3. Forward Outlook \n### Subnational Perspective\n#### [One heading 4 per subnational conflict hotspot]\n### [Include more heading 3 in this section under your discretion] ## 4. Sources\n In sections 1 (Overview) and 2 (Key Security Events), feel free to include whatever heading 3 titles you see fit with the provided `Context`. Furthermore, in section 2, if there is `Context` which can be linked to United Nations humanitarian operations, include a heading 3 which focuses on topic.\nIn section 3 (Forward Outlook), always include the heading 3 mentioned above (Subnational perspective), with one heading 4 per hotspot, where you explain the key events in each region. If there are no hotspots, create the heading 4 sections you see fit. A subnational ADM1 region is considered a hotspot if the number of violent events is expected to increase by at least 25% in the short term. Based on the data available in {current_month_year}, the predicted number of hotspots is {total_hotspots} in {country}. The name of the hotspot regions, the average number of violent events in the last 3 months, the predicted number of violent events in the short term and the percentage increase in the number of violent events is the following:\n{hotspot_regions}\nIf no hotspots are provided, it means that there are no ADM1 regions that can be classified as such at this time."
```

> The structure of the report indicated in the prompt above should be kept in order to ensure compatibility with the final accuracy evaluation of the report (which relies on RegEx for extracting information based on the heading levels to then analyze claims per section).

The final structure of the markdown report is suggested to be as follows (includes the structured sections that are appended to the report without using LLMs): 

```md
# [Title of the report at LLM discretion]

## 1. Overview

### [Section 1 subsections at LLM discretion]

## 2. Key Security Events

### [Section 2 subsections at LLM discretion]

### [Section 2 subsection on UN humanitarian operations suggested, but at LLM discretion]

## 3. Forward Outlook

### Armed Conflict Probability Forecast (Conflict Forecast)

Short paragraph on [ConflictForecast's armed conflict risk predictions](https://conflictforecast.org/) for that country, together with a time series plot on the probability of armed conflict risk 3 months ahead from 2020 onwards.

### Subnational Perspective

#### Predicted Increase in Violent Events in the Short Term (ACLED)

Bar chart of ACLED's predictions of the regions which are expected to have a change in the number of violent events in the next 3 months.

#### [One heading 4 per hotspot at LLM discretion - if no hotspots, subsections at LLM discretion]

### [Additional heading 3 sections under LLM discretion]

## 4. Sources
```

*Example query* (open-ended):

```json
"Generate a comprehensive security report for {country} based on the provided context. The report should cover recent events from the last year and offer a forward-looking perspective on the country's stability. Structure the report with a clear focus on key events, their impact, and the actors involved. Format the entire output as a markdown document. Ensure you cite the sources of information used in the report as provided in the context, with footnote-style citations (e.g., [1], [2]). Whenever possible, the citations should include the author and platform (if the source is from social media) as well as the URL."
```

### examples

Few-shot examples to guide the LLM's output format.

*Recommended value*: Leave this empty (`""`) if the `query_text` is detailed enough. High-quality models often perform well without examples for generation tasks. If the output format is consistently wrong, you can add a short example of the desired report structure here.

### return_context

If `true`, the context used to generate the report will be saved alongside the report itself.

*Recommended value*: set to `true`. This is invaluable for debugging, verification, and understanding why the LLM included certain information in its report.

### Most Impactful Parameters

`search_text` is essential to determine which information is retrieved from the knowledge graph. `query_text` is the most important parameter here, as it directly instructs the LLM on what kind of report to generate. The `llm_config` also plays a key role in the quality and coherence of the final output.

## `evaluation_config.json`

This file configures the accuracy evaluation pipeline, which fact-checks the generated reports against the content included in the knowledge graph.

### section_split

#### *split_pattern*

A regular expression used to split the generated report into individual sections for evaluation.

*Recommended value*: the default `"(?m)^## (.+?)\\s*\\n(.*?)(?=^## |\\Z)"` is designed to split a markdown report by its level-2 headings (`##`).

### accuracy_evaluation

Contains prompts and configurations for the multi-step evaluation process. 

#### *base_claims_prompt*

The prompt used to instruct an LLM to extract verifiable, atomic claims from a report section. The placeholder `{section_text}` is needed inside the prompt for the accuracy evaluation pipeline to work properly.

*Recommended value*: pass a heavily engineered prompt with examples and notes to guide the LLM in extracting high-quality, self-contained claims. It is recommended to use it as is.

*Suggested prompt*:

```json
"You are an AI tasked with extracting verifiable claims from a section of a report. A verifiable claim is an atomic statement that can be checked for accuracy and is relevant to the topic of the report. Here is an example:\nSection: \"The constant attacks of Group A and Group B on civilians in Country X have led to a high number of casualties as well as internally displaced persons (IDPs). In May 2025, there have been more IDPs than any other month that year.\"\n\nFrom this section, you can extract the following verifiable claims:\n1. \"Group A has been carrying constant attacks on civilians in Country X.\"\n2. \"Group B has been carrying constant attacks on civilians in Country X.\"\n3. \"In May 2025, there have been more IDPs than in any of the previous months of 2025.\"\n\nNotes on this:\n- Claims 1 and 2 have to be separate claims, because they refer to different groups, so one could be true while the other is false and that would make the first sentence incorrect.\n- The claim in the original section about a \"high number of casualties\" is not verifiable, because it is not quantifiable/specific. We do not know what \"high\" means and we do not have a reference to compare it too. Claims that are subjective or vague should not be extracted.\n- It is important that you always make all the necessary information explicit in the claims, without pronouns or terms that make references to previous sentences, so that each claim is self-contained and can be verified independently.\n- Also, when abbreviations of any kind are used, always include, if you know it with certainty, the full name/term plus the abbreviation in parentheses.\n\nYou must extract all verifiable claims from the following section of a report:\nSection: \"{section_text}\""
```

> Warning: do *not* suggest the JSON structure of the output inside the prompt, as a structured is already enforced through a schema behind the scenes.  

#### *base_questions_prompt*

The prompt used to generate specific, answerable questions to verify each extracted claim. The placeholder `{claims_list}` is needed inside the prompt. 

*Recommended value*: pass a prompt that instructs the LLM to act as a journalist and create clear, concise questions.

*Suggested prompt*:

```json
"You are a journalist tasked with evaluating the accuracy of a set of claims against a knowledge base.\nFor the given list of claims below, you must generate 1 to 4 questions aimed at leading you to the information needed to verify each claim.\nEach question should be specific, clear, and concise, designed to have a closed-ended objective answer.\n\nWhen abbreviations of any kind are used in the claim, always include in the question, if you know it with certainty, the full name/term plus the abbreviation in parentheses.\n\nHere is the list of claims:\n{claims_list}."
```

> Warning: do *not* suggest the JSON structure of the output inside the prompt, as a structured is already enforced through a schema behind the scenes.  

#### *base_eval_prompt*

The prompt used to make a final judgment (true, false, or mixed) on a claim based on the answers retrieved from the KG (see the [graphrag configuration for accuracy evaluation](#graphrag)) and previously verified claims. The placeholders `{claim_text}`, `{questions_and_answers_json}`, `{previously_true_claims}` and `{hotspot_regions}` need to be included inside the prompt.

*Recommended value*: this prompt defines the final evaluation logic. Provide examples on how should claims be considered.

*Suggested prompt*:

```json
"You are an expert fact-checker. Your task is to evaluate a claim based on a set of questions and their corresponding answers, as well as a list of previously verified true claims. The answers are generated from a knowledge base. Based on all the information provided, determine if the claim is true, false, or a mixture of true and false.\n\n- **true**: The provided information fully supports the claim. The claim can also be considered true if it can be logically inferred from the previously verified true claims.\n- **false**: The provided information explicitly contradicts the claim.\n- **mixed**: The provided information partially supports the claim, supports some parts but not others, or is insufficient to make a full determination.\n\nHere is the claim and the supporting information:\n\n**Claim to Evaluate:**\n\"{claim_text}\"\n\n**Questions and Answers for the Claim:**\n{questions_and_answers_json}\n\n**Previously Verified True Claims (for context):**\n{previously_true_claims}\n\n Finally, here is the additional context for the forecasts on violent conflicts, with the name of the hotspot regions, the average number of violent events in the last 3 months, the predicted number of violent events in the short term and the percentage increase in the number of violent events:\n{hotspot_regions}\n. This data comes from ACLED Conflict Alert System: https://acleddata.com/conflict-alert-system/, so update accordingly the sources to the questions of the claims related to this data. If the list is empty, there are no forecasts in this case."
```

> Warning: do *not* suggest the JSON structure of the output inside the prompt, as a structured is already enforced through a schema behind the scenes.  

#### *llm_claims_config*, *llm_questions_config*, *llm_evaluator_config*

Separate LLM configurations for each step of the evaluation process (claim extraction, question generation, and final evaluation). The structure is the same as in [`kg_building_config.json`](#llm_config).

*Recommended values*: 
- Use fast and cheap models (like `gemini-2.5-flash-lite-preview-06-17`) for these tasks, as the prompts are heavily structured and do not require deep creativity. Furthermore, a very high number of requests will be done for extracting the claims, generating questions, retrieving answers from the knowledge graph and evaluating the claims. 
- A low temperature (`0.0`) is recommended for the claims and evaluator models to ensure deterministic and objective outputs.
- Set `"response_mime_type"` to `"application/json"`, which is needed to enforce the output structure in all of these steps. 

### retrievers

Configures the retriever used during the evaluation phase to find answers to the verification questions in the knowledge graph. The structure is identical to [`kg_retrieval_config.json`](#kg_retrieval_configjson).

*Recommended value*: Use the same high-performance retriever as in `kg_retrieval_config.json` (e.g., `HybridCypherRetriever` or `VectorCypherRetriever`) to ensure the evaluation has access to the best possible context from the knowledge graph. Set `enabled` to `true` for your chosen retriever, `false` to the others.

### graphrag

Configures the RAG process used to answer the verification questions generated in the evaluation pipeline. The structure is the same as in [`graphrag_config.json`](#graphrag_configjson).

#### *llm_config*

The LLM used to synthesize an answer for each question generated for each claim from the context retrieved for a verification question.

*Recommended values*: 
- Use fast and cheap models (like `gemini-2.5-flash-lite-preview-06-17`) for these tasks, as the prompts are heavily structured and do not require deep creativity. Furthermore, a very high number of requests will be done for extracting the claims, generating questions, retrieving answers from the knowledge graph and evaluating the claims. 
- A low temperature (`0.0`) is recommended for the claims and evaluator models to ensure deterministic and objective outputs.
- Set `"response_mime_type"` to `"application/json"`, which is needed to enforce the output structure in this step. 

#### *rag_template_config*

The prompt template for answering the verification questions.

*Recommended value*: the default template and system instructions are optimized to force the LLM to answer only using the provided context, which is critical for an objective evaluation.

*Suggested template*:

```json
"# Question:\n{query_text}\n \n# Context:\n{context}\n \n# Examples:\n{examples}\n \n# Answer:\n"
```

*Suggested system instructions*:

```json
"Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned. If no examples are provided, omit the Examples section in your answer."
```

#### *query_text*

The prompt that instructs the RAG pipeline on how to use the context to answer the verification questions for a given claim. Needs `{claim}` and `{question}` placeholders in order to work properly. 

It is separated from the `search_text`, which in this case will only be the claim that is being evaluated at each time (an individual claim, so embedders with a short context window - like 256 tokens - should suffice to cover the search text at this step).

*Recommended value*: the prompt should be clear and concise to guide the LLM to answer some questions associated to a claim.

*Suggested prompt*:

```json
"Using the provided `Context` below, answer the following question about the `Claim`. Answer concisely and truthfully, without making assumptions or adding information beyond what is provided in the `Context`. Ensure you ONLY cite the sources of information used in the report as provided in the `Context` (whenever you can cite) with the following format (example): <domain>: <url>, <date>. If the URL is not available, just cite with the domain and the date. If there is not enough information in the `Context` to answer a question, state that by answering with \"Not enough information to answer this question.\". Leave the source empty in that case.\n\nHere is the claim and the question:\nClaim: {claim}\nQuestion: {question}",
```

> Warning: do *not* suggest the JSON structure of the output inside the prompt, as a structured is already enforced through a schema behind the scenes.  

#### *examples*

Few-shot examples to guide the LLM's output format.

*Recommended value*: set examples which show the LLM how to answer a set of questions related to a claim, in a structured manner.

*Suggested examples*:

```json
"Claim: \"The UN has reported a significant increase in the number of internally displaced persons (IDPs) in Country X due to ongoing conflicts.\"\nQuestions: [\"What is the current number of IDPs in Country X?\", \"What are the main causes of the increase in IDPs in Country X?\", \"How does the current situation compare to previous years?\"]\n\nExample output: {\"What is the current number of IDPs in Country X?\": \"Not enough information to answer this question.\", \"What are the main causes of the increase in IDPs in Country X?\": \"Ongoing conflicts and violence.\", \"How does the current situation compare to previous years?\": \"There has been a significant increase compared to previous years.\"}"
```

### rewrite_config

#### *enabled*

If `true`, will rewrite the original report with a corrected version using the generated accuracy report.

*Trade-offs*:
- If set to `true`, false or mixed factual claims of the original report will be corrected. Referencing will potentially improve with respect to the original report. However, some format errors may appear more frequently. 
- If set to `false`, the cost of report generation will be lower (less LLM claims, though this rewriting procedure makes far less requests than the accuracy evaluation procedure). False or mixed claims will not be corrected.

*Recommended value*: set to `true`.

#### *rewrite_prompt*

Prompt that will be used to rewrite each of the heading 2 sections of the original report. Must contain 4 placeholders: `{section_title}`, `{original_content}`, `{accuracy_content}` and `{report_sources}`.

*Recommended value*: create prompt that properly instructs the LLM to 1) read the original content of the section, 2) review it with the claim-by-claim evaluataion, 3) rewrite the original content with the corrected content, and 4) set the referencing format.

*Suggested prompt*:

```json
"You are an expert editor tasked with revising a report section based on a factual accuracy analysis. Your goal is to produce a corrected, well-written, and professionally toned narrative for the final report. \n\n**Instructions:**\n1.  Read the **Original Content** of the section.\n2.  Review the **Accuracy Analysis**, which contains a claim-by-claim evaluation (TRUE, FALSE, MIXED) of the original content, along with justifications and specific sources.\n3.  Rewrite the **Original Content** to create a new **Corrected Content**. You must:\n    -   Keep all information that was verified as TRUE.\n    -   Correct or remove information that was identified as FALSE, using the justification from the analysis.\n    -   Clarify and adjust information that was identified as a MIXED.\n    -   Ensure the final text is coherent, objective, and maintains a formal tone suitable for a security report.\n4.  As you rewrite, you must cite the sources provided in the **Accuracy Analysis** for all claims. Use footnote-style citations (e.g., [1], [2]). Ensure you ONLY cite the sources of information used in the report as provided  (whenever you can cite) with the following format (example): <domain>: <url>, <date>. If the URL is not available, just cite with the domain and the date. \n5.  Compile all of the sources you cited in the rewritten text. \n\n**Inputs:**\n\n**1. Section Title:**\n`{section_title}`\n\n**2. Original Content:**\n`{original_content}`\n\n**3. Accuracy Analysis:**\n`{accuracy_content}`\n\n**4. General Report Sources (for context only):**\n`{report_sources}`"
```

> Warning: do *not* suggest the JSON structure of the output inside the prompt, as a structured is already enforced through a schema behind the scenes.  

#### *aggregation_prompt*

Prompt that will be used to aggregate the section-by-section corrected report into one fluent, coherent report. Should contain an `{intermediate_report}` placeholder.

*Recommended value*: prompt that properly instructs the LLM to create a coherent report out of a markdown report which contains a "Sources" heading in each section, and that indicates the LLM to include the sections `### Armed Conflict Probability Forecast (Conflict Forecast)` and `#### Predicted Increase in Violent Events in the Short Term (ACLED)` as-is.

*Suggested prompt*:

```json
"You are an expert technical editor specializing in consolidating and formatting reports. Your task is to process an intermediate markdown report, consolidate its sources, and produce a final, clean version.\n\n**Input Document Structure:**\nThe provided markdown document contains several sections, each starting with a level-2 heading (`##`). Each section contains:\n1.  Rewritten content with in-text, footnote-style citations (e.g., [1], [2]).\n2.  A subsection with a level-3 heading (`### Sources`) that lists the sources for that section. The numbering of these sources is local to each section and may be duplicated across the document.\n\n**Your Instructions:**\n1.  **Consolidate Sources:** Read all the `### Sources` subsections. Identify all unique sources from the entire document and create a single, consolidated list.\n2.  **Re-number Citations:** Re-number the consolidated sources sequentially starting from 1. Then, go through the main text of each section and update all in-text citations (e.g., `[1]`, `[2]`) to match the new, global numbering of the consolidated source list. Ensure you ONLY cite the sources of information used in the report as provided with the following format (example): <domain>: <url>, <date>. If the URL is not available, just cite with the domain and the date. \n3.  **Final Formatting:**\n    -   Remove all the intermediate `### Sources` subsections. \n   -   Keep the ordering and naming of the rest of the sections as-is. \n    -   Create a single, final `## Sources` section at the very end of the document.\n    -   Populate this final section with the consolidated, de-duplicated, and correctly numbered list of sources.\n    -   Ensure the entire document is coherent and correctly formatted in markdown.\n\n**Input Document:**\n```markdown\n{intermediate_report}\n```\n\n**Output:**\nReturn only the full, corrected markdown text of the final report. Do not include any other commentary or explanation."
```

#### *llm_rewriter_config*

The LLM used to rewrite each of the sections of the report with the conclusions from the accuracy report.

*Recommended values*: 
- Use high-quality models (like `gemini-2.5-flash`) for these rewriting tasks, as we want to avoid mistakes of replacing true information or not replacing false information.
- A low temperature (`0.0`) is recommended for more deterministic and objective outputs.
- Set `"response_mime_type"` to `"application/json"`, which is needed to enforce the output structure in this step. 

#### *llm_aggregator_config*

The LLM used to create a final, coherent markdown report corrected with factual evaluation.

*Recommended values*: 
- Use high-quality models (like `gemini-2.5-flash`) for these rewriting tasks, as we want to avoid mistakes of replacing true information or not replacing false information.
- A low temperature (`0.0`) is recommended for more deterministic and objective outputs.

#### *save_intermediate_report*

If set to `true`, the intermediate result of the corrected report will be saved in a separate folder in the directory of the original report.

*Recommended value*: set to `true` for increased transparency and debugging.

### Most Impactful Parameters

The three base prompts (`base_claims_prompt`, `base_questions_prompt`, `base_eval_prompt`) are crucial as they define the logic of the entire evaluation. The last prompt, `base_eval_prompt`, has an especially large impact on the final accuracy report that is generated. The quality of the evaluation depends heavily on how well these prompts guide the LLMs to perform their specific tasks. The choice of `retrievers` is also critical for finding the correct evidence in the graph.