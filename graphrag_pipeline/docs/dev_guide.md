# Knowledge Graph Pipeline for Conflict Analysis

This repository contains an end-to-end pipeline for building and querying a knowledge graph from various data sources related to conflict and security events. It uses the `neo4j-graphrag` library to perform Retrieval-Augmented Generation (RAG) on the constructed graph, enabling complex queries about security situations.

## Architecture Overview

The pipeline is designed with a modular architecture, separating concerns into distinct components. This allows for easier maintenance, configuration, and extension.

-   **`main.py`**: The main entry point of the application. It parses command-line arguments to determine which parts of the pipeline to run.
-   **`application.py`**: The core orchestrator. The `Application` class initializes configurations and calls the appropriate pipeline scripts based on the command-line arguments.
-   **`pipeline/`**: Contains the scripts for each major step of the process:
    -   `01_data_ingestion/`: Scripts to fetch data from sources like ACLED.
    -   `02_kg_building/`: Scripts to process ingested data and construct the knowledge graph.
    -   `03_indexing`: Script to create vector and full-text indexes in Neo4j for efficient retrieval.
    -   `04_ex_post_resolver`: Script for running entity resolution after the graph is built.
    -   `05_graphrag/`: The main logic for the GraphRAG process, including report generation.
    -   `06_evaluation/`: Scripts for evaluating the factual accuracy of the generated reports.
-   **`library/`**: Contains reusable, high-level abstractions:
    -   `data_ingestor/`: Includes an abstract class to scrape Google News data and a date range converter.
    -   `kg_builder/`: Includes the `CustomKGPipeline` which is a tailored version of the `neo4j-graphrag` pipeline.
    -   `kg_indexer/`: A helper class to manage Neo4j graph indexes.
    -   `graphrag/`: Includes a `CustomGraphRAG` class, based on the `GraphRAG` class of the library `neo4j-graphrag`, but with extended functionalities and customization (separation of search query and the final user query and possibility to create structured output).
    -   `evaluator/`: Includes abstract classes that serve as the backbone for the accuracy evaluation of the generated reports.
-   **`config_files/`**: Holds all JSON configuration files and the `.env` file for secrets. This is the primary interface for users to customize the pipeline's behavior without changing code.
-   **`reports/`**: The default output directory where the generated markdown reports are saved.

## How to Use

### 1. Prerequisites

-   Python 3.12 (recommended)
-   Install dependencies: `pip install -r requirements.txt` or through the `uv.lock` if using `uv` as package manager.
-   A Google Gemini API key. You can get a free one [here](https://aistudio.google.com/app/apikey).
-   A Neo4j database. A free cloud-hosted instance from [Neo4j Aura](https://neo4j.com/product/auradb/) is recommended. A self-hosted instance is also possible but may require minor code adjustments.
-   Google Chrome installed to generate the `.svg` plots (through `plotly` and `kaleido`) included in the report.
-   If data ingestion is performed with the sources included by default (Google News, ACLED and Factal), an API key for both ACLED (which can be obtained for free [here](https://acleddata.com/register/)) and [Factal](https://www.factal.com/) (paid tier) are needed.

### 2. Configuration

1. Navigate to the `graphrag_pipeline/config_files/` directory.

2. Create a `.env` file by copying the template or creating a new one.

3. Fill in your credentials:

   ```env
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   GEMINI_API_KEY=your_gemini_api_key
   # Optional API keys for Factal and ACLED only needed if data ingestion is implemented
   ACLED_API_KEY=your_acled_api_key
   ACLED_EMAIL=your_acled_email
   FACTAL_API_KEY=your_factal_api_key
   ```

4. Review the `.json` configuration files (`data_ingestion_config.json`, `kg_building_config.json`, etc.) to customize the pipeline's behavior. For a detailed explanation of each parameter, see the [`config_files_guide.md`](config_files_guide.md).

   > **Important**: All of the values from the JSON configuration files can be adjusted. Nevertheless, **the keys should not be modified under any circumstance**, as the pipeline expects the existence of some keys with particular names. See the section on KG Building for more details on schema configuration.

### 3. Running the Pipeline

The pipeline is executed from the root of the `graphrag_pipeline` directory via `main.py`. You can control which steps are executed using command-line flags.

```bash
# To get help on the available arguments
python main.py --help

# To run the full pipeline: ingest data, build KG, resolve entities ex-post, generate a report for Sudan and create an accuracy evaluation report and a refined report
python main.py --ingest-data "Sudan" --build-kg "Sudan" --resolve-ex-post --retrieval "Sudan" --accuracy-eval

# To only ingest data for a country or a list of countries
python main.py --ingest-data "Sudan" "India" "United States"

# To only build the knowledge graph (assumes data is already ingested) for a country or a list of countries
python main.py --build-kg "Sudan" "India" "United States"

# To generate a report for multiple countries (assumes KG is built and indexed - indexing is automatically done when building the KG)
python main.py --retrieval "Sudan" "UAE"

# To generate a report and save it in a specific output directory
python main.py --retrieval "Sudan" --output-dir "/home/pablo/Downloads"

# To generate a report and then evaluate its accuracy and create a factually corrected report
python main.py --retrieval "Sudan" --accuracy-eval

# To evaluate a specific, existing report
python main.py --accuracy-eval "reports/Sudan/security_report_Sudan_HybridCypher_20250630_120000.md"
```

> **Try out the pipeline without ACLED and Factal API keys**: the repository already includes some sample parquet files with text data for a 3-month period for Factal and ACLED, and for 1 week for Google News (obtained in July 2025), for India, Sudan and the United States. To test the pipeline, you can directly use this data to build a knowledge graph, without having to ingest data previously.

### 4. Pipeline Outputs

## Pipeline Outputs

There are 3 main products that are produced:

1. The first, most resource-efficient product is the **initial country security report**. This report is obtained through GraphRAG and Google Gemini API, i.e., by retrieving contextual information both from text contained in our knowledge graph but also from graph properties (like nodes and edges) and then passing the context to the LLM. The report is generated in markdown format (so it’s totally editable ex-post), and it has 4 sections: a general overview of the country security situation, a section with the key security events in a country, a forward-looking section with ConflictForecast and ACLED’s predictions at subnational level as well as a description of the most relevant subnational events (if available) and, finally, a section with the sources of the report (as extracted from the knowledge graph).
2. The second product is an **accuracy evaluation report** for each and every claim made in the initial report. Each claim is classified either as true, false or mixed, and includes a justification and the sources for the answer (if available). Context retrieval in this case is semantically much more precise (each claim and question is used to retrieve data from the knowledge graph), thus producing more accurate justifications and referencing. Show claim 8 of the first section of the accuracy report.
3. Our last and most refined product (but also most resource-intensive, both in terms of time and LLM requests) is a **corrected version of the initial security report**, where the claims that were detected as false and mixed are modified as well as the sources are adjusted. 

## Pipeline Steps in Detail

### 1. Data Ingestion

-   **Process**: Fetches data from sources defined in `config_files/data_ingestion_config.json`.
-   **Scripts**: Located in `graphrag_pipeline/pipeline/01_data_ingestion/`.
-   **Output**: Standardized data files saved in the `data/` folder.

### 2. Knowledge Graph Building

-   **Process**: Takes the ingested data, extracts entities and relationships using an LLM according to a defined schema, and populates the Neo4j database.
-   **Core Logic**: `CustomKGPipeline` and `KGConstructionPipeline`.
-   **Configuration**: `config_files/kg_building_config.json` controls the LLM, embedding model, text splitting, schema, and prompts.

#### KG Construction Components

```mermaid
flowchart LR

    A --> B[Data Loader]
    B --> C[Text Splitter]
    C --> D[Chunk Embedder]
    D --> E[Entity Relation Extractor]
    E --> F[KG Writer]
    F --> G
    F --> I[Entity Resolver]
    H[Schema Builder] --> E
    I --> G

    subgraph "<b>Document<b>"
        direction LR
        A[Input Text]
    end

    subgraph "<b>Graph Database<b>"
        direction LR
        G[Neo4j]
    end           
```

A Knowledge Graph (KG) construction pipeline requires a few components, most of which are implemented in our `CustomKGPipeline`:

-   **Data loader**: Extracts text from source files.
-   **Text splitter**: Splits text into smaller chunks manageable by the LLM context window.
-   **Chunk embedder**: Computes embeddings for each text chunk.
-   **Schema builder**: Provides a schema to ground the LLM-extracted entities and relations.
-   **Lexical graph builder**: Builds the graph of `Document` and `Chunk` nodes and their relationships.
-   **Entity and relation extractor**: Extracts relevant entities and relations from the text via an LLM.
-   **Knowledge Graph writer**: Saves the entities and relations to the Neo4j database.
-   **Entity resolver**: Merges similar entities into a single node.


#### Why a Custom Pipeline?

The default `SimpleKGPipeline` from `neo4j-graphrag` was not sufficient for two main reasons:

1.  **Metadata Handling**: It lacks robust support for creating `Document` nodes with rich metadata from tabular data. Our `CustomKGPipeline` uses a `LexicalGraphBuilder` to link text chunks to a parent `Document` node that stores this metadata.
2.  **Entity Resolution**: The default resolver (`SinglePropertyExactMatchResolver`) is too conservative. We use the `SpaCySemanticMatchResolver` to merge entities based on semantic similarity, which is more effective for real-world data.

### 3. Knowledge Graph Indexing

-   **Process**: Creates vector and full-text indexes on the `Chunk` nodes in the knowledge graph. This is crucial for efficient similarity searches and keyword lookups during the retrieval phase.
-   **Script**: `pipeline/03_indexing/`
-   **Why index?**:
    -   **Vector indexes** are needed for retrievers that use the numerical representation (embeddings) of text to find the most semantically similar information to a query.
    -   **Text indexes** are useful for retrievers that perform keyword searches on the raw text of the ingested data (e.g., the `HybridCypherRetriever`).

### 4. GraphRAG Query & Report Generation

```mermaid
---
config:
  flowchart:
    subGraphTitleMargin:
      top: 10
      bottom: 10
  look: neo
  layout: dagre
---
flowchart LR
 subgraph s1["<b>Neo4j<b></b></b>"]
        A["Knowledge Graph"]
  end
 subgraph s2["<b>Google Gemini<b></b></b>"]
        n7["LLM"]
  end
 subgraph s3["<b>Context<b></b></b>"]
    direction LR
        n12["Vector embedding"]
        n13["Text"]
        n14["Graph properties"]
  end
    n2["Search prompt"] -.-> n1["Retriever"]
    n2 --> n4["Embedding"]
    n1 --> n12
    n1 -.-> n13
    n4 --> n1
    n6["Report generation prompt"] --> n5["Prompt Formatting"]
    n5 --> n7
    n7 --> n8["Report"]
    n10["Examples"] -.-> n5
    n11["ACLED CAST"] --> n5
    n12 --> n14
    n13 --> n14
    s3 --> n5
    s1 <--> n1
    n16["ACLED CAST and 
    ConflictForecast"] --> n8
    n13@{ shape: rect}
    n2@{ shape: rect}
    n6@{ shape: rect}
    n10@{ shape: rect}
    n11@{ shape: rect}
    n16@{ shape: rect}
```

-   **Process**: This is the final step where the system answers a user's query. It uses a retriever to fetch relevant context from the KG, which is then passed to an LLM to generate a coherent, evidence-based answer.
-   **Core Logic**: `GraphRAGConstructionPipeline`.
-   **Configuration**: `config_files/kg_retrieval_config.json` and `config_files/graphrag_config.json`.
-   **Output**: A detailed markdown report saved in the `reports/` directory.

#### Retrievers

The pipeline supports several retrieval strategies to fetch context from the knowledge graph:

| Retriever                 | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| **VectorRetriever**       | Performs a similarity search using the vector index.         |
| **VectorCypherRetriever** | Extends vector search by running a configurable Cypher query to fetch more context around the matched nodes. |
| **HybridRetriever**       | Combines both vector and full-text search for more robust retrieval. |
| **HybridCypherRetriever** | Same as HybridRetriever with a retrieval query similar to VectorCypherRetriever. |
| **Text2Cypher**           | Translates the natural language question directly into a Cypher query to be run against the graph. |

### 5. Accuracy Evaluation

```mermaid
flowchart LR
 subgraph subGraph1["<b>Final Output<b></b></b>"]
    direction LR
        G["Corrected report"]
  end
 subgraph s2["<b>Claim Generation<b></b></b>"]
        n1["Claim 1"]
        n2["..."]
        n4["Claim C"]
  end
 subgraph s3["<b>Question Generation<b></b></b>"]
        n5["Question 1"]
        n6["..."]
        n7["Question Q"]
  end
 subgraph s4["<b>Claim Assessment<b></b></b>"]
        n12["Assessment of claim"]
        n14["Assessment of claim"]
  end
    n1 --> n5 & n6 & n7
    n5 --> n8["GraphRAG"]
    n8 --> n9["Answer"]
    n7 --> n10["GraphRAG"]
    n10 --> n11["Answer"]
    n9 --> n12
    n11 --> n12
    n4 --> n13["..."]
    n13 --> n14
    n12 --> n15["Accuracy report"]
    n14 --> n15
    n15 --> G
    A["Original Report"] --> n1 & n2 & n4

    n1@{ shape: rect}
    n2@{ shape: rect}
    n4@{ shape: rect}
    style s4 fill:#C8E6C9
    style s2 fill:#BBDEFB
    style s3 fill:#FFCDD2
    style subGraph1 fill:#E1BEE7
```

-   **Process**: Evaluates the factual accuracy of a generated report. It extracts claims from the report, generates verification questions, and queries the knowledge graph to find supporting or refuting evidence. Optionally, it also rewrites the initial report using the accuracy evaluation report, which produces a more factually accurate and refined version, with better referencing (but it is highly resource-intensive in terms of LLM requests).
-   **Core Logic**: `AccuracyEvaluator`.
-   **Configuration**: `config_files/evaluation_config.json`.
-   **Output**: A detailed markdown file(s) with the evaluation results for each claim, and overall factual accuracy conclusions. 

## Common Issues & Troubleshooting

If you encounter unexpected errors, check for these common issues:

-   **No Internet Connection**: The pipeline requires an internet connection to access the Gemini API and potentially other remote resources.
-   **Neo4j Instance Inactive or Inexistent**: 
    -   Ensure your Neo4j Aura instance is existent, active and not paused ("a Free tier instance is considered inactive when there have been no write queries for 3 days", and "a paused Free Instance will be deleted after 30 days, and you won't be able to restore/recover its data" [source](https://support.neo4j.com/s/article/16094506528787-Support-resources-and-FAQ-for-Aura-Free-Tier)). 
    -   If self-hosting, make sure the database is running.
-   **CUDA errors with `torch`** when building the knowledge graph and embedding input texts. Consider using a CPU for better stability (performance should not be downgraded significantly since embedders are just used out-of-the-box).
-   **Gemini API Rate Limits**: The free tier of the Gemini API has rate limits (e.g., tokens per minute, requests per day). Long-running processes or large datasets can exceed these limits, causing errors. Check the current limits [here](https://ai.google.dev/gemini-api/docs/rate-limits#free-tier). Usage of the API can be tracked and checked in [Google AI Studio](https://aistudio.google.com/usage).
-   **Exceeded Tier Limitations of Neo4j Aura Instance**: for the free tier, up to 200,000 nodes and 400,000 relationships (edges) can be stored ([source](https://support.neo4j.com/s/article/16094506528787-Support-resources-and-FAQ-for-Aura-Free-Tier)).

## How to Contribute

To maintain code quality and consistency, please follow these guidelines depending on the type of change you are making.

### Level 1: Configuration Files (`.json`, `.env`)

-   **What**: This is the primary way to customize a pipeline run. It involves changing values in the configuration files to tweak performance, prompts, or models.
-   **Use Case**: Adjusting LLM temperature, changing chunk size, modifying prompt templates, providing few-shot examples.
-   **Rule**: You can change any *value*, but **do not change the keys**.

### Level 2: Implementation Scripts (`pipeline/`)

-   **What**: These scripts contain parameters that are not meant to be tweaked constantly (e.g., batch sizes, specific resolver choices). They define the steps of a pipeline run.
-   **Use Case**: Adding a new data ingestion source, changing the logic of how data is processed for KG building, or altering the report formatting.

### Level 3: Library Scripts (`library/`, `application.py`, `main.py`)

-   **What**: These classes and functions form the basic structure for the entire application. They should not require regular tweaking.
-   **Use Case**: Changing the fundamental pipeline flow, altering the `CustomKGPipeline`'s core components, or adding new command-line arguments. This level requires a deep understanding of the architecture.

## Further work

Here are some of the functionalities that we did not have time to implement but that could be considered to improve the pipeline results:
1. **Data streaming**: generate a cache system for the ingested and KG building functionalities in order to make the program idempotent, so that running the same program twice will only have an effect if there is new data to ingest or from which to update the knowledge graph. 
2. **Knowledge graph updating automation**: create a system by which the knowledge graph is periodically updated.
3. **Containerization with Docker**.
4. **Extended evaluation of the report**: consider evaluating the format, tone and style of the report with LLM-as-a-judge. Furthermore, consider evaluating the GraphRAG results with the Python [`ragas`](https://docs.ragas.io/en/stable/) library.
5. **Product improvement with human feedback**: consider the insights from field experts in order to improve the final products. 

## References

-   [Neo4j GraphRAG Python Library User Guide](https://neo4j.com/docs/neo4j-graphrag-python/current/index.html)
-   [Neo4j GraphRAG Python Library API Documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html)
-   [User Guide: Knowledge Graph Builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)
-   [User Guide: RAG](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)
-   [Examples from the `neo4j-graphrag-python` library](https://github.com/neo4j/neo4j-graphrag-python/tree/main/examples)
