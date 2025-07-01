"""
Useful documentation:
- Structured outputs in Gemini: https://ai.google.dev/gemini-api/docs/structured-output
- Usage of BaseModel and RootModel in Pydantic: https://medium.com/@kishanbabariya101/episode-2-understanding-pydantic-models-basemodel-rootmodel-5e94bf2d2e34
"""

# Utilities
import sys
import os
from pathlib import Path

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = Path(__file__).parent  # Get the directory where this script is located
graphrag_pipeline_dir = script_dir.parent.parent  # Get the graphrag_pipeline directory
if graphrag_pipeline_dir not in sys.path:
    sys.path.append(graphrag_pipeline_dir)

import asyncio
from dotenv import load_dotenv
import os
import json
import re
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from library.kg_builder.utilities import GeminiLLM, get_rate_limit_checker
from neo4j_graphrag.generation import RagTemplate
from library.evaluator import ReportProcessor, AccuracyEvaluator
from library.evaluator.schemas import Claims, Questions, EvaluationResults, GraphRAGResults, RewriteSectionResults
from library.graphrag import CustomGraphRAG
from library.graphrag.utilities import escape_lucene_query
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    HybridRetriever,
    HybridCypherRetriever
)
from library.kg_indexer import KGIndexer

# Neo4j and Neo4j GraphRAG imports
import neo4j

def load_configurations():
    """Loads all necessary configuration files."""
    config_files_path = os.path.join(graphrag_pipeline_dir, 'config_files')
    try:
        # Load environment variables from .env file
        load_dotenv(os.path.join(config_files_path, '.env'), override=True)
        # Load evaluation configurations
        with open(os.path.join(config_files_path, 'evaluation_config.json'), 'r') as f:
            evaluation_config = json.load(f)
        # Load KG building configurations (only to ensure that we use the same embedder at all steps)
        with open(os.path.join(config_files_path, 'kg_building_config.json'), 'r') as f:
            kg_building_config = json.load(f)
        # Load KG retrieval configurations
        with open(os.path.join(config_files_path, 'kg_retrieval_config.json'), 'r') as f:
            kg_retrieval_config = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {e.filename}.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in configuration file: {e.msg}.")
    
    return {
        "evaluation_config": evaluation_config,
        "kg_building_config": kg_building_config,
        "kg_retrieval_config": kg_retrieval_config
    }

def get_report_paths(country, reports_output_directory, kg_retrieval_config):
    """Determines which report files to evaluate."""
    
    report_paths = []
    eval_report_path = os.getenv('GRAPHRAG_EVAL_REPORT_PATH')

    # If eval_report_path is provided, use it to find the specific report to evaluate
    if eval_report_path:
        if os.path.isfile(eval_report_path):
            report_paths.append(eval_report_path)
            print(f"Found specific report to evaluate: {eval_report_path}")
            # Try to parse country from filename, e.g., security_report_United_States_{retriever}_timestamp...
            try:
                # Extract the filename from the provided path
                filename = os.path.basename(eval_report_path)

                # Build a regex pattern to match any of the known retriever names,
                # allowing for an optional "Retriever" suffix.
                retriever_keys = kg_retrieval_config.keys()
                # Creates patterns like 'HybridCypher(?:Retriever)?'
                patterns = [key.replace('Retriever', '(?:Retriever)?') for key in retriever_keys]
                retriever_pattern = '|'.join(patterns)
                
                # The regex captures the country part of the filename.
                # It looks for "security_report_", then captures everything (.+?) non-greedily until it finds
                # one of the known retrievers (with or without "Retriever"), followed by a timestamp and the .md extension.
                match = re.match(rf'security_report_(.+?)_(?:{retriever_pattern})_\d{{8}}_\d{{4}}\.md$', filename)
                
                if match:
                    # The country name is the first captured group. 
                    # It will be in 'safe' format (e.g., "United_States").
                    safe_country = match.group(1)
                    # Convert back to original format for display/use.
                    country = safe_country.replace('_', ' ')
                    print(f"Parsed country '{country}' from filename.")
                else:
                    raise ValueError("Filename does not match expected pattern for parsing country.")
            
            except Exception:
                print("Warning: Could not parse country from report filename.")
                # Fallback to the value passed to the function or from the environment
                country = country or os.getenv('GRAPHRAG_COUNTRY')
        
        else:
            print(f"Error: Specified report path does not exist: {eval_report_path}")
            return [], country
    
    # If eval_report_path is not provided, we will look for the latest reports in the output directory
    else:
        if country is None:
            country = os.getenv('GRAPHRAG_COUNTRY')  # Get country from environment variable if not provided
        if not country:
            print("Error: No country specified. Use --retrieval <country> or set GRAPHRAG_COUNTRY.")
            return [], country

        if reports_output_directory is None:
            reports_output_directory = os.getenv('GRAPHRAG_OUTPUT_DIR')  # Get output directory from environment variable if not provided
        if reports_output_directory:
            country_reports_dir = os.path.join(reports_output_directory, country)
        else:  # Fallback to default reports directory
            reports_base_dir = os.path.join(graphrag_pipeline_dir.parent, 'reports')
            country_reports_dir = os.path.join(reports_base_dir, country)

        if not os.path.isdir(country_reports_dir):
            print(f"Error: No reports directory found for '{country}' at {country_reports_dir}")
            return [], country

        report_files = [os.path.join(country_reports_dir, f) for f in os.listdir(country_reports_dir) if f.endswith('.md')]
        if not report_files:
            print(f"Error: No markdown reports found in {country_reports_dir}")
            return [], country

        # If country is specified, create a sanitized name for the country in the 
        # same way as done for GraphRAGConstructionPipeline
        # Sanitize country name for filesystem
        safe_country = re.sub(r'[^\w\-]', '_', country)  # Replace any non-word characters with underscores
        for retriever, config in kg_retrieval_config.items():
            if config.get('enabled', False):
                # Find all reports in the directory for the current retriever
                retriever_reports = [f for f in report_files if os.path.basename(f).startswith(f'security_report_{safe_country}_{retriever}_')]
                
                if retriever_reports:
                    # The timestamp format YYYYMMDD_HHMM allows finding the latest by simple string comparison
                    latest_report = max(retriever_reports)
                    report_paths.append(latest_report)
                    print(f"Found latest report for '{retriever}': {latest_report}")
                else:
                    print(f"Warning: No reports found for enabled retriever '{retriever}' in {country_reports_dir}")

    if not report_paths:
        print(f"Error: No reports found to evaluate for the specified country and enabled retrievers.")

    return report_paths, country

def initialize_components(configs, gemini_api_key):
    """Initializes all required components like LLMs, evaluators, etc."""

    evaluation_config = configs['evaluation_config']
    kg_building_config = configs['kg_building_config']

    # ----- 3.1. Load evaluation configuration -----

    # Evaluators and Processors
    report_processor = ReportProcessor(pattern=evaluation_config['section_split']['split_pattern'])
    acc_evaluator = AccuracyEvaluator(
        base_claims_prompt=evaluation_config['accuracy_evaluation']['base_claims_prompt'],
        base_questions_prompt=evaluation_config['accuracy_evaluation']['base_questions_prompt']
    )

    # ----- 3.2. LLM configuration -----

    # Initialize LLM for claims
    llm_claims_config = evaluation_config['accuracy_evaluation']['llm_claims_config']
    llm_claims_params = llm_claims_config.get('model_params', {}).copy()
    llm_claims_params['response_schema'] = Claims  # Set the response schema for the LLM claims model
    llm_claims = GeminiLLM(
            model_name=llm_claims_config['model_name'],
            google_api_key=gemini_api_key,
            model_params=llm_claims_params
        )
    
    # Initialize LLM for questions
    llm_questions_config = evaluation_config['accuracy_evaluation']['llm_questions_config']
    llm_questions_params = llm_questions_config.get('model_params', {}).copy()
    llm_questions_params['response_schema'] = Questions  # Set the response schema for the LLM questions model
    llm_questions = GeminiLLM(
            model_name=llm_questions_config['model_name'],
            google_api_key=gemini_api_key,
            model_params=llm_questions_params
        )

    # Initialize LLM with GraphRAG configurations
    llm_graphrag_config = evaluation_config['graphrag']['llm_config']
    llm_graphrag_params = llm_graphrag_config.get('model_params', {}).copy()
    llm_graphrag_params['response_schema'] = GraphRAGResults  # Set the response schema for the LLM GraphRAG model (list of dictionaries, with each dictionary containing a question and its answer)
    llm_graphrag = GeminiLLM(
        model_name=llm_graphrag_config['model_name'],
        google_api_key=gemini_api_key,
        model_params=llm_graphrag_params
    )

    # Initialize LLM for evaluation
    llm_evaluator_config = evaluation_config['accuracy_evaluation']['llm_evaluator_config']
    llm_evaluator_params = llm_evaluator_config.get('model_params', {}).copy()
    llm_evaluator_params['response_schema'] = EvaluationResults  # Set the response schema for the LLM evaluation model
    llm_evaluator = GeminiLLM(
        model_name=llm_evaluator_config['model_name'],
        google_api_key=gemini_api_key,
        model_params=llm_evaluator_params
    )

    # Initialize LLM for rewriting
    llm_rewriter_config = evaluation_config['rewrite_config']['llm_rewriter_config']
    llm_rewriter_params = llm_rewriter_config.get('model_params', {}).copy()
    llm_rewriter_params['response_schema'] = RewriteSectionResults  # Set the response schema for the LLM rewriter model
    llm_rewriter = GeminiLLM(
        model_name=llm_rewriter_config['model_name'],
        google_api_key=gemini_api_key,
        model_params=llm_rewriter_params
    )

    # Initialize LLM for aggregation of rewritten report
    llm_aggregator_config = evaluation_config['rewrite_config']['llm_aggregator_config']
    llm_aggregator_params = llm_aggregator_config.get('model_params', {}).copy()
    llm_aggregator = GeminiLLM(
        model_name=llm_aggregator_config['model_name'],
        google_api_key=gemini_api_key,
        model_params=llm_aggregator_params
    )

    # ----- 3.3. Set LLM requests per minute limit -----

    # Get rate limits for all LLM usages
    claims_rpm = evaluation_config['accuracy_evaluation']['llm_claims_config'].get('max_requests_per_minute', 20)
    questions_rpm = evaluation_config['accuracy_evaluation']['llm_questions_config'].get('max_requests_per_minute', 20)
    graphrag_rpm = evaluation_config['graphrag']['llm_config'].get('max_requests_per_minute', 20)
    evaluator_rpm = evaluation_config['accuracy_evaluation']['llm_evaluator_config'].get('max_requests_per_minute', 20)
    rewriter_rpm = evaluation_config['rewrite_config']['llm_rewriter_config'].get('max_requests_per_minute', 20)
    aggregator_rpm = evaluation_config['rewrite_config']['llm_aggregator_config'].get('max_requests_per_minute', 20)

    # Use the lowest rate limit for all calls to be safe
    min_rpm = min(claims_rpm, questions_rpm, graphrag_rpm, evaluator_rpm, rewriter_rpm, aggregator_rpm)
    
    # Subtract 20% for safety, as Google does not guarantee exact rate limits
    safe_rpm = round(min_rpm - min_rpm * 0.2)
    print(f"Applying a global rate limit of LLM requests of {safe_rpm} requests per minute.")
    
    # Get the rate limit checker function
    check_rate_limit = get_rate_limit_checker(safe_rpm)

    # ----- 3.4. Retrieval and GraphRAG configuration -----

    # GraphRAG components
    rag_template = RagTemplate(
        template=evaluation_config['graphrag']['rag_template_config'].get('template', None),  # Use custom template if specified, otherwise use default
        expected_inputs=['query_text', 'context', 'examples'],  # Define expected inputs for the template
        system_instructions=evaluation_config['graphrag']['rag_template_config'].get('system_instructions', None),  # Use custom system instructions if specified, otherwise use default
    )
    
    # Initialize embedder RAG
    embedder = SentenceTransformerEmbeddings(
        model=kg_building_config['embedder_config']['model_name']
    )

    return {
        "report_processor": report_processor,
        "acc_evaluator": acc_evaluator,
        "llm_claims": llm_claims,
        "llm_questions": llm_questions,
        "llm_graphrag": llm_graphrag,
        "llm_evaluator": llm_evaluator,
        "llm_rewriter": llm_rewriter,
        "llm_aggregator": llm_aggregator,
        "check_rate_limit": check_rate_limit,
        "rag_template": rag_template,
        "embedder": embedder
    }

def initialize_retrievers(driver, configs, embedder):
    """Initializes retriever classes."""
    evaluation_config = configs['evaluation_config']
    
    # ---------- 4.1. Initialization of classes dependent on driver ----------

    indexer = KGIndexer(driver=driver)
    try:
        existing_indexes = indexer.list_all_indexes()
        embeddings_index_name = [index['name'] for index in existing_indexes if index['type'] == 'VECTOR'][0]
        fulltext_index_name = [index['name'] for index in existing_indexes if index['type'] == 'FULLTEXT'][0]
    except IndexError:
        raise ValueError("No vector and/or fulltext indexes found. Please create them before running.")

    # Set default retrieval query for retrievers that do graph traversal
    default_retrieval_query = """
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
    """
    
    retriever_classes = {
        "VectorRetriever": VectorRetriever(
                driver=driver,
                index_name=embeddings_index_name,  # Name of the vector index that will be used for similarity search with the embedded query text
                embedder=embedder,  # Embedder to use for embedding the query text when doing a vector search
                return_properties=evaluation_config['retrievers']['VectorRetriever'].get('return_properties', ['text'])  # Properties to return from the vector search results, apart from the similarity scores (cosine similarity scores by default). Returns the 'text' property by default, which is the text of the document in the knowledge graph.
            ),
        "VectorCypherRetriever": VectorCypherRetriever(
                driver=driver,
                index_name=embeddings_index_name,  # Name of the vector index that will be used for similarity search with the embedded query text
                retrieval_query=evaluation_config['retrievers']['VectorCypherRetriever'].get('retrieval_query', default_retrieval_query), # Cypher query to retrieve the context surrounding the embeddings that are found for the results
                embedder=embedder  # Embedder to use for embedding the query text when doing a vector search
            ),
        "HybridRetriever": HybridRetriever(
                driver=driver,
                vector_index_name=embeddings_index_name,  # Name of the vector index that will be used for similarity search with the embedded query text
                fulltext_index_name=fulltext_index_name,  # Name of the fulltext index that will be used for text search
                embedder=embedder,  # Embedder to use for embedding the query text when doing a vector search
                return_properties=evaluation_config['retrievers']['HybridRetriever'].get('return_properties', ['text'])  # Properties to return from the vector search results, apart from the similarity scores (cosine similarity scores by default). Returns the 'text' property by default, which is the text of the document in the knowledge graph.
            ),
        "HybridCypherRetriever": HybridCypherRetriever(
                driver=driver,
                vector_index_name=embeddings_index_name,  # Name of the vector index that will be used for similarity search with the embedded query text
                fulltext_index_name=fulltext_index_name,  # Name of the fulltext index that will be used for text search
                retrieval_query=evaluation_config['retrievers']['HybridCypherRetriever'].get('retrieval_query', default_retrieval_query), # Cypher query to retrieve the context surrounding the embeddings that are found for the results
                embedder=embedder  # Embedder to use for embedding the query text when doing a vector search
            )        
    }
    return retriever_classes

async def main(country: str = None, reports_output_directory: str = None, accuracy_output_directory: str = None):
    """Main function to run the evaluator pipeline."""
    
    # ==================== 1. Setup ====================
    configs = load_configurations()
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    # ========== 2. Get report(s) to evaluate ==========
    report_paths, country = get_report_paths(country, reports_output_directory, configs['kg_retrieval_config'])
    if not report_paths:
        return

    # ========== 3. Class initialization for report assessment ==========
    components = initialize_components(configs, gemini_api_key)
    acc_evaluator = components['acc_evaluator']
    check_rate_limit = components['check_rate_limit']
    llm_usage = 0  # Variable to track LLM usage for billing purposes

    # ========== 4. Create dictionary of claims and questions to verify claims ==========
    
    with neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:
        
        retriever_classes = initialize_retrievers(driver, configs, components['embedder'])

        for report_path in report_paths:

            print(f"\n=== Evaluating Report: {os.path.basename(report_path)} ===\n")

            # Extract each section as different dictionary entries
            sections = components['report_processor'].get_sections(file_path=report_path)  # sections: Dict[str, str] (key is section title, value is section content)

            # Iterate over each retriever class and initialize it
            for retriever_name, retriever_instance in retriever_classes.items():
                
                if not configs['evaluation_config']['retrievers'][retriever_name].get('enabled', False):
                    continue  # Skip retrievers that are not enabled in the configuration
                
                print(f"Processing with retriever: '{retriever_name}'")

                all_sections_results = []  # List to store results for all sections

                # Iterate over each title and content pair in the sections dictionary
                for section_title, section_content in sections.items():
                    
                    # Avoid evaluating 'Sources' and 'References' sections
                    if section_title.lower() in ['sources', 'references']:
                        continue  # Skip these sections
                    
                    print(f"Processing section: {section_title}")
                    print(f"First 30 characters of section content: {section_content[:30]}...")  # Debugging output
                    
                    try:
                        # From each section, extract claims and questions using the LLMs
                        # Result will be a dictionary with claims (keys) and questions (values).
                        # We will call the LLM twice here (one for claims, one for questions).
                        check_rate_limit()  # Check for claims extraction
                        check_rate_limit()  # Check for questions extraction
                        claims_and_questions = acc_evaluator.get_claims_and_questions_one_section(
                            section_text=section_content,
                            llm_claims=components['llm_claims'],
                            llm_questions=components['llm_questions'],
                            structured_output=True
                        )
                        # This should output a dictionary where each key is 
                        # a claim and the value is a list of questions related 
                        # to that claim (e.g., {'claim_1': ['question_1', 'question_2'], 'claim_2': ['question_1'], ...})
                        llm_usage += 2  # Increment LLM usage for claims and questions extraction for each section (2 calls: one for claims and one for questions)
                    except Exception as e:
                        print(f"Error extracting claims/questions from '{section_title}': {e}")
                        continue
                    
                    print(f"Extracted {len(claims_and_questions)} claims and questions from section '{section_title}'.")
                    print(f"First claim: {list(claims_and_questions.keys())[0] if claims_and_questions else 'No claims found'}")
                    print(f"First question for first claim: {claims_and_questions[list(claims_and_questions.keys())[0]][0] if claims_and_questions and list(claims_and_questions.keys()) else 'No questions found'}")

                    section_claims_list = []  # List to store claims for the current section

                    # ========== 5. Execute GraphRAG pipeline to extract answers for each question ==========

                    # The GraphRAG class is too rigid for this use case, 
                    # as it uses the same `query_text`  both the retriever 
                    # and the LLM prompt. The retriever needs a concise 
                    # search query (the claim, which will be embedded), 
                    # while the LLM needs a  detailed prompt with instructions.
                    # Therefore, we will manually orchestrate the RAG steps: 
                    # retrieve, format prompt with the retrieved context and
                    # finally prompting the LLM with the retrieved context
                    # and the final question in order to generate the answers.
                    # This is all integrated within our CustomGraphRAG class,
                    # created following the code of the GraphRAG class of 
                    # neo4j-graphrag.

                    graphrag = CustomGraphRAG(
                        llm=components['llm_graphrag'],  # LLM for generating answers
                        retriever=retriever_instance,  # Retriever for fetching relevant context 
                        prompt_template=components['rag_template']  # RAG template for formatting the prompt
                    )

                    # We will run GraphRAG for each claim and its associated questions
                    # This will run the LLM for as many times as claims in the report
                    # (e.g., 40 times if there are 40 claims in the report)
                    for claim, questions in claims_and_questions.items():
                        check_rate_limit()  # Check and enforce rate limit before the GraphRAG call
                        safe_claim = escape_lucene_query(claim)  # Sanitize the claim text for use in the query (e.g., escape special characters for Lucene queries)
                        formatted_query_text = configs['evaluation_config']['graphrag']['query_text'].format(
                            claim=claim, 
                            questions=questions
                        )
                        
                        try:
                            # As with neo4j's GraphRAG class, if return_context is
                            # set to True, the `graphrag_results` will be a
                            # dictionary with 2 keys: `answer` and `retriever_result`,
                            # with the context extracted by the retriever
                            graphrag_results = graphrag.search(
                                search_text=safe_claim,  # Search query for the retriever (i.e., the claim)
                                query_text=formatted_query_text,  # User question that is used to search the knowledge graph (i.e., vector search and fulltext search is made based on this question); defaults to empty string if not provided
                                message_history=None,  # Optional message history for conversational context (omitted for now)
                                examples=configs['evaluation_config']['graphrag'].get('examples', ''),  # Optional examples to guide the LLM's response (defaults to empty string)
                                retriever_config=configs['evaluation_config']['retrievers'][retriever_name].get('search_params', None),  # Configuration for the search parameters of the input retriever
                                return_context=True,  # Whether to return the context used for generating the answer (defaults to True). Can be obtained with graphrag_results['retriever_result']
                                structured_output=True  # Whether to return the output in a structured format
                            )

                            llm_usage += 1  # Increment LLM usage by 1 for GraphRAG search
                            
                            # Get the generated answer from the GraphRAG results
                            generated_answers_obj = graphrag_results.answer  # The generated answer is structured as a list of dictionaries, each containing a question (inside the "question" key) and its answer (inside the "answer" key).
                            
                            # The generated_answers_obj.results is a list of Pydantic objects.
                            # We now convert this list to a dictionary where the keys are the questions and the values are the answers.
                            # Example: from a list of objects like [GraphRAGResultsBase(question='Q1', answer='A1', source='S1'), ...]
                            # to a dictionary like {'Q1': ['A1', 'S1'], ...}
                            generated_answers = {}
                            if hasattr(generated_answers_obj, 'results'):  # Access the results attribute of the GraphRAGResults object
                                generated_answers = {
                                    item.question: [item.answer, item.source]  # Store the answer and source in a list 
                                    for item in generated_answers_obj.results 
                                    if hasattr(item, 'question') and hasattr(item, 'answer') and hasattr(item, 'source')
                                }

                            # Create the claim data structure
                            claim_data = {
                                "claim": claim,
                                "questions": generated_answers
                            }

                            section_claims_list.append(claim_data)

                        except Exception as e:
                            print(f"Error during GraphRAG search for claim '{claim[:30]}...': {e}")

                    # Create the section data structure
                    if section_claims_list:
                        all_sections_results.append(
                            {"title_section": section_title, 
                             "claims": section_claims_list}
                        )

                # The results are now stored in all_sections_results, which is a list of dictionaries
                # Each dictionary contains the section title and a list of claims with their questions and answers
                # Sample output structure:
                # [{'title_section': 'section_1', 'claims': [{'claim': 'claim_text', 'questions': {'question_1': ['answer_1', 'source_1], ...}}, ...]}, ...]
                print(f"Completed processing for retriever '{retriever_name}' with {len(all_sections_results)} sections.")
                print("Resulting dictionary of claims, questions, answers and sources for the first section:", all_sections_results[0] if all_sections_results else "No sections found.")

                # ========== 6. Evaluate claims, format, and save the report ==========
                
                # Evaluation is done individually for each of the sections in the report
                if all_sections_results:
                    print(f"Evaluating claims for report: {Path(report_path).name} with {retriever_name}")
                    
                    previously_true_claims = []  # List to store claims evaluated as "true"
                    
                    # Iterate through sections and claims to evaluate each one
                    for section_data in all_sections_results:

                        for claim_data in section_data.get("claims", []):

                            # Add context from previously true claims to the evaluation
                            if previously_true_claims:
                                true_claims_str = "\n".join(f"- {claim}" for claim in previously_true_claims)
                            else:
                                true_claims_str = "No previously verified true claims."
                            
                            # Format the base evaluation prompt with the previously true claims
                            # This will be used to provide context for the evaluation
                            base_eval_prompt_template = configs['evaluation_config']['accuracy_evaluation']['base_eval_prompt']   # Get the base evaluation prompt for the accuracy evaluation from the configuration files
                            try:
                                base_eval_prompt = base_eval_prompt_template.replace("{previously_true_claims}", true_claims_str)  # Use replace instead of format to avoid errors with missing keys
                            except KeyError as e:
                                raise KeyError(f"Missing key in base_eval_prompt: {e}. Please ensure the prompt is correctly formatted with all required placeholders.")
                            
                            # Check and enforce rate limit before the evaluation call
                            check_rate_limit()

                            eval_result = acc_evaluator.evaluate_one_claim(
                                llm_evaluator=components['llm_evaluator'],
                                claim_text=claim_data["claim"],
                                questions_and_answers=claim_data["questions"],  # Here we will pass the dictionary with all of the questions and the corresponding answers and sources associated with a claim
                                base_eval_prompt=base_eval_prompt,
                                structured_output=True
                            )
                            llm_usage += 1  # Increment LLM usage for each claim evaluation
                            claim_data.update(eval_result)  # Update the claim_data with the "justification" and "conclusion" fields
                            # If the claim is true, add it to the list for context in subsequent evaluations
                            if eval_result.get("conclusion") == "true":
                                previously_true_claims.append(claim_data["claim"])

                    print("Formatting accuracy report...")
                    report_content = acc_evaluator.format_accuracy_report(
                        evaluated_data=all_sections_results, 
                        country=country,
                        retriever_type=retriever_name
                    )
                    
                    print("Saving accuracy report...")
                    acc_evaluator.save_accuracy_report(
                        report_content=report_content, 
                        original_report_path=report_path
                    )

                    # Rewrite original report with accuracy evaluation if enabled
                    if configs['evaluation_config']['rewrite_config'].get('enabled', True):
                        
                        # Access the sources section in the original report
                        sources_section = ""
                        for title, content in sections.items():
                            if title.lower() in ['sources', 'references']:
                                sources_section = content
                                break
        
                        # We get a dictionary of sections with their titles (keys) and content (values)
                        # All sections of the accuracy report that refer to the sections
                        # of the original report are formatted as heading 2
                        sections_accuracy = components['report_processor'].get_sections(file_content=report_content)

                        corrected_sections = []  # List to store the rewritten sections

                        # Iterate over each title and content pair in the sections dictionary
                        # of the original report. Since the output for the accuracy evaluation
                        # is structured, the order of the sections and naming in the 
                        # accuracy report is preserved.
                        for section_title, section_content in sections.items():
                            
                            # Check if the section title exists in the accuracy report sections
                            # This is to ensure that we only process sections that have been evaluated
                            # and that the section titles match between the original report and the accuracy report
                            if section_title in sections_accuracy.items():

                                accuracy_content = sections_accuracy[section_title]  # Access the content of the section in the accuracy report
                                
                                # ----- Step 1: Per-Section Rewrite -----
                                
                                rewrite_prompt_template = configs['evaluation_config']['rewrite_config']['rewrite_prompt']
                                
                                # Populate the prompt with the 4 placeholders
                                rewrite_prompt = rewrite_prompt_template.format(
                                    section_title=section_title,
                                    original_content=section_content,
                                    accuracy_content=accuracy_content,
                                    report_sources=sources_section
                                )

                                check_rate_limit()
                                response_rewritten_section = components['llm_rewriter'].invoke(rewrite_prompt)
                                llm_usage += 1
                                
                                if response_rewritten_section:
                                    response_rewritten_section_json = response_rewritten_section.parsed
                                    # The output here will be a dictionary with 3 items:
                                    # - "title_section": the title of the section
                                    # - "corrected_content": the rewritten content of the section
                                    # - "sources": list of dictionaries of the sources section of the report, where each dictionary has 'number' and 'full_source' keys
                                    corrected_sections.append({response_rewritten_section_json})
                        
                        print(f"Rewritten {len(corrected_sections)} sections based on accuracy evaluation.")
                        print("First rewritten section content:", corrected_sections[0] if corrected_sections else "No sections rewritten")

                        # ----- Step 2: Aggregate Rewritten Sections into new report -----
                        
                        if corrected_sections:
                            continue  # Skip aggregation if no sections were rewritten
                        
                        print("Aggregating rewritten sections into a new report...")
                        
                        # Load the original report content to include the initial metadata in the rewritten report
                        with open(report_path, 'r', encoding='utf-8') as f:
                            original_report_content = f.read()

                        # Format intermediate report (in markdown) into a single string
                        intermediate_report = acc_evaluator.format_intermediate_corrected_report(
                            corrected_sections=corrected_sections,  # List of dictionaries with rewritten sections
                            original_report_content=original_report_content  # Original report content to include the initial metadata in the rewritten report
                        )

                        print("Intermediate report content ready for aggregation.")
                        print("First 200 characters of the intermediate report content:", intermediate_report[:100])

                        # Save the intermediate report to a file if enabled
                        if configs['evaluation_config']['rewrite_config'].get('save_intermediate_report', True):
                            print("Saving intermediate report with rewritten sections...")
                            acc_evaluator.save_intermediate_report(
                                report_content=intermediate_report,
                                original_report_path=report_path
                            )
                            print("Intermediate report saved successfully.")

                        # ----- Step 3: Aggregate Rewritten Sections into new report -----

                        print("Making the final report coherent by aggregating sources...")

                        aggregation_prompt_template = configs['evaluation_config']['rewrite_config']['aggregation_prompt']
                        aggregation_prompt = aggregation_prompt_template.format(intermediate_report=intermediate_report)

                        check_rate_limit()
                        final_report_response = components['llm_aggregator'].invoke(aggregation_prompt)
                        llm_usage += 1

                        if final_report_response:
                            final_report_content = final_report_response.content.text  # Get the final report content from the LLM response
                            # Save the final, corrected report
                            acc_evaluator.save_corrected_report(
                                report_content=final_report_content,
                                original_report_path=report_path
                            )
                            print("Final corrected report saved successfully.")

                else:
                    print(f"No results generated for report {report_path}. Skipping evaluation.")

    print(f"\n=== Estimated number of LLM requests: {llm_usage} ===\n")

if __name__ == "__main__":
    # Execute main function when script is run directly
    try:
        result = asyncio.run(main())
        print(f"Evaluation completed.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)