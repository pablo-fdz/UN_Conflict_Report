import sys
from pathlib import Path
from typing import Dict, List

# Utilities
from neo4j_graphrag.llm import LLMInterface
import json
from datetime import datetime

# Neo4j and Neo4j GraphRAG imports
import neo4j

class AccuracyEvaluator:
    def __init__(self, base_claims_prompt: str, base_questions_prompt: str):

        # PROMPTS
        self.base_claims_prompt = base_claims_prompt
        self.base_questions_prompt = base_questions_prompt

    def _extract_verifiable_claims_one_section(self, llm: LLMInterface, section_text: str, structured_output: bool = False) -> List[str]:
        """
        Extracts verifiable claims from the section text. Returns them as a list of strings.

        Args:
            llm: The language model to use for extraction.
            section_text (str): The text of the section from which to extract claims.
            structured_output (bool): If True, expects the LLM to return structured output (e.g., a Pydantic model with a non-empty .parsed attribute). If False, expects a raw JSON string.
        """
        prompt = self.base_claims_prompt.format(section_text=section_text)
        response = llm.invoke(prompt)

        if structured_output == True:

            # Check if the response has a 'parsed' attribute for structured output
            if hasattr(response, 'parsed') and response.parsed:
                # If using a Pydantic RootModel, the data is in the 'root' attribute
                if hasattr(response.parsed, 'claims'):
                    return response.parsed.claims
                # Fallback for other parsed structures like a direct list
                elif isinstance(response.parsed, list):
                    return response.parsed
            
            else:
                raise ValueError("The LLM response does not contain structured output. Please check the LLM configuration. This documentation explains how to configure the LLM for structured output: https://ai.google.dev/gemini-api/docs/structured-output")

        else:

            # Fallback to parsing the raw JSON content if 'parsed' is not available
            try:
                # The response.content is the raw text from the LLM
                claims_list = json.loads(response.content)
                return claims_list if isinstance(claims_list, list) else []
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, return an empty list
                print(f"Warning: Could not parse claims from LLM response content: {response.content}")
                return []

    def _generate_questions_one_section(self, llm: LLMInterface, claims_list: List[str], structured_output: bool = False) -> Dict[str, List[str]]:
        """
        Generates questions for each claim in the list. Returns them as a JSON string.

        Args:
            llm: The language model to use for generating questions.
            claims_list (list): A list of claims for which to generate questions.
            structured_output (bool): If True, expects the LLM to return structured output (e.g., a Pydantic model with a non-empty .parsed attribute). If False, expects a raw JSON string.
        """

        if not claims_list:
            return "{}"

        prompt = self.base_questions_prompt.format(claims_list=claims_list)
        response = llm.invoke(prompt)  # According to the schema that is passed to the LLMInterface, the response will be a list of dictionaries, each with the "claim" and "questions" keys 

        if structured_output == True:

            if hasattr(response, 'parsed') and response.parsed:
                # The new schema is a list of objects, each with a 'claim' and 'questions' field.
                # We need to convert this list back to a dictionary.
                if hasattr(response.parsed, 'c_and_a_list'):
                    questions_list = response.parsed.c_and_a_list
                    # Convert the list of objects to the desired dictionary format
                    return {item.claim: item.questions for item in questions_list}
                # Fallback for other parsed structures like a direct dictionary
                elif isinstance(response.parsed, dict):
                    return dict(response.parsed)
            
            else:
                raise ValueError("The LLM response does not contain structured output. Please check the LLM configuration. This documentation explains how to configure the LLM for structured output: https://ai.google.dev/gemini-api/docs/structured-output")
        
        else:

            # Fallback to parsing the raw JSON content if 'parsed' is not available
            try:
                # The response.content is the raw text from the LLM
                questions_dict = json.loads(response.content)
                return questions_dict if isinstance(questions_dict, dict) else {}
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, return an empty dictionary
                print(f"Warning: Could not parse questions from LLM response content: {response.content}")
                return {}
    
    def get_claims_and_questions_one_section(self, section_text: str, llm_claims: LLMInterface, llm_questions: LLMInterface, structured_output: bool = False) -> Dict[str, List[str]]:
        """
        Evaluates a single section of text by extracting verifiable claims and generating questions.
        
        Args:
            section_text (str): The text of the section to evaluate.
            llm_claims (LLMInterface): The language model for extracting claims.
            llm_questions (LLMInterface): The language model for generating questions.
            structured_output (bool): If True, returns structured output (e.g., a 
                Pydantic model with a non-empty .parsed attribute) for each step
                of the extraction and output. If False, returns a raw JSON string.
        """
        
        claims_list = self._extract_verifiable_claims_one_section(llm_claims, section_text, structured_output)
        questions_dict = self._generate_questions_one_section(llm_questions, claims_list, structured_output)

        return questions_dict

    def evaluate_one_claim(self, llm_evaluator: LLMInterface, claim_text: str, questions_and_answers: dict, base_eval_prompt: str, structured_output: bool = False) -> Dict:
        """
        Evaluates a single claim using an LLM.

        Args:
            llm_evaluator: The language model to use for evaluation.
            claim_text: The text of the claim to evaluate.
            questions_and_answers: A dictionary of questions and answers related to the claim.
            base_eval_prompt: The prompt template for evaluating a claim.
            structured_output: Flag to determine if the LLM should return structured output.

        Returns:
            A dictionary containing the evaluation 'conclusion' and 'justification'.
        """
        # Format the Q&A for the prompt
        q_and_a_str = json.dumps(questions_and_answers, indent=2)

        try:
            # Create the prompt and format it by inserting the claim text and Q&A
            prompt = base_eval_prompt.format(claim_text=claim_text, questions_and_answers_json=q_and_a_str)
        except KeyError as e:
            raise KeyError(f"Missing key in base_eval_prompt: {e}. Please ensure the prompt is correctly formatted with all required placeholders.")

        try:
            response = llm_evaluator.invoke(prompt)  # Get the response content from the LLM
            eval_result = {}

            if structured_output:
                if hasattr(response, 'parsed') and response.parsed:
                    # The 'parsed' attribute is an EvaluationResults object.
                    # We access its attributes to build the dictionary.
                    # We use .value to get the string from the Enum.
                    eval_result = {
                        "conclusion": response.parsed.conclusion.value,
                        "justification": response.parsed.justification,
                    }
                else:
                    raise ValueError("The LLM response does not contain structured output. Please check the LLM configuration. This documentation explains how to configure the LLM for structured output: https://ai.google.dev/gemini-api/docs/structured-output")
            else:
                # Fallback to parsing the raw JSON string from the content.
                # This logic is for when structured output is disabled.
                clean_response = response.content.strip().lstrip("```json").rstrip("```")
                eval_result = json.loads(clean_response)

            return eval_result

        except Exception as e:
            print(f"Failed to evaluate claim: {claim_text}. Error: {e}")
            return {
                "conclusion": "error",
                "justification": f"Failed to parse LLM response: {str(e)}"
            }

    def format_accuracy_report(self, evaluated_data: list, country: str, retriever_type: str) -> str:
        
        """
        Formats the evaluated claims into a structured markdown report.

        Args:
            evaluated_data: The list of sections with evaluated claims. The structure is expected to be:
                [{'title_section': 'section_1', 'claims': [{'claim': 'claim_text', 'questions': {'question_1': 'answer_1', ...}, 'conclusion': 'true/false/mixture/error', 'justification': 'justification_text'}, ...]}, ...]
            country: The country name for the report.
            retriever_type: The retriever type used.

        Returns:
            A string containing the formatted markdown report.
        """

        # Initialize overall stats
        overall_stats = {"true": 0, "false": 0, "mixture": 0, "error": 0, "total": 0}
        report_lines = []

        # First, calculate overall stats
        for section in evaluated_data:  # Iterate through each section dictionary in the evaluated data
            for claim in section.get("claims", []):  # Iterate through each claim in the section
                conclusion = claim.get("conclusion", "error")  # Get the conclusion of the claim, default to "error"
                overall_stats[conclusion] = overall_stats.get(conclusion, 0) + 1  # Add to the respective conclusion count
                overall_stats["total"] += 1  # Increment total claims count

        # Helper to format percentages
        def get_perc(num, total):
            return f"{(num / total * 100):.1f}%" if total > 0 else "0.0%"

        # Add overall header
        report_lines.append(f"# Accuracy Report - {country}")
        report_lines.append(f"**Retriever:** {retriever_type}")
        report_lines.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("---")
        report_lines.append("## Overall Accuracy")
        if overall_stats["total"] > 0:
            report_lines.append(f"- **Total Claims:** {overall_stats['total']}")
            report_lines.append(f"- **True:** {overall_stats['true']} ({get_perc(overall_stats['true'], overall_stats['total'])})")
            report_lines.append(f"- **False:** {overall_stats['false']} ({get_perc(overall_stats['false'], overall_stats['total'])})")
            report_lines.append(f"- **Mixture:** {overall_stats['mixture']} ({get_perc(overall_stats['mixture'], overall_stats['total'])})")
        else:
            report_lines.append("No claims were evaluated.")
        report_lines.append("---")

        # Add section-by-section breakdown
        for section in evaluated_data:
            section_title = section.get("title_section", "Untitled Section")  # Get the title of the section, default to "Untitled Section"
            section_claims = section.get("claims", [])  # Get the list of claims in the section
            section_stats = {"true": 0, "false": 0, "mixture": 0, "error": 0, "total": 0}  # Initialize section stats
            
            for claim in section_claims:  # Iterate through each claim in the section
                conclusion = claim.get("conclusion", "error")  # Get the conclusion of the claim, default to "error"
                section_stats[conclusion] = section_stats.get(conclusion, 0) + 1  # Increment the respective conclusion count
                section_stats["total"] += 1  # Increment total claims count for the section

            report_lines.append(f"## Section: {section_title}")
            if section_stats["total"] > 0:
                report_lines.append(f"**Section Score:** "
                                    f"True: {get_perc(section_stats['true'], section_stats['total'])}, "
                                    f"False: {get_perc(section_stats['false'], section_stats['total'])}, "
                                    f"Mixture: {get_perc(section_stats['mixture'], section_stats['total'])}")
            
            for i, claim in enumerate(section_claims):
                claim_text = claim.get("claim")
                if not claim_text:
                    continue
                conclusion = claim.get("conclusion", "error").upper()
                justification = claim.get("justification")

                report_lines.append(f"\n### Claim {i+1}: {conclusion}")
                report_lines.append(f"> {claim_text}")
                if justification and conclusion in ["FALSE", "MIXTURE"]:
                    report_lines.append(f"**Justification:** {justification}")
            
            report_lines.append("\n---")

        return "\n".join(report_lines)

    def save_accuracy_report(self, report_content: str, original_report_path: str) -> str:
        """
        Saves the accuracy report to a file.

        Args:
            report_content: The markdown content of the report.
            original_report_path: The path to the original report file.

        Returns:
            The path to the saved accuracy report.
        """
        original_path = Path(original_report_path)
        
        # Create 'accuracy_reports' subdirectory
        accuracy_dir = original_path.parent / "accuracy_reports"
        accuracy_dir.mkdir(exist_ok=True)
        
        # Create new filename
        accuracy_filename = f"accuracy_{original_path.name}"
        save_path = accuracy_dir / accuracy_filename
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Accuracy report saved to: {save_path}")
            return str(save_path)
        except Exception as e:
            print(f"Error saving accuracy report to {save_path}: {e}")
            raise