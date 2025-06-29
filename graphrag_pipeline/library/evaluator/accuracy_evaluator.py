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

    def _extract_verifiable_claims_one_section(self, llm: LLMInterface, section_text: str) -> str:
        """
        Extracts verifiable claims from the section text. Returns them as a list of strings.

        Args:
            llm: The language model to use for extraction.
            section_text (str): The text of the section from which to extract claims.
        """
        prompt = self.base_claims_prompt.format(section_text=section_text)
        claims_list = llm.invoke(prompt).content

        return claims_list

    def _generate_questions_one_section(self, llm: LLMInterface, claims_list: str) -> list:
        """
        Generates questions for each claim in the list. Returns them as a list of strings.

        Args:
            llm: The language model to use for generating questions.
            claims_list (list): A list of claims for which to generate questions.
        """

        prompt = self.base_questions_prompt.format(claims_list=claims_list)
        questions_dict = llm.invoke(prompt).content

        return questions_dict
    
    def get_claims_and_questions_one_section(self, section_text: str, llm_claims: LLMInterface, llm_questions: LLMInterface) -> str:
        """
        Evaluates a single section of text by extracting verifiable claims and generating questions.
        
        Args:
            section_text (str): The text of the section to evaluate.
            llm_claims (LLMInterface): The language model for extracting claims.
            llm_questions (LLMInterface): The language model for generating questions.
        """
        
        claims = self._extract_verifiable_claims_one_section(llm_claims, section_text)
        self.questions_dict = self._generate_questions_one_section(llm_questions, claims)

        return self.questions_dict
    
    def evaluate_claims(self, llm_evaluator: LLMInterface, sections_data: list, base_eval_prompt: str) -> list:
        """
        Evaluates each claim in the sections data using an LLM.

        Args:
            llm_evaluator: The language model to use for evaluation.
            sections_data: A list of section dictionaries, each containing claims and questions.
                The structure is expected to be:
                [{'title_section': 'section_1', 'claims': [{'claim': 'claim_text', 'questions': {'question_1': 'answer_1', ...}}, ...]}, ...]
            base_eval_prompt: The prompt template for evaluating a claim.

        Returns:
            The sections_data list, with each claim dictionary augmented with evaluation 'conclusion' and 'justification'.
        """
        # Iterate through each section and its claims
        for section in sections_data:

            # Iterate through each claim in the section
            for claim_data in section.get("claims", []):

                claim_text = claim_data.get("claim")
                if not claim_text:
                    continue  # Skip if claim text is missing
                questions_and_answers = claim_data.get("questions", {})  # Get the questions and answers for the claim

                # Format the Q&A for the prompt
                q_and_a_str = json.dumps(questions_and_answers, indent=2)

                # Create the prompt and format it by inserting the claim text and Q&A
                prompt = base_eval_prompt.format(
                    claim_text=claim_text,
                    questions_and_answers_json=q_and_a_str
                )

                try:
                    response_content = llm_evaluator.invoke(prompt).content  # Get the response content from the LLM
                    # Clean up the response to ensure it's valid JSON
                    clean_response = response_content.strip().lstrip("```json").rstrip("```")
                    eval_result = json.loads(clean_response)
                    
                    # Append the evaluation to the claim data
                    claim_data.update(eval_result)

                except Exception as e:
                    print(f"Failed to evaluate claim: {claim_text}. Error: {e}")
                    claim_data["conclusion"] = "error"
                    claim_data["justification"] = f"Failed to parse LLM response: {str(e)}"
        
        return sections_data  # Return the updated sections data with evaluations

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