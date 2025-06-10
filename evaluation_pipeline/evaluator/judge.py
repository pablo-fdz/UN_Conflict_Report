import json

from evaluator.prompts.base_prompt import base_prompt
from evaluator.prompts.criteria import criteria_dict


class LLMJudge:
    def __init__(self, model):
        self.model = model
        self.base_prompt = base_prompt
        self.criteria_dict = criteria_dict

    def _build_prompt(self, section_name: str, section_text: str) -> str:
        criteria = self.criteria_dict.get(section_name, {})
        criteria_text = "\n".join(f"{i + 1}. ({key}) {question}" for i, (key, question) in enumerate(criteria.items()))
        return base_prompt.format(criteria=criteria_text, section_text=section_text)
    
    def _evaluate_section(self, section_name: str, section_text: str) -> dict:
        if section_name not in self.criteria_dict:
            valid_sections = ", ".join(self.criteria_dict.keys())
            return {
                "error": (
                    f"The section '{section_name}' is not within the list of recognized sections. "
                    "No evaluation criteria were found, so this section was not evaluated.\n"
                    f"Valid sections are: {valid_sections}." 
                    "If there is a closely related name for '{section_name}' within the list of valid sections, change it to that." 
                    "Otherwise, reorganise this information into other sections within the list (whether they exist in the original document or you have to create them)."
                )
            }
        full_prompt = self._build_prompt(section_name, section_text)
        response = self.model.invoke(full_prompt).content
        return self._parse_response(response)

    def _parse_response(self, response_text: str) -> dict | None:
        """
        Parses the LLM's JSON response into a dictionary.
        Returns the dict if successful, or None if parsing fails.
        """
        try:
            # Clean Markdown-style code blocks
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.removeprefix("```json").strip()
            if response_text.startswith("```"):
                response_text = response_text.removeprefix("```").strip()
            if response_text.endswith("```"):
                response_text = response_text.removesuffix("```").strip()

            data = json.loads(response_text)

            # Optional: Basic validation of the structure
            if not isinstance(data, dict):
                print("Parsed JSON is not a dictionary.")
                return None

            for crit, details in data.items():
                if not isinstance(details, dict):
                    print(f"Details for criterion '{crit}' are not a dictionary.")
                    return None
                if "score" not in details or "comment" not in details:
                    print(f"Missing 'score' or 'comment' in criterion '{crit}'.")
                    return None
                score = details["score"]
                if not (isinstance(score, int) and 1 <= score <= 5):
                    print(f"Score for '{crit}' is invalid: {score}")
                    return None
                if not isinstance(details["comment"], str):
                    print(f"Comment for '{crit}' is not a string.")
                    return None

            return data

        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return None

    def evaluate_all_sections(self, sections: dict) -> dict:
        """
        Evaluates multiple sections of a report.

        Parameters:
            sections (dict): A dictionary where keys are section names and values are section content.

        Returns:
            dict: A dictionary where each key is a section name and each value is the evaluation result (dict or None).
        """
        results = {}
        for name, text in sections.items():
            print(f"Evaluating section: {name}")
            try:
                result = self._evaluate_section(name, text)
            except Exception as e:
                print(f"Error evaluating section '{name}': {e}")
                result = None
            results[name] = result
        return results