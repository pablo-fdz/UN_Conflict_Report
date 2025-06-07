import json
from .prompts.base_prompt import base_prompt
from .prompts.criteria import criteria_dict

class LLMJudge:
    def __init__(self, model):
        self.model = model
        self.base_prompt = base_prompt
        self.criteria_dict = criteria_dict

    def build_prompt(self, section_name: str, section_text: str) -> str:
        criteria = self.criteria_dict.get(section_name, {})
        criteria_text = "\n".join(f"{i+1}. ({key}) {question}" for i, (key, question) in enumerate(criteria.items()))
        return base_prompt.format(criteria=criteria_text, section_text=section_text) 
            # criteria_text and section_text added to the respective placeholders
            # self.base_prompt is NOT updated


    def evaluate_section(self, section_name: str, section_text: str) -> dict:
        full_prompt = self.build_prompt(section_name, section_text)
        response = self.model.call(full_prompt)
        return self.parse_response(response)
    
    
    def parse_response(self, response_text: str) -> dict | None:
        """
        Parses the LLM's JSON response into a dictionary.
        Returns the dict if successful, or None if parsing fails.
        """
        try:
            data = json.loads(response_text)

            # Optional: Basic validation of the structure
            if not isinstance(data, dict):
                print("Parsed JSON is not a dictionary.")
                return None

            for crit, details in data.items():
                if not isinstance(details, dict):
                    print(f"Details for criterion '{crit}' are not a dictionary.")
                    return None
                if 'score' not in details or 'comment' not in details:
                    print(f"Missing 'score' or 'comment' in criterion '{crit}'.")
                    return None
                score = details['score']
                if not (isinstance(score, int) and 1 <= score <= 5):
                    print(f"Score for '{crit}' is invalid: {score}")
                    return None
                if not isinstance(details['comment'], str):
                    print(f"Comment for '{crit}' is not a string.")
                    return None

            return data

        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return None