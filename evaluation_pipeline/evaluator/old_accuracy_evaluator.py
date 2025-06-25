import json
from typing import List, Dict

base_claims_prompt = """ 
    You are an AI tasked with extracting verifiable claims from a section of a report. 
    A verifiable claim is an atomic statement that can be checked for accuracy and is relevant to the topic of the report. 
    Here is an example:
    Section: "The constant attacks of Group A and Group B on civilians in Country X have led to a high number of casualties as well as 
            internally displaced persons (IDPs). In May 2025, there have been more IDPs than any other month that year."

    Fromt this section, you can extract the following verifiable claims:
    1. "Group A has been carrying constant attacks on civilians in Country X."
    2. "Group B has been carrying constant attacks on civilians in Country X."
    3. "In May 2025, there have been more IDPs than in any of the previous months of 2025."
    
    Notes on this:
    - Clamis 1 and 2 have to be separate claims, because they refer to different groups, 
    so one could be true while the other is false and that would make the first sentence incorrect.
    - The claim in the original section about a "high number of casualties" is not verifiable, 
    because it is not quantifiable/specific. We do not know what "high" means and we do not have a 
    reference to compare it too. Claims that are subjective or vague should not be extracted.
    - It is important that you always make all the necessary information explicit in the claims,
    without pronouns or terms that make references to previous sentences, so that each claim is 
    self-contained and can be verified independently.
    - Also, when abbreviations of any kind are used, always include, if you know it with certainty,
    the full name/term plus the abbreviation in parentheses.

    You must extract all verifiable claims from the following section of a report:
    Section: "{section_text}"

    Return the the claims in the format of a python list, with each claim as a string. Return only this list, in between square brackets, 
    without any additional text or formatting so i can easily use it as a variable in my code.

    Example output: ["Claim 1", "Claim 2", "Claim 3", ...]
    """

base_questions_prompt = """
    You are a journalist tasked with evaluating the accuracy of a set of claims against a knowledge base.
    For the given list of claims below, you must generate 1 to 4 questions aimed at leading you to the information needed to verify each claim.
    Each question should be specific, clear, and concise, designed to have a closed-ended objective answer.

    When abbreviations of any kind are used in the claim, always include in the question, if you know it with certainty,
    the full name/term plus the abbreviation in parentheses.
    
    Here is the list of claims:
    {claims_list}
    """


class AccuracyEvaluator:
    def __init__(self):

        # PROMPTS
        self.base_claims_prompt = base_claims_prompt
        self.base_questions_prompt = base_questions_prompt

    def extract_verifiable_claims_one_section(self, section_text: str, model) -> list:
        """
        Extracts verifiable claims from the section text. Returns them as a list of strings.
        """
        prompt = self.base_claims_prompt.format(section_text=section_text)
        self.claims_list = model.invoke(prompt).content
    
    def generate_questions_one_section(self, model) -> list:
        """
        Generates questions for each claim in the list. Returns them as a list of strings.
        """

        prompt = self.base_questions_prompt.format(claims_list=self.claims_list)
        self.questions_dict = model.invoke(prompt).content

        