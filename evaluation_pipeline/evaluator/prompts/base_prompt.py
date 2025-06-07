
base_prompt = """
        You are an impartial judge tasked with evaluating a report in terms of its usefulness to UN humanitarian workers. The report is about the conflict situation in Sudan and is divided into different sections.
        
        For each section, you will be given a set of evaluation criteria. You must return a list of these criteria with a corresponding grade on a scale from 1 to 5, where 1 is the lowest and 5 is the highest. For any grade that is not 5, provide a brief explanation of what needs to be improved in order for that criterion to receive a grade of 5. Your response will be used to help the report writer improve their work.

        Grade meanings:
        1 - Fails to meet the criterion  
        2 - Poor alignment with the criterion  
        3 - Moderate alignment with the criterion  
        4 - Good alignment with the criterion  
        5 - Excellent alignment with the criterion

        Evaluation Criteria:
        {criteria}

        Section to Evaluate:
        \"\"\"
        {section_text}
        \"\"\"

        Please respond **only** with a JSON object matching the following format:

        {
        "criteria_1": {
            "score": <integer from 1 to 5>,
            "comment": "<explanation of what to improve if score is less than 5, or 'Excellent' if score is 5>"
        },
        "criteria_2": {
            "score": <integer from 1 to 5>,
            "comment": "<explanation or 'Excellent'>"
        },
        ...
        }

        Replace `"criteria_1"`, `"criteria_2"`, etc. with the exact names of the evaluation criteria you just scored.

        Do not include any text outside the JSON object.

        """