import os
import re

class ReportProcessor:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.content = ""
        self.sections = {}

    def _get_content(self, file_path: str) -> None:
        """
        Loads content from a Markdown file into the processor.
        
        Parameters:
            file_path (str): Path to the .md file.
        """
        if not file_path.endswith('.md'):
            raise ValueError("The file must be a .md Markdown file.")
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

        return self.content

    def get_sections(self, file_path: str=None, file_content=None) -> dict: 
        """
        Splits the markdown content into sections based on the RegEx pattern 
        (e.g., level 2 headings with ##).

        Args:
            file_path (str): Path to the .md file.
            file_content (str, optional): If provided, this content will be used instead of reading from the file.
        
        Returns:
            dict: A dictionary where keys are section titles and values are section bodies.
        """

        if file_path:  # If a file path is provided, load content from the file path

            self._get_content(file_path)  # Load content if not already loaded

            if not self.content:
                raise ValueError("No content loaded.")
        
        elif file_content:  # If content is provided directly

            self.content = file_content  # Use the provided content

            if not self.content:
                raise ValueError("No content provided.")
        
        else:
            raise ValueError("Either file_path or file_content must be provided.")

        matches = re.findall(self.pattern, self.content, re.DOTALL)

        self.sections = {title.strip(): body.strip() for title, body in matches}  # Key is title, value is body

        return self.sections