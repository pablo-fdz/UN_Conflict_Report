import os
import re

class ReportProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = ""
        self.sections = {}

        self._import_markdown()
        self._split_into_sections()

    def _import_markdown(self) -> None:
        """
        Loads content from a Markdown file into the processor.
        
        Parameters:
            file_path (str): Path to the .md file.
        """
        if not self.file_path.endswith('.md'):
            raise ValueError("The file must be a .md Markdown file.")
        
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"No file found at {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

    def _split_into_sections(self) -> dict:
        """
        Splits the markdown content into sections based on level-2 headings (##).
        Returns a dict with section titles as keys and their text as values.
        """
        if not self.content:
            raise ValueError("No content loaded.")

        pattern = r'(?m)^## (.+?)\s*\n(.*?)(?=^## |\Z)'  # Multiline + non-greedy
        matches = re.findall(pattern, self.content, re.DOTALL)

        self.sections = {title.strip(): body.strip() for title, body in matches}

    def get_content(self) -> str:
        return self.content
    
    def get_sections(self) -> dict:
        return self.sections