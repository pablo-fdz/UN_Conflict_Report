import os
import re

class DocumentProcessor:
    def __init__(self):
        self.content = ""

    def import_markdown(self, file_path: str) -> None:
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

    def get_content(self) -> str:
        return self.content
    
    def split_into_sections(self) -> dict:
        """
        Splits the markdown content into sections based on level-2 headings (##).
        Returns a dict with section titles as keys and their text as values.
        """
        if not self.content:
            raise ValueError("No content loaded. Call import_markdown() first.")

        pattern = r'(?m)^## (.+?)\s*\n(.*?)(?=^## |\Z)'  # Multiline + non-greedy
        matches = re.findall(pattern, self.content, re.DOTALL)

        sections = {title.strip(): body.strip() for title, body in matches}
        return sections