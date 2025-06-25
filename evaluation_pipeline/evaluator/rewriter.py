import os
import re

# from evaluator.prompts.rewriter_base_prompt import rewriter_base_prompt


class ReportRewriter:
    def __init__(self, model, file_path: str):
        self.model = model

        self.report_path = file_path
        self.context_path = file_path.replace('.md', '_context.md')

        self.report_content = ""
        self.context_content = ""  # RAG context used to rewrite the initial report -- import to be handled

        self.report_sections = {}

        self._import_markdowns()
        self._split_into_sections()

    def _import_markdowns(self) -> None:
        """
        Loads content of the report and its respective context markdown files into ReporRewriter attributes.
        Assigns values to:
            self.report_content (str)
            self.context_content (str)
        """
        for path, attr in [(self.report_path, "report_content"), (self.context_path, "context_content")]:
            if not path.endswith('.md'):
                raise ValueError("File must be a Markdown file (.md).")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"No file found at {path}")

            with open(path, 'r', encoding='utf-8') as f:
                setattr(self, attr, f.read())

    def _split_into_sections(self) -> dict:
        """
        Splits the markdown report content into sections based on level-2 headings (##).
        Returns a dict with section titles as keys and their text as values.
        """
        if not self.report_content:
            raise ValueError("No content loaded.")

        pattern = r'(?m)^## (.+?)\s*\n(.*?)(?=^## |\Z)'  # Multiline + non-greedy
        matches = re.findall(pattern, self.report_content, re.DOTALL)

        self.report_sections = {title.strip(): body.strip() for title, body in matches}

    def _build_rewrite_prompt(self, section_name: str, original_text: str, feedback: dict) -> str:
        """
        Builds the prompt for the rewriter to write a corrected version of 1 section.
        """
        feedback_text = "\n".join(
            f"- {key}: {detail['comment']}" 
            for key, detail in feedback.items() 
            if isinstance(detail, dict) and 'comment' in detail
        )
        
        prompt = (
            f"You are a humanitarian analyst rewriting a section of a conflict report.\n\n"
            f"### Context:\n{self.context_content}\n\n"
            f"### Original Section [{section_name}]:\n{original_text}\n\n"
            f"### Feedback from Evaluation:\n{feedback_text}\n\n"
            f"### Task:\n"
            f"Rewrite the section to improve it based on the feedback. Be accurate, neutral, and concise. "
            f"Preserve the section's intent, but address all points raised.\n\n"
            f"### Rewritten Section:"
        )
        return prompt

    def _rewrite_section(self, section_name: str, original_text: str, feedback: dict) -> str:
        prompt = self._build_rewrite_prompt(section_name, original_text, feedback)
        response = self.model.invoke(prompt)
        return response.content.strip()

    def rewrite_all_sections(self, evaluations: dict) -> dict:
        rewritten = {}
        for name, text in self.report_sections.items():
            feedback = evaluations.get(name, {})
            if isinstance(feedback, dict) and "error" not in feedback:
                print(f"Rewriting section: {name}")
                rewritten[name] = self._rewrite_section(name, text, feedback)
            else:
                rewritten[name] = text  # fallback: keep original
        return rewritten
