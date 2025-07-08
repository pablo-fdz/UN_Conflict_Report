"""
Citation Hyperlinker Script

This script provides functionality to hyperlink inline citations in markdown documents
to their corresponding sources section. It handles various citation formats including:
- Single citations: [7]
- Multi-citations: [37, 7]
- Duplicate citations: [3, 3]
- Complex multi-citations: [14, 2, 37]

The script converts citations to clickable hyperlinks that jump to anchors in the Sources section.
"""

import re


def hyperlink_citations_to_sources(text):
    """
    Convert inline citations in markdown text to hyperlinks pointing to sources.
    
    Args:
        text (str): The markdown text containing citations and sources
        
    Returns:
        str: The text with hyperlinked citations and anchored sources
    """
    # Find all citation numbers in the Sources section (including those in multi-citation format)
    sources_section = text.split("Sources", 1)[-1] if "Sources" in text else ""
    source_ids = set(re.findall(r"\d+", sources_section))

    # Function to replace matches in the body text
    def replacer(match):
        # Extract the content inside brackets
        content = match.group(1)
        
        # Split the content by commas and process each part
        parts = [part.strip() for part in content.split(',')]
        hyperlinked_parts = []
        
        for part in parts:
            if part.isdigit() and part in source_ids:
                hyperlinked_parts.append(f"[{part}](#{part})")
            else:
                hyperlinked_parts.append(part)
        
        # Join the parts back together
        return f"[{', '.join(hyperlinked_parts)}]"

    # Function to add anchors to sources
    def add_anchor(match):
        number = match.group(1)
        return f'<a id="{number}"></a>[{number}]'

    # Replace citations in the main body (before "Sources")
    if "Sources" in text:
        body, sources = text.split("Sources", 1)
        # Update body with hyperlinks - matches [number] or [number, number, ...] patterns
        updated_body = re.sub(r"\[([0-9,\s]+)\]", replacer, body)
        # Add anchors to sources section
        updated_sources = re.sub(r"\[(\d+)\]", add_anchor, sources)
        return updated_body + "Sources" + updated_sources
    else:
        return text  # Fallback


def test_citation_hyperlinking():
    """
    Test the citation hyperlinking functionality with various test cases.
    """
    # Test specific cases
    test_cases = [
        "This is a test [37, 7] with multiple citations.",
        "Another test [3, 3] with duplicate numbers.",
        "Mixed case [14, 2, 37] with multiple different numbers.",
        "Single citation [7] should also work."
    ]

    print("Testing citation hyperlinking:")
    for i, test in enumerate(test_cases, 1):
        # Create a mini-document with sources for testing
        mini_doc = f"{test}\n\nSources\n[2] Source 2\n[3] Source 3\n[7] Source 7\n[14] Source 14\n[37] Source 37"
        result = hyperlink_citations_to_sources(mini_doc)
        print(f"\nTest {i}:")
        print(f"Input:  {test}")
        print(f"Output: {result.split('Sources')[0].strip()}")


def main():
    """
    Main function to process a specific report file and create hyperlinked version.
    """
    # Input and output file paths
    input_file = r"C:\Users\matia\OneDrive\Escritorio\Nastia_BSE\Master_Thesis\UN_Conflict_Report\reports\Sudan\corrected_reports\corrected_security_report_Sudan_HybridCypher_20250707_2351.md"
    output_file = "C:\Users\matia\OneDrive\Escritorio\Nastia_BSE\Master_Thesis\UN_Conflict_Report\reports\Sudan\corrected_reports\hyper_corrected_security_report_Sudan_HybridCypher_20250707_2351.md"
    
    try:
        # Read the input file
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Process the content to add hyperlinks
        linked_content = hyperlink_citations_to_sources(content)

        # Write the processed content to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(linked_content)
            
        print(f"Successfully processed citations and saved to: {output_file}")
        
        # Run tests to verify functionality
        print("\n" + "="*60)
        test_citation_hyperlinking()
        
        # Check for specific patterns in the output
        print("\n" + "="*60)
        print("Checking actual output for [37, 7] and [3, 3] cases:")

        lines_with_citations = []
        for line in linked_content.split('\n'):
            if '[37, 7]' in line or '[3, 3]' in line:
                lines_with_citations.append(line)

        for line in lines_with_citations[:5]:  # Show first 5 matches
            print(f"Line: {line}")
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
