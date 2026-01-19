#!/usr/bin/env python3
"""
Text Cleaning Pipeline for ML Textbook Analysis
Handles common OCR errors, removes noise, and detects chapter boundaries.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

class TextCleaner:
    """Clean extracted text and detect structure."""

    def __init__(self, text: str, filename: str):
        self.original_text = text
        self.filename = filename
        self.cleaned_text = ""
        self.chapters = []
        self.metadata = {}

    def remove_header_footer(self, text: str) -> str:
        """Remove common header/footer patterns."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip page markers we added
            if line.strip().startswith('--- PAGE'):
                cleaned_lines.append(line)
                continue

            # Skip common header/footer patterns
            stripped = line.strip()

            # Skip if just a number (page number)
            if stripped.isdigit() and len(stripped) < 5:
                continue

            # Skip very short lines that look like headers
            if len(stripped) < 3:
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR and extraction errors."""
        # Fix ligature issues
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            '—': '-',
            '–': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '…': '...',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Fix common word splits
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def detect_chapters(self, text: str) -> List[Dict]:
        """Detect chapter boundaries and extract structure."""
        chapters = []

        # Common chapter patterns
        patterns = [
            r'^(?:CHAPTER|Chapter)\s+(\d+)[:\s]+(.+?)$',
            r'^(\d+)\s+([A-Z][A-Za-z\s]+?)$',
            r'^Part\s+(\w+)[:\s]+(.+?)$',
            r'^PART\s+(\w+)[:\s]+(.+?)$',
        ]

        lines = text.split('\n')
        current_pos = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            for pattern in patterns:
                match = re.match(pattern, stripped)
                if match:
                    chapters.append({
                        'number': match.group(1),
                        'title': match.group(2).strip() if len(match.groups()) > 1 else '',
                        'line_number': i,
                        'position': current_pos
                    })
                    break
            current_pos += len(line) + 1

        # Add end positions
        for i, chapter in enumerate(chapters):
            if i < len(chapters) - 1:
                chapter['end_position'] = chapters[i + 1]['position']
            else:
                chapter['end_position'] = len(text)

        return chapters

    def extract_metadata(self, text: str) -> Dict:
        """Extract basic metadata from text."""
        words = text.split()

        # Count technical terms
        tech_terms = [
            'learning', 'algorithm', 'neural', 'network', 'classification',
            'regression', 'training', 'model', 'data', 'feature', 'prediction',
            'probability', 'bayesian', 'optimization', 'gradient', 'loss',
            'function', 'hypothesis', 'error', 'validation', 'test'
        ]

        term_counts = {}
        text_lower = text.lower()
        for term in tech_terms:
            count = len(re.findall(r'\b' + term + r'\b', text_lower))
            if count > 0:
                term_counts[term] = count

        return {
            'word_count': len(words),
            'char_count': len(text),
            'line_count': len(text.split('\n')),
            'tech_term_counts': term_counts
        }

    def clean(self) -> Tuple[str, Dict]:
        """Run full cleaning pipeline."""
        text = self.original_text

        # Skip the header lines we added during extraction
        if text.startswith('#'):
            lines = text.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('=' * 10):
                    start_idx = i + 1
                    break
            text = '\n'.join(lines[start_idx:])

        # Apply cleaning steps
        text = self.remove_header_footer(text)
        text = self.fix_common_ocr_errors(text)

        self.cleaned_text = text
        self.chapters = self.detect_chapters(text)
        self.metadata = self.extract_metadata(text)
        self.metadata['chapter_count'] = len(self.chapters)
        self.metadata['chapters'] = self.chapters

        return self.cleaned_text, self.metadata


def process_all_texts(input_dir: str, output_dir: str, log_dir: str):
    """Process all extracted text files."""

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    cleaning_log = {
        'processing_date': datetime.now().isoformat(),
        'total_files': len(txt_files),
        'results': []
    }

    print(f"\nCleaning {len(txt_files)} text files...\n")

    for i, txt_file in enumerate(txt_files, 1):
        print(f"[{i}/{len(txt_files)}] {txt_file[:60]}...", end=" ", flush=True)

        input_path = os.path.join(input_dir, txt_file)

        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        cleaner = TextCleaner(text, txt_file)
        cleaned_text, metadata = cleaner.clean()

        # Save cleaned text
        output_path = os.path.join(output_dir, txt_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        # Extract year from filename
        year_match = re.match(r'^(\d{4})_', txt_file)
        year = int(year_match.group(1)) if year_match else 0

        cleaning_log['results'].append({
            'filename': txt_file,
            'year': year,
            'word_count': metadata['word_count'],
            'chapter_count': metadata['chapter_count'],
            'chapters': metadata['chapters'][:10],  # First 10 chapters
            'top_tech_terms': dict(sorted(
                metadata['tech_term_counts'].items(),
                key=lambda x: -x[1]
            )[:10])
        })

        print(f"OK ({metadata['word_count']:,} words, {metadata['chapter_count']} chapters)")

    # Save log
    log_path = os.path.join(log_dir, 'cleaning_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(cleaning_log, f, indent=2)

    print(f"\nCleaning complete! Log saved to: {log_path}")

    return cleaning_log


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_dir = os.path.join(project_root, "extracted_text")
    output_dir = os.path.join(project_root, "cleaned_text")
    log_dir = os.path.join(project_root, "logs")

    os.makedirs(output_dir, exist_ok=True)

    process_all_texts(input_dir, output_dir, log_dir)
