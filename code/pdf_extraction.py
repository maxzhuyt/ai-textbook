#!/usr/bin/env python3
"""
PDF Text Extraction Pipeline for ML Textbook Analysis
Compares multiple extraction methods and selects the best for each document.
"""

import os
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PDF extraction libraries
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF

class PDFExtractor:
    """Multi-method PDF text extraction with quality assessment."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)
        self.results = {}

    def extract_with_pypdf2(self) -> Tuple[str, Dict]:
        """Extract text using PyPDF2."""
        try:
            text_parts = []
            metadata = {}
            with open(self.pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                metadata['num_pages'] = len(reader.pages)
                for i, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text() or ""
                        text_parts.append(f"\n--- PAGE {i+1} ---\n{text}")
                    except Exception as e:
                        text_parts.append(f"\n--- PAGE {i+1} ---\n[EXTRACTION ERROR: {str(e)}]")
            return "\n".join(text_parts), metadata
        except Exception as e:
            return f"[EXTRACTION FAILED: {str(e)}]", {'error': str(e)}

    def extract_with_pdfplumber(self) -> Tuple[str, Dict]:
        """Extract text using pdfplumber (better for tables)."""
        try:
            text_parts = []
            metadata = {}
            with pdfplumber.open(self.pdf_path) as pdf:
                metadata['num_pages'] = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                        text_parts.append(f"\n--- PAGE {i+1} ---\n{text}")
                    except Exception as e:
                        text_parts.append(f"\n--- PAGE {i+1} ---\n[EXTRACTION ERROR: {str(e)}]")
            return "\n".join(text_parts), metadata
        except Exception as e:
            return f"[EXTRACTION FAILED: {str(e)}]", {'error': str(e)}

    def extract_with_pymupdf(self) -> Tuple[str, Dict]:
        """Extract text using PyMuPDF (fitz) - often best quality."""
        try:
            text_parts = []
            metadata = {}
            doc = fitz.open(self.pdf_path)
            metadata['num_pages'] = len(doc)
            for i, page in enumerate(doc):
                try:
                    text = page.get_text() or ""
                    text_parts.append(f"\n--- PAGE {i+1} ---\n{text}")
                except Exception as e:
                    text_parts.append(f"\n--- PAGE {i+1} ---\n[EXTRACTION ERROR: {str(e)}]")
            doc.close()
            return "\n".join(text_parts), metadata
        except Exception as e:
            return f"[EXTRACTION FAILED: {str(e)}]", {'error': str(e)}

    def assess_quality(self, text: str) -> Dict:
        """Assess text extraction quality."""
        if not text or text.startswith("[EXTRACTION FAILED"):
            return {
                'score': 0,
                'char_count': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'alpha_ratio': 0,
                'garbage_ratio': 1.0,
                'issues': ['extraction_failed']
            }

        # Basic counts
        char_count = len(text)
        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return {
                'score': 0,
                'char_count': char_count,
                'word_count': 0,
                'avg_word_length': 0,
                'alpha_ratio': 0,
                'garbage_ratio': 1.0,
                'issues': ['no_words_extracted']
            }

        # Quality metrics
        avg_word_length = sum(len(w) for w in words) / word_count

        # Alpha ratio (proportion of alphabetic characters)
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / char_count if char_count > 0 else 0

        # Garbage detection (unusual characters, encoding issues)
        garbage_chars = sum(1 for c in text if ord(c) > 127 and not c.isalpha())
        garbage_ratio = garbage_chars / char_count if char_count > 0 else 0

        # Check for common OCR/extraction issues
        issues = []

        # Too many single-character "words"
        single_char_words = sum(1 for w in words if len(w) == 1 and not w.isdigit())
        if single_char_words / word_count > 0.2:
            issues.append('excessive_single_chars')

        # Excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / char_count > 0.15:
            issues.append('excessive_special_chars')

        # Low average word length (suggests fragmentation)
        if avg_word_length < 3:
            issues.append('fragmented_words')

        # Calculate overall quality score (0-100)
        score = 100
        score -= garbage_ratio * 50  # Penalty for garbage chars
        score -= (1 - alpha_ratio) * 30  # Penalty for low alpha ratio
        score -= len(issues) * 10  # Penalty per issue

        # Bonus for good word length
        if 4 <= avg_word_length <= 8:
            score += 5

        score = max(0, min(100, score))

        return {
            'score': round(score, 2),
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_length': round(avg_word_length, 2),
            'alpha_ratio': round(alpha_ratio, 4),
            'garbage_ratio': round(garbage_ratio, 4),
            'issues': issues
        }

    def extract_all_methods(self) -> Dict:
        """Run all extraction methods and compare quality."""
        methods = {
            'pypdf2': self.extract_with_pypdf2,
            'pdfplumber': self.extract_with_pdfplumber,
            'pymupdf': self.extract_with_pymupdf
        }

        results = {}
        for method_name, method_func in methods.items():
            print(f"  Extracting with {method_name}...", end=" ", flush=True)
            text, metadata = method_func()
            quality = self.assess_quality(text)
            results[method_name] = {
                'text': text,
                'metadata': metadata,
                'quality': quality
            }
            print(f"Score: {quality['score']:.1f}, Words: {quality['word_count']:,}")

        return results

    def get_best_extraction(self, results: Dict) -> Tuple[str, str, Dict]:
        """Select the best extraction based on quality scores."""
        best_method = None
        best_score = -1

        for method_name, data in results.items():
            score = data['quality']['score']
            if score > best_score:
                best_score = score
                best_method = method_name

        best_data = results[best_method]
        return best_method, best_data['text'], best_data['quality']


def process_all_pdfs(input_dir: str, output_dir: str, log_dir: str):
    """Process all PDFs in the input directory."""

    pdf_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pdf')])

    extraction_log = {
        'processing_date': datetime.now().isoformat(),
        'total_files': len(pdf_files),
        'results': []
    }

    print(f"\nProcessing {len(pdf_files)} PDF files...\n")
    print("=" * 80)

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(input_dir, pdf_file)
        print(f"\n[{i}/{len(pdf_files)}] {pdf_file}")
        print("-" * 60)

        extractor = PDFExtractor(pdf_path)
        results = extractor.extract_all_methods()

        best_method, best_text, quality = extractor.get_best_extraction(results)

        print(f"\n  BEST METHOD: {best_method} (score: {quality['score']:.1f})")

        # Save the best extraction
        # Create a clean filename
        base_name = os.path.splitext(pdf_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Extracted from: {pdf_file}\n")
            f.write(f"# Method: {best_method}\n")
            f.write(f"# Quality Score: {quality['score']}\n")
            f.write(f"# Word Count: {quality['word_count']}\n")
            f.write(f"# Extraction Date: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(best_text)

        # Log results
        extraction_log['results'].append({
            'filename': pdf_file,
            'best_method': best_method,
            'quality_score': quality['score'],
            'word_count': quality['word_count'],
            'char_count': quality['char_count'],
            'issues': quality['issues'],
            'all_method_scores': {
                method: data['quality']['score']
                for method, data in results.items()
            }
        })

        print(f"  Saved to: {output_path}")

    # Save extraction log
    log_path = os.path.join(log_dir, 'extraction_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(extraction_log, f, indent=2)

    print("\n" + "=" * 80)
    print(f"\nExtraction complete! Log saved to: {log_path}")

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)

    total_words = sum(r['word_count'] for r in extraction_log['results'])
    avg_score = sum(r['quality_score'] for r in extraction_log['results']) / len(extraction_log['results'])

    print(f"Total files processed: {len(pdf_files)}")
    print(f"Total words extracted: {total_words:,}")
    print(f"Average quality score: {avg_score:.1f}")

    # Method usage
    method_counts = {}
    for r in extraction_log['results']:
        method = r['best_method']
        method_counts[method] = method_counts.get(method, 0) + 1

    print("\nBest method per file:")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count} files")

    # Files with issues
    files_with_issues = [r for r in extraction_log['results'] if r['issues']]
    if files_with_issues:
        print(f"\nFiles with quality issues ({len(files_with_issues)}):")
        for r in files_with_issues:
            print(f"  - {r['filename']}: {', '.join(r['issues'])}")

    return extraction_log


if __name__ == "__main__":
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_dir = os.path.join(project_root, "raw_pdfs")
    output_dir = os.path.join(project_root, "extracted_text")
    log_dir = os.path.join(project_root, "logs")

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    process_all_pdfs(input_dir, output_dir, log_dir)
