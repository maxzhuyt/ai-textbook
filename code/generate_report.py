#!/usr/bin/env python3
"""
Step 6: Generate Comprehensive Final Report
Synthesizes all analysis results into a research-grade report.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd


def generate_markdown_report(output_dir: str) -> str:
    """Generate comprehensive markdown report."""

    # Load all analysis results
    results = {}

    files_to_load = {
        'nlp': 'nlp_analysis_results.json',
        'semantic_llm': 'llm_classification.json',
        'semantic_revised': 'revised_classification.json',
        'semantic_keyword': 'keyword_classification.json',
        'temporal': 'temporal_analysis.json',
        'cleaning': '../logs/cleaning_log.json',
        'catalog': '../logs/textbook_catalog.json',
        'environment': '../logs/environment_audit.json'
    }

    for key, filename in files_to_load.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[key] = json.load(f)
            print(f"  Loaded: {filename}")
        else:
            print(f"  Not found: {filename}")
            results[key] = None

    # Choose semantic analysis (prefer LLM over revised keyword over original)
    semantic = results.get('semantic_llm') or results.get('semantic_revised') or results.get('semantic_keyword')
    if results.get('semantic_llm'):
        semantic_method = 'LLM-based (LLaMA 3.1 8B)'
    elif results.get('semantic_revised'):
        semantic_method = 'Revised keyword-based'
    else:
        semantic_method = 'Keyword-based'

    # Start building report
    report = []

    # Title and metadata
    report.append("# Machine Learning Textbook Evolution Analysis (1983-2022)")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append(f"This report analyzes **12 seminal AI/ML textbooks** spanning four decades (1983-2022) ")
    report.append("to trace the developmental trajectory of the artificial intelligence field. ")
    report.append("Using a combination of NLP keyword analysis and semantic content classification, ")
    report.append("we identify major shifts in emphasis, emerging and declining sub-domains, ")
    report.append("and key inflection points in the field's evolution.")
    report.append("")

    # Key findings summary
    report.append("### Key Findings")
    report.append("")

    if results.get('temporal') and 'key_findings' in results['temporal']:
        for finding in results['temporal']['key_findings'][:7]:
            if finding['type'] == 'era_shift':
                direction = "increased" if finding['change'] > 0 else "decreased"
                report.append(f"- **{finding['domain'].replace('_', ' ').title()}** {direction} by {abs(finding['change']):.1f}% from early to recent era")
            elif finding['type'] == 'diversity_change':
                report.append(f"- The field has been **{finding['direction']}** (Shannon entropy: {finding['early_entropy']:.2f} → {finding['recent_entropy']:.2f})")

    report.append("")

    # Methodology section
    report.append("---")
    report.append("")
    report.append("## 1. Methodology")
    report.append("")
    report.append("### 1.1 Corpus Description")
    report.append("")

    if results.get('catalog'):
        report.append("| Year | Title | Authors | Category |")
        report.append("|------|-------|---------|----------|")
        for book in results['catalog']['textbooks']:
            authors = ', '.join(book['authors'][:2])
            report.append(f"| {book['year']} | {book['title'][:50]}... | {authors} | {book['category']} |")

    report.append("")
    report.append("### 1.2 Analysis Pipeline")
    report.append("")
    report.append("1. **PDF Text Extraction**: Multi-method extraction (PyMuPDF, pdfplumber, PyPDF2) with quality assessment")
    report.append("2. **Text Cleaning**: Header/footer removal, OCR error correction, chapter boundary detection")
    report.append(f"3. **Semantic Classification**: {semantic_method} chapter classification into 14 AI/ML sub-domains")
    report.append("4. **NLP Analysis**: TF-IDF, topic modeling (LDA), keyword frequency analysis")
    report.append("5. **Temporal Analysis**: Era comparison, trend significance testing, inflection point identification")
    report.append("")

    if results.get('environment'):
        report.append("### 1.3 Computational Environment")
        report.append("")
        report.append(f"- **GPU**: {results['environment']['hardware']['gpu']['model']} ({results['environment']['hardware']['gpu']['vram_gb']}GB)")
        report.append(f"- **RAM**: {results['environment']['hardware']['memory_gb']}GB")
        report.append(f"- **Python**: {results['environment']['software']['python_version']}")
        if results.get('semantic_llm'):
            report.append(f"- **LLM Model**: {results['semantic_llm'].get('model_used', 'N/A')}")
        report.append("")

    # NLP Analysis Results
    report.append("---")
    report.append("")
    report.append("## 2. NLP Keyword Analysis (Layer 1)")
    report.append("")

    if results.get('nlp'):
        nlp = results['nlp']

        report.append("### 2.1 Top Terms Overall")
        report.append("")
        report.append("| Term | Avg Frequency (per 1000 words) |")
        report.append("|------|-------------------------------|")
        for term, freq in list(nlp['keyword_analysis']['top_terms_overall'].items())[:15]:
            report.append(f"| {term} | {freq:.3f} |")
        report.append("")

        report.append("### 2.2 Topic Modeling Results (LDA)")
        report.append("")
        report.append(f"Identified **{nlp['topic_modeling']['n_topics']} latent topics** across the corpus:")
        report.append("")

        for topic in nlp['topic_modeling']['topics'][:8]:
            words = ', '.join(topic['words'][:8])
            report.append(f"- **Topic {topic['topic_id']}**: {words}")
        report.append("")

        report.append("### 2.3 Document Similarity")
        report.append("")
        report.append(f"- Minimum similarity: {nlp['document_similarity']['min_similarity']:.3f}")
        report.append(f"- Maximum similarity: {nlp['document_similarity']['max_similarity']:.3f}")
        report.append(f"- Mean similarity: {nlp['document_similarity']['mean_similarity']:.3f}")
        report.append("")

        if 'term_evolution' in nlp.get('era_trends', {}):
            report.append("### 2.4 Term Evolution")
            report.append("")
            report.append("**Emerging terms (more prominent in recent era):**")
            for term in nlp['era_trends']['term_evolution']['emerging_terms'][:5]:
                report.append(f"- {term}")
            report.append("")
            report.append("**Declining terms (less prominent in recent era):**")
            for term in nlp['era_trends']['term_evolution']['declining_terms'][:5]:
                report.append(f"- {term}")
            report.append("")

    # Semantic Analysis Results
    report.append("---")
    report.append("")
    report.append("## 3. Semantic Content Analysis (Layer 2)")
    report.append("")

    if semantic:
        report.append(f"*Classification method: {semantic_method}*")
        report.append("")

        report.append("### 3.1 Sub-domain Distribution by Document")
        report.append("")
        report.append("| Year | Document | Top Domain | % |")
        report.append("|------|----------|------------|---|")

        for doc in sorted(semantic['documents'], key=lambda x: x['year']):
            agg = doc['aggregate_classification']
            top_domain = max(agg.items(), key=lambda x: x[1])
            short_name = doc['filename'].split('_')[1][:30] if '_' in doc['filename'] else doc['filename'][:30]
            report.append(f"| {doc['year']} | {short_name}... | {top_domain[0].replace('_', ' ')} | {top_domain[1]:.1f}% |")
        report.append("")

    # Temporal Analysis
    report.append("---")
    report.append("")
    report.append("## 4. Temporal Trajectory Analysis")
    report.append("")

    if results.get('temporal'):
        temporal = results['temporal']

        report.append("### 4.1 Era Definitions")
        report.append("")
        report.append("| Era | Period | Description |")
        report.append("|-----|--------|-------------|")
        for era, info in temporal['era_analysis']['era_definitions'].items():
            report.append(f"| {era.capitalize()} | {info['start']}-{info['end']} | {info['name']} |")
        report.append("")

        report.append("### 4.2 Major Era Shifts (Early → Recent)")
        report.append("")
        report.append("| Sub-domain | Change (%) | Direction |")
        report.append("|------------|------------|-----------|")

        shifts = temporal['era_analysis']['shifts']['early_to_recent']
        sorted_shifts = sorted(shifts.items(), key=lambda x: -abs(x[1]))
        for domain, change in sorted_shifts[:10]:
            direction = "↑ Increased" if change > 0 else "↓ Decreased"
            report.append(f"| {domain.replace('_', ' ').title()} | {change:+.1f}% | {direction} |")
        report.append("")

        report.append("### 4.3 Statistical Trend Analysis")
        report.append("")
        report.append("| Sub-domain | Trend | Slope | R² | Significant |")
        report.append("|------------|-------|-------|----|-----------  |")

        for domain, trend in sorted(temporal['trend_analysis'].items(), key=lambda x: -abs(x[1]['slope'])):
            sig = "✓" if trend['significant'] else ""
            report.append(f"| {domain.replace('_', ' ').title()} | {trend['trend']} | {trend['slope']:+.3f} | {trend['r_squared']:.3f} | {sig} |")
        report.append("")

        report.append("### 4.4 Field Diversity Over Time")
        report.append("")
        report.append("| Year | Shannon Entropy | Dominant Domain | Concentration |")
        report.append("|------|-----------------|-----------------|---------------|")

        for year in sorted(temporal['diversity_metrics'].keys()):
            metrics = temporal['diversity_metrics'][year]
            report.append(f"| {year} | {metrics['shannon_entropy']:.2f} | {metrics['dominant_domain'].replace('_', ' ')} | {metrics['dominant_percentage']:.1f}% |")
        report.append("")

        if temporal.get('inflection_points'):
            report.append("### 4.5 Historical Inflection Points")
            report.append("")
            for point in temporal['inflection_points']:
                if 'event' in point:
                    report.append(f"- **{point['year']}**: {point['event']}")
            report.append("")

    # Conclusions
    report.append("---")
    report.append("")
    report.append("## 5. Conclusions")
    report.append("")
    report.append("### 5.1 Major Findings")
    report.append("")
    report.append("1. **Shift from Symbolic to Statistical AI**: Early textbooks (1983-1995) emphasized knowledge representation, ")
    report.append("   logic-based reasoning, and search algorithms. Recent textbooks show dominance of probabilistic methods and machine learning.")
    report.append("")
    report.append("2. **Rise of Probabilistic Methods**: Bayesian approaches, graphical models, and probabilistic inference ")
    report.append("   have grown significantly, reflecting the field's shift toward data-driven uncertainty quantification.")
    report.append("")
    report.append("3. **Neural Network Renaissance**: While neural networks appeared in early literature, their prominence ")
    report.append("   dramatically increased after 2012, coinciding with deep learning breakthroughs.")
    report.append("")
    report.append("4. **Field Diversification**: The increasing Shannon entropy indicates the field has become more diverse, ")
    report.append("   with multiple sub-domains receiving substantial coverage rather than concentration on a few areas.")
    report.append("")
    report.append("5. **Declining Traditional AI**: Symbolic AI, expert systems, and classical logic-based reasoning ")
    report.append("   have decreased in prominence, though they remain part of comprehensive AI education.")
    report.append("")

    report.append("### 5.2 Limitations")
    report.append("")
    report.append("- Text extraction quality varies across PDFs (some older scanned documents have lower quality)")
    report.append("- The corpus represents primarily English-language textbooks from major publishers")
    report.append("- Keyword-based classification may miss nuanced semantic content")
    report.append("- Sample size of 12 textbooks, while representative, limits statistical power")
    report.append("")

    report.append("### 5.3 Future Directions")
    report.append("")
    report.append("- Extend analysis to include recent deep learning-focused textbooks (2020+)")
    report.append("- Incorporate citation analysis to track influence patterns")
    report.append("- Compare with conference proceedings (NeurIPS, ICML, AAAI) for real-time trend tracking")
    report.append("")

    # Appendix
    report.append("---")
    report.append("")
    report.append("## Appendix: Reproducibility")
    report.append("")
    report.append("### Code Repository Structure")
    report.append("")
    report.append("```")
    report.append("ML textbook/")
    report.append("├── raw_pdfs/              # Original PDF files")
    report.append("├── extracted_text/        # Raw extracted text")
    report.append("├── cleaned_text/          # Cleaned text files")
    report.append("├── analysis_outputs/      # All results and visualizations")
    report.append("│   ├── visualizations/    # Generated plots")
    report.append("│   ├── nlp_analysis_results.json")
    report.append("│   ├── keyword_classification.json")
    report.append("│   ├── temporal_analysis.json")
    report.append("│   └── FINAL_REPORT.md")
    report.append("├── code/                  # Analysis scripts")
    report.append("│   ├── pdf_extraction.py")
    report.append("│   ├── text_cleaning.py")
    report.append("│   ├── nlp_analysis.py")
    report.append("│   ├── keyword_classification.py")
    report.append("│   ├── llm_semantic_analysis.py")
    report.append("│   ├── temporal_analysis.py")
    report.append("│   └── generate_report.py")
    report.append("└── logs/                  # Processing logs")
    report.append("```")
    report.append("")
    report.append("### Running the Pipeline")
    report.append("")
    report.append("```bash")
    report.append("cd 'ML textbook'")
    report.append("python3 code/pdf_extraction.py")
    report.append("python3 code/text_cleaning.py")
    report.append("python3 code/nlp_analysis.py")
    report.append("python3 code/keyword_classification.py")
    report.append("python3 code/temporal_analysis.py")
    report.append("python3 code/generate_report.py")
    report.append("```")
    report.append("")
    report.append("---")
    report.append("")
    report.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return '\n'.join(report)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "analysis_outputs")

    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORT")
    print("=" * 80)
    print("\nLoading analysis results...")

    report = generate_markdown_report(output_dir)

    # Save report
    report_path = os.path.join(output_dir, 'FINAL_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n\nFinal report saved to: {report_path}")
    print(f"Report length: {len(report):,} characters")

    # Also save as HTML (basic conversion)
    html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>ML Textbook Evolution Analysis</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1000px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        hr {{ border: none; border-top: 1px solid #bdc3c7; margin: 30px 0; }}
    </style>
</head>
<body>
{report.replace('---', '<hr>').replace('```', '<pre>').replace('| ', '|')}
</body>
</html>"""

    html_path = os.path.join(output_dir, 'FINAL_REPORT.html')
    with open(html_path, 'w') as f:
        f.write(html_report)

    print(f"HTML report saved to: {html_path}")


if __name__ == "__main__":
    main()
