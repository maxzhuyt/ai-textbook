#!/usr/bin/env python3
"""
NLP Keyword Trend Analysis - Continuous Time Flow
Analyzes NLP keyword distribution across ML textbooks from 1983-2022.
Uses keyword counting without LLM for fast, reproducible results.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

# NLP Keywords from comprehensive dictionary
NLP_KEYWORDS = [
    "natural language", "nlp", "text", "corpus", "document",
    "token", "tokenization", "word", "sentence", "vocabulary",
    "parsing", "parse tree", "syntax", "syntactic",
    "part of speech", "pos tagging", "grammar", "constituent",
    "dependency parsing", "semantic", "meaning", "word sense",
    "named entity", "ner", "coreference", "word embedding",
    "word2vec", "language model", "n-gram", "bigram",
    "machine translation", "sentiment analysis", "text classification"
]

# Extended NLP keywords for finer-grained analysis
NLP_SUBCATEGORIES = {
    "core_nlp": [
        "natural language", "nlp", "text processing", "corpus", "linguistic"
    ],
    "tokenization_lexical": [
        "token", "tokenization", "tokenize", "word", "vocabulary", "lexicon",
        "lexical", "morphology", "morphological", "stemming", "lemma"
    ],
    "syntax_parsing": [
        "parsing", "parse tree", "parser", "syntax", "syntactic", "grammar",
        "constituent", "dependency parsing", "phrase structure", "cfg",
        "context-free grammar", "part of speech", "pos tagging", "pos tag"
    ],
    "semantics": [
        "semantic", "meaning", "word sense", "disambiguation", "semantic role",
        "semantic similarity", "ontology", "wordnet", "concept"
    ],
    "ner_ie": [
        "named entity", "ner", "entity recognition", "information extraction",
        "relation extraction", "coreference", "anaphora"
    ],
    "embeddings_representation": [
        "word embedding", "word2vec", "glove", "fasttext", "embedding",
        "distributed representation", "word vector", "dense representation"
    ],
    "language_models": [
        "language model", "n-gram", "bigram", "trigram", "unigram",
        "perplexity", "probability model", "statistical language"
    ],
    "applications": [
        "machine translation", "sentiment analysis", "text classification",
        "question answering", "summarization", "text generation",
        "speech recognition", "dialogue", "chatbot"
    ],
    "neural_nlp": [
        "transformer", "attention", "bert", "gpt", "seq2seq",
        "encoder decoder", "recurrent", "lstm", "neural machine translation"
    ]
}


def count_keywords(text: str, keywords: list) -> dict:
    """Count keyword occurrences in text."""
    text_lower = text.lower()
    counts = {}
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            counts[keyword] = count
    return counts


def analyze_textbook(filepath: str) -> dict:
    """Analyze a single textbook for NLP keywords."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    word_count = len(text.split())
    filename = os.path.basename(filepath)

    # Extract year from filename
    year_match = re.match(r'^(\d{4})_', filename)
    year = int(year_match.group(1)) if year_match else 0

    # Count all NLP keywords
    all_nlp_counts = count_keywords(text, NLP_KEYWORDS)
    total_nlp = sum(all_nlp_counts.values())

    # Count by subcategory
    subcategory_counts = {}
    for subcat, keywords in NLP_SUBCATEGORIES.items():
        counts = count_keywords(text, keywords)
        subcategory_counts[subcat] = {
            'total': sum(counts.values()),
            'normalized': sum(counts.values()) / word_count * 10000,  # per 10k words
            'keywords': counts
        }

    return {
        'filename': filename,
        'year': year,
        'word_count': word_count,
        'total_nlp_keywords': total_nlp,
        'nlp_density': total_nlp / word_count * 10000,  # per 10k words
        'keyword_counts': all_nlp_counts,
        'subcategories': subcategory_counts
    }


def print_results(results: list):
    """Print formatted analysis results."""

    print("\n" + "=" * 80)
    print("NLP KEYWORD TREND ANALYSIS - CONTINUOUS TIME FLOW")
    print("=" * 80)

    # Sort by year for continuous time view
    results = sorted(results, key=lambda x: x['year'])

    # Section 1: Overall NLP Presence Over Time
    print("\n" + "-" * 80)
    print("1. NLP KEYWORD DENSITY OVER TIME (per 10,000 words)")
    print("-" * 80)
    print(f"{'Year':<6} {'Book':<45} {'NLP Density':>12} {'Total KW':>10}")
    print("-" * 80)

    for r in results:
        short_name = r['filename'].split('_')[1][:40] if '_' in r['filename'] else r['filename'][:40]
        print(f"{r['year']:<6} {short_name:<45} {r['nlp_density']:>12.2f} {r['total_nlp_keywords']:>10}")

    # Section 2: Top Keywords Per Book
    print("\n" + "-" * 80)
    print("2. TOP NLP KEYWORDS BY BOOK")
    print("-" * 80)

    for r in results:
        if r['keyword_counts']:
            top5 = sorted(r['keyword_counts'].items(), key=lambda x: -x[1])[:5]
            kw_str = ", ".join([f"{k}({v})" for k, v in top5])
            print(f"\n{r['year']} - {r['filename'][:50]}...")
            print(f"  → {kw_str}")

    # Section 3: Subcategory Analysis Over Time
    print("\n" + "-" * 80)
    print("3. NLP SUBCATEGORY TRENDS (normalized density per 10k words)")
    print("-" * 80)

    subcats = list(NLP_SUBCATEGORIES.keys())

    # Print header
    header = f"{'Year':<6}"
    for sc in subcats:
        short_sc = sc[:8]
        header += f" {short_sc:>9}"
    print(header)
    print("-" * 80)

    for r in results:
        row = f"{r['year']:<6}"
        for sc in subcats:
            density = r['subcategories'][sc]['normalized']
            row += f" {density:>9.2f}"
        print(row)

    # Section 4: Temporal Evolution Summary
    print("\n" + "-" * 80)
    print("4. TEMPORAL EVOLUTION SUMMARY")
    print("-" * 80)

    # Calculate trends using linear regression
    years = np.array([r['year'] for r in results])
    densities = np.array([r['nlp_density'] for r in results])

    if len(years) > 1:
        # Linear fit
        slope, intercept = np.polyfit(years, densities, 1)
        trend_dir = "INCREASING" if slope > 0 else "DECREASING"
        print(f"\nOverall NLP Trend: {trend_dir}")
        print(f"  Rate of change: {slope:.4f} keywords/10k words per year")
        print(f"  1983 estimated: {intercept + slope * 1983:.2f}")
        print(f"  2022 estimated: {intercept + slope * 2022:.2f}")

    # Subcategory trends
    print("\nSubcategory Trends:")
    print(f"{'Subcategory':<25} {'1983-2000 Avg':>14} {'2001-2022 Avg':>14} {'Change':>10}")
    print("-" * 65)

    for sc in subcats:
        early_vals = [r['subcategories'][sc]['normalized'] for r in results if r['year'] <= 2000]
        later_vals = [r['subcategories'][sc]['normalized'] for r in results if r['year'] > 2000]

        early_avg = np.mean(early_vals) if early_vals else 0
        later_avg = np.mean(later_vals) if later_vals else 0
        change = later_avg - early_avg

        arrow = "↑" if change > 0.1 else "↓" if change < -0.1 else "→"
        print(f"{sc:<25} {early_avg:>14.2f} {later_avg:>14.2f} {arrow:>2} {change:>+7.2f}")

    # Section 5: Key Observations
    print("\n" + "-" * 80)
    print("5. KEY OBSERVATIONS")
    print("-" * 80)

    # Find peak NLP book
    max_nlp = max(results, key=lambda x: x['nlp_density'])
    min_nlp = min(results, key=lambda x: x['nlp_density'])

    print(f"\n  • Highest NLP density: {max_nlp['year']} ({max_nlp['nlp_density']:.2f}/10k words)")
    print(f"  • Lowest NLP density: {min_nlp['year']} ({min_nlp['nlp_density']:.2f}/10k words)")

    # Emerging keywords (appear more in later books)
    early_books = [r for r in results if r['year'] <= 2000]
    later_books = [r for r in results if r['year'] > 2000]

    early_kw = defaultdict(int)
    later_kw = defaultdict(int)

    for r in early_books:
        for kw, count in r['keyword_counts'].items():
            early_kw[kw] += count

    for r in later_books:
        for kw, count in r['keyword_counts'].items():
            later_kw[kw] += count

    # Normalize by number of books
    if early_books:
        early_kw = {k: v/len(early_books) for k, v in early_kw.items()}
    if later_books:
        later_kw = {k: v/len(later_books) for k, v in later_kw.items()}

    # Emerging (much higher in later)
    emerging = []
    declining = []
    for kw in set(early_kw.keys()) | set(later_kw.keys()):
        e = early_kw.get(kw, 0)
        l = later_kw.get(kw, 0)
        if l > e * 2 and l > 5:
            emerging.append((kw, e, l))
        elif e > l * 2 and e > 5:
            declining.append((kw, e, l))

    if emerging:
        emerging.sort(key=lambda x: -x[2])
        print(f"\n  • Emerging NLP terms (2001-2022 vs 1983-2000):")
        for kw, e, l in emerging[:7]:
            print(f"      {kw}: {e:.1f} → {l:.1f} (↑ {l/e if e > 0 else float('inf'):.1f}x)")

    if declining:
        declining.sort(key=lambda x: -x[1])
        print(f"\n  • Declining NLP terms:")
        for kw, e, l in declining[:5]:
            print(f"      {kw}: {e:.1f} → {l:.1f} (↓)")

    # Section 6: Year-by-Year Continuous Analysis
    print("\n" + "-" * 80)
    print("6. CONTINUOUS TIME ANALYSIS")
    print("-" * 80)

    print("\nNLP research emphasis trajectory (chronological):\n")

    prev_year = None
    for r in results:
        year_gap = ""
        if prev_year and r['year'] - prev_year > 1:
            year_gap = f"  [+{r['year'] - prev_year} years]"

        # Get dominant subcategory
        top_subcat = max(r['subcategories'].items(), key=lambda x: x[1]['total'])

        print(f"  {r['year']}{year_gap}")
        print(f"    └── Focus: {top_subcat[0]} ({top_subcat[1]['total']} kw)")

        # Top 3 keywords
        if r['keyword_counts']:
            top3 = sorted(r['keyword_counts'].items(), key=lambda x: -x[1])[:3]
            print(f"        Top terms: {', '.join([k for k, v in top3])}")

        prev_year = r['year']

    print("\n" + "=" * 80)
    print("END OF NLP TREND ANALYSIS")
    print("=" * 80)


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cleaned_text_dir = project_root / "cleaned_text"

    print(f"Analyzing textbooks in: {cleaned_text_dir}")

    txt_files = sorted(cleaned_text_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} textbooks\n")

    results = []
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name[:60]}...")
        result = analyze_textbook(str(txt_file))
        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
