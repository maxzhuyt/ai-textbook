#!/usr/bin/env python3
"""
Keyword-based Chapter Classification (Fast Alternative to LLM)
Uses domain-specific keyword matching to classify content.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

# Domain-specific keyword sets
DOMAIN_KEYWORDS = {
    "supervised_learning": [
        "supervised", "classification", "regression", "labeled", "label",
        "training data", "classifier", "decision boundary", "discriminant",
        "logistic regression", "linear regression", "naive bayes"
    ],
    "unsupervised_learning": [
        "unsupervised", "clustering", "cluster", "k-means", "kmeans",
        "dimensionality reduction", "pca", "principal component",
        "density estimation", "unlabeled", "self-organizing"
    ],
    "neural_networks": [
        "neural network", "deep learning", "perceptron", "backpropagation",
        "hidden layer", "activation function", "cnn", "convolutional",
        "recurrent", "rnn", "lstm", "transformer", "attention mechanism",
        "feedforward", "multilayer", "dropout", "batch normalization"
    ],
    "probabilistic_methods": [
        "bayesian", "probability", "probabilistic", "posterior", "prior",
        "likelihood", "graphical model", "belief network", "markov",
        "hidden markov", "hmm", "expectation maximization", "em algorithm",
        "gaussian", "mixture model", "monte carlo", "mcmc", "sampling"
    ],
    "reinforcement_learning": [
        "reinforcement learning", "reward", "policy", "mdp", "markov decision",
        "q-learning", "value function", "action", "state space", "temporal difference",
        "bellman", "exploration", "exploitation", "agent environment"
    ],
    "optimization": [
        "optimization", "gradient descent", "gradient", "loss function",
        "cost function", "objective function", "convex", "convergence",
        "stochastic gradient", "sgd", "momentum", "learning rate",
        "regularization", "constraint", "lagrangian"
    ],
    "learning_theory": [
        "pac learning", "vc dimension", "generalization", "bias variance",
        "computational learning", "sample complexity", "hypothesis space",
        "probably approximately correct", "rademacher", "overfitting bound"
    ],
    "feature_engineering": [
        "feature", "feature selection", "feature extraction", "embedding",
        "representation", "kernel", "kernel method", "kernel trick",
        "dimensionality", "attribute", "transform"
    ],
    "nlp": [
        "natural language", "nlp", "text", "parsing", "grammar", "syntax",
        "semantic", "word", "sentence", "language model", "tokenization",
        "pos tagging", "named entity", "machine translation", "sentiment"
    ],
    "computer_vision": [
        "computer vision", "image", "pixel", "object detection", "recognition",
        "segmentation", "edge detection", "filter", "visual", "scene"
    ],
    "search_planning": [
        "search", "heuristic", "a*", "breadth first", "depth first",
        "planning", "state space", "goal", "path", "tree search",
        "graph search", "minimax", "alpha beta", "game"
    ],
    "knowledge_reasoning": [
        "knowledge representation", "reasoning", "logic", "inference",
        "expert system", "rule", "ontology", "semantic network",
        "frame", "predicate", "propositional", "first order"
    ],
    "robotics_agents": [
        "robot", "robotics", "agent", "multi-agent", "perception",
        "control", "sensor", "actuator", "navigation", "manipulation"
    ],
    "foundations": [
        "matrix", "vector", "linear algebra", "probability theory",
        "statistics", "calculus", "derivative", "integral", "expected value"
    ]
}


def classify_text(text: str) -> Dict[str, float]:
    """Classify text based on keyword frequency."""
    text_lower = text.lower()
    word_count = len(text_lower.split())

    if word_count == 0:
        return {k: 0 for k in DOMAIN_KEYWORDS.keys()}

    scores = {}
    total_matches = 0

    for domain, keywords in DOMAIN_KEYWORDS.items():
        domain_score = 0
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            # Weight longer keywords more
            weight = len(keyword.split())
            domain_score += matches * weight

        scores[domain] = domain_score
        total_matches += domain_score

    # Normalize to percentages
    if total_matches > 0:
        scores = {k: (v / total_matches) * 100 for k, v in scores.items()}

    return scores


def extract_chapters(text: str, max_chapters: int = 30) -> List[Dict]:
    """Extract chapters from text."""
    chapters = []

    patterns = [
        r'^(?:CHAPTER|Chapter)\s+(\d+)[:\.\s]+(.+?)$',
        r'^(\d{1,2})\s+([A-Z][A-Za-z\s,\-]+)$',
    ]

    lines = text.split('\n')
    chapter_positions = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 3 or len(stripped) > 100:
            continue

        for pattern in patterns:
            match = re.match(pattern, stripped)
            if match:
                chapter_positions.append({
                    'line': i,
                    'number': match.group(1),
                    'title': match.group(2).strip() if len(match.groups()) > 1 else ''
                })
                break

    for i, chap in enumerate(chapter_positions[:max_chapters]):
        start_line = chap['line']
        end_line = chapter_positions[i + 1]['line'] if i < len(chapter_positions) - 1 else len(lines)

        content = '\n'.join(lines[start_line:end_line])
        word_count = len(content.split())

        chapters.append({
            'number': chap['number'],
            'title': chap['title'],
            'content': content,
            'word_count': word_count
        })

    return chapters


def analyze_all_textbooks(input_dir: str, output_dir: str):
    """Analyze all textbooks using keyword classification."""
    print("\n" + "=" * 80)
    print("KEYWORD-BASED SEMANTIC CLASSIFICATION")
    print("=" * 80)

    results = {
        'analysis_date': datetime.now().isoformat(),
        'method': 'keyword_frequency',
        'documents': []
    }

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    for file_idx, txt_file in enumerate(txt_files):
        print(f"\n[{file_idx + 1}/{len(txt_files)}] {txt_file[:50]}...")

        filepath = os.path.join(input_dir, txt_file)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        year_match = re.match(r'^(\d{4})_', txt_file)
        year = int(year_match.group(1)) if year_match else 0

        # Classify whole document
        doc_classification = classify_text(text)

        # Extract and classify chapters
        chapters = extract_chapters(text)
        chapter_results = []

        for chap in chapters:
            chap_classification = classify_text(chap['content'])
            chapter_results.append({
                'number': chap['number'],
                'title': chap['title'],
                'word_count': chap['word_count'],
                'classification': chap_classification
            })

        # Calculate aggregate
        aggregate = defaultdict(float)
        if chapter_results:
            for chap in chapter_results:
                for domain, pct in chap['classification'].items():
                    aggregate[domain] += pct
            aggregate = {k: v / len(chapter_results) for k, v in aggregate.items()}
        else:
            aggregate = doc_classification

        results['documents'].append({
            'filename': txt_file,
            'year': year,
            'chapter_count': len(chapter_results),
            'document_classification': doc_classification,
            'aggregate_classification': dict(aggregate),
            'chapters': chapter_results
        })

        # Print top domains
        top_domains = sorted(aggregate.items(), key=lambda x: -x[1])[:3]
        print(f"  Top domains: {', '.join([f'{d}: {p:.1f}%' for d, p in top_domains])}")

    # Save results
    output_path = os.path.join(output_dir, 'keyword_classification.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_dir = os.path.join(project_root, "cleaned_text")
    output_dir = os.path.join(project_root, "analysis_outputs")

    os.makedirs(output_dir, exist_ok=True)
    analyze_all_textbooks(input_dir, output_dir)
