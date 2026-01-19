#!/usr/bin/env python3
"""
Layer 2: LLM-Based Semantic Content Analysis for ML Textbooks
Uses local LLM to classify chapter content into AI/ML sub-domains.
"""

import os
import re
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer

# AI/ML Sub-domain taxonomy
SUBDOMAIN_TAXONOMY = {
    "supervised_learning": "Supervised Learning (classification, regression, labeled data)",
    "unsupervised_learning": "Unsupervised Learning (clustering, dimensionality reduction, density estimation)",
    "neural_networks": "Neural Networks & Deep Learning (perceptrons, MLPs, CNNs, RNNs, transformers)",
    "probabilistic_methods": "Probabilistic Methods (Bayesian inference, graphical models, HMMs)",
    "reinforcement_learning": "Reinforcement Learning (MDPs, Q-learning, policy gradient)",
    "optimization": "Optimization Theory (gradient descent, convex optimization, loss functions)",
    "learning_theory": "Learning Theory (PAC learning, VC dimension, generalization bounds)",
    "feature_engineering": "Feature Engineering & Representation (feature selection, embeddings, kernels)",
    "nlp": "Natural Language Processing (text processing, parsing, language models)",
    "computer_vision": "Computer Vision (image processing, object detection, recognition)",
    "search_planning": "Search & Planning (state space search, heuristics, game playing)",
    "knowledge_reasoning": "Knowledge Representation & Reasoning (logic, expert systems, ontologies)",
    "robotics_agents": "Robotics & Intelligent Agents (perception, control, multi-agent systems)",
    "foundations": "Mathematical Foundations (linear algebra, probability, statistics basics)"
}


class ChapterExtractor:
    """Extract and segment chapters from textbook text."""

    def __init__(self, text: str, filename: str):
        self.text = text
        self.filename = filename

    def extract_chapters(self, max_chapters: int = 30) -> List[Dict]:
        """Extract chapters with their content."""
        chapters = []

        # Multiple chapter detection patterns
        patterns = [
            (r'^(?:CHAPTER|Chapter)\s+(\d+)[:\.\s]+(.+?)$', 'numbered'),
            (r'^(\d{1,2})\s+([A-Z][A-Za-z\s,\-]+)$', 'simple_numbered'),
            (r'^Part\s+(\w+)[:\s]+(.+?)$', 'part'),
        ]

        lines = self.text.split('\n')
        chapter_positions = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) < 3 or len(stripped) > 100:
                continue

            for pattern, ptype in patterns:
                match = re.match(pattern, stripped)
                if match:
                    chapter_positions.append({
                        'line': i,
                        'number': match.group(1),
                        'title': match.group(2).strip() if len(match.groups()) > 1 else '',
                        'type': ptype
                    })
                    break

        # Extract content between chapters
        for i, chap in enumerate(chapter_positions[:max_chapters]):
            start_line = chap['line']
            end_line = chapter_positions[i + 1]['line'] if i < len(chapter_positions) - 1 else len(lines)

            # Get first ~3000 words of chapter for classification
            chapter_lines = lines[start_line:min(end_line, start_line + 500)]
            content = '\n'.join(chapter_lines)

            # Truncate to ~2000 words for LLM
            words = content.split()[:2000]
            content = ' '.join(words)

            chapters.append({
                'number': chap['number'],
                'title': chap['title'],
                'content_preview': content,
                'word_count': len(words)
            })

        return chapters


class LLMClassifier:
    """Use local LLM to classify chapter content."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the LLM model."""
        print(f"Loading model from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print("Model loaded successfully!")

    def classify_chapter(self, chapter_content: str, chapter_title: str) -> Dict:
        """Classify a chapter into AI/ML subdomains."""
        # Build taxonomy string
        taxonomy_str = "\n".join([f"- {k}: {v}" for k, v in SUBDOMAIN_TAXONOMY.items()])

        prompt = f"""You are an expert AI/ML researcher analyzing textbook chapters. Given the chapter title and content excerpt, classify what percentage of this chapter belongs to each AI/ML sub-domain.

SUB-DOMAIN TAXONOMY:
{taxonomy_str}

CHAPTER TITLE: {chapter_title}

CHAPTER CONTENT (excerpt):
{chapter_content[:3000]}

TASK: Return a JSON object with the classification. Each sub-domain should have a percentage (0-100). Percentages should sum to approximately 100.

Return ONLY valid JSON in this exact format, no other text:
{{"supervised_learning": 0, "unsupervised_learning": 0, "neural_networks": 0, "probabilistic_methods": 0, "reinforcement_learning": 0, "optimization": 0, "learning_theory": 0, "feature_engineering": 0, "nlp": 0, "computer_vision": 0, "search_planning": 0, "knowledge_reasoning": 0, "robotics_agents": 0, "foundations": 0}}"""

        messages = [{"role": "user", "content": prompt}]

        # Format for the model
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Parse JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                # Normalize to ensure all keys exist
                for key in SUBDOMAIN_TAXONOMY.keys():
                    if key not in classification:
                        classification[key] = 0
                return classification
        except json.JSONDecodeError:
            pass

        # Return default if parsing fails
        return {k: 0 for k in SUBDOMAIN_TAXONOMY.keys()}

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            print("Model unloaded, GPU memory freed.")


def analyze_textbooks(input_dir: str, output_dir: str, model_path: str, max_chapters_per_book: int = 20):
    """Run semantic analysis on all textbooks."""
    print("\n" + "=" * 80)
    print("LAYER 2: LLM-BASED SEMANTIC CONTENT ANALYSIS")
    print("=" * 80)

    # Initialize classifier
    classifier = LLMClassifier(model_path)
    classifier.load_model()

    results = {
        'analysis_date': datetime.now().isoformat(),
        'model_used': os.path.basename(model_path),
        'taxonomy': SUBDOMAIN_TAXONOMY,
        'documents': []
    }

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    for file_idx, txt_file in enumerate(txt_files):
        print(f"\n[{file_idx + 1}/{len(txt_files)}] {txt_file[:60]}...")

        filepath = os.path.join(input_dir, txt_file)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Extract year
        year_match = re.match(r'^(\d{4})_', txt_file)
        year = int(year_match.group(1)) if year_match else 0

        # Extract chapters
        extractor = ChapterExtractor(text, txt_file)
        chapters = extractor.extract_chapters(max_chapters=max_chapters_per_book)

        print(f"  Found {len(chapters)} chapters")

        doc_result = {
            'filename': txt_file,
            'year': year,
            'chapter_count': len(chapters),
            'chapters': [],
            'aggregate_classification': defaultdict(float)
        }

        for chap_idx, chapter in enumerate(chapters):
            print(f"    Classifying chapter {chap_idx + 1}/{len(chapters)}: {chapter['title'][:40]}...", end=" ", flush=True)

            classification = classifier.classify_chapter(
                chapter['content_preview'],
                chapter['title']
            )

            # Store chapter result
            doc_result['chapters'].append({
                'number': chapter['number'],
                'title': chapter['title'],
                'word_count': chapter['word_count'],
                'classification': classification
            })

            # Aggregate
            for domain, pct in classification.items():
                doc_result['aggregate_classification'][domain] += pct

            print("Done")

        # Normalize aggregate
        if len(chapters) > 0:
            for domain in doc_result['aggregate_classification']:
                doc_result['aggregate_classification'][domain] /= len(chapters)

        doc_result['aggregate_classification'] = dict(doc_result['aggregate_classification'])
        results['documents'].append(doc_result)

    # Cleanup
    classifier.unload_model()

    # Save results
    output_path = os.path.join(output_dir, 'llm_semantic_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Generate summary
    generate_summary(results, output_dir)

    return results


def generate_summary(results: Dict, output_dir: str):
    """Generate summary statistics and visualizations."""
    import matplotlib.pyplot as plt
    import numpy as np

    print("\nGenerating semantic analysis summary...")

    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Prepare data
    years = []
    domain_data = defaultdict(list)

    for doc in results['documents']:
        years.append(doc['year'])
        for domain, pct in doc['aggregate_classification'].items():
            domain_data[domain].append(pct)

    # Create temporal evolution plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by year for plotting
    sorted_indices = np.argsort(years)
    sorted_years = [years[i] for i in sorted_indices]

    # Top domains plot
    ax = axes[0, 0]
    top_domains = ['neural_networks', 'probabilistic_methods', 'supervised_learning',
                   'search_planning', 'knowledge_reasoning', 'reinforcement_learning']

    for domain in top_domains:
        if domain in domain_data:
            sorted_values = [domain_data[domain][i] for i in sorted_indices]
            ax.plot(sorted_years, sorted_values, marker='o', label=domain.replace('_', ' ').title())

    ax.set_xlabel('Year')
    ax.set_ylabel('Average % Allocation')
    ax.set_title('Evolution of Major AI/ML Sub-domains')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Stacked area plot for all domains
    ax = axes[0, 1]
    domain_names = list(SUBDOMAIN_TAXONOMY.keys())
    data_matrix = np.array([[domain_data[d][i] for i in sorted_indices] for d in domain_names])

    ax.stackplot(sorted_years, data_matrix, labels=[d.replace('_', ' ') for d in domain_names], alpha=0.8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative % Allocation')
    ax.set_title('Sub-domain Distribution Over Time (Stacked)')
    ax.legend(loc='upper left', fontsize=6, ncol=2)

    # Era comparison bar chart
    ax = axes[1, 0]
    era_data = {'early': defaultdict(list), 'middle': defaultdict(list), 'recent': defaultdict(list)}

    for doc in results['documents']:
        year = doc['year']
        if year <= 1995:
            era = 'early'
        elif year <= 2010:
            era = 'middle'
        else:
            era = 'recent'

        for domain, pct in doc['aggregate_classification'].items():
            era_data[era][domain].append(pct)

    # Calculate era averages
    era_avgs = {}
    for era in ['early', 'middle', 'recent']:
        era_avgs[era] = {d: np.mean(v) if v else 0 for d, v in era_data[era].items()}

    # Plot top 8 domains
    top8_domains = sorted(domain_names, key=lambda d: sum(era_avgs[e].get(d, 0) for e in era_avgs), reverse=True)[:8]

    x = np.arange(len(top8_domains))
    width = 0.25

    for i, era in enumerate(['early', 'middle', 'recent']):
        values = [era_avgs[era].get(d, 0) for d in top8_domains]
        ax.bar(x + i * width, values, width, label=f"{era.capitalize()}")

    ax.set_xlabel('Sub-domain')
    ax.set_ylabel('Average % Allocation')
    ax.set_title('Sub-domain Emphasis by Era')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('_', '\n') for d in top8_domains], fontsize=8)
    ax.legend()

    # Summary statistics text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "SEMANTIC ANALYSIS SUMMARY\n" + "=" * 40 + "\n\n"
    summary_text += f"Documents analyzed: {len(results['documents'])}\n"
    summary_text += f"Model used: {results['model_used']}\n\n"

    summary_text += "TOP DOMAINS OVERALL:\n"
    overall_avgs = {d: np.mean(domain_data[d]) for d in domain_names if domain_data[d]}
    for d, v in sorted(overall_avgs.items(), key=lambda x: -x[1])[:6]:
        summary_text += f"  {d.replace('_', ' ').title()}: {v:.1f}%\n"

    summary_text += "\nKEY OBSERVATIONS:\n"
    if era_avgs['early'].get('neural_networks', 0) < era_avgs['recent'].get('neural_networks', 0):
        summary_text += "  - Neural networks emphasis increased over time\n"
    if era_avgs['early'].get('knowledge_reasoning', 0) > era_avgs['recent'].get('knowledge_reasoning', 0):
        summary_text += "  - Knowledge/reasoning emphasis decreased\n"
    if era_avgs['recent'].get('probabilistic_methods', 0) > era_avgs['early'].get('probabilistic_methods', 0):
        summary_text += "  - Probabilistic methods became more prominent\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'semantic_analysis_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Summary visualization saved to: {viz_dir}/semantic_analysis_summary.png")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_dir = os.path.join(project_root, "cleaned_text")
    output_dir = os.path.join(project_root, "analysis_outputs")

    # Use Mistral Small for classification (good balance of speed and quality)
    model_path = "/project/jevans/maxzhuyt/models/Mistral-Small-3.2-24B-Instruct-2506"

    # Check if model exists, fallback to alternatives
    if not os.path.exists(model_path):
        alternatives = [
            "/project/jevans/maxzhuyt/models/Meta-Llama-3.1-8B-Instruct",
            "/project/jevans/maxzhuyt/models/Qwen2.5-7B-Instruct"
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                break

    os.makedirs(output_dir, exist_ok=True)

    analyze_textbooks(input_dir, output_dir, model_path, max_chapters_per_book=15)


if __name__ == "__main__":
    main()
