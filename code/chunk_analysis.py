#!/usr/bin/env python3
"""
Chunk-Based ML Textbook Analysis (Optimized)
1. NLP exploration to build comprehensive keyword dictionaries
2. Split books into fixed-size chunks
3. Count keywords per chunk (fast, no LLM)
4. Aggregate counts per document
5. LLM interprets aggregated counts ONCE per document
6. Sample chunks for validation examples
"""

import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================================
# COMPREHENSIVE KEYWORD DICTIONARIES
# ============================================================================

DOMAIN_KEYWORDS = {
    "supervised_learning": {
        "description": "Learning from labeled examples to predict outputs",
        "keywords": [
            "supervised learning", "labeled data", "training labels", "ground truth",
            "classification", "classifier", "regression", "regressor",
            "prediction", "predict", "target variable", "class label",
            "decision tree", "random forest", "logistic regression", "linear regression",
            "naive bayes", "k-nearest neighbor", "knn", "discriminant analysis",
            "training set", "test set", "validation set", "cross-validation",
            "confusion matrix", "precision", "recall", "f1 score", "accuracy",
            "true positive", "false positive", "roc curve", "auc",
            "overfitting", "underfitting", "loss function", "squared error",
            "cross entropy", "hinge loss", "mean squared"
        ]
    },
    "unsupervised_learning": {
        "description": "Finding patterns in unlabeled data",
        "keywords": [
            "unsupervised learning", "unlabeled data", "clustering", "cluster",
            "dimensionality reduction", "latent variable", "hidden structure",
            "k-means", "kmeans", "hierarchical clustering", "agglomerative",
            "dbscan", "spectral clustering", "gaussian mixture", "mixture model",
            "cluster center", "centroid", "silhouette",
            "pca", "principal component", "eigenvalue", "eigenvector",
            "singular value", "svd", "factor analysis", "ica",
            "manifold learning", "t-sne", "autoencoder", "representation learning",
            "em algorithm", "expectation maximization", "density estimation"
        ]
    },
    "neural_networks": {
        "description": "Connectionist models with layered computation",
        "keywords": [
            "neural network", "deep learning", "perceptron", "multilayer",
            "hidden layer", "input layer", "output layer", "feedforward",
            "convolutional", "cnn", "convnet", "recurrent", "rnn",
            "lstm", "gru", "transformer", "attention mechanism", "self-attention",
            "encoder", "decoder", "seq2seq", "activation function",
            "relu", "sigmoid", "tanh", "softmax", "dropout", "batch normalization",
            "pooling", "max pooling", "backpropagation", "backprop",
            "gradient flow", "vanishing gradient", "weight initialization",
            "residual network", "resnet", "skip connection",
            "generative adversarial", "gan", "variational autoencoder",
            "pretrained", "fine-tuning", "transfer learning", "embedding"
        ]
    },
    "probabilistic_graphical_models": {
        "description": "Structured probabilistic models with graph representations",
        "keywords": [
            "graphical model", "bayesian network", "belief network", "bayes net",
            "markov random field", "mrf", "undirected graphical",
            "directed graphical", "factor graph", "hidden markov", "hmm",
            "state space model", "conditional random field", "crf",
            "belief propagation", "message passing", "sum-product",
            "variable elimination", "junction tree", "clique tree",
            "loopy belief", "variational inference", "mean field",
            "conditional independence", "d-separation", "markov blanket",
            "marginalization", "partition function"
        ]
    },
    "bayesian_inference": {
        "description": "Probabilistic reasoning with prior beliefs and evidence",
        "keywords": [
            "bayesian", "bayes rule", "bayes theorem", "posterior",
            "prior", "likelihood", "evidence", "marginal likelihood",
            "posterior distribution", "prior distribution",
            "maximum a posteriori", "map estimate", "bayesian estimation",
            "credible interval", "posterior predictive",
            "conjugate prior", "informative prior", "hierarchical bayes",
            "mcmc", "markov chain monte carlo", "gibbs sampling",
            "metropolis", "metropolis-hastings", "importance sampling",
            "bayes factor", "model evidence", "bayesian model"
        ]
    },
    "reinforcement_learning": {
        "description": "Learning through interaction with environment and rewards",
        "keywords": [
            "reinforcement learning", "reward", "policy", "agent",
            "environment", "state", "action", "episode",
            "markov decision", "mdp", "pomdp", "partially observable",
            "discount factor", "return", "cumulative reward",
            "value function", "state value", "action value", "q-value",
            "q-learning", "sarsa", "temporal difference", "td learning",
            "bellman equation", "policy gradient", "actor-critic",
            "exploration", "exploitation", "epsilon greedy",
            "upper confidence", "ucb", "monte carlo tree", "mcts",
            "dynamic programming", "value iteration", "policy iteration"
        ]
    },
    "optimization": {
        "description": "Numerical methods for finding optimal parameters",
        "keywords": [
            "gradient descent", "stochastic gradient", "sgd", "mini-batch",
            "learning rate", "step size", "momentum", "nesterov",
            "adam", "rmsprop", "adagrad", "optimizer",
            "loss function", "cost function", "objective function",
            "minimize", "maximization", "global minimum", "local minimum",
            "convergence", "convex optimization", "convex function",
            "lagrangian", "lagrange multiplier", "dual problem", "duality",
            "kkt conditions", "constraint", "constrained optimization",
            "newton method", "hessian", "quasi-newton", "bfgs",
            "regularization", "l1 regularization", "l2 regularization",
            "lasso", "ridge", "elastic net", "weight decay"
        ]
    },
    "statistical_learning_theory": {
        "description": "Theoretical foundations of learning and generalization",
        "keywords": [
            "generalization", "generalization error", "generalization bound",
            "sample complexity", "hypothesis space", "hypothesis class",
            "pac learning", "probably approximately correct",
            "vc dimension", "vapnik-chervonenkis", "shattering",
            "rademacher complexity", "growth function", "covering number",
            "uniform convergence", "empirical risk", "true risk",
            "structural risk", "bias variance", "bias-variance tradeoff",
            "approximation error", "estimation error", "no free lunch"
        ]
    },
    "kernel_methods": {
        "description": "Nonlinear learning using kernel functions",
        "keywords": [
            "kernel", "kernel function", "kernel method", "kernel trick",
            "reproducing kernel", "rkhs", "hilbert space",
            "rbf kernel", "radial basis", "gaussian kernel",
            "polynomial kernel", "linear kernel", "string kernel",
            "support vector", "svm", "support vector machine",
            "margin", "maximum margin", "soft margin", "hard margin",
            "slack variable", "kernel pca", "kernel ridge",
            "mercer", "positive definite", "gram matrix", "feature map"
        ]
    },
    "natural_language_processing": {
        "description": "Computational processing of human language",
        "keywords": [
            "natural language", "nlp", "text", "corpus", "document",
            "token", "tokenization", "word", "sentence", "vocabulary",
            "parsing", "parse tree", "syntax", "syntactic",
            "part of speech", "pos tagging", "grammar", "constituent",
            "dependency parsing", "semantic", "meaning", "word sense",
            "named entity", "ner", "coreference", "word embedding",
            "word2vec", "language model", "n-gram", "bigram",
            "machine translation", "sentiment analysis", "text classification"
        ]
    },
    "computer_vision": {
        "description": "Visual perception and image understanding",
        "keywords": [
            "computer vision", "image", "pixel", "visual",
            "image processing", "image analysis", "object detection",
            "image classification", "recognition", "image segmentation",
            "semantic segmentation", "face recognition", "face detection",
            "edge detection", "feature extraction", "sift", "hog",
            "corner detection", "blob detection", "texture", "shape",
            "bounding box", "region proposal", "optical flow", "stereo"
        ]
    },
    "symbolic_ai_logic": {
        "description": "Knowledge representation and logical reasoning",
        "keywords": [
            "logic", "logical", "propositional logic", "predicate logic",
            "first order logic", "predicate", "quantifier",
            "inference rule", "modus ponens", "resolution",
            "knowledge representation", "knowledge base", "ontology",
            "semantic network", "frame", "schema", "slot",
            "reasoning", "deduction", "induction", "abduction",
            "forward chaining", "backward chaining", "inference engine",
            "theorem proving", "automated reasoning",
            "expert system", "rule-based", "production rule"
        ]
    },
    "search_algorithms": {
        "description": "Systematic exploration of state spaces",
        "keywords": [
            "search", "search algorithm", "state space",
            "breadth first", "bfs", "depth first", "dfs",
            "iterative deepening", "uniform cost search",
            "heuristic", "heuristic search", "a star", "a*",
            "best first", "greedy search", "admissible heuristic",
            "search tree", "search graph", "node expansion",
            "frontier", "branching factor", "completeness", "optimality",
            "local search", "hill climbing", "simulated annealing",
            "genetic algorithm", "evolutionary", "beam search"
        ]
    },
    "planning_decision_making": {
        "description": "Sequential decision problems and planning",
        "keywords": [
            "planning", "plan", "planner", "action sequence",
            "goal", "precondition", "effect", "operator",
            "strips", "pddl", "classical planning",
            "decision making", "decision theory", "utility",
            "expected utility", "utility function", "preference",
            "rational agent", "game", "game theory", "game playing",
            "minimax", "alpha-beta", "alpha beta pruning",
            "nash equilibrium", "zero-sum", "payoff",
            "multi-agent", "multiagent", "cooperative", "negotiation"
        ]
    },
    "ensemble_methods": {
        "description": "Combining multiple models for better performance",
        "keywords": [
            "ensemble", "bagging", "bootstrap", "bootstrap aggregating",
            "random forest", "boosting", "adaboost", "gradient boosting",
            "xgboost", "lightgbm", "weak learner", "strong learner",
            "voting", "majority voting", "weighted voting",
            "stacking", "blending", "model averaging", "mixture of experts"
        ]
    }
}


def split_into_chunks(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def count_keywords_in_text(text: str, keywords_dict: Dict) -> Dict[str, Dict]:
    """Count all domain keywords in text."""
    text_lower = text.lower()
    word_count = len(text.split())

    results = {}
    for domain, info in keywords_dict.items():
        domain_counts = {}
        total_count = 0

        for keyword in info['keywords']:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                domain_counts[keyword] = count
                total_count += count

        results[domain] = {
            'total_count': total_count,
            'normalized': (total_count / word_count * 1000) if word_count > 0 else 0,
            'keywords': domain_counts
        }

    return results


def llm_interpret_document_counts(keyword_counts: Dict, filename: str, year: int,
                                   model, tokenizer) -> Dict[str, float]:
    """Use LLM to interpret aggregated keyword counts for one document."""

    # Build summary
    summary_lines = []
    sorted_domains = sorted(keyword_counts.items(), key=lambda x: -x[1]['total_count'])

    for domain, counts in sorted_domains:
        if counts['total_count'] > 0:
            top_kw = sorted(counts['keywords'].items(), key=lambda x: -x[1])[:7]
            kw_str = ', '.join([f"{k}({v})" for k, v in top_kw])
            summary_lines.append(f"- {domain}: total={counts['total_count']}, top keywords: {kw_str}")

    keyword_summary = '\n'.join(summary_lines[:12])  # Top 12 domains

    prompt = f"""Analyze this AI/ML textbook and estimate what percentage of content belongs to each domain.

TEXTBOOK: {filename} (Year: {year})

KEYWORD COUNTS DETECTED:
{keyword_summary}

DOMAINS:
1. supervised_learning: Classification, regression, labeled data
2. unsupervised_learning: Clustering, dimensionality reduction
3. neural_networks: Deep learning, CNNs, RNNs, transformers
4. probabilistic_graphical_models: Bayesian networks, HMMs
5. bayesian_inference: Priors, posteriors, Bayesian estimation
6. reinforcement_learning: Rewards, policies, MDPs
7. optimization: Gradient descent, loss functions
8. statistical_learning_theory: PAC learning, VC dimension
9. kernel_methods: SVMs, kernel functions
10. natural_language_processing: Text, parsing, language models
11. computer_vision: Image processing, object detection
12. symbolic_ai_logic: Logic, knowledge representation
13. search_algorithms: A*, BFS, DFS, heuristics
14. planning_decision_making: Planning, game theory, utility
15. ensemble_methods: Boosting, bagging, random forests

Based on keyword counts, estimate percentage for each domain (sum to 100).
Higher keyword counts suggest more content on that topic.
Return JSON only:"""

    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, 'apply_chat_template'):
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text_input = prompt

    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Parse JSON
    distribution = {d: 0.0 for d in keyword_counts.keys()}
    try:
        json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            for k, v in parsed.items():
                if k in distribution:
                    distribution[k] = float(v)
    except:
        # Fallback: use normalized keyword counts
        total = sum(c['normalized'] for c in keyword_counts.values())
        if total > 0:
            distribution = {d: (c['normalized']/total)*100 for d, c in keyword_counts.items()}

    return distribution


def get_validation_examples(chunks: List[str], keywords_dict: Dict, n_examples: int = 3) -> List[Dict]:
    """Get validation examples from chunks with clear domain signals."""
    examples = []

    for i, chunk in enumerate(chunks):
        counts = count_keywords_in_text(chunk, keywords_dict)

        # Find dominant domain
        top_domain = max(counts.items(), key=lambda x: x[1]['total_count'])
        domain_name, domain_data = top_domain

        if domain_data['total_count'] >= 10:  # At least 10 keyword matches
            examples.append({
                'chunk_index': i,
                'domain': domain_name,
                'total_keywords': domain_data['total_count'],
                'top_keywords': dict(sorted(domain_data['keywords'].items(), key=lambda x: -x[1])[:10]),
                'chunk_preview': chunk[:500] + '...'
            })

    # Return top examples by keyword count
    examples.sort(key=lambda x: -x['total_keywords'])
    return examples[:n_examples]


def run_analysis(input_dir: str, output_dir: str, model_path: str):
    """Run the complete analysis."""

    print("=" * 80, flush=True)
    print("ML TEXTBOOK CHUNK-BASED ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    # Load LLM
    print("\nLoading LLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded!", flush=True)

    results = {
        'analysis_date': datetime.now().isoformat(),
        'method': 'chunk_keyword_count_with_llm_interpretation',
        'model': os.path.basename(model_path),
        'chunk_size_words': 2000,
        'taxonomy': {k: v['description'] for k, v in DOMAIN_KEYWORDS.items()},
        'documents': [],
        'validation_examples': {}
    }

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    print(f"\nAnalyzing {len(txt_files)} documents...\n", flush=True)

    for file_idx, txt_file in enumerate(txt_files):
        print(f"[{file_idx + 1}/{len(txt_files)}] {txt_file[:55]}...", flush=True)

        filepath = os.path.join(input_dir, txt_file)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        year_match = re.match(r'^(\d{4})_', txt_file)
        year = int(year_match.group(1)) if year_match else 0

        # Split into chunks
        chunks = split_into_chunks(text, chunk_size=2000)
        print(f"  {len(chunks)} chunks, {len(text.split())} words", flush=True)

        # Count keywords across entire document
        doc_keyword_counts = count_keywords_in_text(text, DOMAIN_KEYWORDS)

        # Get LLM interpretation (ONE call per document)
        print(f"  Getting LLM interpretation...", flush=True)
        distribution = llm_interpret_document_counts(doc_keyword_counts, txt_file, year, model, tokenizer)

        # Get validation examples
        validation = get_validation_examples(chunks, DOMAIN_KEYWORDS, n_examples=2)

        # Store results
        results['documents'].append({
            'filename': txt_file,
            'year': year,
            'word_count': len(text.split()),
            'num_chunks': len(chunks),
            'keyword_counts': {k: {'total': v['total_count'], 'normalized': round(v['normalized'], 2)}
                              for k, v in doc_keyword_counts.items()},
            'distribution': distribution
        })

        # Store validation examples by domain
        for ex in validation:
            domain = ex['domain']
            if domain not in results['validation_examples']:
                results['validation_examples'][domain] = []
            if len(results['validation_examples'][domain]) < 2:
                ex['source'] = txt_file
                ex['year'] = year
                del ex['chunk_preview']  # Remove to save space
                results['validation_examples'][domain].append(ex)

        # Print summary
        top3 = sorted(distribution.items(), key=lambda x: -x[1])[:3]
        print(f"  Top: {', '.join([f'{d}: {p:.1f}%' for d, p in top3])}", flush=True)

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # Compute aggregate statistics
    print("\nComputing aggregate statistics...", flush=True)
    results['aggregate'] = compute_aggregates(results['documents'])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}", flush=True)

    # Generate report
    generate_report(results, output_dir)

    return results


def compute_aggregates(documents: List[Dict]) -> Dict:
    """Compute aggregate statistics."""
    eras = {'early': (1983, 1995), 'middle': (1996, 2010), 'recent': (2011, 2022)}

    era_data = {era: defaultdict(list) for era in eras}
    overall = defaultdict(list)

    for doc in documents:
        year = doc['year']
        for domain, pct in doc['distribution'].items():
            overall[domain].append(pct)
            for era, (start, end) in eras.items():
                if start <= year <= end:
                    era_data[era][domain].append(pct)

    stats = {
        'overall': {d: {'mean': round(np.mean(v), 2), 'std': round(np.std(v), 2)}
                   for d, v in overall.items()},
        'by_era': {era: {d: round(np.mean(v), 2) if v else 0 for d, v in data.items()}
                  for era, data in era_data.items()},
        'shifts': {}
    }

    # Era shifts
    if stats['by_era'].get('early') and stats['by_era'].get('recent'):
        stats['shifts'] = {
            d: round(stats['by_era']['recent'].get(d, 0) - stats['by_era']['early'].get(d, 0), 2)
            for d in overall.keys()
        }

    return stats


def generate_report(results: Dict, output_dir: str):
    """Generate markdown report."""

    lines = [
        "# ML Textbook Analysis Report",
        "",
        f"**Date:** {results['analysis_date']}",
        f"**Method:** Keyword counting + LLM interpretation",
        f"**Model:** {results['model']}",
        f"**Documents:** {len(results['documents'])}",
        "",
        "---",
        "",
        "## Overall Domain Distribution",
        "",
        "| Domain | Mean % | Std |",
        "|--------|--------|-----|"
    ]

    overall = results['aggregate']['overall']
    for domain, stats in sorted(overall.items(), key=lambda x: -x[1]['mean']):
        lines.append(f"| {domain.replace('_', ' ').title()} | {stats['mean']:.1f} | {stats['std']:.1f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Era Comparison",
        "",
        "| Domain | Early (83-95) | Middle (96-10) | Recent (11-22) | Shift |",
        "|--------|---------------|----------------|----------------|-------|"
    ])

    by_era = results['aggregate']['by_era']
    shifts = results['aggregate']['shifts']

    for domain, _ in sorted(overall.items(), key=lambda x: -x[1]['mean'])[:12]:
        early = by_era.get('early', {}).get(domain, 0)
        middle = by_era.get('middle', {}).get(domain, 0)
        recent = by_era.get('recent', {}).get(domain, 0)
        shift = shifts.get(domain, 0)
        arrow = "↑" if shift > 1 else "↓" if shift < -1 else "→"
        lines.append(f"| {domain.replace('_', ' ').title()} | {early:.1f} | {middle:.1f} | {recent:.1f} | {arrow} {shift:+.1f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Per-Document Results",
        "",
        "| Year | Document | Top Domain | % | Second | % |",
        "|------|----------|------------|---|--------|---|"
    ])

    for doc in sorted(results['documents'], key=lambda x: x['year']):
        top2 = sorted(doc['distribution'].items(), key=lambda x: -x[1])[:2]
        name = doc['filename'].split('_')[1][:20] if '_' in doc['filename'] else doc['filename'][:20]
        lines.append(f"| {doc['year']} | {name}... | {top2[0][0].replace('_', ' ')} | {top2[0][1]:.0f} | {top2[1][0].replace('_', ' ')} | {top2[1][1]:.0f} |")

    lines.extend([
        "",
        "---",
        "",
        "## Validation Examples",
        "",
        "Keywords that led to domain classifications:",
        ""
    ])

    for domain, examples in sorted(results['validation_examples'].items()):
        if examples:
            lines.append(f"### {domain.replace('_', ' ').title()}")
            for ex in examples:
                kw_list = ', '.join([f"{k}({v})" for k, v in list(ex['top_keywords'].items())[:8]])
                lines.append(f"- **{ex['source'][:30]}** ({ex['year']}): {kw_list}")
            lines.append("")

    report_path = os.path.join(output_dir, 'ANALYSIS_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report saved to: {report_path}", flush=True)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_dir = os.path.join(project_root, "cleaned_text")
    output_dir = os.path.join(project_root, "analysis_outputs")
    model_path = "/project/jevans/maxzhuyt/models/Meta-Llama-3.1-8B-Instruct"

    run_analysis(input_dir, output_dir, model_path)
