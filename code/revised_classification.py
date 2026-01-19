#!/usr/bin/env python3
"""
Revised Semantic Classification with Conceptually Isolated Sub-domains
Uses keyword-based approach on full text (no chapter segmentation).
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# REVISED TAXONOMY - Conceptually Isolated Categories
SUBDOMAIN_TAXONOMY = {
    "supervised_learning": {
        "description": "Supervised Learning: learning from labeled data",
        "keywords": [
            "supervised", "classification", "classifier", "regression", "labeled data",
            "training labels", "decision boundary", "discriminant", "logistic regression",
            "linear regression", "naive bayes", "decision tree", "random forest",
            "class label", "target variable", "prediction", "training set"
        ]
    },
    "unsupervised_learning": {
        "description": "Unsupervised Learning: discovering structure in unlabeled data",
        "keywords": [
            "unsupervised", "clustering", "cluster analysis", "k-means", "kmeans",
            "hierarchical clustering", "dimensionality reduction", "pca",
            "principal component", "density estimation", "unlabeled", "self-organizing",
            "latent variable", "mixture model", "em algorithm", "expectation maximization"
        ]
    },
    "neural_networks_deep_learning": {
        "description": "Neural Networks & Deep Learning: connectionist models",
        "keywords": [
            "neural network", "deep learning", "perceptron", "backpropagation",
            "hidden layer", "activation function", "convolutional", "cnn",
            "recurrent", "rnn", "lstm", "transformer", "attention mechanism",
            "feedforward", "multilayer", "dropout", "batch normalization",
            "autoencoder", "deep neural", "relu", "sigmoid", "softmax"
        ]
    },
    "probabilistic_graphical_models": {
        "description": "Probabilistic Graphical Models: structured probabilistic inference",
        "keywords": [
            "bayesian network", "belief network", "graphical model", "markov random field",
            "hidden markov", "hmm", "belief propagation", "message passing",
            "conditional independence", "d-separation", "junction tree",
            "factor graph", "undirected graphical", "directed graphical"
        ]
    },
    "bayesian_inference": {
        "description": "Bayesian Inference: probabilistic reasoning with priors",
        "keywords": [
            "bayesian", "posterior", "prior", "likelihood", "bayes rule",
            "conjugate prior", "bayesian inference", "credible interval",
            "maximum a posteriori", "map estimate", "bayesian estimation",
            "hierarchical bayes", "empirical bayes"
        ]
    },
    "reinforcement_learning": {
        "description": "Reinforcement Learning: learning from rewards through interaction",
        "keywords": [
            "reinforcement learning", "reward", "policy", "mdp", "markov decision",
            "q-learning", "value function", "temporal difference", "td learning",
            "bellman equation", "exploration exploitation", "policy gradient",
            "actor critic", "sarsa", "monte carlo tree"
        ]
    },
    "optimization_methods": {
        "description": "Optimization Methods: numerical optimization techniques",
        "keywords": [
            "gradient descent", "stochastic gradient", "sgd", "optimization",
            "loss function", "cost function", "objective function", "convex optimization",
            "convergence", "learning rate", "momentum", "adam optimizer",
            "lagrangian", "constraint optimization", "newton method"
        ]
    },
    "statistical_learning_theory": {
        "description": "Statistical Learning Theory: theoretical foundations of learning",
        "keywords": [
            "pac learning", "vc dimension", "generalization bound", "bias variance",
            "computational learning", "sample complexity", "hypothesis space",
            "rademacher complexity", "probably approximately correct",
            "structural risk", "empirical risk", "uniform convergence"
        ]
    },
    "kernel_methods": {
        "description": "Kernel Methods: nonlinear learning via kernel functions",
        "keywords": [
            "kernel", "kernel method", "kernel trick", "support vector",
            "svm", "reproducing kernel", "rkhs", "kernel function",
            "radial basis", "rbf kernel", "polynomial kernel", "kernel pca",
            "kernel ridge", "gaussian kernel"
        ]
    },
    "natural_language_processing": {
        "description": "Natural Language Processing: computational linguistics",
        "keywords": [
            "natural language", "nlp", "parsing", "grammar", "syntax",
            "semantic", "language model", "tokenization", "pos tagging",
            "named entity", "machine translation", "sentiment analysis",
            "word embedding", "word2vec", "text classification"
        ]
    },
    "computer_vision": {
        "description": "Computer Vision: visual perception and image understanding",
        "keywords": [
            "computer vision", "image processing", "object detection", "image recognition",
            "segmentation", "edge detection", "feature extraction", "sift",
            "image classification", "object recognition", "scene understanding",
            "visual recognition", "face recognition", "optical flow"
        ]
    },
    "symbolic_ai_logic": {
        "description": "Symbolic AI & Logic: knowledge representation and logical reasoning",
        "keywords": [
            "knowledge representation", "expert system", "logical reasoning",
            "first order logic", "propositional logic", "predicate logic",
            "theorem proving", "inference rule", "forward chaining", "backward chaining",
            "ontology", "semantic network", "frame", "rule-based", "symbolic"
        ]
    },
    "search_algorithms": {
        "description": "Search Algorithms: systematic exploration of state spaces",
        "keywords": [
            "search algorithm", "a-star", "a*", "breadth first", "depth first",
            "heuristic search", "best first", "iterative deepening", "branch and bound",
            "graph search", "tree search", "state space", "search tree",
            "admissible heuristic", "uniform cost"
        ]
    },
    "planning_decision_making": {
        "description": "Planning & Decision Making: sequential decision problems",
        "keywords": [
            "planning", "decision making", "utility", "decision theory",
            "game playing", "minimax", "alpha beta", "game tree",
            "partial observability", "pomdp", "sequential decision",
            "goal-based", "means-ends", "strips", "classical planning"
        ]
    },
    "ensemble_methods": {
        "description": "Ensemble Methods: combining multiple models",
        "keywords": [
            "ensemble", "boosting", "adaboost", "gradient boosting", "xgboost",
            "bagging", "bootstrap aggregating", "random forest", "voting",
            "model averaging", "stacking", "model combination", "weak learner"
        ]
    }
}


def classify_text_keywords(text: str) -> Dict[str, float]:
    """Classify text using revised keyword taxonomy."""
    text_lower = text.lower()
    word_count = len(text_lower.split())

    if word_count == 0:
        return {k: 0 for k in SUBDOMAIN_TAXONOMY.keys()}

    scores = {}
    total_matches = 0

    for domain, info in SUBDOMAIN_TAXONOMY.items():
        domain_score = 0
        for keyword in info['keywords']:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            weight = len(keyword.split())  # Weight multi-word terms more
            domain_score += matches * weight

        scores[domain] = domain_score
        total_matches += domain_score

    # Normalize to percentages
    if total_matches > 0:
        scores = {k: (v / total_matches) * 100 for k, v in scores.items()}

    return scores


def run_keyword_classification(input_dir: str, output_dir: str) -> Dict:
    """Run keyword-based classification with revised taxonomy on full text."""
    print("\n" + "=" * 80)
    print("REVISED KEYWORD-BASED CLASSIFICATION (FULL TEXT)")
    print("=" * 80)

    results = {
        'analysis_date': datetime.now().isoformat(),
        'method': 'revised_keyword_taxonomy_full_text',
        'taxonomy': {k: v['description'] for k, v in SUBDOMAIN_TAXONOMY.items()},
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

        word_count = len(text.split())

        # Classify full document
        classification = classify_text_keywords(text)

        results['documents'].append({
            'filename': txt_file,
            'year': year,
            'word_count': word_count,
            'classification': classification
        })

        # Print top domains
        top_domains = sorted(classification.items(), key=lambda x: -x[1])[:3]
        top_str = ', '.join([f"{d.replace('_', ' ')}: {p:.1f}%" for d, p in top_domains])
        print(f"  Top: {top_str}")

    # Save results
    output_path = os.path.join(output_dir, 'revised_classification.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    return results


def run_llm_classification(input_dir: str, output_dir: str, model_path: str) -> Dict:
    """Run LLM-based classification with revised taxonomy."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 80)
    print("LLM-BASED CLASSIFICATION (LLaMA 3.1 8B)")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded!")

    results = {
        'analysis_date': datetime.now().isoformat(),
        'method': 'llm_classification',
        'model': os.path.basename(model_path),
        'taxonomy': {k: v['description'] for k, v in SUBDOMAIN_TAXONOMY.items()},
        'documents': []
    }

    taxonomy_str = "\n".join([f"- {k}: {v['description']}" for k, v in SUBDOMAIN_TAXONOMY.items()])

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    for file_idx, txt_file in enumerate(txt_files):
        print(f"\n[{file_idx + 1}/{len(txt_files)}] {txt_file[:50]}...")

        filepath = os.path.join(input_dir, txt_file)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        year_match = re.match(r'^(\d{4})_', txt_file)
        year = int(year_match.group(1)) if year_match else 0

        chapters = extract_chapters(text, max_chapters=12)
        print(f"  Found {len(chapters)} chapters")

        chapter_results = []
        aggregate = defaultdict(float)

        for chap_idx, chapter in enumerate(chapters[:10]):  # Process first 10 chapters
            print(f"    Ch {chap_idx + 1}: {chapter['title'][:35]}...", end=" ", flush=True)

            # Build prompt
            content_preview = ' '.join(chapter['content'].split()[:1500])

            prompt = f"""Classify this textbook chapter into AI/ML sub-domains.

TAXONOMY:
{taxonomy_str}

CHAPTER: {chapter['title']}
CONTENT (excerpt): {content_preview}

Return JSON with percentage for each domain (sum to 100). Example:
{{"supervised_learning": 20, "neural_networks_deep_learning": 30, ...}}

JSON only:"""

            messages = [{"role": "user", "content": prompt}]

            if hasattr(tokenizer, 'apply_chat_template'):
                text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text_input = prompt

            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Parse JSON
            classification = {k: 0 for k in SUBDOMAIN_TAXONOMY.keys()}
            try:
                json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    for k in classification:
                        if k in parsed:
                            classification[k] = float(parsed[k])
            except:
                pass

            chapter_results.append({
                'number': chapter['number'],
                'title': chapter['title'],
                'classification': classification
            })

            for domain, pct in classification.items():
                aggregate[domain] += pct

            print("Done")

        # Normalize aggregate
        if chapter_results:
            aggregate = {k: v / len(chapter_results) for k, v in aggregate.items()}

        results['documents'].append({
            'filename': txt_file,
            'year': year,
            'chapters_analyzed': len(chapter_results),
            'aggregate_classification': dict(aggregate),
            'chapters': chapter_results
        })

        top = sorted(aggregate.items(), key=lambda x: -x[1])[:3]
        print(f"  Summary: {', '.join([f'{d}: {p:.1f}%' for d, p in top])}")

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # Save results
    output_path = os.path.join(output_dir, 'llm_classification.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nLLM results saved to: {output_path}")
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_dir = os.path.join(project_root, "cleaned_text")
    output_dir = os.path.join(project_root, "analysis_outputs")

    os.makedirs(output_dir, exist_ok=True)

    # Run keyword classification on full text (fast, no chapter segmentation)
    keyword_results = run_keyword_classification(input_dir, output_dir)

    # LLM classification skipped - using keyword-based full text analysis only
    print("\n[INFO] LLM classification skipped - using keyword-based full text analysis")


if __name__ == "__main__":
    main()
