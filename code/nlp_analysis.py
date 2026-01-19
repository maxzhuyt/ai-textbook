#!/usr/bin/env python3
"""
Layer 1: NLP Keyword & Distribution Analysis for ML Textbooks
Comprehensive analysis of terminology evolution across decades.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class MLTextbookAnalyzer:
    """Comprehensive NLP analysis of ML textbooks."""

    # AI/ML domain-specific terms to track
    DOMAIN_TERMS = {
        # Core ML concepts
        'machine learning', 'artificial intelligence', 'deep learning',
        'neural network', 'neural networks', 'supervised learning',
        'unsupervised learning', 'reinforcement learning', 'semi-supervised',

        # Algorithms & methods
        'decision tree', 'random forest', 'gradient descent', 'backpropagation',
        'support vector', 'svm', 'naive bayes', 'k-means', 'clustering',
        'regression', 'classification', 'perceptron', 'multilayer perceptron',
        'convolutional', 'recurrent', 'lstm', 'transformer', 'attention',
        'ensemble', 'boosting', 'bagging', 'dropout', 'batch normalization',

        # Probabilistic/Statistical
        'bayesian', 'probability', 'probabilistic', 'gaussian', 'markov',
        'hidden markov', 'graphical model', 'belief network', 'inference',
        'maximum likelihood', 'expectation maximization', 'em algorithm',
        'monte carlo', 'mcmc', 'sampling',

        # Optimization
        'optimization', 'gradient', 'loss function', 'cost function',
        'objective function', 'convex', 'stochastic gradient',
        'adam', 'momentum', 'learning rate',

        # Learning theory
        'generalization', 'overfitting', 'underfitting', 'bias variance',
        'vc dimension', 'pac learning', 'regularization', 'cross validation',

        # Features & representations
        'feature', 'feature extraction', 'feature selection', 'dimensionality reduction',
        'pca', 'principal component', 'embedding', 'representation learning',
        'kernel', 'kernel method',

        # Applications
        'natural language', 'nlp', 'computer vision', 'image recognition',
        'speech recognition', 'robotics', 'game playing', 'expert system',

        # Data concepts
        'training data', 'test data', 'validation', 'dataset', 'big data',
        'data mining', 'pattern recognition',

        # Modern concepts
        'autoencoder', 'generative', 'discriminative', 'gan',
        'pretrained', 'transfer learning', 'fine tuning'
    }

    # Era definitions
    ERAS = {
        'early': (1983, 1995),
        'middle': (1996, 2010),
        'recent': (2011, 2022)
    }

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.documents = {}
        self.years = {}
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Add domain-specific stop words
        self.stop_words.update([
            'also', 'one', 'two', 'three', 'first', 'second', 'would', 'could',
            'may', 'might', 'must', 'shall', 'using', 'used', 'use', 'given',
            'figure', 'table', 'chapter', 'section', 'page', 'example'
        ])

    def load_documents(self):
        """Load all cleaned text documents."""
        print("Loading documents...")

        for filename in sorted(os.listdir(self.input_dir)):
            if not filename.endswith('.txt'):
                continue

            filepath = os.path.join(self.input_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            # Extract year from filename
            year_match = re.match(r'^(\d{4})_', filename)
            year = int(year_match.group(1)) if year_match else 0

            # Create document ID
            doc_id = filename.replace('.txt', '')

            self.documents[doc_id] = text
            self.years[doc_id] = year

            print(f"  Loaded: {filename[:50]}... ({len(text):,} chars, year: {year})")

        print(f"\nTotal documents loaded: {len(self.documents)}")

    def preprocess_text(self, text: str, for_ngrams: bool = False) -> List[str]:
        """Preprocess text for analysis."""
        # Lowercase
        text = text.lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', ' ', text)

        # Tokenize
        tokens = text.split()

        if for_ngrams:
            # For n-gram analysis, don't remove stop words
            return tokens

        # Remove stop words and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def compute_keyword_frequencies(self) -> pd.DataFrame:
        """Compute frequency of domain-specific terms across documents."""
        print("\nComputing keyword frequencies...")

        results = []

        for doc_id, text in self.documents.items():
            text_lower = text.lower()
            year = self.years[doc_id]
            word_count = len(text.split())

            for term in self.DOMAIN_TERMS:
                # Count occurrences (handle multi-word terms)
                pattern = r'\b' + re.escape(term) + r'\b'
                count = len(re.findall(pattern, text_lower))

                if count > 0:
                    results.append({
                        'doc_id': doc_id,
                        'year': year,
                        'term': term,
                        'count': count,
                        'frequency_per_1000': (count / word_count) * 1000 if word_count > 0 else 0
                    })

        df = pd.DataFrame(results)
        print(f"  Found {len(df)} term occurrences across documents")
        return df

    def compute_tfidf(self) -> Tuple[np.ndarray, List[str], TfidfVectorizer]:
        """Compute TF-IDF matrix for all documents."""
        print("\nComputing TF-IDF matrix...")

        # Prepare documents
        doc_ids = sorted(self.documents.keys())
        texts = [self.documents[doc_id] for doc_id in doc_ids]

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
        return tfidf_matrix, list(feature_names), vectorizer

    def compute_document_similarity(self, tfidf_matrix: np.ndarray) -> pd.DataFrame:
        """Compute pairwise document similarity."""
        print("\nComputing document similarity...")

        doc_ids = sorted(self.documents.keys())
        similarity_matrix = cosine_similarity(tfidf_matrix)

        df = pd.DataFrame(
            similarity_matrix,
            index=doc_ids,
            columns=doc_ids
        )

        return df

    def run_topic_modeling(self, n_topics: int = 10) -> Tuple[LatentDirichletAllocation, np.ndarray, List[str]]:
        """Run LDA topic modeling."""
        print(f"\nRunning LDA topic modeling ({n_topics} topics)...")

        doc_ids = sorted(self.documents.keys())
        texts = [self.documents[doc_id] for doc_id in doc_ids]

        # Create count vectorizer for LDA
        vectorizer = CountVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.9,
            stop_words='english'
        )

        count_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Fit LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='batch'
        )

        doc_topics = lda.fit_transform(count_matrix)

        print(f"  LDA converged with {lda.n_iter_} iterations")
        return lda, doc_topics, list(feature_names)

    def get_topic_words(self, lda: LatentDirichletAllocation, feature_names: List[str], n_words: int = 15) -> List[Dict]:
        """Extract top words for each topic."""
        topics = []

        for idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]

            topics.append({
                'topic_id': idx,
                'words': top_words,
                'weights': [float(w) for w in top_weights]
            })

        return topics

    def analyze_temporal_trends(self, keyword_df: pd.DataFrame) -> Dict:
        """Analyze how terminology changes over time."""
        print("\nAnalyzing temporal trends...")

        # Group by era
        era_trends = {}

        for era_name, (start_year, end_year) in self.ERAS.items():
            era_data = keyword_df[
                (keyword_df['year'] >= start_year) &
                (keyword_df['year'] <= end_year)
            ]

            if len(era_data) == 0:
                continue

            # Top terms by frequency
            term_totals = era_data.groupby('term')['frequency_per_1000'].mean()
            top_terms = term_totals.nlargest(20).to_dict()

            era_trends[era_name] = {
                'year_range': f"{start_year}-{end_year}",
                'document_count': era_data['doc_id'].nunique(),
                'top_terms': top_terms
            }

        # Identify emerging vs declining terms
        if 'early' in era_trends and 'recent' in era_trends:
            early_terms = set(era_trends['early']['top_terms'].keys())
            recent_terms = set(era_trends['recent']['top_terms'].keys())

            emerging = recent_terms - early_terms
            declining = early_terms - recent_terms
            persistent = early_terms & recent_terms

            era_trends['term_evolution'] = {
                'emerging_terms': list(emerging),
                'declining_terms': list(declining),
                'persistent_terms': list(persistent)
            }

        return era_trends

    def generate_visualizations(self, keyword_df: pd.DataFrame, similarity_df: pd.DataFrame,
                               doc_topics: np.ndarray, topics: List[Dict]):
        """Generate all visualizations."""
        print("\nGenerating visualizations...")

        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # 1. Key terms evolution over time
        self._plot_term_evolution(keyword_df, viz_dir)

        # 2. Document similarity heatmap
        self._plot_similarity_heatmap(similarity_df, viz_dir)

        # 3. Topic distribution over time
        self._plot_topic_distribution(doc_topics, topics, viz_dir)

        # 4. Word count by year
        self._plot_word_counts(viz_dir)

        # 5. Era comparison
        self._plot_era_comparison(keyword_df, viz_dir)

        print(f"  Visualizations saved to: {viz_dir}")

    def _plot_term_evolution(self, keyword_df: pd.DataFrame, viz_dir: str):
        """Plot evolution of key terms over time."""
        # Select important terms
        key_terms = [
            'neural network', 'deep learning', 'machine learning',
            'bayesian', 'reinforcement learning', 'supervised learning',
            'classification', 'regression', 'clustering'
        ]

        fig, ax = plt.subplots(figsize=(14, 8))

        for term in key_terms:
            term_data = keyword_df[keyword_df['term'] == term].copy()
            if len(term_data) > 0:
                term_data = term_data.sort_values('year')
                ax.plot(term_data['year'], term_data['frequency_per_1000'],
                       marker='o', label=term, linewidth=2, markersize=6)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Frequency per 1000 words', fontsize=12)
        ax.set_title('Evolution of Key ML/AI Terms (1983-2022)', fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'term_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_similarity_heatmap(self, similarity_df: pd.DataFrame, viz_dir: str):
        """Plot document similarity heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create shorter labels
        labels = []
        for idx in similarity_df.index:
            year = self.years.get(idx, '')
            short_name = idx.split('_')[1][:15] if '_' in idx else idx[:15]
            labels.append(f"{year} {short_name}")

        sns.heatmap(similarity_df.values, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=labels, ax=ax)

        ax.set_title('Document Similarity Matrix (Cosine Similarity)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'similarity_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_topic_distribution(self, doc_topics: np.ndarray, topics: List[Dict], viz_dir: str):
        """Plot topic distribution over documents."""
        doc_ids = sorted(self.documents.keys())
        years = [self.years[d] for d in doc_ids]

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create stacked bar chart
        bottom = np.zeros(len(doc_ids))
        colors = plt.cm.tab10(np.linspace(0, 1, len(topics)))

        for i, topic in enumerate(topics):
            topic_weights = doc_topics[:, i]
            topic_label = f"T{i}: {', '.join(topic['words'][:3])}"
            ax.bar(range(len(doc_ids)), topic_weights, bottom=bottom,
                  label=topic_label, color=colors[i])
            bottom += topic_weights

        ax.set_xlabel('Document', fontsize=12)
        ax.set_ylabel('Topic Proportion', fontsize=12)
        ax.set_title('Topic Distribution Across Textbooks', fontsize=14)
        ax.set_xticks(range(len(doc_ids)))
        ax.set_xticklabels([str(y) for y in years], rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'topic_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_word_counts(self, viz_dir: str):
        """Plot word counts by year."""
        data = []
        for doc_id, text in self.documents.items():
            data.append({
                'year': self.years[doc_id],
                'word_count': len(text.split()),
                'doc_id': doc_id
            })

        df = pd.DataFrame(data).sort_values('year')

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(df)), df['word_count'], color='steelblue')

        ax.set_xlabel('Document', fontsize=12)
        ax.set_ylabel('Word Count', fontsize=12)
        ax.set_title('Document Length by Publication Year', fontsize=14)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([str(y) for y in df['year']], rotation=45)

        # Add value labels
        for bar, count in zip(bars, df['word_count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                   f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'word_counts.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_era_comparison(self, keyword_df: pd.DataFrame, viz_dir: str):
        """Plot term frequency comparison across eras."""
        # Calculate era averages for top terms
        era_data = defaultdict(lambda: defaultdict(list))

        for _, row in keyword_df.iterrows():
            year = row['year']
            for era_name, (start, end) in self.ERAS.items():
                if start <= year <= end:
                    era_data[era_name][row['term']].append(row['frequency_per_1000'])
                    break

        # Get top 15 terms overall
        term_totals = keyword_df.groupby('term')['frequency_per_1000'].sum()
        top_terms = term_totals.nlargest(15).index.tolist()

        # Prepare data for plotting
        plot_data = []
        for era in ['early', 'middle', 'recent']:
            for term in top_terms:
                if term in era_data[era]:
                    avg_freq = np.mean(era_data[era][term])
                    plot_data.append({
                        'era': era,
                        'term': term,
                        'frequency': avg_freq
                    })

        df = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create grouped bar chart
        x = np.arange(len(top_terms))
        width = 0.25

        for i, era in enumerate(['early', 'middle', 'recent']):
            era_df = df[df['era'] == era]
            freqs = [era_df[era_df['term'] == t]['frequency'].values[0]
                    if len(era_df[era_df['term'] == t]) > 0 else 0
                    for t in top_terms]
            ax.bar(x + i*width, freqs, width, label=f"{era.capitalize()} ({self.ERAS[era][0]}-{self.ERAS[era][1]})")

        ax.set_xlabel('Term', fontsize=12)
        ax.set_ylabel('Average Frequency per 1000 words', fontsize=12)
        ax.set_title('Term Frequency Comparison Across Eras', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(top_terms, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'era_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def run_full_analysis(self) -> Dict:
        """Run the complete NLP analysis pipeline."""
        print("\n" + "=" * 80)
        print("LAYER 1: NLP KEYWORD & DISTRIBUTION ANALYSIS")
        print("=" * 80)

        # Load documents
        self.load_documents()

        # Keyword frequency analysis
        keyword_df = self.compute_keyword_frequencies()

        # TF-IDF analysis
        tfidf_matrix, feature_names, vectorizer = self.compute_tfidf()

        # Document similarity
        similarity_df = self.compute_document_similarity(tfidf_matrix)

        # Topic modeling
        lda, doc_topics, lda_features = self.run_topic_modeling(n_topics=10)
        topics = self.get_topic_words(lda, lda_features)

        # Temporal trends
        era_trends = self.analyze_temporal_trends(keyword_df)

        # Generate visualizations
        self.generate_visualizations(keyword_df, similarity_df, doc_topics, topics)

        # Compile results
        results = {
            'analysis_date': datetime.now().isoformat(),
            'documents_analyzed': len(self.documents),
            'year_range': f"{min(self.years.values())}-{max(self.years.values())}",
            'keyword_analysis': {
                'total_term_occurrences': len(keyword_df),
                'unique_terms_found': keyword_df['term'].nunique(),
                'top_terms_overall': keyword_df.groupby('term')['frequency_per_1000'].mean().nlargest(20).to_dict()
            },
            'tfidf_analysis': {
                'vocabulary_size': len(feature_names),
                'top_tfidf_terms': list(feature_names[:50])
            },
            'topic_modeling': {
                'n_topics': len(topics),
                'topics': topics
            },
            'era_trends': era_trends,
            'document_similarity': {
                'min_similarity': float(similarity_df.values[np.triu_indices(len(similarity_df), k=1)].min()),
                'max_similarity': float(similarity_df.values[np.triu_indices(len(similarity_df), k=1)].max()),
                'mean_similarity': float(similarity_df.values[np.triu_indices(len(similarity_df), k=1)].mean())
            }
        }

        # Save results
        self._save_results(results, keyword_df, similarity_df, doc_topics, topics)

        return results

    def _save_results(self, results: Dict, keyword_df: pd.DataFrame,
                     similarity_df: pd.DataFrame, doc_topics: np.ndarray, topics: List[Dict]):
        """Save all analysis results."""
        print("\nSaving results...")

        # Save main results JSON
        with open(os.path.join(self.output_dir, 'nlp_analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Save keyword frequency CSV
        keyword_df.to_csv(os.path.join(self.output_dir, 'keyword_frequencies.csv'), index=False)

        # Save similarity matrix CSV
        similarity_df.to_csv(os.path.join(self.output_dir, 'document_similarity.csv'))

        # Save topic assignments
        doc_ids = sorted(self.documents.keys())
        topic_df = pd.DataFrame(doc_topics, index=doc_ids,
                               columns=[f'topic_{i}' for i in range(doc_topics.shape[1])])
        topic_df['year'] = [self.years[d] for d in doc_ids]
        topic_df.to_csv(os.path.join(self.output_dir, 'document_topics.csv'))

        print(f"  Results saved to: {self.output_dir}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_dir = os.path.join(project_root, "cleaned_text")
    output_dir = os.path.join(project_root, "analysis_outputs")

    os.makedirs(output_dir, exist_ok=True)

    analyzer = MLTextbookAnalyzer(input_dir, output_dir)
    results = analyzer.run_full_analysis()

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nDocuments analyzed: {results['documents_analyzed']}")
    print(f"Year range: {results['year_range']}")
    print(f"\nTop 10 terms overall (avg frequency per 1000 words):")
    for term, freq in list(results['keyword_analysis']['top_terms_overall'].items())[:10]:
        print(f"  - {term}: {freq:.3f}")

    print(f"\nDocument similarity range: {results['document_similarity']['min_similarity']:.3f} - {results['document_similarity']['max_similarity']:.3f}")

    if 'term_evolution' in results['era_trends']:
        print(f"\nEmerging terms (recent vs early):")
        for term in results['era_trends']['term_evolution']['emerging_terms'][:5]:
            print(f"  + {term}")
        print(f"\nDeclining terms:")
        for term in results['era_trends']['term_evolution']['declining_terms'][:5]:
            print(f"  - {term}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
