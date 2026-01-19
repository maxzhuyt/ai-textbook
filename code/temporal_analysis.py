#!/usr/bin/env python3
"""
Step 5: Temporal Trajectory Analysis
Synthesizes NLP and semantic analysis to trace AI field evolution.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class TemporalAnalyzer:
    """Analyze temporal evolution of AI/ML field."""

    ERAS = {
        'early': (1983, 1995, 'Symbolic AI Era'),
        'middle': (1996, 2010, 'Statistical ML Era'),
        'recent': (2011, 2022, 'Deep Learning Era')
    }

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.nlp_results = None
        self.semantic_results = None

    def load_results(self):
        """Load analysis results from previous steps."""
        print("Loading analysis results...")

        # Load NLP analysis
        nlp_path = os.path.join(self.output_dir, 'nlp_analysis_results.json')
        with open(nlp_path, 'r') as f:
            self.nlp_results = json.load(f)

        # Load semantic classification (prefer LLM, fallback to revised keyword)
        llm_path = os.path.join(self.output_dir, 'llm_classification.json')
        revised_path = os.path.join(self.output_dir, 'revised_classification.json')
        keyword_path = os.path.join(self.output_dir, 'keyword_classification.json')

        if os.path.exists(llm_path):
            with open(llm_path, 'r') as f:
                self.semantic_results = json.load(f)
            print("  Using LLM-based semantic classification (LLaMA 3.1 8B)")
        elif os.path.exists(revised_path):
            with open(revised_path, 'r') as f:
                self.semantic_results = json.load(f)
            print("  Using revised keyword-based semantic classification")
        elif os.path.exists(keyword_path):
            with open(keyword_path, 'r') as f:
                self.semantic_results = json.load(f)
            print("  Using keyword-based semantic classification")
        else:
            print("  WARNING: No semantic classification found!")
            self.semantic_results = None

        # Load keyword frequencies
        keyword_path = os.path.join(self.output_dir, 'keyword_frequencies.csv')
        self.keyword_df = pd.read_csv(keyword_path)

        print(f"  Loaded {len(self.semantic_results['documents'])} documents for analysis")

    def analyze_era_shifts(self) -> Dict:
        """Analyze major shifts between eras."""
        print("\nAnalyzing era shifts...")

        era_data = {era: defaultdict(list) for era in self.ERAS.keys()}

        for doc in self.semantic_results['documents']:
            year = doc['year']
            # Support both old (aggregate_classification) and new (classification) format
            classification = doc.get('aggregate_classification') or doc.get('classification', {})
            for era_name, (start, end, _) in self.ERAS.items():
                if start <= year <= end:
                    for domain, pct in classification.items():
                        era_data[era_name][domain].append(pct)
                    break

        # Calculate era averages and changes
        era_averages = {}
        for era, data in era_data.items():
            era_averages[era] = {domain: np.mean(values) if values else 0
                                for domain, values in data.items()}

        # Calculate shifts between eras
        shifts = {
            'early_to_middle': {},
            'middle_to_recent': {},
            'early_to_recent': {}
        }

        domains = list(era_averages['early'].keys())
        for domain in domains:
            early = era_averages['early'].get(domain, 0)
            middle = era_averages['middle'].get(domain, 0)
            recent = era_averages['recent'].get(domain, 0)

            shifts['early_to_middle'][domain] = middle - early
            shifts['middle_to_recent'][domain] = recent - middle
            shifts['early_to_recent'][domain] = recent - early

        return {
            'era_averages': era_averages,
            'shifts': shifts,
            'era_definitions': {k: {'start': v[0], 'end': v[1], 'name': v[2]}
                              for k, v in self.ERAS.items()}
        }

    def identify_inflection_points(self) -> List[Dict]:
        """Identify major inflection points in the field."""
        print("\nIdentifying inflection points...")

        inflection_points = []

        # Known historical inflection points to validate
        known_events = [
            (1997, "Deep Blue defeats Kasparov"),
            (2006, "Deep Learning renaissance (Hinton et al.)"),
            (2012, "AlexNet wins ImageNet"),
            (2017, "Transformer architecture introduced"),
        ]

        # Analyze keyword frequency changes
        years = sorted(self.keyword_df['year'].unique())

        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]

            prev_data = self.keyword_df[self.keyword_df['year'] == prev_year]
            curr_data = self.keyword_df[self.keyword_df['year'] == curr_year]

            # Check for significant changes in key terms
            key_terms = ['neural network', 'deep learning', 'bayesian',
                        'reinforcement learning', 'machine learning']

            for term in key_terms:
                prev_freq = prev_data[prev_data['term'] == term]['frequency_per_1000'].mean()
                curr_freq = curr_data[curr_data['term'] == term]['frequency_per_1000'].mean()

                if pd.notna(prev_freq) and pd.notna(curr_freq) and prev_freq > 0:
                    change_ratio = curr_freq / prev_freq
                    if change_ratio > 2.0 or change_ratio < 0.5:
                        inflection_points.append({
                            'year': curr_year,
                            'term': term,
                            'change_ratio': float(change_ratio),
                            'direction': 'increase' if change_ratio > 1 else 'decrease'
                        })

        # Add known historical events
        for year, event in known_events:
            inflection_points.append({
                'year': year,
                'event': event,
                'type': 'historical_milestone'
            })

        return sorted(inflection_points, key=lambda x: x['year'])

    def compute_diversity_metrics(self) -> Dict:
        """Compute field diversity and specialization over time."""
        print("\nComputing diversity metrics...")

        diversity_by_year = {}

        for doc in self.semantic_results['documents']:
            year = doc['year']
            # Support both old (aggregate_classification) and new (classification) format
            classification = doc.get('aggregate_classification') or doc.get('classification', {})

            # Calculate Shannon entropy (diversity)
            values = [v for v in classification.values() if v > 0]
            if values:
                total = sum(values)
                probs = [v/total for v in values]
                entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)

                # Calculate Gini coefficient (concentration)
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                cumsum = np.cumsum(sorted_vals)
                gini = (2 * sum((i+1) * v for i, v in enumerate(sorted_vals)) /
                       (n * sum(sorted_vals))) - (n + 1) / n if sum(sorted_vals) > 0 else 0

                diversity_by_year[year] = {
                    'shannon_entropy': float(entropy),
                    'gini_coefficient': float(gini),
                    'dominant_domain': max(classification.items(), key=lambda x: x[1])[0],
                    'dominant_percentage': max(classification.values())
                }

        return diversity_by_year

    def statistical_trend_analysis(self) -> Dict:
        """Perform statistical tests on temporal trends."""
        print("\nPerforming statistical trend analysis...")

        # Prepare data for regression
        years = []
        domain_values = defaultdict(list)

        for doc in self.semantic_results['documents']:
            years.append(doc['year'])
            # Support both old (aggregate_classification) and new (classification) format
            classification = doc.get('aggregate_classification') or doc.get('classification', {})
            for domain, pct in classification.items():
                domain_values[domain].append(pct)

        # Linear regression for each domain
        trend_results = {}
        for domain, values in domain_values.items():
            if len(values) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
                trend_results[domain] = {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'trend': 'increasing' if slope > 0 else 'decreasing',
                    'significant': p_value < 0.05
                }

        return trend_results

    def generate_synthesis_report(self, era_shifts: Dict, inflection_points: List,
                                 diversity_metrics: Dict, trend_analysis: Dict) -> Dict:
        """Generate comprehensive synthesis report."""
        print("\nGenerating synthesis report...")

        # Identify key findings
        key_findings = []

        # Trend findings
        sig_trends = [(d, t) for d, t in trend_analysis.items() if t['significant']]
        if sig_trends:
            for domain, trend in sig_trends:
                key_findings.append({
                    'type': 'significant_trend',
                    'domain': domain,
                    'direction': trend['trend'],
                    'r_squared': trend['r_squared']
                })

        # Era shift findings
        major_shifts = []
        for domain, change in era_shifts['shifts']['early_to_recent'].items():
            if abs(change) > 5:  # More than 5% change
                major_shifts.append({
                    'domain': domain,
                    'change': change,
                    'direction': 'increased' if change > 0 else 'decreased'
                })

        key_findings.extend([
            {'type': 'era_shift', **shift}
            for shift in sorted(major_shifts, key=lambda x: -abs(x['change']))[:5]
        ])

        # Diversity trend
        years_sorted = sorted(diversity_metrics.keys())
        if len(years_sorted) >= 2:
            early_entropy = np.mean([diversity_metrics[y]['shannon_entropy']
                                    for y in years_sorted[:len(years_sorted)//3]])
            recent_entropy = np.mean([diversity_metrics[y]['shannon_entropy']
                                     for y in years_sorted[-len(years_sorted)//3:]])
            key_findings.append({
                'type': 'diversity_change',
                'early_entropy': float(early_entropy),
                'recent_entropy': float(recent_entropy),
                'direction': 'diversifying' if recent_entropy > early_entropy else 'specializing'
            })

        return {
            'analysis_date': datetime.now().isoformat(),
            'era_analysis': era_shifts,
            'inflection_points': inflection_points,
            'diversity_metrics': diversity_metrics,
            'trend_analysis': trend_analysis,
            'key_findings': key_findings
        }

    def generate_visualizations(self, synthesis: Dict):
        """Generate comprehensive visualizations."""
        print("\nGenerating visualizations...")

        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Era comparison heatmap
        self._plot_era_heatmap(synthesis['era_analysis'], viz_dir)

        # 2. Diversity over time
        self._plot_diversity_timeline(synthesis['diversity_metrics'], viz_dir)

        # 3. Trend significance plot
        self._plot_trend_significance(synthesis['trend_analysis'], viz_dir)

        # 4. Comprehensive timeline
        self._plot_comprehensive_timeline(synthesis, viz_dir)

    def _plot_era_heatmap(self, era_analysis: Dict, viz_dir: str):
        """Plot era comparison heatmap."""
        fig, ax = plt.subplots(figsize=(14, 8))

        domains = list(era_analysis['era_averages']['early'].keys())
        eras = ['early', 'middle', 'recent']

        data = np.array([[era_analysis['era_averages'][era].get(d, 0) for d in domains]
                        for era in eras])

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(eras)))
        ax.set_yticklabels([f"{era.capitalize()}\n({era_analysis['era_definitions'][era]['start']}-{era_analysis['era_definitions'][era]['end']})"
                          for era in eras])

        # Add value annotations
        for i in range(len(eras)):
            for j in range(len(domains)):
                ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', fontsize=7)

        ax.set_title('Sub-domain Emphasis Across Eras (% Allocation)', fontsize=14)
        plt.colorbar(im, ax=ax, label='Percentage')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'era_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_diversity_timeline(self, diversity_metrics: Dict, viz_dir: str):
        """Plot diversity metrics over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        years = sorted(diversity_metrics.keys())
        entropy = [diversity_metrics[y]['shannon_entropy'] for y in years]
        gini = [diversity_metrics[y]['gini_coefficient'] for y in years]

        ax1.plot(years, entropy, 'b-o', linewidth=2, markersize=8)
        ax1.set_ylabel('Shannon Entropy', fontsize=12)
        ax1.set_title('Field Diversity Over Time', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Add era boundaries
        for era, (start, end, name) in self.ERAS.items():
            ax1.axvspan(start, end, alpha=0.1, label=name)

        ax2.plot(years, gini, 'r-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Gini Coefficient', fontsize=12)
        ax2.set_title('Field Concentration Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'diversity_timeline.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_trend_significance(self, trend_analysis: Dict, viz_dir: str):
        """Plot trend significance for each domain."""
        fig, ax = plt.subplots(figsize=(12, 8))

        domains = list(trend_analysis.keys())
        slopes = [trend_analysis[d]['slope'] for d in domains]
        r_squared = [trend_analysis[d]['r_squared'] for d in domains]
        significant = [trend_analysis[d]['significant'] for d in domains]

        colors = ['green' if s else 'gray' for s in significant]
        sizes = [r * 300 + 50 for r in r_squared]

        scatter = ax.scatter(range(len(domains)), slopes, c=colors, s=sizes, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels([d.replace('_', '\n') for d in domains], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Slope (% per year)', fontsize=12)
        ax.set_title('Temporal Trends by Sub-domain\n(Green = Statistically Significant, Size = R²)', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'trend_significance.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_comprehensive_timeline(self, synthesis: Dict, viz_dir: str):
        """Plot comprehensive timeline with all insights."""
        fig, ax = plt.subplots(figsize=(16, 10))

        # Plot dominant domains over time
        years = sorted(synthesis['diversity_metrics'].keys())
        dominant_domains = [synthesis['diversity_metrics'][y]['dominant_domain'] for y in years]
        dominant_pct = [synthesis['diversity_metrics'][y]['dominant_percentage'] for y in years]

        # Create color map for domains
        unique_domains = list(set(dominant_domains))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_domains)))
        domain_colors = {d: colors[i] for i, d in enumerate(unique_domains)}

        for i, (year, domain, pct) in enumerate(zip(years, dominant_domains, dominant_pct)):
            ax.bar(year, pct, color=domain_colors[domain], width=3, alpha=0.7)
            ax.text(year, pct + 1, domain.replace('_', '\n')[:15], ha='center', va='bottom',
                   fontsize=6, rotation=45)

        # Add era boundaries
        for era, (start, end, name) in self.ERAS.items():
            ax.axvline(x=start, color='red', linestyle='--', alpha=0.5)
            ax.text(start + 2, ax.get_ylim()[1] * 0.95, name, fontsize=10, rotation=90, va='top')

        # Add inflection points
        for point in synthesis['inflection_points']:
            if 'event' in point:
                ax.annotate(point['event'], xy=(point['year'], 10),
                          xytext=(point['year'], 20), fontsize=8,
                          arrowprops=dict(arrowstyle='->', color='blue'),
                          rotation=45)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Dominant Domain % Allocation', fontsize=12)
        ax.set_title('AI/ML Field Evolution Timeline (1983-2022)', fontsize=14)
        ax.set_xlim(1980, 2025)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'comprehensive_timeline.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def run_full_analysis(self) -> Dict:
        """Run complete temporal analysis."""
        print("\n" + "=" * 80)
        print("STEP 5: TEMPORAL TRAJECTORY ANALYSIS")
        print("=" * 80)

        self.load_results()

        era_shifts = self.analyze_era_shifts()
        inflection_points = self.identify_inflection_points()
        diversity_metrics = self.compute_diversity_metrics()
        trend_analysis = self.statistical_trend_analysis()

        synthesis = self.generate_synthesis_report(
            era_shifts, inflection_points, diversity_metrics, trend_analysis
        )

        self.generate_visualizations(synthesis)

        # Save results
        output_path = os.path.join(self.output_dir, 'temporal_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(synthesis, f, indent=2, default=str)

        print(f"\nTemporal analysis saved to: {output_path}")

        return synthesis


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "analysis_outputs")

    analyzer = TemporalAnalyzer(output_dir)
    synthesis = analyzer.run_full_analysis()

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    for finding in synthesis['key_findings'][:10]:
        if finding['type'] == 'significant_trend':
            print(f"  - {finding['domain']}: {finding['direction']} (R²={finding['r_squared']:.3f})")
        elif finding['type'] == 'era_shift':
            print(f"  - {finding['domain']}: {finding['direction']} by {abs(finding['change']):.1f}% from early to recent era")
        elif finding['type'] == 'diversity_change':
            print(f"  - Field is {finding['direction']} (entropy: {finding['early_entropy']:.2f} -> {finding['recent_entropy']:.2f})")


if __name__ == "__main__":
    main()
