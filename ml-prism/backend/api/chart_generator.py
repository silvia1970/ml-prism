"""
Chart Generator for PRISM ML Predictions
Generates prediction charts for both MIMIC and SepsisExp models
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
import os
from typing import Dict, List, Optional
import logging

from api import minio_storage

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate prediction visualization charts"""
    
    def __init__(self, output_dir: str = "api_data/charts"):
        """
        Initialize chart generator
        
        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color scheme matching frontend
        self.colors = {
            'low_risk': '#28a745',      # Green
            'moderate_risk': '#ffc107', # Yellow/Orange
            'high_risk': '#dc3545',     # Red
            'primary': '#4A90E2',       # Blue
            'text': '#2c3e50'
        }
    
    def generate_risk_distribution_chart(
        self,
        records: List[Dict] = None,
        *,
        results: List[Dict] = None,
        db_type: str = None,
        db_name: str = None,
        title: str = None,
        submission_id: str = None,
    ) -> Optional[str]:
        """
        Generate risk distribution pie chart.

        Accepts both the legacy positional signature
        ``(results, db_type, submission_id)`` and the newer keyword-only
        signature used by the blueprint
        ``(records=…, db_name=…, title=…)``.

        Returns:
            Absolute path to the generated PNG file, or None on failure.
        """
        # --- normalise arguments so the rest of the method is uniform ---
        data = records if records is not None else (results or [])
        db_label = db_name or db_type or 'unknown'
        chart_title = title or f'Distribuzione Rischio - {db_label.upper()}'
        sub_id = submission_id or 'adhoc'
        try:
            # Count risk levels (3-class: low_risk, moderate_risk, high_risk)
            risk_counts = {'low_risk': 0, 'moderate_risk': 0, 'high_risk': 0}
            for result in data:
                risk_class = result.get('class', 'low_risk')
                if risk_class in risk_counts:
                    risk_counts[risk_class] += 1
            
            # Prepare data
            labels = ['Basso (< 0.50)', 'Moderato (0.50–0.75)', 'Alto (≥ 0.75)']
            sizes = [risk_counts['low_risk'], risk_counts['moderate_risk'], risk_counts['high_risk']]
            colors = [self.colors['low_risk'], self.colors['moderate_risk'], self.colors['high_risk']]
            
            # Filter out zero values
            filtered_data = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
            if not filtered_data:
                return None
                
            labels, sizes, colors = zip(*filtered_data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            
            # Styling
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
            
            ax.set_title(
                f'{chart_title}\n{len(data)} pazienti analizzati',
                fontsize=14,
                weight='bold',
                color=self.colors['text'],
                pad=20
            )
            
            plt.tight_layout()
            
            # Save chart
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"{db_label}_risk_distribution_{sub_id}_{ts}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Mirror to MinIO
            minio_storage.put_file(f"charts/{filename}", filepath, content_type='image/png')
            
            logger.info(f"Generated risk distribution chart: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating risk distribution chart: {str(e)}")
            return None
    
    def generate_score_distribution_chart(
        self,
        records: List[Dict] = None,
        *,
        results: List[Dict] = None,
        db_type: str = None,
        db_name: str = None,
        title: str = None,
        submission_id: str = None,
        bins: int = 20,
    ) -> Optional[str]:
        """
        Generate score distribution histogram.

        Accepts both ``(results, db_type, submission_id)`` (legacy) and
        ``(records=…, db_name=…, title=…, bins=…)`` (blueprint) calling styles.

        Returns:
            Absolute path to the generated PNG, or None on failure.
        """
        data = records if records is not None else (results or [])
        db_label = db_name or db_type or 'unknown'
        chart_title = title or f'Distribuzione Score - {db_label.upper()}'
        sub_id = submission_id or 'adhoc'
        try:
            # Extract scores
            scores = [result.get('score', 0.0) for result in data]
            
            if not scores:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Histogram
            n, bins, patches = ax.hist(
                scores, 
                bins=20, 
                edgecolor='white', 
                alpha=0.7,
                color=self.colors['primary']
            )
            
            # Color bars by risk level
            for i, patch in enumerate(patches):
                bin_center = (bins[i] + bins[i+1]) / 2
                if bin_center < 0.50:
                    patch.set_facecolor(self.colors['low_risk'])
                elif bin_center < 0.75:
                    patch.set_facecolor(self.colors['moderate_risk'])
                else:
                    patch.set_facecolor(self.colors['high_risk'])
            
            # Threshold lines
            ax.axvline(x=0.50, color='#ffc107', linestyle='--', linewidth=2, label='Soglia Moderato (0.50)')
            ax.axvline(x=0.75, color='#dc3545', linestyle='--', linewidth=2, label='Soglia Alto (0.75)')
            
            # Statistics
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            ax.axvline(x=mean_score, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Media ({mean_score:.1f})')
            ax.axvline(x=median_score, color='purple', linestyle=':', linewidth=2, alpha=0.7, label=f'Mediana ({median_score:.1f})')
            
            # Styling
            ax.set_xlabel('Score Predizione', fontsize=12, weight='bold')
            ax.set_ylabel('Numero Pazienti', fontsize=12, weight='bold')
            ax.set_title(
                f'{chart_title}\n{len(data)} pazienti',
                fontsize=14,
                weight='bold',
                color=self.colors['text'],
                pad=20
            )
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            plt.tight_layout()
            
            # Save chart
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"{db_label}_score_distribution_{sub_id}_{ts}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Mirror to MinIO
            minio_storage.put_file(f"charts/{filename}", filepath, content_type='image/png')
            
            logger.info(f"Generated score distribution chart: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating score distribution chart: {str(e)}")
            return None
    
    def generate_combined_chart(
        self,
        records: List[Dict] = None,
        *,
        results: List[Dict] = None,
        db_type: str = None,
        db_name: str = None,
        title: str = None,
        submission_id: str = None,
    ) -> Optional[str]:
        """
        Generate combined risk + score visualization.

        Accepts both ``(results, db_type, submission_id)`` (legacy) and
        ``(records=…, db_name=…, title=…)`` (blueprint) calling styles.

        Returns:
            Absolute path to the generated PNG, or None on failure.
        """
        data = records if records is not None else (results or [])
        db_label = db_name or db_type or 'unknown'
        sub_id = submission_id or 'adhoc'
        try:
            # Extract data
            scores = [result.get('score', 0.0) for result in data]
            risk_counts = {'low_risk': 0, 'moderate_risk': 0, 'high_risk': 0}
            
            for result in data:
                risk_class = result.get('class', 'low_risk')
                if risk_class in risk_counts:
                    risk_counts[risk_class] += 1
            
            if not scores:
                return None
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 6))
            
            # === LEFT: Risk Distribution Pie ===
            ax1 = plt.subplot(1, 2, 1)
            
            labels = ['Basso (< 0.50)', 'Moderato (0.50–0.75)', 'Alto (≥ 0.75)']
            sizes = [risk_counts['low_risk'], risk_counts['moderate_risk'], risk_counts['high_risk']]
            colors = [self.colors['low_risk'], self.colors['moderate_risk'], self.colors['high_risk']]
            
            # Filter zeros
            filtered_data = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
            if filtered_data:
                labels, sizes, colors = zip(*filtered_data)
                wedges, texts, autotexts = ax1.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 11, 'weight': 'bold'}
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(13)
            
            ax1.set_title('Distribuzione Livelli di Rischio', fontsize=12, weight='bold', pad=15)
            
            # === RIGHT: Score Histogram ===
            ax2 = plt.subplot(1, 2, 2)
            
            n, bins, patches = ax2.hist(
                scores,
                bins=15,
                edgecolor='white',
                alpha=0.8,
                color=self.colors['primary']
            )
            
            # Color by risk
            for i, patch in enumerate(patches):
                bin_center = (bins[i] + bins[i+1]) / 2
                if bin_center < 0.50:
                    patch.set_facecolor(self.colors['low_risk'])
                elif bin_center < 0.75:
                    patch.set_facecolor(self.colors['moderate_risk'])
                else:
                    patch.set_facecolor(self.colors['high_risk'])
            
            # Thresholds
            ax2.axvline(x=0.50, color='#ffc107', linestyle='--', linewidth=1.5, alpha=0.7, label='Moderato (0.50)')
            ax2.axvline(x=0.75, color='#dc3545', linestyle='--', linewidth=1.5, alpha=0.7, label='Alto (0.75)')
            
            # Stats
            mean_score = np.mean(scores)
            ax2.axvline(x=mean_score, color='blue', linestyle=':', linewidth=2, alpha=0.6)
            ax2.text(
                mean_score, 
                ax2.get_ylim()[1] * 0.95, 
                f'μ={mean_score:.3f}',
                ha='center',
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            ax2.set_xlabel('Score Predizione', fontsize=11, weight='bold')
            ax2.set_ylabel('Frequenza', fontsize=11, weight='bold')
            ax2.set_title('Distribuzione Score', fontsize=12, weight='bold', pad=15)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
            
            # Main title
            fig.suptitle(
                f'Analisi Predizioni {db_label.upper()} - {len(data)} Pazienti',
                fontsize=16,
                weight='bold',
                color=self.colors['text'],
                y=0.98
            )
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"{db_label}_combined_analysis_{sub_id}_{ts}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Remove old file if exists to ensure fresh data
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Removed old chart: {filepath}")
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Mirror to MinIO
            minio_storage.put_file(f"charts/{filename}", filepath, content_type='image/png')
            
            logger.info(f"Generated combined chart: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating combined chart: {str(e)}")
            return None
