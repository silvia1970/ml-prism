"""Chart Generator for PRISM API."""
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


class ChartGenerator:
    """Generate visualization charts for PRISM data."""

    def __init__(self):
        sns.set_style('whitegrid')

    def generate_score_distribution(self, scores: List[float], title: str = "Score Distribution") -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.5, color='orange', linestyle='--', label='Moderate Risk')
        ax.axvline(x=0.75, color='red', linestyle='--', label='High Risk')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        return self._fig_to_base64(fig)

    def generate_risk_pie_chart(self, classes: List[str], title: str = "Risk Distribution") -> str:
        fig, ax = plt.subplots(figsize=(8, 8))
        class_counts = {c: classes.count(c) for c in set(classes)}
        colors = {'low_risk': '#2ecc71', 'moderate_risk': '#f39c12', 'high_risk': '#e74c3c'}
        ax.pie(class_counts.values(), labels=class_counts.keys(),
               colors=[colors.get(c, '#95a5a6') for c in class_counts.keys()],
               autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
        return self._fig_to_base64(fig)

    def generate_timeline_chart(self, timestamps: List[str], scores: List[float],
                                 title: str = "Risk Score Timeline") -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(scores)), scores, marker='o', linewidth=2, markersize=4)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk')
        ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='High Risk')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Risk Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._fig_to_base64(fig)

    def generate_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> str:
        from sklearn.metrics import confusion_matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"