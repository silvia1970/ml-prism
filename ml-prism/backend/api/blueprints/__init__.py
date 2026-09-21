"""
PRISM API Blueprints

Modular route handlers organized by domain:
- data: Patient data submission, retrieval, and updates
- csv: CSV file upload, templates, and batch predictions
- submissions: Submission management and recalculation
- patients: Patient history, statistics, and sequence predictions
- charts: Visualization chart generation and retrieval
- stats: System statistics, scores, and model information
"""


def register_all_blueprints(app, db, ml_scorer, model_loader, inference_engine, chart_generator):
    """
    Register all API blueprints with the Flask application.

    Args:
        app: Flask application instance
        db: Database instance
        ml_scorer: MLScorer instance
        model_loader: ModelLoader instance
        inference_engine: InferenceEngine instance
        chart_generator: ChartGenerator instance
    """
    from api.blueprints.data import register_data_routes
    from api.blueprints.csv import register_csv_routes
    from api.blueprints.submissions import register_submissions_routes
    from api.blueprints.patients import register_patients_routes
    from api.blueprints.charts import register_charts_routes
    from api.blueprints.stats import register_stats_routes

    register_data_routes(app, db, ml_scorer)
    register_csv_routes(app, db, ml_scorer, inference_engine)
    register_submissions_routes(app, db, ml_scorer)
    register_patients_routes(app, db, ml_scorer, model_loader)
    register_charts_routes(app, db, chart_generator)
    register_stats_routes(app, db, ml_scorer, model_loader)