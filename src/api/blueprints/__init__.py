"""Register all blueprints with the Flask app."""
from src.api.blueprints.routes import register_all_routes

def register_all_blueprints(app, db, ml_scorer, model_loader, inference_engine, chart_generator):
    register_all_routes(app, db, ml_scorer, model_loader, inference_engine, chart_generator)