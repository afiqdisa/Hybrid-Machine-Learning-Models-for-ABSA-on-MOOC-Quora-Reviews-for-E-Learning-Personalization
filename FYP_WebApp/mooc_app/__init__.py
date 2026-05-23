from flask import Flask
from datetime import datetime
import os

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, 
                instance_relative_config=True,
                template_folder='../templates', # Look for templates in the folder 'up' one level
                static_folder='../static'      # Look for static files in the folder 'up' one level
               )

    # This context processor makes the 'now' variable available to all templates
    # so we don't have to pass it in every render_template() call.
    @app.context_processor
    def inject_now():
        return {'now': datetime.now()}

    # The 'with app.app_context()' ensures that the application is ready
    # before we try to register our routes.
    with app.app_context():
        # Import and register the blueprint from our routes.py file
        from . import routes
        app.register_blueprint(routes.main_bp)

        # The NLTK download checks can also be moved here to run on startup
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')


    return app