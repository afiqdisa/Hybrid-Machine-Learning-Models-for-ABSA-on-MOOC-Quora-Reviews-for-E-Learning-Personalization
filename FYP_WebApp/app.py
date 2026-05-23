from mooc_app import create_app

# Create the application instance using the factory
app = create_app()

if __name__ == "__main__":
    # The debug=True flag is useful for development.
    # It provides detailed error pages and auto-reloads the server when you save a file.
    app.run(debug=True)