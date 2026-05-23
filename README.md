# Hybrid Machine Learning Models for Aspect-Based Sentiment Analysis on MOOC Quora Reviews for E-Learning Personalization

This repository contains my Final Year Project: an end-to-end machine learning pipeline designed to parse, classify, and visualize student sentiments across granular e-learning aspects using a hybrid approach.

## 📁 Repository Structure

1. **`Data/`**: Divided into raw scraped text and progressive processing checkpoints (`preprocess_1` to `preprocess_final`) showing data cleansing staging, alongside LLM-assisted (`gemini_aspect_labels`) ground-truth validation datasets.
2. **`Notebooks/`**: Jupyter Notebooks detailing the web scraping, text preprocessing (NLTK, Pandas), and feature engineering (TF-IDF, LDA) workflows.
3. **`FYP_WebApp/`**: Production source code, HTML templates, and backend configuration for the interactive Flask dashboard.

---

## 🛠️ Tech Stack & Key Features

* **Data Engineering:** Web Scraping (Selenium), Text Processing (NLTK, Pandas), Topic Modeling (LDA), Text Vectorization (TF-IDF).
* **Machine Learning:** Optimized hybrid predictive modeling using **Decision Trees** and **Support Vector Machines (SVM)** with hyperparameter tuning via Grid Search.
* **Web Deployment:** A lightweight **Flask** web application backed by an **SQLite** database, delivering data insights using **Plotly** and **WordCloud** visualizations.

---

## 🚀 How to Run the Web Application Locally

To test the interactive dashboard on your local machine, follow these steps to set up your environment and launch the application:

1. **Clone or download** this repository to your local machine and open your terminal.
2. Navigate to the web application directory, initialize a fresh virtual environment, activate it, install dependencies, and launch the Flask server by executing the following command sequence:
   ```bash
   cd FYP_WebApp
   python -m venv venv
   .\venv\Scripts\activate
   pip install flask pandas nltk plotly wordcloud scikit-learn selenium
   py app.py
