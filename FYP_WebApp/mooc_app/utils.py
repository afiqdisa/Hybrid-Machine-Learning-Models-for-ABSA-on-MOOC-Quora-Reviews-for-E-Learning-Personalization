import sqlite3
import pandas as pd
import os
import io
import base64
import re
import string
import json
import plotly
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# --- DATABASE AND GLOBAL CONSTANTS ---
DATABASE_NAME = "mooc_analyzer.db"
SENTIMENT_COLORS = {'Positive': '#2ECC71', 'Negative': '#E74C3C', 'Neutral': '#F1C40F'}

# --- PREPROCESSING SETUP (Objects created once for efficiency) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- HELPER FUNCTIONS ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    # This path is adjusted to go up one level from the 'mooc_app' folder to find the database.
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', DATABASE_NAME)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_wordnet_pos(treebank_tag):
    """Maps NLTK's POS tags to WordNet's format."""
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def preprocess_text(text):
    """Cleans, tokenizes, removes stopwords, and lemmatizes text."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    pos_tags = pos_tag(filtered_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized_tokens)

def generate_wordcloud_base64(text_series, sentiment_colormap='viridis', width=380, height=230):
    """Generates a word cloud from a pandas Series and returns it as a base64 string."""
    if text_series.empty or text_series.str.strip().eq('').all():
        return None

    text = " ".join(review for review in text_series.dropna())
    if not text.strip():
        return None

    stopwords_for_wc = set(STOPWORDS)
    stopwords_for_wc.update([
        "course", "learn", "udemy", "coursera", "edx", "platform", "review", "student",
        "online", "teacher", "lecture", "material", "assignment", "exam", "mooc",
        "data", "skill", "project", "time", "good", "great", "really", "well", "also",
        "use", "get", "make", "see", "need", "try", "find", "take", "give", "look",
        "say", "tell", "ask", "answer", "question", "quora", "recommend", "experience",
        "content", "structure", "assessment", "interaction", "instructor", "topic",
        "subject", "information", "detail", "example", "point", "thing", "way", "many",
        "much", "lot", "bit", "provide", "offer", "include", "available", "help",
        "class", "video", "module", "week"
    ])

    try:
        wordcloud = WordCloud(
            width=width, height=height, background_color='white',
            min_font_size=10, max_words=50, stopwords=stopwords_for_wc,
            prefer_horizontal=0.95, colormap=sentiment_colormap, collocations=False
        ).generate(text)

        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8')
    except Exception as e:
        print(f"WordCloud generation error: {e}")
        return None

def parse_mixed_dates(date_str):
    """Robustly parses different date string formats."""
    if not isinstance(date_str, str):
        return None

    # Format 1: "Month Day, Year at HH:MM:SS AM/PM"
    try:
        # This format string directly matches "October 25, 2024 at 10:33:45 PM"
        return pd.to_datetime(date_str, format='%B %d, %Y at %I:%M:%S %p')
    except ValueError:
        # If the first format fails, try other common formats as a fallback
        try:
            # This handles formats like '2024-10-25' and others pandas recognizes automatically
            return pd.to_datetime(date_str)
        except ValueError:
            return None # Return None if all attempts fail

def get_platform_overview_data(platform_name):
    """Fetches overview data (sentiment counts and word clouds) for a specific platform."""
    conn = get_db_connection()
    overview_data = {
        'total_positive': 0, 'total_neutral': 0, 'total_negative': 0, 'grand_total': 0,
        'positive_wordcloud': None, 'neutral_wordcloud': None, 'negative_wordcloud': None
    }
    try:
        df_counts = pd.read_sql_query(
            "SELECT predicted_sentiment, COUNT(*) as count FROM reviews WHERE Platform = ? GROUP BY predicted_sentiment",
            conn, params=(platform_name,)
        )
        counts_map = {row['predicted_sentiment']: row['count'] for _, row in df_counts.iterrows()}
        overview_data.update({f"total_{s.lower()}": counts_map.get(s, 0) for s in ['Positive', 'Neutral', 'Negative']})
        overview_data['grand_total'] = sum(v for k, v in overview_data.items() if k.startswith('total_'))

        sentiment_color_maps = {'Positive': 'Greens', 'Neutral': 'YlOrBr', 'Negative': 'Reds'}
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            df_texts = pd.read_sql_query(
                "SELECT Lemmatized_Text FROM reviews WHERE Platform = ? AND predicted_sentiment = ?",
                conn, params=(platform_name, sentiment)
            )
            if not df_texts.empty:
                color = sentiment_color_maps.get(sentiment)
                overview_data[f'{sentiment.lower()}_wordcloud'] = generate_wordcloud_base64(df_texts['Lemmatized_Text'], sentiment_colormap=color)
    finally:
        conn.close()
    return overview_data

def get_platform_main_charts(platform_name):
    """Generates JSON and dynamic explanations for pie and line charts for a specific platform."""
    conn = get_db_connection()
    charts_data = {
        'pie_chart_json': '{}', 'line_chart_json': '{}',
        'pie_chart_explanation': 'No data available for sentiment distribution.',
        'line_chart_explanation': 'No trend data available.'
    }
    try:
        # --- Pie Chart Logic (This part is correct) ---
        df_pie = pd.read_sql_query(
            "SELECT predicted_sentiment, COUNT(*) as count FROM reviews WHERE Platform = ? GROUP BY predicted_sentiment",
            conn, params=(platform_name,)
        )
        if not df_pie.empty and df_pie['count'].sum() > 0:
            labels = df_pie['predicted_sentiment'].tolist()
            values = df_pie['count'].tolist()
            colors = [SENTIMENT_COLORS.get(l, '#cccccc') for l in labels]
            pie_fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), hole=0.3)])
            pie_fig.update_layout(legend=dict(bgcolor="white", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
            charts_data['pie_chart_json'] = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Pie chart explanation
            counts = {row['predicted_sentiment']: row['count'] for _, row in df_pie.iterrows()}
            total_reviews = sum(counts.values())
            sorted_sentiments = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            summary_parts = [f"<strong>{s}</strong> ({ (c / total_reviews) * 100 :.1f}%)" for s, c in sorted_sentiments]
            explanation = f"For <strong>{platform_name}</strong>, the sentiment breakdown shows that {summary_parts[0]} is the most common."
            if len(summary_parts) > 1:
                explanation += f" This is followed by {summary_parts[1]}"
            if len(summary_parts) > 2:
                explanation += f" and {summary_parts[2]}."
            charts_data['pie_chart_explanation'] = explanation

        # --- FINAL Line Chart Logic ---
        df_trends = pd.read_sql_query(
            "SELECT Date, predicted_sentiment FROM reviews WHERE Platform = ? AND Date IS NOT NULL",
            conn, params=(platform_name,)
        )
        
        # Use errors='coerce' to handle any date format issues gracefully
        df_trends['Parsed_Date'] = pd.to_datetime(df_trends['Date'], errors='coerce')
        df_trends.dropna(subset=['Parsed_Date'], inplace=True)

        if not df_trends.empty:
            df_trends['YearMonth'] = df_trends['Parsed_Date'].dt.to_period('M')
            trends = df_trends.groupby(['YearMonth', 'predicted_sentiment']).size().unstack(fill_value=0)

            min_date, max_date = trends.index.min().to_timestamp(), trends.index.max().to_timestamp()
            all_months = pd.period_range(start=min_date, end=max_date, freq='M')
            trends = trends.reindex(all_months, fill_value=0).reset_index()
            trends = trends.rename(columns={'index': 'YearMonth'})
            trends['YearMonthStr'] = trends['YearMonth'].astype(str)

            traces = [
                go.Scatter(
                    x=trends['YearMonthStr'].tolist(),
                    y=trends.get(s, [0] * len(trends)).tolist(),
                    name=s,
                    mode='lines+markers',
                    line=dict(color=SENTIMENT_COLORS.get(s))
                )
                for s in ['Positive', 'Neutral', 'Negative']
            ]

            line_layout = go.Layout(
                xaxis_title='<b>Month and Year</b>',
                yaxis_title='<b>Number of Reviews</b>',
                legend=dict(bgcolor='white', bordercolor='gray', borderwidth=1)
            )
            line_fig = go.Figure(data=traces, layout=line_layout)
            charts_data['line_chart_json'] = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Explanation logic
            trends['Total'] = trends[['Positive', 'Negative', 'Neutral']].sum(axis=1)
            peak_engagement_month = trends.loc[trends['Total'].idxmax()]
            peak_positive_month = trends.loc[trends['Positive'].idxmax()]
            charts_data['line_chart_explanation'] = f"For <b>{platform_name}</b>, engagement peaked in <b>{peak_engagement_month['YearMonthStr']}</b> with a total of {int(peak_engagement_month['Total'])} reviews. The highest volume of positive feedback was recorded in <b>{peak_positive_month['YearMonthStr']}</b>."

    except Exception as e:
        print(f"An error occurred in get_platform_main_charts for {platform_name}: {e}")
    finally:
        conn.close()
        
    return charts_data