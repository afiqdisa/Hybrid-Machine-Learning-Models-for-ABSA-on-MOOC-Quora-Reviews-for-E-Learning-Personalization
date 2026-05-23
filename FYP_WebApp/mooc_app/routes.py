from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json
import plotly
import plotly.graph_objs as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import our own custom modules
from .utils import (
    get_db_connection, 
    generate_wordcloud_base64, 
    parse_mixed_dates, 
    preprocess_text,
    get_platform_overview_data,
    get_platform_main_charts,
    SENTIMENT_COLORS
)
from .models import SENTIMENT_MODELS

# --- GLOBAL CONSTANTS FOR ROUTES ---
PLATFORM_LOGOS = {
    "Coursera": "coursera logo.png",
    "edX": "edX logo.png",
    "Udemy": "udemy logo.png",
    "Default": "default_logo.png"
}
ASPECT_LIST_ORDER = ['Content', 'Structure', 'Assessment', 'Interaction', 'Instructor']

# Create a Blueprint object. All routes will be registered with this blueprint.
main_bp = Blueprint('main', __name__)

# --- Main App Routes ---

@main_bp.route("/")
def landing_page():
    return render_template("landing_page.html", active_page = "Welcome")

@main_bp.route("/home")
def home_page():
    return render_template("home_page.html", page_title="MOOC Platform Overview")

@main_bp.route("/dashboard")
def dashboard_page():
    conn = get_db_connection()
    context = {
        'page_title': "MOOC Review Insights",
        'page_explanation': "A summary of user sentiments from Quora reviews",
        'overall_pie_data_json': '{}',
        'line_chart_data_json': '{}',
        'pie_chart_explanation': 'No data available for sentiment distribution.',
        'line_chart_explanation': 'This line chart tracks the volume of reviews over time.',
        'overall_positive_wordcloud': None,
        'overall_neutral_wordcloud': None,
        'overall_negative_wordcloud': None,
        'aspect_list_for_explorer': ASPECT_LIST_ORDER
    }
    try:
        # Pie Chart Data
        df_pie = pd.read_sql_query(
            "SELECT predicted_sentiment, COUNT(*) as count FROM reviews WHERE predicted_sentiment IS NOT NULL GROUP BY predicted_sentiment",
            conn
        )
        if not df_pie.empty:
            labels = df_pie['predicted_sentiment'].tolist()
            colors = [SENTIMENT_COLORS.get(l.capitalize(), '#cccccc') for l in labels]
            fig = go.Figure(data=[go.Pie(labels=labels, values=df_pie['count'].tolist(), marker=dict(colors=colors), hole=0.3)], layout=go.Layout(legend=dict(bgcolor="white", bordercolor="lightgray", borderwidth=2)))
            context['overall_pie_data_json'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            counts = {row['predicted_sentiment']: row['count'] for _, row in df_pie.iterrows()}
            total_reviews = sum(counts.values())
            if total_reviews > 0:
                sorted_sentiments = sorted(counts.items(), key=lambda item: item[1], reverse=True)
                summary_parts = [f"<strong>{s}</strong> which is ({ (c / total_reviews) * 100 :.1f}%)" for s, c in sorted_sentiments]
                dynamic_sentence = f" The analysis shows that {summary_parts[0]} has the highest percentage"
                if len(summary_parts) > 1:
                    dynamic_sentence += f", followed by {summary_parts[1]}"
                if len(summary_parts) > 2:
                    dynamic_sentence += f", and lastly {summary_parts[2]}."
                else: 
                    dynamic_sentence += "."
                context['pie_chart_explanation'] = "This pie chart illustrates the overall sentiment distribution across all collected Quora reviews." + dynamic_sentence

        # Line Chart
        df_trends = pd.read_sql_query("SELECT Date, predicted_sentiment FROM reviews WHERE Date IS NOT NULL", conn)
        if not df_trends.empty:
            df_trends['Parsed_Date'] = df_trends['Date'].apply(parse_mixed_dates)
            df_trends.dropna(subset=['Parsed_Date'], inplace=True)
            if not df_trends.empty:
                df_trends['YearMonth'] = df_trends['Parsed_Date'].dt.to_period('M')
                trends = df_trends.groupby(['YearMonth', 'predicted_sentiment']).size().unstack(fill_value=0)
                min_date, max_date = trends.index.min().to_timestamp(), trends.index.max().to_timestamp()
                all_months = pd.period_range(start=min_date, end=max_date, freq='M')
                trends = trends.reindex(all_months, fill_value=0).reset_index()
                trends = trends.rename(columns={'index': 'YearMonth'})
                trends['YearMonthStr'] = trends['YearMonth'].astype(str)
                traces = [go.Scatter(x=trends['YearMonthStr'].tolist(), y=trends.get(s, [0] * len(trends)).tolist(), name=s, mode='lines+markers', line=dict(color=SENTIMENT_COLORS.get(s))) for s in ['Positive', 'Neutral', 'Negative']]
                fig = go.Figure(data=traces, layout=go.Layout(xaxis_title_text='<b>Month and Year</b>', yaxis_title_text='<b>Number of Reviews</b>', legend=dict(bgcolor="white", bordercolor="#d1d5db", borderwidth=2)))
                context['line_chart_data_json'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                peak_positive_month = trends.loc[trends['Positive'].idxmax()]
                peak_negative_month = trends.loc[trends['Negative'].idxmax()]
                peak_neutral_month = trends.loc[trends['Neutral'].idxmax()]
                context['line_chart_explanation'] = f"The analysis of sentiment trends reveals the peak months for each category. <strong>Positive</strong> sentiment peaked in <strong>{peak_positive_month['YearMonthStr']}</strong> with <strong>{peak_positive_month['Positive']}</strong> reviews. <strong>Negative</strong> sentiment saw its highest volume in <strong>{peak_negative_month['YearMonthStr']}</strong> with <strong>{peak_negative_month['Negative']}</strong> reviews, while <strong>Neutral</strong> sentiment was most frequent in <strong>{peak_neutral_month['YearMonthStr']}</strong> with <strong>{peak_neutral_month['Neutral']}</strong> reviews."
                
        # Word Clouds
        sentiment_color_maps = {'Positive': 'Greens', 'Neutral': 'YlOrBr', 'Negative': 'Reds'}
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            df_texts = pd.read_sql_query("SELECT Lemmatized_Text FROM reviews WHERE predicted_sentiment = ?", conn, params=(sentiment,))
            if not df_texts.empty:
                color = sentiment_color_maps.get(sentiment)
                context[f'overall_{sentiment.lower()}_wordcloud'] = generate_wordcloud_base64(df_texts['Lemmatized_Text'], sentiment_colormap=color)
    finally:
        conn.close()
    return render_template("dashboard_page.html", **context)


@main_bp.route("/coursera")
def coursera_page():
    return render_template("coursera_page.html",
                           page_title="Coursera Analysis", platform_name="Coursera",
                           platform_logo=PLATFORM_LOGOS.get("Coursera"),
                           overview_data=get_platform_overview_data("Coursera"),
                           main_charts_data=get_platform_main_charts("Coursera"),
                           aspect_list=ASPECT_LIST_ORDER)

@main_bp.route("/edx")
def edx_page():
    return render_template("edx_page.html",
                           page_title="edX Analysis", platform_name="edX",
                           platform_logo=PLATFORM_LOGOS.get("edX"),
                           overview_data=get_platform_overview_data("edX"),
                           main_charts_data=get_platform_main_charts("edX"),
                           aspect_list=ASPECT_LIST_ORDER)

@main_bp.route("/udemy")
def udemy_page():
    return render_template("udemy_page.html",
                           page_title="Udemy Analysis", platform_name="Udemy",
                           platform_logo=PLATFORM_LOGOS.get("Udemy"),
                           overview_data=get_platform_overview_data("Udemy"),
                           main_charts_data=get_platform_main_charts("Udemy"),
                           aspect_list=ASPECT_LIST_ORDER)

@main_bp.route("/recommendation", methods=['GET', 'POST'])
def recommendation_page():
    conn = get_db_connection()
    recommended_platform, selected_aspect, pie_chart_json, star_chart_json = None, None, '{}', '{}'
    all_platform_scores = [] 

    if request.method == 'POST':
        selected_aspect = request.form.get('selected_aspect')
        if selected_aspect:
            platform_scores_calc = []
            for platform in ['Coursera', 'edX', 'Udemy']:
                query = "SELECT predicted_sentiment, COUNT(*) as count FROM reviews WHERE Platform = ? AND LOWER(Aspect_Label) = LOWER(?) GROUP BY predicted_sentiment"
                df = pd.read_sql_query(query, conn, params=(platform, selected_aspect))
                counts = {row['predicted_sentiment']: row['count'] for _, row in df.iterrows()}
                total_reviews = sum(counts.values())
                positive_percentage = (counts.get('Positive', 0) / total_reviews) * 100 if total_reviews > 0 else 0
                platform_scores_calc.append({
                    'name': platform, 'positive_percentage': positive_percentage, 
                    'counts': counts, 'logo': PLATFORM_LOGOS.get(platform)
                })
            
            all_platform_scores = sorted(platform_scores_calc, key=lambda x: x['positive_percentage'], reverse=True)

            if all_platform_scores:
                best_platform_info = all_platform_scores[0]
                if best_platform_info['positive_percentage'] > 0:
                    recommended_platform = best_platform_info
                    recommended_platform['explanation'] = f"<strong>{recommended_platform['name']}</strong> is recommended for '<strong>{selected_aspect}</strong>' because it achieved the highest positive sentiment score of <strong>{recommended_platform['positive_percentage']:.1f}%</strong>"
                    
                    pie_labels, pie_values = list(recommended_platform['counts'].keys()), list(recommended_platform['counts'].values())
                    pie_colors = [SENTIMENT_COLORS.get(l, '#cccccc') for l in pie_labels]
                    pie_fig = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, marker=dict(colors=pie_colors), hole=0.3)])
                    pie_fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                    pie_chart_json = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

                    star_counts = recommended_platform['counts']
                    total = sum(star_counts.values())
                    r_values = [(star_counts.get(s, 0) / total * 100) if total > 0 else 0 for s in ['Positive','Neutral','Negative']]
                    star_fig = go.Figure(data=[go.Scatterpolar(r=r_values, theta=['Positive','Neutral','Negative'], fill='toself', marker_color='#2563eb')])
                    star_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(t=40, b=40, l=40, r=40))
                    star_chart_json = json.dumps(star_fig, cls=plotly.utils.PlotlyJSONEncoder)

    conn.close()
    return render_template("recommendation_page.html",
                           page_title="Analysis Hub", aspects_list=ASPECT_LIST_ORDER,
                           selected_aspect=selected_aspect, recommended_platform=recommended_platform,
                           all_platform_scores=all_platform_scores, pie_chart_json=pie_chart_json,
                           star_chart_json=star_chart_json)

@main_bp.route('/sentiment_analyzer', methods=['GET', 'POST'])
def sentiment_analyzer():
    results, review_text = [], ""
    if request.method == 'POST':
        review_text = request.form.get('review_text', "").strip()
        if review_text and SENTIMENT_MODELS:
            try:
                vectorizer = SENTIMENT_MODELS.get('tfidf_vectorizer')
                aspect_model = SENTIMENT_MODELS.get('aspect_extraction_model')
                dt_models = SENTIMENT_MODELS.get('dt_sentiment_models', {})
                svm_models = SENTIMENT_MODELS.get('svm_sentiment_models', {})
                weights = SENTIMENT_MODELS.get('hybrid_sentiment_weights', {})
                sentiment_classes = SENTIMENT_MODELS.get('sentiment_classes')

                processed_text = preprocess_text(review_text)
                text_vector = vectorizer.transform([processed_text])
                predicted_aspect = aspect_model.predict(text_vector)[0]
                
                dt_model, svm_model, aspect_weights = dt_models.get(predicted_aspect), svm_models.get(predicted_aspect), weights.get(predicted_aspect)
                predicted_sentiment = 'Error' 
                
                if all([dt_model, svm_model, aspect_weights, sentiment_classes is not None]):
                    prob_dt = dt_model.predict_proba(text_vector)[0]
                    prob_svm = svm_model.predict_proba(text_vector)[0]
                    avg_prob = aspect_weights['DT'] * prob_dt + aspect_weights['SVM'] * prob_svm
                    predicted_sentiment = sentiment_classes[np.argmax(avg_prob)]
                
                results = [{'aspect': predicted_aspect, 'sentiment': predicted_sentiment}]

            except Exception as e:
                print(f"[ERROR] An error occurred in the sentiment analyzer: {e}")
                results = [{'aspect': 'Error', 'sentiment': 'Could not process the request.'}]
    return render_template('sentiment_analyzer.html', results=results, review_text=review_text)

# --- AJAX API Endpoints ---

@main_bp.route("/get_aspect_visualization", methods=['POST'])
def get_aspect_visualization():
    data, conn = request.get_json(), get_db_connection()
    selected_aspect, selected_viz_type = data.get('aspect'), data.get('viz_type')
    response_data = {}
    try:
        if selected_viz_type == "Sentiment Distribution":
            df = pd.read_sql_query("SELECT predicted_sentiment, COUNT(*) as count FROM reviews WHERE LOWER(Aspect_Label) = LOWER(?) GROUP BY predicted_sentiment", conn, params=(selected_aspect,))
            counts = {row['predicted_sentiment']: row['count'] for _, row in df.iterrows()}
            if any(counts.values()):
                fig = go.Figure(data=[go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color=[SENTIMENT_COLORS.get(s) for s in counts.keys()])])
                fig.update_layout(title_text=f'Bar Chart: {selected_aspect} Sentiment Distribution', title_x=0.5)
                response_data['chart_json'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                total_reviews, dominant_sentiment = sum(counts.values()), max(counts, key=counts.get)
                response_data['explanation_html'] = f"<h4 class='font-semibold text-lg mb-2'>Overall Sentiment Distribution for '{selected_aspect}'</h4><p>This bar chart shows the sentiment breakdown for reviews related to <strong>{selected_aspect}</strong>. Out of a total of <strong>{total_reviews}</strong> reviews, the most common sentiment is <strong>'{dominant_sentiment}'</strong> with <strong>{counts[dominant_sentiment]}</strong> mentions.</p>"
            else:
                response_data['message'] = f"No sentiment data available for the aspect: {selected_aspect}."
        elif selected_viz_type == "Most Frequent Mentioned Words":
            df = pd.read_sql_query("SELECT Lemmatized_Text FROM reviews WHERE LOWER(Aspect_Label) = LOWER(?)", conn, params=(selected_aspect,))
            wordcloud_img = generate_wordcloud_base64(df['Lemmatized_Text'])
            if wordcloud_img:
                response_data['wordcloud_image'] = wordcloud_img
                response_data['explanation_html'] = f"<h4 class='font-semibold text-lg mb-2'>Overall Frequent Words for '{selected_aspect}'</h4><p>This word cloud displays the most frequently used terms in reviews specifically discussing the <strong>{selected_aspect}</strong>. The larger the word, the more often it was mentioned.</p>"
            else:
                response_data['message'] = f"Not enough data to generate a word cloud for {selected_aspect}."
    finally:
        conn.close()
    return jsonify(response_data)

@main_bp.route("/get_aspect_details", methods=['POST'])
def get_aspect_details():
    data, conn = request.get_json(), get_db_connection()
    platform, aspect = data.get('platform'), data.get('aspect')
    response = {}
    try:
        df = pd.read_sql_query("SELECT predicted_sentiment s, COUNT(*) c FROM reviews WHERE Platform=? AND LOWER(TRIM(Aspect_Label)) LIKE LOWER(?) GROUP BY s", conn, params=(platform, f"%{aspect}%"))
        counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        for _, row in df.iterrows():
            sentiment = str(row['s']).strip().capitalize()
            if sentiment in counts:
                counts[sentiment] = row['c']
        
        bar_fig = go.Figure(data=[go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color=[SENTIMENT_COLORS.get(s) for s in counts.keys()])])
        bar_fig.update_layout(showlegend=False, margin=dict(t=5, b=5, l=5, r=5))
        response['bar_chart_json'] = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

        total = sum(counts.values())
        r_values = [(counts.get(s, 0) / total * 100) if total > 0 else 0 for s in ['Positive', 'Neutral', 'Negative']]
        star_fig = go.Figure(data=[go.Scatterpolar(r=r_values, theta=['Positive', 'Neutral', 'Negative'], fill='toself')])
        star_fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=False, margin=dict(t=40,b=40,l=40,r=40))
        response['star_plot_json'] = json.dumps(star_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        df_wc = pd.read_sql_query("SELECT Lemmatized_Text FROM reviews WHERE Platform=? AND LOWER(TRIM(Aspect_Label)) LIKE LOWER(?)", conn, params=(platform, f"%{aspect}%"))
        response['wordcloud_image'] = generate_wordcloud_base64(df_wc['Lemmatized_Text'])
        
        if total > 0:
            dominant_sentiment, dominant_count = max(counts, key=counts.get), max(counts.values())
            response['bar_chart_explanation'] = f"For '<strong>{aspect}</strong>' on <strong>{platform}</strong>, the sentiment is predominantly '<strong>{dominant_sentiment}</strong>', making up <strong>{dominant_count}</strong> out of <strong>{total}</strong> reviews."
            response['star_chart_explanation'] = f"This radar shows the sentiment balance. <strong>Positive</strong> sentiment accounts for <strong>{r_values[0]:.1f}%</strong> of all reviews for this aspect."
        else:
            response['bar_chart_explanation'] = "No review data found."
            response['star_chart_explanation'] = "No review data found."
        response['wordcloud_explanation'] = f"The most frequent words in '<strong>{aspect}</strong>' reviews are shown here."
    finally:
        conn.close()
    return jsonify(response)


@main_bp.route("/compare_aspects", methods=['POST'])
def compare_aspects():
    data, conn = request.get_json(), get_db_connection()
    platform, aspect1, aspect2 = data.get('platform'), data.get('aspect1'), data.get('aspect2')
    response = {'visuals': {}, 'conclusion': {}}
    def get_metrics(p, a):
        df = pd.read_sql_query("SELECT predicted_sentiment s, COUNT(*) c FROM reviews WHERE Platform=? AND LOWER(TRIM(Aspect_Label)) LIKE LOWER(?) GROUP BY s", conn, params=(p, f"%{a}%"))
        c = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        for _, row in df.iterrows():
            sentiment = str(row['s']).strip().capitalize()
            if sentiment in c: c[sentiment] = row['c']
        t = sum(c.values())
        m = {'positive_percentage': (c['Positive']/t*100) if t>0 else 0, 'pn_ratio': c['Positive']/c['Negative'] if c.get('Negative',0)>0 else float('inf'), 'net_sentiment_score': c['Positive'] - c.get('Negative',0)}
        return c, m
    try:
        counts1, metrics1 = get_metrics(platform, aspect1)
        counts2, metrics2 = get_metrics(platform, aspect2)
        response['visuals']['data_table_data'] = {'headers': ["Metric", aspect1, aspect2], 'rows': [{'label': l, 'values': [v1, v2]} for l,v1,v2 in [('Positive Reviews',counts1['Positive'],counts2['Positive']), ('Neutral Reviews',counts1['Neutral'],counts2['Neutral']), ('Negative Reviews',counts1['Negative'],counts2['Negative']), ('Total Reviews',sum(counts1.values()),sum(counts2.values()))]]}
        pn1 = '∞' if metrics1['pn_ratio']==float('inf') else f"{metrics1['pn_ratio']:.2f}:1"
        pn2 = '∞' if metrics2['pn_ratio']==float('inf') else f"{metrics2['pn_ratio']:.2f}:1"
        response['visuals']['key_metrics_data'] = {'headers': ["Metric", aspect1, aspect2], 'rows': [{'label': l, 'values': [v1, v2]} for l,v1,v2 in [('Positive Sentiment %',f"{metrics1['positive_percentage']:.1f}%", f"{metrics2['positive_percentage']:.1f}%"), ('Positive-to-Negative Ratio',pn1, pn2), ('Net Sentiment Score',metrics1['net_sentiment_score'], metrics2['net_sentiment_score'])]]}
        fig = go.Figure(data=[go.Bar(name=s, x=[aspect1, aspect2], y=[counts1.get(s, 0), counts2.get(s, 0)], marker_color=SENTIMENT_COLORS.get(s)) for s in ['Positive', 'Neutral', 'Negative']])
        fig.update_layout(barmode='group', title_x=0.5, xaxis_title_text='<b>Aspects</b>', yaxis_title_text='<b>Review Count</b>', legend_title_text='Sentiment', legend=dict(bgcolor='white', bordercolor='gray', borderwidth=1))
        response['visuals']['bar_chart_json'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        response['visuals']['bullet_chart_data'] = [{'name': a, 'value': m['positive_percentage']} for a, m in [(aspect1, metrics1), (aspect2, metrics2)]]
        winner, winner_metrics = (aspect1, metrics1) if metrics1['net_sentiment_score'] > metrics2['net_sentiment_score'] else (aspect2, metrics2)
        winner_pnr_text = 'an overwhelmingly positive ratio' if winner_metrics['pn_ratio'] == float('inf') else f"a ratio of <strong>{winner_metrics['pn_ratio']:.2f}</strong> positive reviews for every one negative review"
        response['conclusion']['text'] = f"Based on the analysis, <strong>{winner}</strong> has a more favorable sentiment profile."
        response['conclusion']['explanation'] = f"<ul class='list-disc list-inside text-left space-y-2 mt-2'><li><b>Positive Sentiment Percentage:</b> <b>{winner}</b> leads with <b>{winner_metrics['positive_percentage']:.1f}%</b>.</li><li><b>Net Sentiment Score:</b> <b>{winner}</b> has a stronger positive balance with a score of <b>{winner_metrics['net_sentiment_score']}</b>.</li><li><b>Positive-to-Negative Ratio:</b> <b>{winner}</b> shows less controversy with {winner_pnr_text}.</li></ul>"
    finally:
        conn.close()
    return jsonify(response)


@main_bp.route("/get_platform_ranking", methods=['POST'])
def get_platform_ranking():
    data, conn = request.get_json(), get_db_connection()
    aspect, metric = data.get('aspect'), data.get('metric')
    if not aspect or not metric: return jsonify({'error': 'Missing aspect or metric'}), 400
    platform_scores = []
    for platform in ['Coursera', 'edX', 'Udemy']:
        df = pd.read_sql_query("SELECT predicted_sentiment, COUNT(*) as count FROM reviews WHERE Platform = ? AND LOWER(Aspect_Label) = LOWER(?) GROUP BY predicted_sentiment", conn, params=(platform, aspect))
        counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for _, row in df.iterrows(): counts[row['predicted_sentiment']] = row['count']
        total = sum(counts.values())
        platform_scores.append({'name': platform, 'logo': PLATFORM_LOGOS.get(platform), 'positive_percentage': (counts['Positive']/total*100) if total>0 else 0, 'negative_percentage': (counts['Negative']/total*100) if total>0 else 0, 'pn_ratio': counts['Positive']/counts['Negative'] if counts['Negative']>0 else float('inf'), 'net_score': counts['Positive'] - counts['Negative'], 'total_reviews': total})
    conn.close()
    platform_scores.sort(key=lambda x: x[metric], reverse=(metric != 'negative_percentage'))
    return jsonify(platform_scores)

@main_bp.route("/compare_platforms", methods=['POST'])
def compare_platforms():
    data, conn = request.get_json(), get_db_connection()
    platform_a, platform_b, aspect = data.get('platform_a'), data.get('platform_b'), data.get('aspect')
    if not all([platform_a, platform_b, aspect]): return jsonify({'error': 'Missing fields'}), 400
    if platform_a == platform_b: return jsonify({'error': 'Please select two different platforms'}), 400
    results = {'metrics': {}, 'visuals': {}, 'similarity': {}, 'platform_a_name': platform_a, 'platform_b_name': platform_b, 'aspect': aspect}
    def get_metrics(p, a):
        df = pd.read_sql_query("SELECT predicted_sentiment, COUNT(*) as count FROM reviews WHERE Platform = ? AND LOWER(Aspect_Label) = LOWER(?) GROUP BY predicted_sentiment", conn, params=(p, a))
        c = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for _, row in df.iterrows(): c[row['predicted_sentiment'].strip().capitalize()] = row['count']
        t = sum(c.values())
        m = {'positive_percentage': (c['Positive']/t*100) if t>0 else 0, 'pn_ratio': c['Positive']/c['Negative'] if c['Negative']>0 else float('inf'), 'net_score': c['Positive'] - c['Negative']}
        return c, m
    try:
        counts_a, metrics_a = get_metrics(platform_a, aspect)
        counts_b, metrics_b = get_metrics(platform_b, aspect)
        results['metrics'][platform_a], results['metrics'][platform_b] = metrics_a, metrics_b
        winner = platform_a if metrics_a['net_score'] > metrics_b['net_score'] else platform_b if metrics_b['net_score'] > metrics_a['net_score'] else ""
        if winner:
            winner_metrics = metrics_a if winner == platform_a else metrics_b
            winner_pnr = '∞' if winner_metrics['pn_ratio'] == float('inf') else f"{winner_metrics['pn_ratio']:.2f}:1"
            results['comparison_conclusion'] = f"<p class='mb-3'>Based on the analysis, <strong>{winner}</strong> has a more favorable sentiment profile for the '<strong>{aspect}</strong>' aspect. Here's a breakdown:</p><ul class='list-disc list-inside space-y-2 text-black'><li><strong>Positive Sentiment Percentage:</strong> <strong>{winner}</strong> leads with <strong>{winner_metrics['positive_percentage']:.1f}%</strong></li><li><strong>Net Sentiment Score:</strong> <strong>{winner}</strong> has a stronger positive balance with a score of <strong>{winner_metrics['net_score']}</strong></li><li><strong>Positive-to-Negative Ratio:</strong> <strong>{winner}</strong> has a ratio of <strong>{winner_pnr}</strong></li></ul>"
        else:
            results['comparison_conclusion'] = f"<p>For the '<strong>{aspect}</strong>' aspect, both platforms are perceived very similarly.</p>"
        
        fig = go.Figure()
        for s in ['Positive', 'Neutral', 'Negative']: fig.add_trace(go.Bar(name=s, x=[platform_a, platform_b], y=[counts_a[s], counts_b[s]], marker_color=SENTIMENT_COLORS.get(s), hovertemplate='<b>'+s+'</b><br>Count: %{y}<extra></extra>'))
        fig.update_layout(barmode='group', xaxis_title='<b>MOOC Platform</b>', yaxis_title='<b>Number of Reviews</b>', legend_title='<b>Sentiment</b>', legend=dict(orientation='v', x=1.02, y=1, xanchor='left', yanchor='top', bgcolor='white', bordercolor='black', borderwidth=1), margin=dict(r=120))
        results['visuals']['bar_chart_json'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        results['visuals']['table'] = {'headers': ['Metric', platform_a, platform_b], 'rows': [['Positive', counts_a['Positive'], counts_b['Positive']], ['Neutral', counts_a['Neutral'], counts_b['Neutral']], ['Negative', counts_a['Negative'], counts_b['Negative']], ['Total', sum(counts_a.values()), sum(counts_b.values())]]}
        results['visuals']['bullet_chart_data'] = [{'name': platform_a, 'value': metrics_a['positive_percentage']}, {'name': platform_b, 'value': metrics_b['positive_percentage']}]
        
        df_a_text, df_b_text = pd.read_sql_query("SELECT Lemmatized_Text FROM reviews WHERE Platform = ? AND LOWER(Aspect_Label) = LOWER(?)", conn, params=(platform_a, aspect)), pd.read_sql_query("SELECT Lemmatized_Text FROM reviews WHERE Platform = ? AND LOWER(Aspect_Label) = LOWER(?)", conn, params=(platform_b, aspect))
        if not df_a_text.empty and not df_b_text.empty:
            corpus_a, corpus_b = " ".join(df_a_text['Lemmatized_Text'].dropna()), " ".join(df_b_text['Lemmatized_Text'].dropna())
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([corpus_a, corpus_b])
                sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                results['similarity'] = {'score': f"{sim_score:.2f}", 'explanation': f"The language used in reviews for this aspect on both platforms is {sim_score:.0%} similar"}
            except ValueError:
                 results['similarity'] = {'score': "N/A", 'explanation': "Similarity could not be calculated (vocabulary might be empty)."}
        else:
            results['similarity'] = {'score': "N/A", 'explanation': "Could not calculate similarity due to insufficient data."}
    finally:
        conn.close()
    return jsonify(results)