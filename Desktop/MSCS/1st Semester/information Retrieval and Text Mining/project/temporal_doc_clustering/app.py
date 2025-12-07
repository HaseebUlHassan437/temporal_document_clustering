"""
Enhanced Dashboard with Gemini AI Integration
Temporal Document Clustering for E-Discovery
Authors: Haseeb Ul Hassan (MSCS25003)
Features: Dynamic clustering, ranked search, Gemini-powered insights
"""

import dash
from dash import dcc, html, Input, Output, State, ctx, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re
import requests
import traceback
import os
try:
    from dotenv import load_dotenv
except Exception:
    # If python-dotenv is not installed, provide a no-op loader and warn the user.
    def load_dotenv(*args, **kwargs):
        print("⚠️ python-dotenv not installed; skipping `.env` load. Install with: `pip install python-dotenv`")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# (joblib persistence removed per user request)

# ==================== GEMINI API CONFIG ====================
# Load environment variables from a local .env file if present (no-op if dotenv missing)
load_dotenv()

# Read the API key from the environment. Set `GEMINI_API_KEY` in your shell
# or place it in a `.env` file at project root (see `.env.example`).
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = "models/gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent"

if not GEMINI_API_KEY:
    print("⚠️ GEMINI_API_KEY not found in environment. Gemini API calls will fail.")

# NOTE: For security, keep API keys out of source control. Use environment
# variables or secret management in production.

# ==================== HELPER FUNCTIONS ====================

def query_gemini(prompt: str, context: str = "", max_retries: int = 3) -> str:
    """
    Query Gemini API with context management for RAG-like behavior
    Passes selected email content as context to ensure answers are based on selected documents only.
    """
    if context:
        full_prompt = f"""You are an intelligent assistant analyzing email documents.

## Email Context (selected documents):
{context}

## User Question:
{prompt}

**Important**: Answer ONLY based on the email content provided above. 
Do not use external knowledge. If the answer is not in the emails, say so clearly."""
    else:
        full_prompt = prompt
    
    # HTTP headers and payload for the Gemini REST API
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code == 429:
                print(f"Rate limit hit, retrying... (attempt {attempt+1}/{max_retries})")
                continue
            else:
                return f"⚠️ API Error ({response.status_code}): {response.text[:200]}"
        
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt+1}/{max_retries}")
            continue
        except Exception as e:
            # Log the error and return a short user-visible error string
            print(f"Error querying Gemini: {e}")
            return f"❌ Error: {str(e)[:200]}"
    
    return "❌ Failed after retries. Please try again."


def compute_relevance_scores(texts: list, query: str, vectorizer=None) -> np.ndarray:
    """
    Compute TF-IDF relevance scores for documents vs query.
    Returns normalized scores (0-1) based on cosine similarity.
    """
    # We try to compute TF-IDF vectors and cosine similarities.
    # If vectorizer is None we fit a small local TF-IDF (useful for isolated calls).
    try:
        if vectorizer is None:
            # Fit a temporary TF-IDF on (texts + query). This is slower but self-contained.
            vectorizer = TfidfVectorizer(max_features=500, max_df=0.8, min_df=2)
            tfidf_matrix = vectorizer.fit_transform(texts + [query])
        else:
            # Use the provided (already-fitted) vectorizer to map texts+query into same space.
            tfidf_matrix = vectorizer.transform(texts + [query])
        
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        scores = cosine_similarity(doc_vecs, query_vec).flatten()
        
        # Normalize so top score becomes 1.0 (makes thresholding intuitive)
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    except Exception:
        # On unexpected failure, return a neutral score (0.5) for every document.
        return np.ones(len(texts)) * 0.5


def format_email_context(email_rows: list) -> str:
    """Format selected emails into structured context for Gemini API"""
    if not email_rows or len(email_rows) == 0:
        return ""
    
    context_parts = []
    for idx, row in enumerate(email_rows, 1):
        email_text = f"""
--- Email {idx} ---
Date: {row.get('date', 'N/A')}
From: {row.get('from', 'Unknown')}
Subject: {row.get('subject', 'No Subject')}
Body: {row.get('cleaned_text', row.get('body', 'No content'))}
"""
        context_parts.append(email_text)
    
    return "\n".join(context_parts)


# ==================== APP INITIALIZATION ====================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)
app.title = "E-Discovery: Temporal Document Clustering + AI Insights"

# The Dash app object is the main application server. `suppress_callback_exceptions`
# allows callbacks referencing ids that are added dynamically later in the layout.

# Load data
print("="*70)
print("INITIALIZING DASHBOARD")
print("="*70)

DATA_DIR = Path("data")
# Prefer merged file with raw messages when available (created by scripts/merge_raw_into_clustered.py)
RAW_CLUSTERED = DATA_DIR / 'clustered_emails_raw.csv'
CLUSTERED = DATA_DIR / 'clustered_emails.csv'
if RAW_CLUSTERED.exists():
    DATA_PATH = RAW_CLUSTERED
    print(f"Loading merged clustered file with raw bodies: {DATA_PATH}")
elif CLUSTERED.exists():
    DATA_PATH = CLUSTERED
    print(f"Loading clustered file: {DATA_PATH}")
else:
    DATA_PATH = DATA_DIR / 'clustered_emails.csv'
    print(f"Warning: clustered file not found, attempted fallback to {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
print(f"✓ Loaded {len(df):,} documents from {DATA_PATH.name}")

# `df` is the canonical in-memory DataFrame used across callbacks.
# Columns expected: date, from, subject, body, cleaned_text, processed_text

min_date = df['date'].min()
max_date = df['date'].max()
days_range = (max_date - min_date).days

# Initialize TF-IDF
print("Initializing TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    max_df=0.8,
    min_df=5,
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'].fillna(''))
print(f"✓ TF-IDF matrix: {tfidf_matrix.shape}")

# `tfidf_vectorizer` stores the vocabulary and IDF weights (useful for transforming queries).
# `tfidf_matrix` is a sparse matrix (n_documents x n_features) used for clustering/search.

# Default clustering
print("Performing initial clustering...")
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(tfidf_matrix.toarray())
df['pca_x'] = pca_coords[:, 0]
df['pca_y'] = pca_coords[:, 1]

# PCA is used only for visualization (2D). Note: tfidf_matrix is converted to dense
# with `.toarray()` which may be memory intensive for large corpora. For large data
# use `TruncatedSVD` instead (sparse-friendly).

# PCA is used only for visualization (2D). Note: tfidf_matrix is converted to dense
# with `.toarray()` which may be memory intensive for large corpora. For large data
# use `TruncatedSVD` instead (sparse-friendly).

# Get top terms per cluster
def get_cluster_top_terms(kmeans_model, vectorizer, n_terms=5):
    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for i in range(kmeans_model.n_clusters):
        center = kmeans_model.cluster_centers_[i]
        top_indices = center.argsort()[-n_terms:][::-1]
        cluster_terms[i] = [feature_names[idx] for idx in top_indices]
    return cluster_terms

cluster_terms = get_cluster_top_terms(kmeans, tfidf_vectorizer)
print("✓ Clustering complete")
print("="*70 + "\n")

# App state
app_state = {
    'current_kmeans': kmeans,
    'current_pca': pca,
    'current_tfidf': tfidf_vectorizer,
    'current_tfidf_matrix': tfidf_matrix,
    'current_clusters': 8,
    'cluster_terms': cluster_terms,
    'selected_emails': [],
    'conversation_history': []
}

# ==================== LAYOUT ====================

app.layout = dbc.Container([
    # ---------- HEADER + STATS ----------
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    "🤖 AI-Powered Temporal Document Clustering",
                    html.Br(),
                    html.Span("with Gemini Insights", style={'font-size': '0.6em', 'color': '#666'})
                ], className="text-center mb-2 mt-4"),
                html.H5("E-Discovery System with Intelligent Analysis",
                       className="text-center text-muted mb-1"),
                html.P("CS 516: Information Retrieval & Text Mining | ITU Fall 2025",
                      className="text-center text-muted small"),
                html.Hr()
            ])
        ])
    ]),
    
    # Stats Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{len(df):,}", className="text-primary mb-0"),
                    html.P("Total Emails", className="text-muted mb-0 small")
                ])
            ], className="text-center")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='cluster-count-display', className="text-success mb-0"),
                    html.P("Clusters", className="text-muted mb-0 small")
                ])
            ], className="text-center")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{days_range}", className="text-info mb-0"),
                    html.P("Days Span", className="text-muted mb-0 small")
                ])
            ], className="text-center")
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='filtered-count', className="text-warning mb-0"),
                    html.P("Visible Docs", className="text-muted mb-0 small")
                ])
            ], className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='selected-count', className="text-danger mb-0"),
                    html.P("Selected", className="text-muted mb-0 small")
                ])
            ], className="text-center")
        ], width=3)
    ], className="mb-4"),
    # The header and stats row above show global counts and quick metrics.
    # The main page uses a two-column layout: left controls (filters/search)
    # and right panel for visualizations and the AI (Gemini) tab.
    
    # Main 2-Column Layout
    dbc.Row([
        # LEFT PANEL - Controls
        dbc.Col([
            # Help Card
            dbc.Alert([
                html.Strong("📘 How to Use:", className="d-block mb-2"),
                html.Small([
                    "1️⃣ Adjust filters (date/cluster/search)", html.Br(),
                    "2️⃣ Click scatter points to select emails", html.Br(),
                    "3️⃣ Go to AI tab → Ask questions"
                ], style={'fontSize': '11px'})
            ], color="info", className="mb-3 py-2"),
            
            # Clustering Control
            dbc.Card([
                dbc.CardHeader(html.H6("⚙️ Clustering", className="mb-0")),
                dbc.CardBody([
                    dcc.Slider(
                        id='k-slider',
                        min=3, max=15, step=1, value=8,
                        marks={i: str(i) for i in range(3, 16, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    dbc.Button("🔄 Re-Cluster", id='recluster-btn',
                              color="warning", className="w-100 mt-2", size="sm"),
                    html.Div(id='recluster-status', className="mt-2 small")
                ], className="p-3")
            ], className="mb-3"),
            
            # Temporal Filter
            dbc.Card([
                dbc.CardHeader(html.H6("📅 Time Range", className="mb-0")),
                dbc.CardBody([
                    dcc.RangeSlider(
                        id='date-slider',
                        min=0, max=days_range, value=[0, days_range],
                        marks={
                            0: min_date.strftime('%Y-%m'),
                            days_range//2: (min_date + pd.Timedelta(days=days_range//2)).strftime('%Y-%m'),
                            days_range: max_date.strftime('%Y-%m')
                        },
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id='date-text', className="text-center text-muted mt-2 small")
                ], className="p-3")
            ], className="mb-3"),
            
            # Ranked Keyword Search
            # - User enters keywords then presses the Search button.
            # - Slider controls set minimum relevance threshold and top-K results.
            dbc.Card([
                dbc.CardHeader(html.H6("🔍 Ranked Search", className="mb-0")),
                dbc.CardBody([
                    dbc.Textarea(
                        id='search-box',
                        placeholder='Keywords (space-separated)\ne.g., "energy california"',
                        style={'height': '70px', 'font-size': '12px'},
                        className="mb-2"
                    ),
                    html.Label("Relevance Threshold:", className="small fw-bold"),
                    html.P("Min score to show results (higher = stricter)", 
                          className="text-muted small mb-1", style={'fontSize': '10px'}),
                    dcc.Slider(
                        id='threshold-slider',
                        min=0, max=0.99, step=0.01, value=0.1,
                        marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 0.99: '0.99'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Label("Max Results:", className="small fw-bold mt-2"),
                    html.P("Maximum documents to display", 
                          className="text-muted small mb-1", style={'fontSize': '10px'}),
                    dcc.Slider(
                        id='topk-slider',
                        min=10, max=500, step=10, value=50,
                        marks={10: '10', 250: '250', 500: '500'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    dbc.Button("🔎 Search", id='search-button',
                              color="primary", className="w-100 mt-3", size="sm"),
                    html.Div(id='search-status', className="mt-2 small")
                ], className="p-3")
            ], className="mb-3"),
            
            # Cluster Filter
            dbc.Card([
                dbc.CardHeader(html.H6("🎯 Cluster Filter", className="mb-0")),
                dbc.CardBody([
                    dcc.Dropdown(id='cluster-filter', clearable=False),
                    html.Div(id='cluster-info', className="mt-2 small")
                ], className="p-3")
            ], className="mb-3"),
            
            # Email Selection for Gemini
            # This panel shows which emails the user has added to the Gemini context.
            dbc.Card([
                dbc.CardHeader(html.H6("📧 Selected Emails", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='selected-emails-display',
                            children=[html.P("No emails selected. Click points to select.", 
                                           className="text-muted small")]),
                    dbc.Button("🗑️ Clear Selection", id='clear-selection-btn',
                              color="danger", className="w-100 mt-2", size="sm")
                ], className="p-3")
            ])
        ], width=3),
        
        # RIGHT PANEL - Visualizations + AI Chat
        dbc.Col([
            dbc.Tabs([
                # TAB 1: Visualizations
                dbc.Tab(label="📊 Visualizations", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("Timeline", className="mb-0")),
                                dbc.CardBody([
                                    dcc.Loading(
                                        dcc.Graph(id='timeline-chart',
                                                 style={'height': '200px'},
                                                 config={'responsive': True})
                                    )
                                ], className="p-0")
                            ], className="mb-3")
                        ])
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("Cluster Scatter (PCA)", className="mb-0")),
                                dbc.CardBody([
                                    html.P("Click points to select emails for analysis. "
                                          "Size = relevance score.",
                                          className="text-muted small mb-2"),
                                    dcc.Loading(
                                        dcc.Graph(id='cluster-scatter',
                                                 style={'height': '500px'},
                                                 config={'responsive': True})
                                    )
                                ], className="p-2")
                            ])
                        ])
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("Selected Email (from scatter click)", className="mb-0")),
                                dbc.CardBody([
                                    dcc.Loading(
                                        html.Div(
                                            id='inline-doc-details',
                                            children=[
                                                html.P("Click a point in the scatter to view the email here.",
                                                       className="text-muted text-center p-4")
                                            ]
                                        )
                                    )
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # TAB 2: Document Details
                # Shows full email body when a point is clicked or when user selects a doc.
                dbc.Tab(label="📄 Document", children=[
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Loading([
                                html.Div(id='document-details',
                                        children=[html.P("Select an email to view.",
                                                       className="text-muted text-center p-5")])
                            ])
                        ], style={'maxHeight': '700px', 'overflowY': 'auto'})
                    ])
                ]),
                
                # TAB 3: Gemini AI Analysis
                # - Displays context summary for selected emails
                # - Lets user ask questions which are sent to Gemini with selected emails as context
                dbc.Tab(label="🤖 AI Insights (Gemini)", children=[
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Analyze Selected Emails with Gemini", className="mb-3"),
                            
                            # Context Summary
                            dbc.Alert(
                                [html.Strong("Selected Documents Summary: "),
                                 html.Span(id='gemini-context-summary', className="small")],
                                id='context-alert',
                                color="info",
                                dismissable=False,
                                className="mb-3"
                            ),
                            
                            # Query Input
                            html.Label("Ask about the selected emails:", 
                                      className="fw-bold small mb-2"),
                            dbc.InputGroup([
                                dbc.Textarea(
                                    id='gemini-query',
                                    placeholder='e.g., "What are the main topics?" '
                                              '"Summarize key points." '
                                              '"What actions were discussed?"',
                                    style={'height': '80px', 'font-size': '12px'}
                                ),
                            ]),
                            
                            dbc.Button([html.I(className="bi bi-brain me-2"), "Ask Gemini"],
                                      id='gemini-button',
                                      color="success",
                                      className="w-100 mt-2",
                                      size="sm"),
                            
                            # Response Display
                            dbc.Alert(
                                id='gemini-loading',
                                is_open=False,
                                color="warning",
                                className="mt-3"
                            ),
                            
                            html.Div(id='gemini-response',
                                    className="mt-3 p-3 bg-light rounded",
                                    style={'minHeight': '200px',
                                          'maxHeight': '500px',
                                          'overflowY': 'auto',
                                          'whiteSpace': 'pre-wrap',
                                          'wordBreak': 'break-word',
                                          'fontSize': '13px',
                                          'lineHeight': '1.6'},
                                    children=[
                                        html.P("No analysis yet. "
                                              "Select emails and ask a question.",
                                              className="text-muted")
                                    ]),
                            
                            # Conversation History
                            html.Hr(className="mt-4"),
                            html.H6("Recent Queries:", className="small mb-2"),
                            html.Div(id='query-history',
                                    style={'maxHeight': '200px',
                                          'overflowY': 'auto'},
                                    children=[
                                        html.P("No queries yet.",
                                              className="text-muted small")
                                    ])
                        ], style={'minHeight': '700px'})
                    ])
                ])
            ])
        ], width=9)
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(className="mt-4"),
            html.P([
                html.Strong("CS 516 Project: "), 
                "Temporal Document Clustering for E-Discovery | ",
                html.Strong("Team: "), "Haseeb Ul Hassan (MSCS25003) ",
                html.Strong("Instructor: "), "Dr. Ahmad Mustafa"
            ], className="text-center text-muted small mb-3")
        ])
    ]),
    
    # Stores
    dcc.Store(id='selected-emails-store'),
    dcc.Store(id='filtered-df-store'),
    dcc.Store(id='current-doc-store'),
    dcc.Store(id='gemini-history-store', data=[])
    
], fluid=True, style={'backgroundColor': '#f8f9fa'})

# ==================== CALLBACKS ====================

# The following callbacks manage interactive behavior and data flow.
# Data flow summary:
# - `filtered-df-store`: holds the JSON-serialized dataframe currently visible
#    (after date/cluster/search filters). This store preserves `relevance_score`.
# - `current-doc-store`: stores the most recently clicked document (dict with id + email)
# - `selected-emails-store`: stores a list of selected document ids (strings)
# - `gemini-history-store`: keeps last N Gemini queries and responses for UI history

# Each callback has a concise docstring below; comments explain inputs/outputs and
# main side-effects (updates to `app_state` or stores).

# --- Re-cluster callback ---
# Purpose: Re-compute KMeans + PCA when user changes K and clicks Re-Cluster.
# Inputs:
# - `recluster-btn.n_clicks`, `k-slider.value`
# Outputs:
# - `cluster-filter.options`: dropdown options for clusters
# - `cluster-count-display.children`: show new cluster count
@app.callback(
    [Output('cluster-filter', 'options', allow_duplicate=True),
     Output('cluster-count-display', 'children')],
    Input('recluster-btn', 'n_clicks'),
    State('k-slider', 'value'),
    prevent_initial_call=True
)
def recluster(n_clicks, k):
    """Re-cluster with new K"""
    kmeans_new = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans_new.fit_predict(tfidf_matrix)
    
    pca_new = PCA(n_components=2, random_state=42)
    coords = pca_new.fit_transform(tfidf_matrix.toarray())
    df['pca_x'] = coords[:, 0]
    df['pca_y'] = coords[:, 1]
    
    terms = get_cluster_top_terms(kmeans_new, tfidf_vectorizer)
    
    app_state.update({
        'current_kmeans': kmeans_new,
        'current_pca': pca_new,
        'current_clusters': k,
        'cluster_terms': terms
    })
    
    options = [{'label': 'All Clusters', 'value': 'all'}] + \
              [{'label': f"C{i}: {', '.join(terms[i][:3])}", 'value': i} for i in range(k)]
    
    return options, str(k)


# --- Date label update ---
# Purpose: Convert the RangeSlider numeric range into readable date text.
# Input: `date-slider.value` (days offset)
# Output: `date-text.children` (human readable range)
@app.callback(
    Output('date-text', 'children'),
    Input('date-slider', 'value')
)
def update_date_text(val):
    start = min_date + pd.Timedelta(days=val[0])
    end = min_date + pd.Timedelta(days=val[1])
    return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"


# --- Update visualizations & filtered dataframe ---
# Purpose: Central update function that applies date/cluster filters, performs
# ranked TF-IDF search when requested, computes relevance scores, and returns
# updated timeline and scatter figures plus serialized filtered dataframe.
# Inputs:
# - `date-slider.value`, `search-button.n_clicks`, `cluster-filter.value`, `recluster-btn.n_clicks`
# States:
# - `search-box.value`, `topk-slider.value`, `threshold-slider.value`
# Outputs:
# - `timeline-chart.figure`, `cluster-scatter.figure`, `filtered-count.children`,
#   `search-status.children`, `filtered-df-store.data`
@app.callback(
    [Output('timeline-chart', 'figure'),
     Output('cluster-scatter', 'figure'),
     Output('filtered-count', 'children'),
     Output('search-status', 'children'),
     Output('filtered-df-store', 'data')],
    [Input('date-slider', 'value'),
     Input('search-button', 'n_clicks'),
     Input('cluster-filter', 'value'),
     Input('recluster-btn', 'n_clicks')],
    [State('search-box', 'value'),
     State('topk-slider', 'value'),
     State('threshold-slider', 'value')],
    prevent_initial_call=False
)
def update_charts(date_val, search_clicks, cluster_val, recluster_clicks,
                 search_text, topk, threshold):
    """Update all visualizations with ranking"""
    
    # Filter by date
    start_dt = min_date + pd.Timedelta(days=date_val[0])
    end_dt = min_date + pd.Timedelta(days=date_val[1])
    filt_df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
    
    # Filter by cluster
    if cluster_val and cluster_val != 'all':
        filt_df = filt_df[filt_df['cluster'] == int(cluster_val)]
    
    # Ranked search with relevance scores
    relevance_scores = np.zeros(len(filt_df))
    search_msg = None
    
    if search_text and len(search_text.strip()) > 0:
        # Compute TF-IDF relevance
        scores = compute_relevance_scores(
            filt_df['processed_text'].fillna('').tolist(),
            search_text.lower(),
            app_state.get('current_tfidf')
        )
        
        # Apply threshold - ONLY show emails above threshold
        high_rel_indices = np.where(scores >= threshold)[0]
        
        if len(high_rel_indices) > 0:
            # Sort by score descending, take top K
            sorted_indices = high_rel_indices[np.argsort(scores[high_rel_indices])[::-1][:topk]]
            filt_df = filt_df.iloc[sorted_indices].copy()
            relevance_scores = scores[sorted_indices]
            search_msg = dbc.Alert([
                html.Strong(f"✓ Found {len(filt_df)} relevant emails"),
                html.Br(),
                html.Small(f"Threshold: ≥{threshold} | Showing top {min(len(filt_df), topk)} results", 
                          style={'fontSize': '11px'})
            ], color="success", className="mb-0 py-2", dismissable=True)
        else:
            # NO RESULTS - show empty state
            filt_df = pd.DataFrame()  # Empty dataframe
            relevance_scores = np.array([])
            search_msg = dbc.Alert([
                html.Strong("⚠️ No results found"),
                html.Br(),
                html.Small(f"Try: Lower threshold (currently {threshold}) or different keywords", 
                          style={'fontSize': '11px'})
            ], color="warning", className="mb-0 py-2", dismissable=True)
    
    if len(filt_df) > 0:
        filt_df['relevance_score'] = relevance_scores
    
    # Timeline
    fig_timeline = go.Figure()
    if len(filt_df) > 0:
        timeline_df = filt_df.groupby(filt_df['date'].dt.date).size()
        fig_timeline.add_trace(go.Scatter(
            x=timeline_df.index, y=timeline_df.values,
            mode='lines+markers', name='Emails per Day',
            line=dict(color='#0d6efd', width=2),
            marker=dict(size=6)
        ))
    else:
        fig_timeline.add_annotation(
            text="No data to display", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False
        )
    fig_timeline.update_layout(
        height=200, hovermode='x unified',
        xaxis_title='Date', yaxis_title='Count',
        plot_bgcolor='white', margin=dict(l=40, r=20, t=20, b=40)
    )
    
    # Scatter with size = relevance
    fig_scatter = go.Figure()
    
    if len(filt_df) > 0:
        filt_df['cluster_label'] = filt_df['cluster'].apply(
            lambda c: f"C{c}: {', '.join(app_state['cluster_terms'].get(c, ['unknown'])[:2])}"
        )
        
        fig_scatter = px.scatter(
            filt_df, x='pca_x', y='pca_y',
            color='cluster_label',
            size=filt_df['relevance_score'].values * 30 + 3,
            hover_data={'date': '|%Y-%m-%d', 'subject': True, 'from': True,
                       'relevance_score': ':.3f', 'pca_x': False, 'pca_y': False,
                       'cluster_label': False},
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Clusters ({len(filt_df):,} docs)"
        )
        fig_scatter.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='white')))
    else:
        fig_scatter.add_annotation(text="No docs match filters", x=0.5, y=0.5,
                                  xref="paper", yref="paper", showarrow=False)
    
    fig_scatter.update_layout(
        height=500, plot_bgcolor='white',
        xaxis_title='PCA 1', yaxis_title='PCA 2',
        hovermode='closest'
    )
    
    return fig_timeline, fig_scatter, f"{len(filt_df):,}", search_msg, \
           filt_df.to_json(date_format='iso', orient='split')


# --- Toggle/Add Remove selected email ---
# Purpose: Add or remove the currently viewed document (from `current-doc-store`)
# to `selected-emails-store`. Also builds the UI preview list and selected count.
# Inputs:
# - `add-gemini-btn.n_clicks`
# States:
# - `selected-emails-store.data`, `current-doc-store.data`
# Outputs:
# - `selected-emails-store.data`, `selected-emails-display.children`, `selected-count.children`
@app.callback(
    [Output('selected-emails-store', 'data', allow_duplicate=True),
     Output('selected-emails-display', 'children', allow_duplicate=True),
     Output('selected-count', 'children', allow_duplicate=True)],
    Input('add-gemini-btn', 'n_clicks'),
    [State('selected-emails-store', 'data'),
     State('current-doc-store', 'data')],
    prevent_initial_call='initial_duplicate'
)
def select_email(add_clicks, stored_selection, current_doc):
    """Toggle selection using the inline Gemini button"""
    stored_selection = stored_selection or []
    if not current_doc:
        selection = stored_selection
    elif not add_clicks:
        selection = stored_selection
    else:
        doc_id = current_doc.get('id')
        if doc_id in stored_selection:
            selection = [d for d in stored_selection if d != doc_id]
        else:
            selection = stored_selection + [doc_id]
    # Get email details
    selected_emails = []
    for doc_id in selection:
        try:
            pca_x, pca_y = map(float, doc_id.split('_'))
        except Exception:
            continue
        email_row = df[(df['pca_x'] == pca_x) & (df['pca_y'] == pca_y)]
        if len(email_row) > 0:
            selected_emails.append(email_row.iloc[0].to_dict())
    app_state['selected_emails'] = selected_emails
    # Display selected
    display_items = []
    for email in selected_emails:
        display_items.append(
            dbc.ListGroupItem([
                html.Small(pd.to_datetime(email['date']).strftime('%Y-%m-%d'), className="text-muted"),
                html.Br(),
                html.Strong(str(email.get('subject', 'No Subject'))[:40] + ('...' if len(str(email.get('subject', ''))) > 40 else ''),
                           className="small")
            ])
        )
    display_div = dbc.ListGroup(display_items) if display_items else \
                  html.P("No emails selected. Click points.", className="text-muted small")
    return selection, display_div, str(len(selection))


# --- Track most recently clicked document ---
# Purpose: When the user clicks a point in the PCA scatter, store the clicked
# document (id + dict) into `current-doc-store`. Prefer the filtered DF (so
# relevance_score is preserved) and fall back to the global `df` otherwise.
# Inputs: `cluster-scatter.clickData`, `filtered-df-store.data`
# Output: `current-doc-store.data`
@app.callback(
    Output('current-doc-store', 'data'),
    [Input('cluster-scatter', 'clickData'),
     Input('filtered-df-store', 'data')],
    prevent_initial_call=False
)
def set_current_doc(click_data, filtered_df_json):
    """Track the most recently clicked document from filtered dataframe."""
    if not click_data:
        return None
    pca_x, pca_y = click_data['points'][0]['x'], click_data['points'][0]['y']
    
    # Try to get from filtered df first (has relevance scores)
    if filtered_df_json:
        try:
            filt_df = pd.read_json(filtered_df_json, orient='split')
            # Parse dates
            if 'date' in filt_df.columns:
                filt_df['date'] = pd.to_datetime(filt_df['date'])
            
            # Use tolerance for float comparison
            tolerance = 1e-9
            email_match = filt_df[
                (abs(filt_df['pca_x'] - pca_x) < tolerance) & 
                (abs(filt_df['pca_y'] - pca_y) < tolerance)
            ]
            if len(email_match) > 0:
                doc_id = f"{pca_x}_{pca_y}"
                email_dict = email_match.iloc[0].to_dict()
                # Ensure relevance_score is present
                if 'relevance_score' not in email_dict or pd.isna(email_dict.get('relevance_score')):
                    email_dict['relevance_score'] = 0.0
                return {'id': doc_id, 'email': email_dict}
        except Exception as e:
            print(f"⚠️ Error loading filtered df: {e}")
    
    # Fallback to main df (no relevance score)
    email = df[(df['pca_x'] == pca_x) & (df['pca_y'] == pca_y)].iloc[0]
    doc_id = f"{pca_x}_{pca_y}"
    email_dict = email.to_dict()
    email_dict['relevance_score'] = None  # Mark as no search
    return {'id': doc_id, 'email': email_dict}


# --- Show document details in UI ---
# Purpose: Render the full email body and metadata both in the Document tab
# and in the inline viewer under the scatter. Also creates the Add/Remove
# button (id=`add-gemini-btn`) and reflects current selection state.
# Inputs: `current-doc-store.data`, `selected-emails-store.data`
# Outputs: `document-details.children`, `inline-doc-details.children`
@app.callback(
    [Output('document-details', 'children', allow_duplicate=True),
     Output('inline-doc-details', 'children')],
    [Input('current-doc-store', 'data'),
     Input('selected-emails-store', 'data')],
    prevent_initial_call='initial_duplicate'
)
def show_document(current_doc, selected_ids):
    """Display selected document in both tabs and reflect selection state"""
    if not current_doc:
        placeholder = html.P("Select an email to view.", className="text-muted text-center p-5")
        return placeholder, placeholder
    email_dict = current_doc.get('email', {})
    doc_id = current_doc.get('id')
    selected_ids = selected_ids or []
    is_selected = doc_id in selected_ids
    email = pd.Series(email_dict)
    date_val = pd.to_datetime(email.get('date')) if 'date' in email else None
    # Prefer the original raw email body when available (column `raw_body`).
    # Fall back to cleaned_text (readable) or body if raw not present.
    body = email.get('raw_body') or email.get('cleaned_text') or email.get('body', 'No content')
    body = '' if isinstance(body, float) and pd.isna(body) else body
    select_btn = dbc.Button(
        "Add to Gemini Selection" if not is_selected else "Remove from Gemini Selection",
        id="add-gemini-btn",
        color="success" if not is_selected else "secondary",
        size="sm",
        className="mb-2",
        n_clicks=0
    )
    content = [
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small("DATE", className="text-muted d-block"),
                    html.Strong(date_val.strftime('%Y-%m-%d %H:%M') if date_val is not None else 'N/A')
                ], className="mb-2")
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small("RELEVANCE", className="text-muted d-block"),
                    html.Strong(
                        f"{email.get('relevance_score'):.3f}" 
                        if 'relevance_score' in email and email.get('relevance_score') is not None and pd.notna(email.get('relevance_score')) and email.get('relevance_score') > 0 
                        else "N/A (no search)"
                    )
                ], className="mb-2")
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small("FROM", className="text-muted d-block"),
                    html.Strong(email['from'] if pd.notna(email['from']) else 'N/A', className="small")
                ], className="mb-2")
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small("CLUSTER", className="text-muted d-block"),
                    dbc.Badge(f"C{email['cluster']}", color="primary")
                ], className="mb-2")
            ], width=6)
        ]),
        html.Hr(),
        html.Div([
            html.Small("SUBJECT", className="text-muted d-block mb-1"),
            html.H6(email['subject'] if pd.notna(email['subject']) else 'No Subject')
        ], className="mb-3"),
        html.Div([
            html.Small("BODY", className="text-muted d-block mb-1"),
            html.Div(
                body[:2000] + ('...' if len(str(body)) > 2000 else ''),
                className="p-3 bg-light rounded",
                style={'maxHeight': '400px', 'overflowY': 'auto', 'fontSize': '13px', 'lineHeight': '1.6'}
            )
        ]),
        html.Hr(),
        html.Div([
            html.Small("Gemini Selection", className="text-muted d-block mb-1"),
            select_btn
        ], className="mb-2")
    ]
    return content, content


# --- Query Gemini with selected emails as context ---
# Purpose: Send a user question + formatted selected emails to Gemini, receive
# the response, append to history, and update the UI with the response and
# recent query list.
# Inputs: `gemini-button.n_clicks`
# States: `gemini-query.value`, `selected-emails-store.data`, `gemini-history-store.data`
# Outputs: `gemini-response.children`, `gemini-loading.is_open`, `gemini-loading.children`,
# `query-history.children`, `gemini-history-store.data`
@app.callback(
    [Output('gemini-response', 'children'),
     Output('gemini-loading', 'is_open'),
     Output('gemini-loading', 'children'),
     Output('query-history', 'children'),
     Output('gemini-history-store', 'data')],
    Input('gemini-button', 'n_clicks'),
    [State('gemini-query', 'value'),
     State('selected-emails-store', 'data'),
     State('gemini-history-store', 'data')],
    prevent_initial_call=True
)
def query_gemini_callback(n_clicks, query, selected_ids, history):
    """Query Gemini with selected emails as context"""
    
    history = history or []
    
    # Validate inputs
    if not query or not query.strip():
        return [dbc.Alert("Please enter a question.", color="warning")], False, "", \
               html.P("No queries yet.", className="text-muted small"), history
    
    if not selected_ids or len(selected_ids) == 0:
        return [dbc.Alert("Select at least one email first.", color="warning")], False, "", \
               html.P("No queries yet.", className="text-muted small"), history
    
    # Fetch selected emails
    selected_emails = []
    for doc_id in selected_ids:
        pca_x, pca_y = map(float, doc_id.split('_'))
        email = df[(df['pca_x'] == pca_x) & (df['pca_y'] == pca_y)]
        if len(email) > 0:
            selected_emails.append(email.iloc[0].to_dict())
    
    # Format context
    context = format_email_context(selected_emails)
    
    # Query Gemini
    response = query_gemini(query.strip(), context)
    
    # Store in history
    history_item = {
        'query': query.strip(),
        'timestamp': datetime.now().isoformat(),
        'email_count': len(selected_emails),
        'response_preview': response[:100] + '...' if len(response) > 100 else response
    }
    history.append(history_item)
    history = history[-10:]  # Keep last 10
    
    # Format response
    response_display = [
        html.P([html.Strong("Your Question: "), query], className="mb-3"),
        html.Hr(),
        html.Div(
            response,
            className="mb-3",
            style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word'}
        ),
        html.Hr(),
        dbc.Alert([
            html.Small(f"✓ Analysis complete. Based on {len(selected_emails)} email(s).")
        ], color="success", className="mt-2 mb-0")
    ]
    
    # Format history display
    history_display = [
        dbc.ListGroupItem([
            html.Strong(item['query'][:50] + '...', className="small"),
            html.Br(),
            html.Small(f"{item['email_count']} email(s) | {item['timestamp'][:10]}",
                      className="text-muted")
        ])
        for item in reversed(history)
    ] if history else [html.P("No queries yet.", className="text-muted small")]
    
    return response_display, False, "", dbc.ListGroup(history_display), history


# --- Update short context summary for Gemini tab ---
# Purpose: Show a concise summary of the selected documents (count, date range,
# preview subjects) to give the user context before querying Gemini.
# Input: `selected-emails-store.data`
# Outputs: `gemini-context-summary.children`, `context-alert.color`
@app.callback(
    [Output('gemini-context-summary', 'children'),
     Output('context-alert', 'color')],
    Input('selected-emails-store', 'data')
)
def update_context_summary(selected_ids):
    """Update context summary for Gemini tab"""
    
    if not selected_ids:
        return "No emails selected", "secondary"
    selected_count = len(selected_ids)
    date_range = "Loading..."
    subjects_preview = "Loading..."
    if selected_count > 0:
        selected_emails = []
        for doc_id in selected_ids:
            pca_x, pca_y = map(float, doc_id.split('_'))
            email_row = df[(df['pca_x'] == pca_x) & (df['pca_y'] == pca_y)]
            if len(email_row) > 0:
                selected_emails.append(email_row.iloc[0].to_dict())
        if selected_emails:
            dates = [pd.to_datetime(e['date']) for e in selected_emails]
            min_d = min(dates).strftime('%Y-%m-%d')
            max_d = max(dates).strftime('%Y-%m-%d')
            date_range = f"{min_d} to {max_d}"
            subjects = [str(e.get('subject', 'No Subject'))[:30] for e in selected_emails[:3]]
            subjects_preview = ", ".join([f'"{s}"' for s in subjects])
            if len(selected_emails) > 3:
                subjects_preview += f" +{len(selected_emails)-3} more"
    summary_text = f"{selected_count} email(s) ({date_range}) - Topics: {subjects_preview}"
    color = "success" if selected_count > 0 else "secondary"
    return summary_text, color


# --- Clear selection ---
# Purpose: Reset selected emails (clear selection store and UI preview)
# Input: `clear-selection-btn.n_clicks`
# Outputs: `selected-emails-store.data`, `selected-emails-display.children`, `selected-count.children`
@app.callback(
    [Output('selected-emails-store', 'data', allow_duplicate=True),
     Output('selected-emails-display', 'children', allow_duplicate=True),
     Output('selected-count', 'children', allow_duplicate=True)],
    Input('clear-selection-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_selection(n_clicks):
    """Clear selected emails and reset display"""
    app_state['selected_emails'] = []
    return [], html.P("No emails selected. Click points to select.", className="text-muted small"), "0"


# --- Initialize cluster dropdown on app start ---
# Purpose: Populate the cluster dropdown options using `app_state` when the
# app mounts. This uses `cluster-count-display` as a trigger but only runs once
# at initialization due to `prevent_initial_call='initial_duplicate'`.
@app.callback(
    [Output('cluster-filter', 'options'),
     Output('cluster-filter', 'value')],
    Input('cluster-count-display', 'children'),
    prevent_initial_call='initial_duplicate'
)
def init_cluster_dropdown(_):
    k = app_state['current_clusters']
    terms = app_state['cluster_terms']
    options = [{'label': 'All Clusters', 'value': 'all'}] + \
              [{'label': f"C{i}: {', '.join(terms[i][:3])}", 'value': i} for i in range(k)]
    return options, 'all'


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 DASHBOARD WITH GEMINI AI READY")
    print("="*70)
    print(f"📊 Documents: {len(df):,}")
    print(f"🎯 Clusters: {app_state['current_clusters']}")
    print(f"🤖 Gemini Integration: Enabled")
    print("🌐 Open: http://127.0.0.1:8050/")
    print("="*70 + "\n")
    
    app.run(debug=True, port=8050)
