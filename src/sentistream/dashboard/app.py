import asyncio
import contextlib
import json
import logging
import re
import threading
import uuid
from collections import deque

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dcc, html
from redis.asyncio import Redis
from sqlalchemy import desc, select

from sentistream.shared.config import settings
from sentistream.shared.db import AsyncSessionLocal
from sentistream.shared.kafka_client import get_kafka_producer
from sentistream.shared.models import ReviewRecord
from sentistream.shared.schemas import ReviewRaw


# Setup standard formatting logger
logging.basicConfig(level=settings.app.log_level)
logger = logging.getLogger("sentistream.dashboard")

# Global state for the Dash app
MAX_POINTS = 2000
data_points = deque(maxlen=MAX_POINTS)
cluster_names = {}
DEBUG_ENABLED = settings.app.env == "development"

THEME_SETTINGS = {
    "light": {
        "bg": "#f6f2ec",
        "bg_alt": "#e7ecef",
        "panel": "#ffffff",
        "border": "#d2d7de",
        "text": "#141414",
        "muted": "#5f5f5f",
        "accent": "#006d77",
        "accent_soft": "#e0f2f1",
        "plot_template": "plotly_white",
    },
    "dark": {
        "bg": "#0f1318",
        "bg_alt": "#11161d",
        "panel": "#1a222b",
        "border": "#2c3642",
        "text": "#f2f3f5",
        "muted": "#a2a8b0",
        "accent": "#f4a261",
        "accent_soft": "#3a2a1b",
        "plot_template": "plotly_dark",
    },
}

STYLE_CSS = """
body {
    margin: 0;
    font-family: "IBM Plex Sans", sans-serif;
    background: var(--bg);
    color: var(--text);
}
h1, h2, h3, h4, h5, h6 {
    color: var(--text);
}
.app-shell * {
    color: inherit;
}
.app-shell {
    min-height: 100vh;
    background: radial-gradient(circle at 10% 10%, rgba(255, 255, 255, 0.25), transparent 45%),
        radial-gradient(circle at 90% 15%, rgba(0, 109, 119, 0.18), transparent 40%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-alt) 100%);
    padding: 24px;
    transition: background 0.3s ease;
}
.title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 28px;
    letter-spacing: -0.02em;
    margin: 0 0 12px 0;
    color: var(--text);
}
.subtitle {
    color: var(--muted);
    margin-bottom: 20px;
}
.controls-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    background: var(--panel);
    padding: 16px;
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
    animation: fadeIn 0.5s ease;
}
label {
    color: var(--text);
}
.dash-radio-items label,
.dash-checkbox label,
.dash-radio-items span,
.dash-checkbox span {
    color: var(--text);
}
input, select, textarea {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
}
input::placeholder, textarea::placeholder {
    color: var(--muted);
}
.rc-slider {
    color: var(--text);
}
.rc-slider-rail {
    background-color: var(--border);
}
.rc-slider-track {
    background-color: var(--accent);
}
.rc-slider-handle {
    border-color: var(--accent);
}
.panel {
    background: var(--panel);
    padding: 16px;
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.stat-card {
    background: var(--accent_soft);
    padding: 12px 14px;
    border-radius: 12px;
    color: var(--text);
}
.stat-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}
.stat-value {
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
}
.section-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 18px;
    margin: 0 0 8px 0;
    color: var(--text);
}
.anomaly-list li {
    margin-bottom: 6px;
    color: var(--muted);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
"""


# Background task to consume Redis Pub/Sub async stream into standard memory
async def listen_to_redis():
    logger.info("Connecting to Redis Pub/Sub for live visualization...")

    # Load historical points from Postgres on startup
    try:
        async with AsyncSessionLocal() as session:
            stmt = (
                select(ReviewRecord)
                .order_by(desc(ReviewRecord.timestamp))
                .limit(MAX_POINTS)
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
            for r in reversed(records):
                data_points.append(
                    {
                        "id": r.id,
                        "text": r.text,
                        "coords": r.reduced_coords,
                        "cluster_id": r.cluster_id,
                    }
                )
                if r.cluster_name:
                    cluster_names[r.cluster_id] = r.cluster_name
            logger.info(
                f"Successfully pre-loaded {len(records)} recent reviews from PostgreSQL."
            )
    except Exception as e:
        logger.error(f"Failed to fetch historical Data from DB: {e}")

    try:
        redis_client = Redis.from_url(
            settings.database.redis_url, decode_responses=True
        )
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("dash_stream", "dash_names_update")

        logger.info("Successfully subscribed to real-time streams.")
        async for message in pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"]
                data = json.loads(message["data"])

                if channel == "dash_stream":
                    data_points.append(data)
                elif channel == "dash_names_update":
                    cluster_names[data["cluster_id"]] = data["cluster_name"]

    except Exception as e:
        logger.error(f"Redis listener failed: {e}")


def start_redis_listener():
    """Runs the asyncio Redis listener in a dedicated background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen_to_redis())


# Start the background listener before launching the Dash HTTP server
listener_thread = threading.Thread(target=start_redis_listener, daemon=True)
listener_thread.start()


async def _publish_debug_review(review_id: str, text: str, metadata: dict | None):
    topic = settings.kafka.topics.get("reviews_raw", "reviews_raw")
    payload = ReviewRaw(id=review_id, text=text, metadata=metadata)
    producer = await get_kafka_producer()
    try:
        await producer.send(
            topic,
            value=payload.model_dump(mode="json"),
            key=review_id.encode("utf-8"),
        )
    finally:
        await producer.stop()


def publish_debug_review(text: str, metadata: dict | None) -> str:
    if not DEBUG_ENABLED:
        raise RuntimeError("Debug publishing is disabled outside development mode.")

    review_id = str(uuid.uuid4())
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_publish_debug_review(review_id, text, metadata))
    finally:
        loop.close()
    return review_id


# Initialize the Plotly Dash Application
external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap"
]
app = dash.Dash(
    __name__,
    title="SentiStream Dashboard",
    external_stylesheets=external_stylesheets,
)
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{STYLE_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

# Layout configuration
app.layout = html.Div(
    id="app-shell",
    className="app-shell",
    children=[
        html.Div(
            [
                html.H1("SentiStream", className="title"),
                html.Div(
                    "Real-time review clustering and trend discovery.",
                    className="subtitle",
                ),
            ]
        ),
        html.Div(
            className="controls-bar",
            children=[
                html.Div(
                    [
                        html.Div("Theme", className="stat-label"),
                        dcc.RadioItems(
                            id="theme-mode",
                            options=[
                                {"label": "Light", "value": "light"},
                                {"label": "Dark", "value": "dark"},
                            ],
                            value="light",
                            inline=True,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div("Search", className="stat-label"),
                        dcc.Input(
                            id="search-query",
                            type="text",
                            placeholder="Search reviews or keywords",
                            debounce=True,
                            style={"width": "100%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div("Noise", className="stat-label"),
                        dcc.Checklist(
                            id="show-noise",
                            options=[
                                {
                                    "label": "Include noise / anomalies",
                                    "value": "include",
                                }
                            ],
                            value=["include"],
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div("Items per cluster", className="stat-label"),
                        dcc.Slider(
                            id="max-items",
                            min=3,
                            max=25,
                            step=1,
                            value=10,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
            ],
        ),
        # Interval component controls how often the page queries the `data_points` deque
        dcc.Interval(
            id="interval-component",
            interval=2000,  # in milliseconds (update every 2 seconds)
            n_intervals=0,
        ),
        html.Div(
            className="stats-grid",
            children=[
                html.Div(
                    [
                        html.Div("Total reviews", className="stat-label"),
                        html.Div("0", id="stats-total", className="stat-value"),
                    ],
                    className="stat-card",
                ),
                html.Div(
                    [
                        html.Div("Filtered", className="stat-label"),
                        html.Div("0", id="stats-filtered", className="stat-value"),
                    ],
                    className="stat-card",
                ),
                html.Div(
                    [
                        html.Div("Active clusters", className="stat-label"),
                        html.Div("0", id="stats-clusters", className="stat-value"),
                    ],
                    className="stat-card",
                ),
                html.Div(
                    [
                        html.Div("Noise points", className="stat-label"),
                        html.Div("0", id="stats-noise", className="stat-value"),
                    ],
                    className="stat-card",
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "flexDirection": "row", "gap": "20px"},
            children=[
                # Main Scatter Plot Canvas
                html.Div(
                    style={"flex": "3"},
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.Div(
                                    "Live UMAP Projection", className="section-title"
                                ),
                                dcc.Graph(id="live-scatter", style={"height": "70vh"}),
                            ],
                        )
                    ],
                ),
                # Live Metrics / Feed Sidebar
                html.Div(
                    style={"flex": "1", "maxHeight": "75vh", "overflowY": "auto"},
                    children=[
                        html.Div(
                            className="panel",
                            style={"marginBottom": "16px"},
                            children=[
                                html.Div("Debug Publisher", className="section-title"),
                                html.Div(
                                    "Enabled in development mode only.",
                                    className="subtitle",
                                    style={"marginBottom": "10px"},
                                ),
                                dcc.Textarea(
                                    id="debug-text",
                                    placeholder="Sample review text",
                                    style={"width": "100%", "height": "80px"},
                                ),
                                dcc.Input(
                                    id="debug-metadata",
                                    type="text",
                                    placeholder='Metadata JSON (optional): {"source":"demo"}',
                                    style={"width": "100%", "marginTop": "10px"},
                                ),
                                html.Button(
                                    "Send to Kafka",
                                    id="debug-send",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "10px",
                                        "background": "var(--accent)",
                                        "color": "#fff",
                                        "border": "none",
                                        "padding": "8px 12px",
                                        "borderRadius": "8px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Div(
                                    id="debug-status",
                                    style={"marginTop": "8px", "fontSize": "0.85em"},
                                ),
                            ],
                        )
                        if DEBUG_ENABLED
                        else html.Div(
                            style={"display": "none"},
                            children=[
                                dcc.Textarea(id="debug-text"),
                                dcc.Input(id="debug-metadata"),
                                html.Button(id="debug-send"),
                                html.Div(id="debug-status"),
                            ],
                        ),
                        html.Div(
                            className="panel",
                            children=[
                                html.Div("Anomalies", className="section-title"),
                                html.Ul(id="anomaly-list", className="anomaly-list"),
                            ],
                        ),
                        html.Div(
                            className="panel",
                            style={"marginTop": "16px"},
                            children=[
                                html.Div("Cluster Feed", className="section-title"),
                                html.Div(id="live-feed"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    [
        Output("live-scatter", "figure"),
        Output("live-feed", "children"),
        Output("stats-total", "children"),
        Output("stats-filtered", "children"),
        Output("stats-clusters", "children"),
        Output("stats-noise", "children"),
        Output("anomaly-list", "children"),
        Output("app-shell", "style"),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("theme-mode", "value"),
        Input("search-query", "value"),
        Input("show-noise", "value"),
        Input("max-items", "value"),
    ],
)
def update_graph_and_feed(n, theme_mode, query, show_noise, max_items):
    """
    Called every 2 seconds. Pulls the latest records from the globally shared
    deque and redraws the Plotly map and sidebar.
    """
    theme = THEME_SETTINGS.get(theme_mode, THEME_SETTINGS["light"])
    shell_style = {
        "--bg": theme["bg"],
        "--bg-alt": theme["bg_alt"],
        "--panel": theme["panel"],
        "--border": theme["border"],
        "--text": theme["text"],
        "--muted": theme["muted"],
        "--accent": theme["accent"],
        "--accent_soft": theme["accent_soft"],
    }

    if not data_points:
        empty_fig = px.scatter(title="Waiting for incoming text streams...")
        empty_fig.update_layout(
            template=theme["plot_template"],
            paper_bgcolor=theme["panel"],
            plot_bgcolor=theme["panel"],
            font={"color": theme["text"]},
        )
        return (
            empty_fig,
            html.P("No data yet."),
            "0",
            "0",
            "0",
            "0",
            [html.Li("No anomalies yet.")],
            shell_style,
        )

    # Convert the deque of dicts into a pandas DataFrame for fast vector plotting
    df = pd.DataFrame(list(data_points))
    total_points = len(df)

    # Extract the first two dimensions of the 5-dim UMAP embedding for 2D plotting
    df["dim_x"] = df["coords"].apply(lambda c: c[0])
    df["dim_y"] = df["coords"].apply(lambda c: c[1])

    # Map raw cluster IDs to LLM cluster names dynamically
    df["Topic"] = df["cluster_id"].apply(
        lambda cid: cluster_names.get(cid, f"Cluster {cid} (Evaluating...)")
    )

    # Filter based on UI controls
    include_noise = "include" in (show_noise or [])
    if not include_noise:
        df = df[df["cluster_id"] != -1]

    if query:
        safe_query = str(query).strip()
        if safe_query:
            with contextlib.suppress(re.error):
                df = df[
                    df["text"].str.contains(
                        re.escape(safe_query), case=False, na=False, regex=True
                    )
                ]

    filtered_points = len(df)

    # Hover text wrapping for long reviews
    df["hover_text"] = df["text"].apply(
        lambda t: t[:100] + "..." if len(t) > 100 else t
    )

    # Re-render Scatter
    fig = px.scatter(
        df,
        x="dim_x",
        y="dim_y",
        color="Topic",
        hover_data={"dim_x": False, "dim_y": False, "hover_text": True, "Topic": True},
        title=f"Live Micro-Clusters (Showing last {len(df)} records)",
        template=theme["plot_template"],
    )

    # Prevent auto-rescale on update, preserve user zoom/selection
    fig.update_layout(
        transition_duration=0,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=theme["panel"],
        plot_bgcolor=theme["panel"],
        font={"color": theme["text"]},
        uirevision="sentistream-persist",  # This preserves zoom/selection
        xaxis_autorange=False,
        yaxis_autorange=False,
    )

    # Re-render Sidebar Stats
    topic_counts = df["Topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]

    feed_html = []
    for _, row in topic_counts.iterrows():
        topic = row["Topic"]
        count = row["Count"]

        # Get the most recent reviews for this topic
        topic_reviews = df[df["Topic"] == topic].tail(max_items or 10)

        review_items = [
            html.Li(
                r["text"],
                style={
                    "fontSize": "0.85em",
                    "marginBottom": "8px",
                    "color": theme["text"],
                    "lineHeight": "1.3",
                },
            )
            for _, r in topic_reviews.iterrows()
        ]

        feed_html.append(
            html.Details(
                style={
                    "border": f"1px solid {theme['border']}",
                    "padding": "10px",
                    "marginBottom": "8px",
                    "borderRadius": "6px",
                    "backgroundColor": theme["panel"],
                },
                children=[
                    html.Summary(
                        style={
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "outline": "none",
                            "color": theme["text"],
                        },
                        children=[
                            f"{topic} ",
                            html.Span(
                                f"({count} records)",
                                style={
                                    "color": theme["muted"],
                                    "fontWeight": "normal",
                                    "fontSize": "0.9em",
                                },
                            ),
                        ],
                    ),
                    html.Ul(
                        style={
                            "paddingLeft": "20px",
                            "marginTop": "10px",
                            "marginBottom": "0",
                        },
                        children=review_items,
                    ),
                ],
            )
        )

    active_clusters = max(topic_counts.shape[0], 0)
    noise_count = int((df["cluster_id"] == -1).sum()) if not df.empty else 0
    anomalies = df[df["cluster_id"] == -1].tail(10)
    anomaly_items = (
        [
            html.Li(
                row["text"],
                style={
                    "fontSize": "0.85em",
                    "marginBottom": "6px",
                    "lineHeight": "1.3",
                    "color": theme["muted"],
                },
            )
            for _, row in anomalies.iterrows()
        ]
        if not anomalies.empty
        else [html.Li("No anomalies in the current window.")]
    )

    return (
        fig,
        feed_html,
        str(total_points),
        str(filtered_points),
        str(active_clusters),
        str(noise_count),
        anomaly_items,
        shell_style,
    )


@app.callback(
    Output("debug-status", "children"),
    Input("debug-send", "n_clicks"),
    State("debug-text", "value"),
    State("debug-metadata", "value"),
    prevent_initial_call=True,
)
def send_debug_message(n_clicks, text, metadata_raw):
    if not DEBUG_ENABLED:
        return "Debug publishing is disabled."

    if not text or not str(text).strip():
        return "Enter review text before sending."

    metadata = None
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            return "Metadata must be valid JSON."

    try:
        review_id = publish_debug_review(str(text).strip(), metadata)
        return f"Sent review {review_id}."
    except Exception as e:
        logger.error(f"Debug publish failed: {e}")
        return "Failed to send review to Kafka."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True, use_reloader=False)
