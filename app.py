"""
Wildfire Risk Check
===================
A one-page Streamlit web app that asks for the weather of one day in California,
one question at a time, and tells you how likely it is that a wildfire STARTS
on that day.

School project - SKEMA Global BBA, "AI for Business & Society".

How the app is organised (top to bottom):
  1. Settings and constants (file name, features, colours, questions)
  2. The model: loaded and trained ONCE thanks to @st.cache_resource
  3. Helper functions: prediction, explanation of the prediction, formatting
  4. inject_css(theme): ALL the visual design lives in this one function
  5. One function per screen (welcome, date, weather question, recap, result)
  6. main(): reads st.session_state["step"] and shows the right screen
"""

import calendar
import datetime as dt
import math
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# 1. SETTINGS AND CONSTANTS
# ---------------------------------------------------------------------------

# The data file sits next to app.py. We accept both possible file names.
HERE = os.path.dirname(os.path.abspath(__file__))
FILE_CANDIDATES = ["Data.xlsx", "Copy_of_Data_set_wild_fire__v2.xlsx"]
SHEET = "CA_Weather_Fire_Dataset_1984-20"

# The 10 inputs of the model. YEAR is deliberately NOT here: in our first model
# it was important for the wrong reasons (missing fire labels after 2023 and a
# rising trend over the decades), so we removed it.
FEATURES = ['PRECIPITATION', 'MAX_TEMP', 'MIN_TEMP', 'AVG_WIND_SPEED', 'TEMP_RANGE', 'WIND_TEMP_RATIO',
            'MONTH', 'LAGGED_PRECIPITATION', 'LAGGED_AVG_WIND_SPEED', 'DAY_OF_YEAR']

# At 30% or more we raise an alert (not 50%): missing a fire costs far more
# than a false alarm.
ALERT_THRESHOLD = 0.30

# Presentation colours
NAVY = "#14283b"
ORANGE = "#f05a28"
OFF_WHITE = "#f6f3ed"

# The four risk levels: (upper limit, name, gradient start, gradient end, message)
LEVELS = [
    (0.30, "LOW", "#0f5132", "#198754", "No alert. Normal readiness."),
    (0.50, "ELEVATED – ALERT", "#b45309", "#f59e0b", "Alert: put crews on standby."),
    (0.70, "HIGH", "#c2410c", "#f97316", "High risk: pre-position engines."),
    (1.01, "EXTREME", "#7f1d1d", "#dc2626", "Extreme risk: full mobilisation, warn the public."),
]

# Screen numbers kept in st.session_state["step"]
STEP_WELCOME = 0
STEP_DATE = 1          # screens 1..7 are the questions
STEP_RECAP = 8
STEP_LOADING = 9
STEP_RESULT = 10
TOTAL_QUESTIONS = 7

# The six weather questions (screens 2 to 7), in order.
# "conv" says which live conversion to show under the field.
QUESTIONS = [
    dict(key="MAX_TEMP", short="Max temperature", unit="°F", conv="temp", step=1.0, fmt="%.1f",
         title="What was the maximum temperature of the day?",
         helper="The hottest moment of the day, in degrees Fahrenheit.",
         typical="Typical day: 56–91 °F · seen in the data: 50–106 °F"),
    dict(key="MIN_TEMP", short="Night temperature", unit="°F", conv="temp", step=1.0, fmt="%.1f",
         title="How cold did it get during the night?",
         helper="The minimum (night) temperature, in °F. It cannot be higher than the maximum.",
         typical="Typical night: 40–70 °F · seen in the data: 33–77 °F"),
    dict(key="AVG_WIND_SPEED", short="Wind speed", unit="mph", conv="wind", step=0.5, fmt="%.1f",
         title="How windy was it, on average?",
         helper="The average wind speed over the whole day, in miles per hour.",
         typical="Median day: 7 mph · seen in the data: 1.8–26 mph"),
    dict(key="PRECIPITATION", short="Rain that day", unit="in", conv="rain", step=0.05, fmt="%.2f",
         title="How much rain fell on that day?",
         helper="Total rain of the day, in inches. Type 0 if it stayed dry.",
         typical="Most days: 0 in · seen in the data: 0–4.5 in"),
    dict(key="LAGGED_PRECIPITATION", short="Rain, previous 7 days", unit="in", conv="rain", step=0.05, fmt="%.2f",
         title="How much rain fell in TOTAL over the previous 7 days?",
         helper="Add up the rain of the 7 days before this one (a 7-day total), in inches.",
         typical="Median: 0 in · seen in the data: 0–8 in"),
    dict(key="LAGGED_AVG_WIND_SPEED", short="Wind, previous 7 days", unit="mph", conv="wind", step=0.5, fmt="%.1f",
         title="What was the AVERAGE wind over the previous 7 days?",
         helper="The mean of the daily wind speeds of the 7 days before this one, in mph.",
         typical="Median: 7.5 mph · seen in the data: 3.2–14 mph"),
]

# Sanity limits: values outside these are surely typing mistakes, so we refuse them.
# (Values inside these limits but outside the data range are accepted with a warning.)
SANITY = {
    "MAX_TEMP": (1, 140), "MIN_TEMP": (-60, 140), "AVG_WIND_SPEED": (0, 150),
    "PRECIPITATION": (0, 30), "LAGGED_PRECIPITATION": (0, 80), "LAGGED_AVG_WIND_SPEED": (0, 150),
}


# ---------------------------------------------------------------------------
# 2. THE MODEL (trained only once, then kept in memory)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training the Random Forest on 40 years of data…")
def load_model():
    """Read the data, train the Random Forest exactly like in our presentation,
    and remember the min/max of every column (for the out-of-range warning)."""
    file = next((os.path.join(HERE, f) for f in FILE_CANDIDATES if os.path.exists(os.path.join(HERE, f))), None)
    if file is None:
        raise FileNotFoundError("The data file (Data.xlsx) was not found next to app.py.")

    df = pd.read_excel(file, sheet_name='CA_Weather_Fire_Dataset_1984-20').dropna()
    df = df[df['DATE'] <= '2023-12-22']          # after this date fires are not labelled -> leakage
    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    model.fit(df[FEATURES], df['FIRE_START_DAY'].astype(int))

    ranges = {col: (float(df[col].min()), float(df[col].max())) for col in FEATURES}
    return model, ranges, len(df)


# ---------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def build_features(answers):
    """Turn the user's answers into the 10 model inputs (one-row table)."""
    day = answers["DATE"]
    row = {
        "PRECIPITATION": answers["PRECIPITATION"],
        "MAX_TEMP": answers["MAX_TEMP"],
        "MIN_TEMP": answers["MIN_TEMP"],
        "AVG_WIND_SPEED": answers["AVG_WIND_SPEED"],
        "TEMP_RANGE": answers["MAX_TEMP"] - answers["MIN_TEMP"],             # max - min
        "WIND_TEMP_RATIO": answers["AVG_WIND_SPEED"] / answers["MAX_TEMP"],  # wind / max temp
        "MONTH": day.month,
        "LAGGED_PRECIPITATION": answers["LAGGED_PRECIPITATION"],    # 7-day TOTAL rain
        "LAGGED_AVG_WIND_SPEED": answers["LAGGED_AVG_WIND_SPEED"],  # 7-day AVERAGE wind
        "DAY_OF_YEAR": day.timetuple().tm_yday,
    }
    return pd.DataFrame([row], columns=FEATURES)


def explain_prediction(model, X):
    """'What drove this prediction' with the tree-path method.

    For each tree we follow the path of this day from the root to its leaf.
    At every split we look at the share of fire days in the child node minus
    the share in the parent node, and give that change to the feature used
    by the split. We then average over the 300 trees.
    Result: probability = baseline (average of the roots) + sum of contributions.
    """
    x = X.to_numpy(dtype=np.float32)
    contributions = np.zeros(len(FEATURES))
    baseline = 0.0
    for tree_model in model.estimators_:
        tree = tree_model.tree_
        counts = tree.value[:, 0, :]                          # class weights in each node
        fire_share = counts[:, 1] / counts.sum(axis=1)        # proportion of fire days per node
        path = tree_model.decision_path(x)                    # nodes visited by this day
        nodes = np.sort(path.indices[path.indptr[0]:path.indptr[1]])
        for parent, child in zip(nodes[:-1], nodes[1:]):
            contributions[tree.feature[parent]] += fire_share[child] - fire_share[parent]
        baseline += fire_share[0]
    contributions /= len(model.estimators_)
    baseline /= len(model.estimators_)

    probability = model.predict_proba(X)[0, 1]
    # Safety check: the pieces must add up exactly to the prediction.
    assert abs(baseline + contributions.sum() - probability) < 1e-6, "Contributions do not add up!"
    return probability, baseline, dict(zip(FEATURES, contributions))


def risk_level(p):
    """Return (name, colour1, colour2, message) for a probability p."""
    for limit, name, c1, c2, msg in LEVELS:
        if p < limit:
            return name, c1, c2, msg
    return LEVELS[-1][1:]


def nice(v):
    """Print a number without useless zeros: 95.0 -> '95', 0.30 -> '0.3'."""
    return f"{round(v, 2):g}"


def conversion_text(kind, v):
    """Live conversion shown under the field."""
    if v is None:
        return "&nbsp;"
    if kind == "temp":
        return f"= {(v - 32) * 5 / 9:.1f} °C"
    if kind == "wind":
        return f"= {v * 1.609344:.1f} km/h"
    return f"= {v * 25.4:.1f} mm"


def validate_number(key, v, answers):
    """Return an error message, or None if the answer is fine."""
    if v is None:
        return "Please type a number to continue."
    low, high = SANITY[key]
    if key in ("PRECIPITATION", "LAGGED_PRECIPITATION") and v < 0:
        return "Rain cannot be negative. Type 0 if it did not rain."
    if key in ("AVG_WIND_SPEED", "LAGGED_AVG_WIND_SPEED") and v < 0:
        return "Wind speed cannot be negative."
    if key == "MAX_TEMP" and v < low:
        return "The model needs a maximum temperature above 0 °F."
    if not (low <= v <= high):
        return f"{nice(v)} looks like a typing mistake. Please check the value (expected between {low} and {high})."
    if key == "MIN_TEMP" and answers.get("MAX_TEMP") is not None and v > answers["MAX_TEMP"]:
        return f"The night temperature cannot be higher than the day's maximum ({nice(answers['MAX_TEMP'])} °F)."
    if key == "MAX_TEMP" and answers.get("MIN_TEMP") is not None and v < answers["MIN_TEMP"]:
        return f"The maximum cannot be lower than the night temperature you entered ({nice(answers['MIN_TEMP'])} °F)."
    return None


def parse_date(dd, mm, yyyy):
    """Check the typed date. Return (date, None) or (None, friendly error)."""
    dd, mm, yyyy = (s.strip() for s in (dd or "", mm or "", yyyy or ""))
    if not (dd and mm and yyyy):
        return None, "Please fill in the day, the month and the year."
    if not (dd.isdigit() and mm.isdigit() and yyyy.isdigit()):
        return None, "Please use numbers only, for example 15 / 08 / 2024."
    d, m, y = int(dd), int(mm), int(yyyy)
    if len(yyyy) != 4 or y < 1:
        return None, "Please type the year with 4 digits, for example 2024."
    if not 1 <= m <= 12:
        return None, "The month must be between 01 and 12."
    days_in_month = calendar.monthrange(y, m)[1]
    if not 1 <= d <= days_in_month:
        month_name = calendar.month_name[m]
        return None, f"This date does not exist: {month_name} {y} has only {days_in_month} days."
    return dt.date(y, m, d), None


# ---------------------------------------------------------------------------
# 4. DESIGN: all the CSS in one function
# ---------------------------------------------------------------------------

def inject_css(theme, step=0):
    """Inject the whole design. `theme` is "light" for the welcome / question
    screens, or a (colour1, colour2) pair for the result page, which turns the
    background of the ENTIRE page into the colour of the risk.
    `step` is only used to restart the fade-in animation on every new screen."""
    if theme == "light":
        background = OFF_WHITE
        text_on_bg = NAVY
    else:
        c1, c2 = theme
        background = f"linear-gradient(160deg, {c1} 0%, {c2} 100%) fixed"
        text_on_bg = "#ffffff"
    anim = f"fadein{step}"   # a new animation name restarts the animation

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ---- hide the default Streamlit chrome ---- */
#MainMenu, footer, header {{visibility: hidden;}}
header[data-testid="stHeader"] {{height: 0; background: transparent;}}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{display: none;}}

/* ---- page background and fonts ---- */
.stApp {{background: {background}; color: {text_on_bg}; font-family: 'Inter', sans-serif;}}
html, body, p, div, label, input, button {{font-family: 'Inter', sans-serif;}}
h1, h2, h3, .wr-title {{font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em;}}
.block-container, [data-testid="stMainBlockContainer"] {{max-width: 680px; padding: 2.5rem 1rem 3rem;}}

/* ---- fade-in on every screen ---- */
@keyframes {anim} {{ from {{opacity: 0; transform: translateY(10px);}} to {{opacity: 1; transform: none;}} }}
.wr-anim, [class*="st-key-card"] {{animation: {anim} .5s ease-out both;}}

/* ---- white rounded cards ---- */
[class*="st-key-card"] {{
  background: #ffffff; color: {NAVY}; border-radius: 24px; padding: 2.4rem 2.2rem 2rem;
  box-shadow: 0 1px 2px rgba(20,40,59,.05), 0 18px 50px -12px rgba(20,40,59,.18);
}}
[class*="st-key-card"] p, [class*="st-key-card"] label {{color: {NAVY};}}

/* ---- progress bar ---- */
.wr-progress-top {{display: flex; justify-content: space-between; font-size: .8rem; font-weight: 600;
  color: #6b7a88; text-transform: uppercase; letter-spacing: .06em; margin-bottom: .5rem;}}
.wr-track {{height: 4px; background: #efe9df; border-radius: 99px; overflow: hidden; margin-bottom: 2rem;}}
.wr-fill {{height: 100%; background: {ORANGE}; border-radius: 99px; transition: width .4s ease;}}

/* ---- questions ---- */
.wr-question {{font-family: 'Space Grotesk', sans-serif; font-size: 2rem; line-height: 1.15; font-weight: 700;
  color: {NAVY}; margin: 0 0 .6rem; letter-spacing: -0.02em;}}
.wr-helper {{color: #55667a; font-size: 1rem; margin: 0 0 1.6rem;}}
.wr-typical {{display: inline-block; background: #fdf0ea; color: #b8431c; font-size: .85rem; font-weight: 600;
  padding: .35rem .75rem; border-radius: 99px; margin: .2rem 0 .2rem;}}
.wr-conv {{color: #6b7a88; font-size: 1rem; font-weight: 500; margin: .1rem 0 1.4rem .2rem; min-height: 1.4rem;}}
.wr-note {{color: #6b7a88; font-size: .88rem; margin: .4rem 0 1.4rem;}}
.wr-unit {{font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 600; color: #8a97a5;
  line-height: 70px;}}
.wr-small-label {{font-size: .78rem; font-weight: 600; color: #6b7a88; text-transform: uppercase;
  letter-spacing: .06em; margin-bottom: .2rem;}}

/* ---- big inputs ---- */
.stNumberInput input, .stTextInput input {{font-size: 28px !important; font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600; color: {NAVY}; height: 64px; padding-left: 1rem;}}
.stNumberInput [data-baseweb="input"], .stTextInput [data-baseweb="input"], [data-testid="stNumberInputContainer"],
[data-testid="stTextInputRootElement"] {{border-radius: 14px !important; border: 2px solid #e6e0d6 !important;
  background: #fbfaf7 !important; height: auto !important; min-height: 66px;}}
.stNumberInput [data-baseweb="input"]:focus-within, .stTextInput [data-baseweb="input"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within {{border-color: {ORANGE} !important;}}
.stNumberInput input, .stTextInput input {{background: transparent !important;}}
[data-testid="stHeaderActionElements"] {{display: none;}}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"], [data-testid="InputInstructions"] {{display: none;}}
.stTextInput input {{text-align: center; padding-left: 0;}}

/* ---- buttons: full width, rounded ---- */
.stButton button {{width: 100%; min-height: 3.4rem; border-radius: 999px; font-weight: 600; font-size: 1.05rem;
  transition: transform .15s ease, box-shadow .15s ease, background .15s ease;}}
.stButton button p {{font-size: 1.05rem; font-weight: 600;}}
.stButton button[kind="primary"], [data-testid="stBaseButton-primary"] {{background: {ORANGE}; border: none; color: #fff;
  box-shadow: 0 10px 24px -10px rgba(240,90,40,.8);}}
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{background: #d94a1b; transform: translateY(-1px);}}
.stButton button[kind="secondary"], [data-testid="stBaseButton-secondary"] {{background: transparent; color: {NAVY};
  border: 2px solid #d9d2c6;}}
.stButton button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {{border-color: {NAVY}; color: {NAVY};}}
.stButton button[kind="tertiary"], [data-testid="stBaseButton-tertiary"] {{min-height: 2.2rem; color: {ORANGE}; width: 100%;
  padding: 0; justify-content: flex-end; box-shadow: none;}}
.stButton button[kind="tertiary"] p, [class*="st-key-card"] .stButton button[kind="tertiary"] p {{font-size: .95rem; color: {ORANGE};
  text-decoration: underline; text-underline-offset: 3px;}}
.stButton button[kind="primary"] p, [class*="st-key-card"] .stButton button[kind="primary"] p {{color: #fff;}}
.stButton button[kind="secondary"] p, [class*="st-key-card"] .stButton button[kind="secondary"] p {{color: {NAVY};}}

/* ---- keep columns side by side, even on phones ---- */
[data-testid="stHorizontalBlock"] {{flex-wrap: nowrap !important; gap: .75rem;}}
[data-testid="stColumn"], [data-testid="column"] {{min-width: 0 !important;}}

/* ---- errors ---- */
[data-testid="stAlert"] {{border-radius: 14px;}}

/* ---- welcome (hero) ---- */
.wr-hero {{text-align: center; padding: 3.5rem 0 1.5rem;}}
.wr-flame {{width: 84px; height: 84px; margin: 0 auto 1.6rem; border-radius: 26px; display: grid; place-items: center;
  font-size: 44px; background: linear-gradient(145deg, #ff8a4c, {ORANGE}); box-shadow: 0 18px 40px -14px rgba(240,90,40,.9);}}
.wr-eyebrow {{font-size: .8rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: {ORANGE};
  margin-bottom: 1rem;}}
.wr-h1 {{font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.03em; font-size: 4.2rem; line-height: 1; margin: 0 0 1.2rem; color: {NAVY}; font-weight: 700;}}
.wr-h1 span {{color: {ORANGE};}}
.wr-sub {{font-size: 1.15rem; color: #4d5e70; max-width: 470px; margin: 0 auto 2.2rem; line-height: 1.5;}}
.wr-stats {{display: flex; justify-content: center; gap: .6rem; flex-wrap: wrap; margin: 2.2rem 0 1rem;}}
.wr-stat {{background: #fff; border-radius: 16px; padding: .8rem 1.1rem; min-width: 0; flex: 0 1 140px; text-align: left;
  box-shadow: 0 6px 20px -10px rgba(20,40,59,.25);}}
.wr-stat b {{display: block; font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; color: {NAVY};}}
.wr-stat span {{font-size: .78rem; color: #6b7a88;}}

/* ---- recap ---- */
[class*="st-key-recaprow"] {{border-bottom: 1px solid #f0ebe3; padding: .35rem 0; gap: 0;}}
[class*="st-key-recaprow"] [data-testid="stMarkdownContainer"] {{margin: 0;}}
[class*="st-key-recaprow"] .stElementContainer, [class*="st-key-recaprow"] [data-testid="stElementContainer"] {{margin: 0;}}
.wr-rlabel {{color: #6b7a88; font-size: .95rem;}}
.wr-rvalue {{color: {NAVY}; font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem;}}

/* ---- loading ---- */
.wr-loading {{text-align: center; padding: 7rem 0;}}
@keyframes wrpulse {{0%,100% {{transform: scale(1); opacity: .9;}} 50% {{transform: scale(1.18); opacity: 1;}}}}
@keyframes wrbar {{from {{width: 0;}} to {{width: 100%;}}}}
.wr-loading .wr-flame {{animation: wrpulse .8s ease-in-out infinite;}}
.wr-loading .wr-track {{max-width: 260px; margin: 1.5rem auto 0;}}
.wr-loading .wr-fill {{animation: wrbar 1s ease-out forwards;}}

/* ---- result page ---- */
.wr-result {{text-align: center; color: #fff; padding: 1.5rem 0 1.8rem;}}
.wr-kicker {{font-size: .85rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; opacity: .85;}}
.wr-pct {{font-family: 'Space Grotesk', sans-serif; font-size: 140px; font-weight: 700; line-height: 1;
  margin: .6rem 0 .8rem; letter-spacing: -0.05em; text-shadow: 0 10px 40px rgba(0,0,0,.18);}}
.wr-pct span {{font-size: 64px; margin-left: 4px; opacity: .9;}}
.wr-badge {{display: inline-block; padding: .45rem 1.1rem; border-radius: 99px; background: rgba(255,255,255,.18);
  border: 1px solid rgba(255,255,255,.45); font-weight: 700; letter-spacing: .1em; font-size: .9rem;}}
.wr-msg {{font-size: 1.3rem; font-weight: 600; margin: 1.1rem auto 0; max-width: 460px; color: #fff;}}
.wr-card-title {{font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; color: {NAVY}; margin: 0 0 .3rem;}}
.wr-card-sub {{color: #6b7a88; font-size: .92rem; margin: 0 0 1.2rem;}}
[class*="st-key-card"] {{margin-bottom: .4rem;}}
.wr-drow {{margin: 0 0 1rem;}}
.wr-dtop {{display: flex; justify-content: space-between; gap: .8rem; font-size: .95rem; color: {NAVY}; margin-bottom: .35rem;}}
.wr-dtop b {{white-space: nowrap;}}
.wr-dtrack {{position: relative; height: 12px; background: #f3efe8; border-radius: 99px;}}
.wr-dtrack::after {{content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px; background: #c9c1b4;}}
.wr-dbar {{position: absolute; top: 0; bottom: 0; border-radius: 99px;}}
.wr-dfoot {{display: flex; justify-content: space-between; font-size: .78rem; color: #8a97a5; margin-top: .2rem;}}
.wr-sum {{background: #faf8f4; border-radius: 14px; padding: .8rem 1rem; font-size: .9rem; color: #4d5e70; margin-top: .6rem;}}
.wr-warn {{background: #fef3c7; border-left: 5px solid #f59e0b; color: #78350f; border-radius: 16px;
  padding: 1rem 1.2rem; margin: 0 0 1rem; font-size: .95rem; line-height: 1.5;}}
.wr-info {{background: rgba(255,255,255,.95); border-left: 5px solid {ORANGE}; color: {NAVY}; border-radius: 16px;
  padding: 1rem 1.2rem; margin: 0 0 1rem; font-size: .95rem; line-height: 1.5;}}
.wr-chips {{display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .55rem;}}
.wr-chip {{background: #faf8f4; border-radius: 12px; padding: .55rem .8rem;}}
.wr-chip span {{display: block; font-size: .75rem; color: #6b7a88;}}
.wr-chip b {{font-family: 'Space Grotesk', sans-serif; color: {NAVY}; font-size: 1.05rem;}}
.wr-footer {{text-align: center; font-size: .82rem; color: {text_on_bg}; opacity: .8; margin: 2.2rem 0 0; line-height: 1.6;}}

/* ---- phones (e.g. 390 px wide) ---- */
@media (max-width: 640px) {{
  .block-container, [data-testid="stMainBlockContainer"] {{padding: 1.2rem 16px 2rem;}}
  [class*="st-key-card"] {{padding: 1.6rem 1.2rem 1.4rem; border-radius: 22px;}}
  .wr-question {{font-size: 1.55rem;}}
  .wr-hero {{padding-top: 2rem;}}
  .wr-h1 {{font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.03em; font-size: 2.9rem;}}
  .wr-sub {{font-size: 1rem;}}
  .wr-pct {{font-size: 112px;}}
  .wr-pct span {{font-size: 48px;}}
  .wr-msg {{font-size: 1.1rem;}}
  .stNumberInput input, .stTextInput input {{font-size: 24px !important;}}
  [data-testid="stHorizontalBlock"] {{gap: .5rem;}}
  .wr-stats {{flex-wrap: nowrap;}}
  .wr-stat {{padding: .7rem .75rem;}}
  .wr-stat b {{font-size: 1.15rem;}}
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 5. THE SCREENS
# ---------------------------------------------------------------------------

def go_to(step):
    """Move to another screen."""
    st.session_state["step"] = step
    st.rerun()


def progress_bar(n, label):
    """Thin orange bar 'Step n of 7'."""
    pct = n / TOTAL_QUESTIONS * 100
    st.markdown(f"""
<div class="wr-progress-top"><span>Step {n} of {TOTAL_QUESTIONS}</span><span>{label}</span></div>
<div class="wr-track"><div class="wr-fill" style="width:{pct:.0f}%"></div></div>
""", unsafe_allow_html=True)


def back_next(next_label="Next →"):
    """Back / Next buttons side by side. Returns (back_clicked, next_clicked)."""
    if st.session_state.get("editing"):
        next_label = "Save and back to recap"
    left, right = st.columns([1, 2])
    back = left.button("← Back", key="btn_back", width="stretch")
    nxt = right.button(next_label, key="btn_next", type="primary", width="stretch")
    return back, nxt


def after_valid_answer(next_step):
    """When editing from the recap, go straight back to the recap."""
    if st.session_state.get("editing"):
        st.session_state["editing"] = False
        go_to(STEP_RECAP)
    go_to(next_step)


def screen_welcome(n_days):
    inject_css("light", STEP_WELCOME)
    st.markdown(f"""
<div class="wr-hero wr-anim">
  <div class="wr-flame">🔥</div>
  <div class="wr-eyebrow">Wildfire Risk Check · California</div>
  <div class="wr-h1">Will a wildfire <span>start?</span></div>
  <div class="wr-sub">A Random Forest trained on 40 years of California weather and fire records, 1984–2023.</div>
</div>
""", unsafe_allow_html=True)
    if st.button("Start  →", type="primary", key="btn_start", width="stretch"):
        go_to(STEP_DATE)
    st.markdown(f"""
<div class="wr-stats wr-anim">
  <div class="wr-stat"><b>{n_days:,}</b><span>days of records</span></div>
  <div class="wr-stat"><b>300</b><span>decision trees</span></div>
  <div class="wr-stat"><b>7</b><span>quick questions</span></div>
</div>
<div class="wr-footer">Educational model – SKEMA Global BBA, AI for Business &amp; Society.<br>AI advises, humans decide.</div>
""", unsafe_allow_html=True)


def screen_date():
    inject_css("light", STEP_DATE)
    answers = st.session_state["answers"]
    raw = st.session_state["date_raw"]      # what the user typed, kept when going back
    with st.container(key="card_q"):
        progress_bar(1, "Date")
        st.markdown('<div class="wr-question">Which day do you want to check?</div>'
                    '<div class="wr-helper">Type the date by hand: day, month and year.</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1.5])
        with c1:
            st.markdown('<div class="wr-small-label">Day</div>', unsafe_allow_html=True)
            dd = st.text_input("Day (DD)", value=raw[0], max_chars=2, placeholder="DD",
                               key="in_dd", label_visibility="collapsed")
        with c2:
            st.markdown('<div class="wr-small-label">Month</div>', unsafe_allow_html=True)
            mm = st.text_input("Month (MM)", value=raw[1], max_chars=2, placeholder="MM",
                               key="in_mm", label_visibility="collapsed")
        with c3:
            st.markdown('<div class="wr-small-label">Year</div>', unsafe_allow_html=True)
            yyyy = st.text_input("Year (YYYY)", value=raw[2], max_chars=4, placeholder="YYYY",
                                 key="in_yyyy", label_visibility="collapsed")
        st.markdown('<div class="wr-note">ⓘ Any year works, past or future. The year itself is not used by the '
                    'model – only the season (month and day of the year).</div>', unsafe_allow_html=True)

        back, nxt = back_next()
        st.session_state["date_raw"] = [dd, mm, yyyy]
        if back:
            st.session_state["editing"] = False
            go_to(STEP_RECAP if answers.get("DATE") and st.session_state.get("came_from_recap") else STEP_WELCOME)
        if nxt:
            day, error = parse_date(dd, mm, yyyy)
            if error:
                st.error(error, icon="📅")
            else:
                answers["DATE"] = day
                after_valid_answer(STEP_DATE + 1)


def screen_question(step):
    inject_css("light", step)
    q = QUESTIONS[step - 2]
    answers = st.session_state["answers"]
    with st.container(key="card_q"):
        progress_bar(step, q["short"])
        st.markdown(f'<div class="wr-question">{q["title"]}</div><div class="wr-helper">{q["helper"]}</div>',
                    unsafe_allow_html=True)
        field, unit = st.columns([5, 1], vertical_alignment="top")
        with field:
            value = st.number_input(q["title"], value=answers.get(q["key"]), step=q["step"], format=q["fmt"],
                                    placeholder="Type a number", key=f"in_{q['key']}",
                                    label_visibility="collapsed")
        unit.markdown(f'<div class="wr-unit">{q["unit"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wr-conv">{conversion_text(q["conv"], value)}</div>'
                    f'<div class="wr-typical">{q["typical"]}</div><div style="height:1.4rem"></div>',
                    unsafe_allow_html=True)

        back, nxt = back_next()
        if back:
            answers[q["key"]] = value            # keep what was typed
            st.session_state["editing"] = False
            go_to(step - 1)
        if nxt:
            error = validate_number(q["key"], value, answers)
            if error:
                st.error(error, icon="⚠️")
            else:
                answers[q["key"]] = value
                after_valid_answer(step + 1)


def screen_recap():
    inject_css("light", STEP_RECAP)
    answers = st.session_state["answers"]
    st.session_state["came_from_recap"] = True
    with st.container(key="card_recap"):
        st.markdown('<div class="wr-progress-top"><span>All done</span><span>Recap</span></div>'
                    '<div class="wr-track"><div class="wr-fill" style="width:100%"></div></div>'
                    '<div class="wr-question">Check your answers</div>'
                    '<div class="wr-helper">Tap “Edit” to change a line, then calculate the risk.</div>',
                    unsafe_allow_html=True)
        rows = [("Date", answers["DATE"].strftime("%d/%m/%Y"), STEP_DATE)]
        for i, q in enumerate(QUESTIONS):
            rows.append((q["short"], f"{nice(answers[q['key']])} {q['unit']}", i + 2))
        for label, value, target in rows:
            with st.container(key=f"recaprow_{target}"):
                a, b, c = st.columns([5, 3.2, 1.6], vertical_alignment="center")
                a.markdown(f'<div class="wr-rlabel">{label}</div>', unsafe_allow_html=True)
                b.markdown(f'<div class="wr-rvalue">{value}</div>', unsafe_allow_html=True)
                if c.button("Edit", key=f"edit_{target}", type="tertiary"):
                    st.session_state["editing"] = True
                    go_to(target)
        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)

        # Final safety check (e.g. the min temperature was edited above the max)
        error = None
        for q in QUESTIONS:
            error = error or validate_number(q["key"], answers[q["key"]], answers)
        if error:
            st.error(error, icon="⚠️")
        left, right = st.columns([1, 2])
        if left.button("← Back", key="btn_back", width="stretch"):
            go_to(STEP_RECAP - 1)
        if right.button("🔥 Calculate the risk", key="btn_calc", type="primary", width="stretch",
                        disabled=error is not None):
            go_to(STEP_LOADING)


def screen_loading(model):
    """Short loading animation (about 1 second) while the 300 trees vote."""
    inject_css("light", STEP_LOADING)
    st.markdown("""
<div class="wr-loading wr-anim">
  <div class="wr-flame">🔥</div>
  <div class="wr-question" style="font-size:1.5rem">Asking 300 decision trees…</div>
  <div class="wr-helper">Comparing your day with 40 years of California weather.</div>
  <div class="wr-track"><div class="wr-fill"></div></div>
</div>""", unsafe_allow_html=True)
    start = time.time()
    X = build_features(st.session_state["answers"])
    st.session_state["result"] = explain_prediction(model, X)
    time.sleep(max(0.0, 1.0 - (time.time() - start)))
    go_to(STEP_RESULT)


def gauge_figure(p):
    """Semi-circular gauge 0-100% with the 4 risk zones and a marker at 30%."""
    fig = go.Figure(go.Indicator(
        mode="gauge", value=p * 100,
        gauge=dict(
            shape="angular",
            # the label of the 30% tick sits right next to the orange marker, at any screen width
            axis=dict(range=[0, 100], tickvals=[0, 30, 50, 70, 100],
                      ticktext=["0%", "our alert<br>threshold <b>30%</b>", "50%", "70%", "100%"],
                      tickwidth=0, tickfont=dict(size=12, color=NAVY, family="Inter")),
            bar=dict(color=NAVY, thickness=0.3),
            bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, 30], color="#bfe3cf"), dict(range=[30, 50], color="#fde3a7"),
                   dict(range=[50, 70], color="#fdc9a4"), dict(range=[70, 100], color="#f7b4b4")],
            threshold=dict(line=dict(color=ORANGE, width=5), thickness=1, value=30),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(height=240, margin=dict(l=62, r=38, t=34, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter"))
    return fig


def driver_rows(contrib, answers):
    """Group the contributions into plain-English lines and keep the 6 biggest."""
    a = answers
    X = build_features(a).iloc[0]
    groups = [
        (f"Night temperature {nice(a['MIN_TEMP'])}°F", contrib["MIN_TEMP"]),
        (f"Max temperature {nice(a['MAX_TEMP'])}°F", contrib["MAX_TEMP"]),
        (f"Wind speed {nice(a['AVG_WIND_SPEED'])} mph", contrib["AVG_WIND_SPEED"]),
        (f"Rain that day {nice(a['PRECIPITATION'])} in", contrib["PRECIPITATION"]),
        (f"Total rain, previous 7 days {nice(a['LAGGED_PRECIPITATION'])} in", contrib["LAGGED_PRECIPITATION"]),
        (f"Average wind, previous 7 days {nice(a['LAGGED_AVG_WIND_SPEED'])} mph", contrib["LAGGED_AVG_WIND_SPEED"]),
        (f"Day–night temperature gap {nice(X['TEMP_RANGE'])}°F", contrib["TEMP_RANGE"]),
        (f"Wind-to-heat ratio {X['WIND_TEMP_RATIO']:.2f}", contrib["WIND_TEMP_RATIO"]),
        (f"Season (date) {a['DATE'].strftime('%d %B')}", contrib["MONTH"] + contrib["DAY_OF_YEAR"]),
    ]
    groups.sort(key=lambda g: abs(g[1]), reverse=True)
    return groups[:6]


def points(c):
    """0.183 -> '+18 pts' ; small values keep one decimal."""
    v = c * 100
    return f"{v:+.1f} pts" if abs(v) < 1 else f"{v:+.0f} pts"


def out_of_range(answers, ranges):
    """List the answers that are outside what the model saw in 1984-2023."""
    X = build_features(answers).iloc[0]
    names = {q["key"]: q["short"] for q in QUESTIONS}
    names.update(TEMP_RANGE="Day–night temperature gap", WIND_TEMP_RATIO="Wind-to-heat ratio")
    found = []
    for key, label in names.items():
        low, high = ranges[key]
        if not (low - 1e-9 <= X[key] <= high + 1e-9):
            found.append(f"{label} (seen: {nice(low)}–{nice(high)})")
    return found


def screen_result(ranges):
    answers = st.session_state["answers"]
    p, baseline, contrib = st.session_state["result"]
    level, c1, c2, message = risk_level(p)
    inject_css((c1, c2), STEP_RESULT)
    pct = math.floor(round(p * 100, 6))   # floor: 29.7% shows "29", never a misleading "30% LOW"

    # 1. Huge percentage, level badge and message
    st.markdown(f"""
<div class="wr-result wr-anim">
  <div class="wr-kicker">Chance a wildfire starts · {answers['DATE'].strftime('%d %B')}</div>
  <div class="wr-pct">{pct}<span>%</span></div>
  <div class="wr-badge">{level}</div>
  <div class="wr-msg">{message}</div>
</div>""", unsafe_allow_html=True)

    # 2. Gauge
    with st.container(key="card_gauge"):
        st.markdown('<div class="wr-card-title">Where this day sits</div>'
                    '<div class="wr-card-sub">We raise an alert from 30%, not 50%: missing a fire costs far '
                    'more than a false alarm.</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_figure(p), config={"displayModeBar": False, "staticPlot": True}, width="stretch")

    # 3. What drove this prediction
    rows = driver_rows(contrib, answers)
    biggest = max(abs(c) for _, c in rows) or 1
    html = ""
    for label, c in rows:
        width = abs(c) / biggest * 50          # half of the bar = the biggest effect
        color = "#dc2626" if c > 0 else "#16a34a"
        side = f"left:50%; width:{width:.1f}%" if c > 0 else f"right:50%; width:{width:.1f}%"
        html += (f'<div class="wr-drow"><div class="wr-dtop"><span>{label}</span>'
                 f'<b style="color:{color}">→ {points(c)}</b></div>'
                 f'<div class="wr-dtrack"><div class="wr-dbar" style="{side}; background:{color}"></div></div></div>')
    with st.container(key="card_drivers"):
        st.markdown(f"""
<div class="wr-card-title">What drove this prediction</div>
<div class="wr-card-sub">How much each answer moved the risk for this day
(<span style="color:#dc2626;font-weight:600">red pushes it up</span>,
<span style="color:#16a34a;font-weight:600">green pulls it down</span>).</div>
{html}
<div class="wr-dfoot"><span>← lowers the risk</span><span>raises the risk →</span></div>
<div class="wr-sum">Average day in the data: <b>{baseline * 100:.1f}%</b> &nbsp;+&nbsp; effect of your answers:
<b>{(p - baseline) * 100:+.1f} pts</b> &nbsp;=&nbsp; <b>{p * 100:.1f}%</b>.
Computed by following this day's path through each of the 300 trees.</div>
""", unsafe_allow_html=True)

    # 4. Out-of-range warning
    outside = out_of_range(answers, ranges)
    if outside:
        st.markdown(f"""<div class="wr-warn wr-anim">⚠️ <b>Some values are outside what the model has seen (1984–2023).</b>
A Random Forest cannot extrapolate, so it treats them like the closest values it knows.<br>
<span style="font-size:.88rem">Concerned: {", ".join(outside)}.</span></div>""", unsafe_allow_html=True)

    # 5. Seasonal note (June to September)
    if 6 <= answers["DATE"].month <= 9:
        st.markdown("""<div class="wr-info wr-anim">☀️ <b>Summer note.</b> In summer the risk stays high even with
mild weather: in our data 70% of summer days had a fire start, and summer nights never went below 51°F.</div>""",
                    unsafe_allow_html=True)

    # 6. Inputs summary + restart
    chips = f'<div class="wr-chip"><span>Date</span><b>{answers["DATE"].strftime("%d/%m/%Y")}</b></div>'
    for q in QUESTIONS:
        chips += f'<div class="wr-chip"><span>{q["short"]}</span><b>{nice(answers[q["key"]])} {q["unit"]}</b></div>'
    with st.container(key="card_inputs"):
        st.markdown(f'<div class="wr-card-title">Your inputs</div><div class="wr-chips">{chips}</div>'
                    '<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        if st.button("Check another day  →", key="btn_restart", type="primary", width="stretch"):
            reset_answers()
            go_to(STEP_DATE)

    # 7. Footer
    st.markdown('<div class="wr-footer">Educational model – SKEMA Global BBA, AI for Business &amp; Society.<br>'
                '<b>AI advises, humans decide.</b></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 6. MAIN: show the right screen
# ---------------------------------------------------------------------------

def reset_answers():
    st.session_state["answers"] = {}
    st.session_state["date_raw"] = ["", "", ""]
    st.session_state["editing"] = False
    st.session_state["came_from_recap"] = False
    st.session_state.pop("result", None)


def main():
    st.set_page_config(page_title="Wildfire Risk Check", page_icon="🔥", layout="centered")
    if "step" not in st.session_state:
        st.session_state["step"] = STEP_WELCOME
        reset_answers()

    model, ranges, n_days = load_model()    # trained once, then reused on every click
    step = st.session_state["step"]

    if step == STEP_WELCOME:
        screen_welcome(n_days)
    elif step == STEP_DATE:
        screen_date()
    elif 2 <= step <= TOTAL_QUESTIONS:
        screen_question(step)
    elif step == STEP_RECAP:
        screen_recap()
    elif step == STEP_LOADING:
        screen_loading(model)
    else:
        screen_result(ranges)


# Streamlit runs this file as "__main__". Importing it (e.g. for tests) does not start the UI.
if __name__ == "__main__":
    main()
