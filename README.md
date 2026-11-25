# 🔥 Wildfire Risk Check

A one-page web app that asks for the weather of **one day in California**, one question at a time,
and estimates the probability that a **wildfire starts** on that day.

School project – SKEMA Global BBA, *AI for Business & Society*.

## What it does
1. You type a date (DD / MM / YYYY) and 6 weather values: max temperature, night temperature,
   wind speed, rain that day, **total** rain over the previous 7 days, **average** wind over the previous 7 days.
2. A **Random Forest** (300 trees, max depth 12) trained on California weather and fire records
   1984–2023 (`Data.xlsx`) computes the probability of a fire start.
3. The whole page changes colour with the risk:

| Probability | Level | Action |
|---|---|---|
| < 30% | LOW | No alert. Normal readiness. |
| 30–50% | ELEVATED – ALERT | Put crews on standby. |
| 50–70% | HIGH | Pre-position engines. |
| ≥ 70% | EXTREME | Full mobilisation, warn the public. |

Our alert threshold is **30%**, not 50%: missing a fire costs far more than a false alarm.

4. "What drove this prediction" shows, for this specific day, how much each answer pushed the risk
   up or down (tree-path method: baseline + contributions = prediction).

Notes on the model:
- Data after 22/12/2023 is removed (fires were not labelled yet → leakage).
- `YEAR` is deliberately **not** used (it was important for the wrong reasons). Only the season matters.
- A Random Forest cannot extrapolate: values outside the 1984–2023 range get a warning.

## Run it on your computer
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501

## Files
- `app.py` – the whole app (model, screens, design)
- `Data.xlsx` – the dataset (sheet `CA_Weather_Fire_Dataset_1984-20`)
- `requirements.txt` – Python libraries
- `.streamlit/config.toml` – light theme, orange accent

*Educational model. AI advises, humans decide.*
