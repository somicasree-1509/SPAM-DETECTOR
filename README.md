# 📩 Spam Message Detector

A beginner-friendly web app that predicts whether a message is **Spam** or **Not Spam (Ham)**.
Built with Python, Streamlit, Scikit-learn and Pandas.

## How it works
1. `train_model.py` reads labeled messages from `spam.csv`.
2. **TF-IDF** converts each message into numbers (how important each word is).
3. **Multinomial Naive Bayes** learns which words tend to appear in spam vs. normal messages.
4. The trained model is saved as `spam_model.joblib`.
5. `app.py` loads the model and classifies whatever you type.

No spam keywords are hardcoded. The model learns only from the labeled examples.

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit user interface and prediction |
| `train_model.py` | Trains and saves the model |
| `spam.csv` | Labeled dataset (`label`, `message`) |
| `requirements.txt` | Python libraries needed |

## Setup and run (VS Code terminal)

**Windows**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

**macOS / Linux**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The app opens at http://localhost:8501.

## Using a bigger dataset (recommended)
The included `spam.csv` is a small sample so the project stays simple, so accuracy is limited.
For better results, download the public **SMS Spam Collection** from Kaggle or the UCI repository
and save it as `spam.csv` (replace the existing file). The training script already understands its
format (columns `v1` and `v2`). Then run `python train_model.py` again.

## Notes
- If you see "Model file not found", run `python train_model.py` first.
- This is a learning project, so don't rely on it for real security.
