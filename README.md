# Diabetes Classifier

A small machine-learning project that compares five classification algorithms on the Pima Indians Diabetes dataset and uses the best-performing MLP model in a Streamlit app to estimate diabetes risk from eight clinical measurements. This project is for educational purposes and is not a medical diagnostic tool.

## Install and Run

Install the UV-managed environment:

```bash
uv sync
```

Start the Streamlit app:

```bash
uv run streamlit run src/streamlit_app.py
```

Open the training and evaluation notebook:

```bash
uv run jupyter lab src/diabetes_model_training.ipynb
```
