# mlops-blueprint

Production-grade ML template repo that trains and serves a simple income classifier using the Adult Census Income dataset.
This project is intentionally “small model, big engineering” — the point is to show clean structure, reproducible runs, and strong hygiene (tests, formatting, CI).

---

## What this project does

### Goal
Train a model that predicts whether a person’s income is:
- `>50K` (positive class)
- `<=50K` (negative class)

### Dataset
Hugging Face dataset:
- `scikit-learn/adult-census-income`

This dataset is derived from the classic UCI Adult dataset and includes demographic + work-related features such as education, age, occupation, hours per week, etc.

---

## Repo structure (important files)

mlops-blueprint/
├─ src/app/
│ ├─ data.py # loads dataset + splits into train/test
│ ├─ model.py # builds sklearn Pipeline (preprocess + classifier)
│ ├─ train.py # trains model + evaluates + saves artifacts
│ ├─ predict.py # loads saved model + predicts from JSON input
│ ├─ eval.py # evaluation metrics (accuracy, F1)
│ ├─ io.py # save/load model utilities
│ └─ utils.py # shared helpers
├─ tests/ # unit tests
├─ configs/ # optional config files (train/predict)
├─ models/ # saved model artifacts (gitignored)
└─ reports/
├─ results.md # training runs log (appended on each train run)
└─ predictions.json (gitignored output)


---

## How the code works (the pipeline)

### 1) Data loading + schema normalization (`src/app/data.py`)
- Downloads dataset from Hugging Face.
- Converts to a pandas DataFrame.
- Normalizes column names into a consistent schema.

Why?  
The Adult dataset sometimes appears with dotted column names like:
- `education.num`, `capital.gain`, `hours.per.week`

But our model expects:
- `education_num`, `capital_gain`, `hours_per_week`

So we map dotted → underscore schema to keep the training code stable.

Then it:
- separates `X` (features) and `y` (label)
- performs a deterministic split using `random_state`
- returns a `DatasetSplit` bundle:
  - `X_train, X_test, y_train, y_test`
  - feature lists: numeric + categorical

### 2) Model construction (`src/app/model.py`)
We build a scikit-learn `Pipeline`:

**Preprocessing**
- numeric columns → `StandardScaler()`
- categorical columns → `OneHotEncoder(handle_unknown="ignore")`

**Classifier**
- `LogisticRegression(max_iter=2000, solver="lbfgs")`

Pipeline is used so the exact preprocessing is saved together with the model.

### 3) Training + evaluation (`src/app/train.py`)
Training script:
- calls `load_adult_census_hf(...)`
- builds pipeline from known feature lists
- fits pipeline
- predicts on test set
- computes metrics:
  - accuracy
  - F1 (with correct positive class label)

Artifacts produced:
- `models/model.joblib` (saved pipeline)
- `reports/results.md` gets a new row appended with:
  - timestamp
  - dataset name
  - model name
  - accuracy
  - F1
  - model path

### 4) Prediction from JSON (`src/app/predict.py`)
Prediction script:
- loads saved `models/model.joblib`
- reads input JSON (single object OR list of objects)
- handles both:
  - underscore schema (`education_num`)
  - dotted schema (`education.num`)
- also handles UTF-8 BOM safely (common on Windows)
- returns:
  - predicted label
  - probability for positive class (`>50K`)
- writes results to output JSON
