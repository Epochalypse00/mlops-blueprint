# mlops-blueprint

A **production-grade machine learning template** that trains and serves a simple income classifier using the Adult Census Income dataset.

This repository is intentionally **“small model, big engineering.”**
The goal is not model complexity, but to demonstrate **how ML should be built, tested, versioned, and used in practice**.

If you can understand and extend this repo, you can work on real ML systems.

## What this project does

### Goal

Train a binary classifier that predicts whether a person’s income is:

* `>50K` → positive class
* `<=50K` → negative class

### Dataset

Hugging Face dataset:

```
scikit-learn/adult-census-income
```

This is a cleaned version of the classic UCI Adult dataset, containing demographic and employment features such as:

* age
* education
* occupation
* hours worked per week
* marital status

---

## Repo structure (important files)

```
mlops-blueprint/
├─ src/app/
│  ├─ data.py      # dataset loading, schema normalization, train/test split
│  ├─ model.py     # sklearn Pipeline (preprocessing + classifier)
│  ├─ train.py     # training entrypoint + evaluation + artifact logging
│  ├─ predict.py   # inference from JSON input
│  ├─ eval.py      # evaluation metrics (accuracy, F1)
│  ├─ io.py        # model save/load utilities
│  └─ utils.py     # shared helpers
│
├─ tests/          # unit tests for data, model IO, preprocessing
├─ configs/        # optional YAML configs (train / predict)
├─ models/         # trained model artifacts (gitignored)
└─ reports/
   ├─ results.md   # append-only training run log
   └─ predictions.json  # inference output (gitignored)
```

This layout separates **concerns cleanly**:

* data
* model definition
* training
* inference
* evaluation
* I/O

---

## How the code works (end-to-end)

### 1) Data loading & schema normalization (`src/app/data.py`)

What happens here:

1. Dataset is downloaded from Hugging Face
2. Converted to a pandas DataFrame
3. Column names are **normalized**

#### Why schema normalization matters

The Adult dataset sometimes uses dotted column names:

```
education.num
capital.gain
hours.per.week
```

But production code should **never depend on dataset quirks**.

So we normalize everything to underscore style:

```
education_num
capital_gain
hours_per_week
```

This ensures:

* stable feature names
* predictable pipelines
* safer inference inputs

The function returns a `DatasetSplit` object containing:

* `X_train`, `X_test`
* `y_train`, `y_test`
* `numeric_features`
* `categorical_features`

Splitting is **deterministic** via `random_state`.

---

### 2) Model construction (`src/app/model.py`)

We use a **single scikit-learn Pipeline** so preprocessing and model are saved together.

**Preprocessing**

* Numeric features → `StandardScaler`
* Categorical features → `OneHotEncoder(handle_unknown="ignore")`

**Classifier**

* `LogisticRegression`

  * stable
  * interpretable
  * fast
  * production-friendly

Why Logistic Regression?

* This repo tests engineering, not deep learning
* Simple models remove noise and highlight structure

---

### 3) Training & evaluation (`src/app/train.py`)

The training entrypoint does the following:

1. Loads dataset
2. Builds the pipeline
3. Fits on training data
4. Evaluates on test data
5. Computes:

   * Accuracy
   * F1 score (with correct positive class)
6. Saves the trained pipeline to:

   ```
   models/model.joblib
   ```
7. Appends results to:

   ```
   reports/results.md
   ```

Each training run is logged with:

* timestamp
* dataset name
* model type
* accuracy
* F1 score
* artifact path

This mimics **experiment tracking**, even without a full MLflow setup.

---

### 4) Inference from JSON (`src/app/predict.py`)

The prediction script is intentionally robust.

It supports:

* single JSON object
* list of JSON objects
* underscore schema
* dotted schema
* UTF-8 BOM (common on Windows)

Example input:

```json
{
  "age": 39,
  "workclass": "State-gov",
  "fnlwgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
```

Output includes:

* predicted label
* probability of positive class (`>50K`)

---

## How to run the project

### Install (editable + dev tools)

```bash
pip install -e ".[dev]"
```

### Train the model

```bash
python -m app.train
```

Outputs:

* `models/model.joblib`
* updated `reports/results.md`

### Run prediction

```bash
python -m app.predict --input sample.json --output reports/predictions.json
```

---

## Testing & code quality

This repo enforces hygiene:

* `pytest` for unit tests
* `ruff` + `black` for formatting
* `pre-commit` hooks for consistency

Run tests:

```bash
pytest
```

---

## What this repo demonstrates

* Clean ML project structure
* Deterministic data handling
* Reproducible training
* Safe inference interfaces
* Strong separation of concerns
* Professional Git history

This is the kind of repo you **build on**, not throw away.



