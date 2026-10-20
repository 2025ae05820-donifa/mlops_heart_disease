# Heart Disease Prediction — End-to-End MLOps Project
 
An end-to-end Machine Learning Operations (MLOps) project for binary heart disease classification using the UCI Cleveland Heart Disease dataset. The repository includes data collection and preprocessing, exploratory data analysis, model training and evaluation, MLflow tracking, a FastAPI inference service, a Dockerfile, and Kubernetes manifests for the API, Prometheus, and Grafana. The manifests are provided for deployment; successful execution depends on a configured local Kubernetes environment.
 
---
 
## Table of Contents
 
1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Key Features](#3-key-features)
4. [Technology Stack](#4-technology-stack)
5. [System Architecture](#5-system-architecture)
6. [Dataset Description](#6-dataset-description)
7. [Project Structure](#7-project-structure)
8. [Installation and Setup](#8-installation-and-setup)
9. [Data Collection and Preprocessing](#9-data-collection-and-preprocessing)
10. [Exploratory Data Analysis](#10-exploratory-data-analysis)
11. [Model Development](#11-model-development)
12. [Model Evaluation and Results](#12-model-evaluation-and-results)
13. [MLflow Experiment Tracking](#13-mlflow-experiment-tracking)
14. [FastAPI Model Serving](#14-fastapi-model-serving)
15. [API Documentation and Testing](#15-api-documentation-and-testing)
16. [Docker, Kubernetes, and Monitoring](#16-docker-kubernetes-and-monitoring)
17. [Git and Version Control](#17-git-and-version-control)
18. [Reproducibility](#18-reproducibility)
19. [Current Project Status](#19-current-project-status)
20. [Future Enhancements](#20-future-enhancements)
21. [Limitations](#21-limitations)
22. [Disclaimer](#22-disclaimer)
23. [Author](#23-author)
 
---
 
## 1. Project Overview
 
Heart disease is a major global health concern. Machine learning can help analyze clinical measurements and identify patterns associated with the presence of heart disease.
 
This project develops a binary classification pipeline that predicts whether a patient belongs to the heart disease positive or negative class based on selected clinical attributes.
 
The project follows an MLOps-oriented workflow, connecting the model development lifecycle with experiment tracking, model persistence, API-based inference, and containerization configuration.
 
The original dataset contains 303 observations. The original target variable, `num`, represents multiple target values from 0 to 4. For this project, it is transformed into a binary target:
 
- `0`: No heart disease indicated by the dataset.
- `1`: Heart disease indicated by the dataset.
 
Two classification algorithms are trained and compared:
 
- Logistic Regression
- Random Forest Classifier
 
Hyperparameter tuning and stratified five-fold cross-validation are used to compare model performance. Logistic Regression is selected as the final model based on its mean cross-validation ROC-AUC score.
 
The selected model is saved as a serialized artifact and integrated into a FastAPI application that exposes prediction, health, and readiness endpoints.
 
**Project goal:** Build a reproducible, maintainable, and deployable machine learning application rather than stopping at model training alone.
 
## 2. Objectives
 
The main objectives are:
 
1. Collect the heart disease dataset and organize raw and processed data.
2. Clean the dataset and handle missing values.
3. Perform exploratory data analysis to understand the dataset and feature relationships.
4. Build a preprocessing pipeline for numerical and categorical features.
5. Train and compare Logistic Regression and Random Forest models.
6. Apply hyperparameter tuning and stratified cross-validation.
7. Evaluate classification performance using multiple metrics.
8. Track model experiments using MLflow with a local SQLite backend.
9. Persist the selected model and associated metadata.
10. Develop a REST API using FastAPI for model inference.
11. Provide health and readiness endpoints for service checks.
12. Package the API with Docker and provide Kubernetes and monitoring manifests.
13. Establish a foundation for automated testing, continuous integration, deployment, and model monitoring.
 
## 3. Key Features
 
### Data pipeline
 
- Raw and processed dataset separation.
- Data cleaning and missing-value handling.
- Binary target transformation.
- Reusable preprocessing components.
 
### Model development
 
- Logistic Regression and Random Forest comparison.
- Stratified five-fold cross-validation.
- Hyperparameter tuning.
- Accuracy, precision, recall, F1-score, and ROC-AUC evaluation.
- Model comparison report and evaluation visualizations.
 
### Experiment tracking
 
- MLflow experiment organization.
- SQLite-backed tracking storage.
- Parameter and metric logging.
- Model and artifact logging.
 
### Model serving
 
- FastAPI REST API.
- Input validation using Pydantic.
- Prediction and positive-class probability output.
- Health and readiness checks.
- Interactive Swagger documentation.
 
### Deployment preparation
 
- Python dependency management.
- Dockerfile.
- Kubernetes Deployment and Service manifests for the API.
- Prometheus scrape configuration and Grafana Prometheus data-source provisioning.
- Git ignore rules for environments, caches, and generated files.
 
## 4. Technology Stack
 
| Category                | Technology                                                               |
| ----------------------- | ------------------------------------------------------------------------ |
| Programming language    | Python 3.11 in the Docker image; use a compatible Python version locally |
| Data manipulation       | Pandas, NumPy                                                            |
| Machine learning        | scikit-learn                                                             |
| Model serialization     | Joblib                                                                   |
| Experiment tracking     | MLflow                                                                   |
| Tracking database       | SQLite                                                                   |
| API framework           | FastAPI                                                                  |
| API server              | Uvicorn                                                                  |
| Input validation        | Pydantic                                                                 |
| Data visualization      | Matplotlib and Seaborn, where used in EDA                                |
| Containerization        | Docker                                                                   |
| Orchestration           | Kubernetes (Minikube manifests provided)                                 |
| Metrics                 | prometheus-fastapi-instrumentator and Prometheus                         |
| Dashboards              | Grafana (Prometheus data source provisioned)                             |
| Version control         | Git                                                                      |
| Development environment | Python virtual environment (`.venv`)                                     |
| Dataset                 | UCI Cleveland Heart Disease dataset                                      |
 
Python package versions are pinned in `requirements.txt`. The Docker image uses Python 3.11; the repository does not specify a separate local Python version constraint.
 
## 5. System Architecture
 
The project is organized into five major stages.
 
### Stage 1: Data ingestion and preprocessing
 
The raw dataset is collected and stored under `data/raw/`. A preprocessing script cleans the data, handles missing values, and generates the processed dataset under `data/processed/`.
 
### Stage 2: Model development
 
Exploratory data analysis is performed to understand the data. A reusable preprocessing pipeline handles numerical and categorical features. Logistic Regression and Random Forest models are trained and evaluated using cross-validation and a held-out test set.
 
### Stage 3: Experiment tracking
 
Training parameters, metrics, and relevant artifacts are recorded using MLflow. A local SQLite database stores experiment metadata.
 
### Stage 4: Model serving
 
The selected model is saved to `artifacts/best_model.joblib`. The FastAPI application loads this model and exposes REST endpoints for prediction and service health checks.
 
### Stage 5: Containerization and deployment
 
A Dockerfile defines how to package the API, dependencies, source code, and model artifact. Kubernetes manifests define the API deployment and service, plus Prometheus and Grafana resources. The repository does not include a Docker Compose file.
 
### Architecture flow
 
                 UCI Heart Disease Dataset
                            |
                     Data Ingestion
                            |
                    Raw Dataset (CSV)
                            |
                  Data Cleaning and
                    Preprocessing
                            |
                  Processed Dataset
                            |
                 Exploratory Data Analysis
                            |
              Feature Preprocessing Pipeline
                            |
              +-------------+-------------+
              |                           |
      Logistic Regression          Random Forest
              |                           |
              +-------------+-------------+
                            |
                Cross-Validation and
                  Hyperparameter Tuning
                            |
                  Model Comparison
                            |
                  Best Model Selection
                            |
                +-----------+-----------+
                |                       |
          MLflow Tracking         Model Artifacts
          Experiment Metrics      best_model.joblib
                |                       |
                +-----------+-----------+
                            |
                     FastAPI Service
                            |
                    REST Prediction API
                            |
                  Client / Swagger UI
                            |
                       Docker Image
                            |
                Kubernetes Deployment
                      (Minikube)
                            |
                  Kubernetes Service
                            |
                    Prediction Endpoint
                            |
                 Prometheus Metrics
                      (/metrics)
                            |
                  Grafana Visualization

## 6. Dataset Description
 
### Dataset source
 
The project uses the Cleveland subset of the UCI Heart Disease dataset.
 
Dataset information and original documentation:
 
- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/45/heart+disease
 
The dataset contains 303 observations and clinical attributes used for heart disease classification.
 
### Input features
 
The following 13 input features are used by the prediction API.
 
| Feature    | Description                                        |
| ---------- | -------------------------------------------------- |
| `age`      | Age in years                                       |
| `sex`      | Encoded sex value                                  |
| `cp`       | Chest pain type                                    |
| `trestbps` | Resting blood pressure                             |
| `chol`     | Serum cholesterol                                  |
| `fbs`      | Fasting blood sugar indicator                      |
| `restecg`  | Resting electrocardiographic result                |
| `thalach`  | Maximum heart rate achieved                        |
| `exang`    | Exercise-induced angina indicator                  |
| `oldpeak`  | ST depression induced by exercise relative to rest |
| `slope`    | Slope of the peak exercise ST segment              |
| `ca`       | Number of major vessels visualized by fluoroscopy  |
| `thal`     | Thalassemia-related categorical encoding           |
 
The categorical values use the encodings expected by the dataset and trained model. They should not be interpreted as arbitrary integers or replaced with different encodings without updating the preprocessing pipeline.
 
### Target variable
 
The original target variable is `num`.
 
It is transformed as follows:
 
```python
target = 0 if num == 0 else 1
```
 
This creates a binary classification problem.
 
- Class `0`: No heart disease indicated by the dataset.
- Class `1`: Heart disease indicated by the dataset.
 
### Dataset handling
 
The project separates raw and processed data:
 
- `data/raw/heart_disease.csv`: downloaded raw dataset.
- `data/processed/heart_disease_clean.csv`: cleaned dataset prepared for analysis and training.
 
The raw dataset should be retained as the source of truth. Cleaning and target transformation should be repeatable through the preprocessing script.
 
## 7. Project Structure
 
The project is organized into scripts, source modules, model artifacts, reports, and deployment configuration.
 
```text
mlops_heart_disease/
│
├── data/
│   ├── raw/
│   │   └── heart_disease.csv
│   └── processed/
│       └── heart_disease_clean.csv
│
├── scripts/
│   ├── download_data.py
│   └── preprocess_data.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── mlflow_tracking.py
│   ├── train.py
│   │
│   └── models/
│       ├── logistic_regression/
│       │   └── model.py
│       └── random_forest/
│           └── model.py
│
├── artifacts/
│   ├── best_model.joblib
│   └── model_info.json
│
├── reports/
│   ├── model_comparison.csv
│   └── figures/
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── monitoring.yaml
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── mlflow.db                  # Generated local tracking database
└── README.md
```
 
Reports, plots, model artifacts, and `mlflow.db` are generated or updated by the corresponding scripts. There is currently no `tests/` directory, Docker Compose file, or GitHub Actions workflow in the repository.
 
## 8. Installation and Setup
 
### Prerequisites
 
Install the following tools as needed:
 
- Python 3.11 (the Docker base image uses Python 3.11)
- Git
- Docker for container execution
- Minikube and `kubectl` for the Kubernetes deployment
 
### Step 1: Clone the repository
 
Clone the project:
 
```bash
git clone https://github.com/2025ae05820-donifa/mlops_heart_disease.git
cd mlops_heart_disease
```
 
### Step 2: Create and activate a virtual environment
 
```bash
python3 -m venv .venv
source .venv/bin/activate
```
 
Verify the Python version:
 
```bash
python --version
```
 
### Step 3: Install dependencies
 
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
 
### Step 4: Verify the environment
 
```bash
python -c "import sys; print(sys.executable)"
```
 
The output should point to the project's `.venv` environment. On Windows, use `.venv\\Scripts\\activate` to activate the environment.
 
## 9. Data Collection and Preprocessing
 
### Download the dataset
 
Run the data download script from the project root:
 
```bash
python scripts/download_data.py
```
 
The script downloads the raw dataset to:
 
```text
data/raw/heart_disease.csv
```
 
Verify that the file exists before running preprocessing.
 
### Preprocess the data
 
Run:
 
```bash
python scripts/preprocess_data.py
```
 
The preprocessing step creates the cleaned dataset:
 
```text
data/processed/heart_disease_clean.csv
```
 
The workflow handles missing values and prepares the target for binary classification.
 
To create the one-hot-encoded CSV as well, run `python scripts/preprocess_data.py`; it writes `data/processed/heart_disease_encoded.csv`. Model training uses the clean CSV and applies its own fitted preprocessing pipeline, so do not train on the already encoded CSV.
 
### Model preprocessing pipeline
 
The reusable preprocessing implementation is in `src/preprocessing.py`.
 
It uses a `ColumnTransformer` with separate transformations:
 
**Numerical features**
 
- Median imputation for missing values.
- Standard scaling.
 
**Categorical features**
 
- Most-frequent-value imputation.
- One-hot encoding with `handle_unknown="ignore"`.
 
The preprocessing transformations are incorporated into the model pipeline so that training and inference use consistent feature processing.
 
Using a pipeline also reduces the risk of applying different transformations to training data and prediction requests.
 
## 10. Exploratory Data Analysis
 
Exploratory data analysis is implemented in:
 
```text
src/eda.py
```
 
EDA helps investigate the dataset before model development.
 
Areas to examine include:
 
- Dataset dimensions and feature types.
- Missing-value distribution.
- Numerical feature distributions.
- Target class distribution.
- Relationships between input features and the target.
- Correlations among numerical features.
- Potential outliers and unusual values.
 
Visualizations generated by the project are stored under:
 
```text
reports/figures/
```
 
Run the EDA workflow from the repository root:
 
```bash
python -m src.eda
```
 
It creates `class_balance.png`, `feature_distributions.png`, `correlation_heatmap.png`, and `missing_values.png`, along with `descriptive_statistics.csv`, `missing_values.csv`, and `eda_summary.json`. Training additionally writes model comparison, ROC, confusion-matrix, and metric plots to the same figures directory.
 
EDA is useful for understanding the data, but observed correlations should not be interpreted as evidence of causation.
 
## 11. Model Development
 
The training workflow is implemented in:
 
```text
src/train.py
```
 
Model definitions are separated into their respective modules.
 
### 11.1 Logistic Regression
 
Implementation:
 
```text
src/models/logistic_regression/model.py
```
 
The model uses scikit-learn's `LogisticRegression` with a maximum iteration limit of 2,000 and a random state of 42 in its default configuration.
 
Logistic Regression is a useful baseline for binary classification because it is relatively simple, efficient, and easier to interpret than many more complex models.
 
The tuning workflow evaluates different hyperparameter settings, including the regularization parameter `C` and class weighting.
 
### 11.2 Random Forest
 
Implementation:
 
```text
src/models/random_forest/model.py
```
 
The default configuration uses:
 
- 200 decision trees.
- Balanced class weights.
- Random state 42.
- Parallel processing through `n_jobs=-1`.
 
Random Forest can capture nonlinear relationships and interactions between features.
 
### 11.3 Training and validation strategy
 
The training workflow uses:
 
- A held-out test set.
- Stratified five-fold cross-validation.
- Grid search for hyperparameter tuning.
- A common preprocessing approach.
- Multiple performance metrics for model comparison.
 
Stratification helps preserve class proportions across the validation folds.
 
The held-out test set is used to assess the selected model's performance on data not used to fit that model. Cross-validation provides an additional basis for comparing model configurations.
 
### 11.4 Model selection
 
The selection criterion is the mean cross-validation ROC-AUC score.
 
Logistic Regression was selected because it achieved a mean cross-validation ROC-AUC of **0.9113**, higher than the Random Forest result recorded in the training run.
 
The selected model is saved to:
 
```text
artifacts/best_model.joblib
```
 
Model metadata is saved to:
 
```text
artifacts/model_info.json
```
 
The model comparison report is saved to:
 
```text
reports/model_comparison.csv
```
 
Run the complete training, tuning, evaluation, and MLflow logging workflow from the repository root:
 
```bash
python -m src.train
```
 
The script reads `data/processed/heart_disease_clean.csv`, creates an 80/20 stratified split with random state 42, and writes the selected model, metadata, comparison report, classification reports, plots, and MLflow runs. Re-running it can replace the current model and report artifacts.
 
## 12. Model Evaluation and Results
 
The following results are from the recorded successful training run. They should be treated as experiment results for this particular dataset split and configuration, not as guaranteed performance on other populations.
 
### 12.1 Cross-validation results
 
| Metric            | Logistic Regression | Random Forest |
| ----------------- | ------------------: | ------------: |
| Mean CV accuracy  |              0.8511 |        0.8139 |
| Mean CV precision |              0.8776 |        0.8210 |
| Mean CV recall    |              0.7921 |        0.7644 |
| Mean CV ROC-AUC   |              0.9113 |        0.8992 |
 
### 12.2 Held-out test results
 
| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |              0.8852 |        0.8689 |
| Precision |              0.8387 |        0.8333 |
| Recall    |              0.9286 |        0.8929 |
| F1-score  |              0.8814 |        0.8621 |
| ROC-AUC   |              0.9610 |        0.9470 |
 
### 12.3 Interpretation
 
**Accuracy** measures the proportion of correct predictions across both classes.
 
**Precision** measures the proportion of positive predictions that are actually positive in the evaluation data.
 
**Recall** measures the proportion of actual positive cases correctly identified by the model.
 
**F1-score** balances precision and recall using their harmonic mean.
 
**ROC-AUC** measures how well the model ranks positive cases above negative cases across classification thresholds.
 
Logistic Regression achieved better recorded cross-validation ROC-AUC, accuracy, precision, recall, and ROC-AUC on the held-out test set than Random Forest.
 
The test recall of 0.9286 indicates that the selected model identified a high proportion of positive cases in this test split. However, the precision of 0.8387 also indicates that some positive predictions were false positives.
 
Because this is a health-related classification problem, threshold selection and the relative costs of false negatives and false positives require careful consideration before any real-world application.
 
The held-out test metrics should be interpreted alongside the dataset size, split strategy, and limitations of the underlying data.
 
## 13. MLflow Experiment Tracking
 
MLflow is used to track model training experiments.
 
The tracking utilities are implemented in:
 
```text
src/mlflow_tracking.py
```
 
The experiment name is:
 
```text
Heart Disease Classification
```
 
### 13.1 Tracking backend
 
The project uses a local SQLite tracking database:
 
```text
mlflow.db
```
 
The tracking URI is configured using a SQLite connection string.
 
This approach avoids relying on MLflow's legacy filesystem-based tracking backend.
 
### 13.2 Information tracked
 
The tracking utilities support logging:
 
- Model and experiment tags.
- Training parameters.
- Evaluation metrics.
- Model artifacts.
- Other selected files generated during training.
 
The training script integrates with these utilities to record model runs.
 
### 13.3 Start the MLflow UI
 
From the project root, with the virtual environment activated:
 
```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000 --workers 1
```
 
Open:
 
http://127.0.0.1:5000
 
The single-worker setting was used to avoid a Windows socket error encountered with the default worker configuration.
 
The UI can be used to inspect recorded runs, compare metrics, and review logged parameters and artifacts.
 
Stop the UI with `Ctrl+C` when finished.
 
### 13.4 Model serialization
 
The model is saved locally with Joblib. MLflow model logging uses cloudpickle serialization in the configured helper.
 
Serialized model files should only be loaded from trusted sources because deserialization can execute unsafe code in some circumstances.
 
### 13.5 Tracking persistence
 
The SQLite database contains local experiment metadata. It should be handled separately from the model artifact and source code.
 
For a shared or production environment, a managed tracking server and appropriate artifact storage can be considered later.
 
## 14. FastAPI Model Serving
 
The API implementation is:
 
```text
src/api.py
```
 
The application loads the saved model from:
 
```text
artifacts/best_model.joblib
```
 
The API provides prediction functionality and service checks.
 
### 14.1 Start the API locally
 
Activate the virtual environment and run:
 
```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
 
Keep the terminal open while testing the service.
 
### 14.2 Available endpoints
 
| Method | Endpoint   | Purpose                                 |
| ------ | ---------- | --------------------------------------- |
| GET    | `/`        | Returns basic API information           |
| GET    | `/health`  | Checks whether the API process responds |
| GET    | `/ready`   | Checks whether the model is loaded      |
| POST   | `/predict` | Returns a model prediction              |
 
### 14.3 Root endpoint
 
Request:
 
```http
GET /
```
 
Returns basic service information, including links to the API documentation and health endpoints.
 
### 14.4 Health endpoint
 
Request:
 
```http
GET /health
```
 
Expected response:
 
```json
{
  "status": "ok"
}
```
 
This endpoint checks whether the API process is responding. It does not independently establish that the model is ready for predictions.
 
### 14.5 Readiness endpoint
 
Request:
 
```http
GET /ready
```
 
Expected response when the model is loaded:
 
```json
{
  "status": "ready",
  "model": "best_model.joblib"
}
```
 
If the model is unavailable, the endpoint returns HTTP 503.
 
### 14.6 Prediction endpoint
 
Endpoint:
 
```http
POST /predict
```
 
The request body contains the 13 model input features.
 
Example request:
 
```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```
 
This is a sample software-testing payload, not real patient data.
 
### 14.7 Prediction response
 
The response contains fields similar to:
 
```json
{
  "prediction": 1,
  "label": "disease_positive",
  "confidence": 0.85,
  "confidence_percent": 85.0,
  "class_probabilities": {
    "0": 0.15,
    "1": 0.85
  },
  "disclaimer": [
    "This output is generated by a machine-learning model for analysis only. It is not a medical diagnosis and must not replace professional medical advice."
  ]
}
```
 
The prediction and probabilities shown above are illustrative only. Actual values depend on the model and supplied input.
 
Response fields:
 
- `prediction`: predicted binary class, either `0` or `1`.
- `label`: human-readable representation of the predicted class.
- `confidence`: model-estimated probability of the predicted class.
- `confidence_percent`: the same confidence expressed as a percentage.
- `class_probabilities`: model-estimated probabilities keyed by class (`"0"` and `"1"`).
- `disclaimer`: reminder that the output is not a medical diagnosis.
 
The probability is model-dependent and is not necessarily a calibrated estimate of an individual's clinical risk.
 
The API also exposes `GET /metrics` for Prometheus scraping. This endpoint is not listed in the generated Swagger schema.
 
### 14.8 Input validation
 
The request schema uses Pydantic field constraints and encoded categories to validate the request.
 
Examples of validation include:
 
- Age must be within the configured range.
- Binary features must use the expected binary values.
- Numerical measurements must satisfy their configured constraints.
- All required features must be provided.
 
Invalid requests should return a validation error rather than being silently accepted.
 
## 15. API Documentation and Testing
 
FastAPI automatically generates an interactive OpenAPI documentation interface.
 
Once the API is running, open:
 
http://127.0.0.1:8000/docs
 
The Swagger UI provides interactive access to the four implemented endpoints.
 
 
These are manual smoke-test steps, not checked-in test results. There is no automated test suite in the current repository. Although `pytest` is listed in `requirements.txt`, `python -m pytest -v` will not run project tests until test files are added. Ruff is not currently listed as a project dependency, and no GitHub Actions workflow is present.
 
## 16. Docker, Kubernetes, and Monitoring
 
Docker is intended to package the application and its dependencies into a reproducible environment.
 
### 16.1 Dockerfile
 
The Dockerfile uses `python:3.11-slim` and:
 
- Set the working directory.
- Install dependencies from `requirements.txt`.
- Copy the application source code.
- Copy the saved model artifacts.
- Expose port 8000.
- Start the API using Uvicorn.
 
The container starts the application using:
 
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
 
Binding to `0.0.0.0` inside the container allows the service to accept connections through a published Docker port.
 
### 16.2 Build and run the Docker image
 
There is no Docker Compose file. Build and run the API image directly:
 
```bash
docker build -t heart-disease-api:dev .
docker run --rm --name heart-disease-api-container -p 8000:8000 heart-disease-api:dev
```
 
The `dev` tag is intentional: it matches `image: heart-disease-api:dev` in `k8s/deployment.yaml`. The Dockerfile copies `artifacts/` into the image, so `artifacts/best_model.joblib` must exist before building.
 
### 16.3 Verify the containerized application
 
After the container starts, test:
 
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/ready
 
Then execute a prediction through Swagger.
 
A successful local API test does not by itself prove that the Docker image works. Containerized execution must be verified independently.
 
Verify Docker is installed and running:
 
```bash
docker --version
docker info
```
 
### 16.4 Deploy to Minikube
 
Start Minikube using Docker, build the image in Minikube's image environment, and apply all manifests:
 
```bash
minikube start --driver=docker
minikube image build -t heart-disease-api:dev .
kubectl apply -f k8s/
kubectl get nodes
kubectl get deployments,pods,services
```
 
The API Service is `heart-disease-api` on port `8000` (NodePort `30080`). Access it using:
 
```bash
minikube service heart-disease-api --url
```
 
Or forward the service to local port 8000:
 
```bash
kubectl port-forward service/heart-disease-api 8000:8000
```
 
Inspect workload logs if needed:
 
```bash
kubectl logs deployment/heart-disease-api
kubectl describe deployment heart-disease-api
```
 
### 16.5 Prometheus and Grafana
 
`k8s/monitoring.yaml` deploys Prometheus and Grafana in the default namespace. Prometheus is configured to scrape the API at `/metrics`; the FastAPI app exposes that endpoint through `prometheus-fastapi-instrumentator`. Grafana's Prometheus data source is provisioned automatically, but no dashboard is included in the repository.
 
Forward the UIs to localhost (run each command in a separate terminal):
 
```bash
kubectl port-forward service/prometheus 9090:9090
kubectl port-forward service/grafana 3000:3000
```
 
Open `http://127.0.0.1:9090` and `http://127.0.0.1:3000`. In Prometheus, check `up{job="heart-disease-api"}` and confirm it returns `1`; inspect `/metrics` or Prometheus's metric browser for the exact instrumented metric names before querying request or latency data. The YAML configures these components but does not prove that a cluster deployment or Grafana dashboard has been run successfully.
 
Check monitoring resources and logs with:
 
```bash
kubectl get deployments,pods,services
kubectl logs deployment/prometheus
kubectl logs deployment/grafana
```
 
## 17. Git and Version Control
 
Git is used to manage the project source code and track changes.
 
### Initialize Git
 
From the project root:
 
```bash
git init
```
 
Check the repository state:
 
```bash
git status
```
 
### `.gitignore`
 
The `.gitignore` file excludes files that should not normally be tracked, including:
 
- Python virtual environments.
- Python cache files.
- Editor and operating-system files.
- Environment variables and local secrets.
- Local MLflow database files.
- Generated model files when configured to exclude them.
- Generated plots and temporary files.
 
This prevents thousands of environment and cache files from being added accidentally.
 
### `.dockerignore`
 
The `.dockerignore` file excludes unnecessary files from the Docker build context, such as:
 
- `.venv/`
- `.git/`
- `__pycache__/`
- Local MLflow tracking files.
 
The two ignore files serve different purposes: `.gitignore` controls Git tracking, while `.dockerignore` controls what Docker sends to the build process.
 
### Recommended Git workflow
 
```bash
git status
git add .gitignore .dockerignore Dockerfile requirements.txt README.md src scripts k8s
git status
```
 
Review the staged changes before committing. Add tests and other required project files once they exist.
 
Then commit:
 
```bash
git commit -m "Add heart disease MLOps pipeline and FastAPI service"
```
 
Do not commit secrets, private environment files, or an entire virtual environment.
 
If the saved model is excluded from Git, document how to regenerate or obtain it so another developer can reproduce the deployment.
 
## 18. Reproducibility
 
The project includes several mechanisms that support repeatable execution:
 
- A dedicated Python virtual environment.
- A dependency file.
- Separate data ingestion and preprocessing scripts.
- Reusable preprocessing components.
- Fixed random-state settings in the model configurations.
- Stratified cross-validation.
- Recorded training parameters and evaluation metrics.
- A saved model artifact.
- A structured API input schema.
- Docker configuration for packaging the service.
 
For stronger reproducibility, future improvements should include:
 
- Pinning compatible package versions.
- Recording the dataset source and version.
- Tracking data and model artifact checksums.
- Recording the train/test split configuration.
- Persisting the selected model's complete hyperparameters.
- Testing inference consistency between local Python and Docker.
- Using a reproducible build and automated test pipeline.
 
## 19. Current Project Status
 
The table below distinguishes repository implementation and generated artifacts from runtime checks that must be performed in the target environment.
 
| Component                        | Status                                                                                     |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| Data download and preprocessing  | Scripts and raw/processed CSV files are present                                            |
| EDA                              | Implemented; summary, tables, and figures are present                                      |
| Model training and evaluation    | Implemented; model, metadata, comparison results, and reports are present                  |
| MLflow tracking                  | SQLite-backed tracking is configured; local tracking data is present                       |
| MLflow UI                        | Command documented; successful UI access is not verifiable from the repository             |
| FastAPI                          | Implemented; run the API to verify it in the target environment                            |
| Swagger/API smoke tests          | Instructions provided; no automated tests or checked-in test output                        |
| Docker                           | Dockerfile present; image execution is environment-dependent                               |
| Kubernetes and monitoring        | API, Prometheus, and Grafana manifests present; cluster execution is environment-dependent |
| Prometheus metrics               | API instrumentation exposes `/metrics`                                                     |
| Grafana dashboard                | Not included; only the Prometheus data source is provisioned                               |
| Automated tests / GitHub Actions | Not present; pytest is a dependency, but no test files or workflow are checked in          |
 
## 20. Future Enhancements
 
### Automated testing
 
Implement pytest tests for preprocessing, model predictions, API responses, invalid inputs, and model loading.
 
### Continuous integration and delivery
 
Configure GitHub Actions to install dependencies and run automated tests on pushes and pull requests.
 
A later pipeline can build the Docker image and publish it to a container registry after all required checks pass.
 
### Deployment improvements
 
The repository includes Kubernetes Deployments and Services, probes, and resource requests and limits. Useful next steps are to validate the manifests in CI, add secrets/configuration management where needed, and document successful Minikube results.
 
### Monitoring and observability
 
Extend the service with structured logging and monitoring for:
 
- Request counts and errors.
- Response latency.
- Model availability.
- Input validation failures.
- Prediction distribution changes.
- Data drift and model performance when suitable ground-truth labels become available.
 
The API already exposes Prometheus metrics at `/metrics`, and the Kubernetes monitoring manifest configures a scrape target and Grafana data source. A dashboard, alert rules, and deployment evidence remain future work.
 
### Model governance
 
Improve model governance by recording model versions, training datasets, evaluation results, and approval criteria. Any model replacement should pass the agreed evaluation and validation checks before deployment.
 
### Model improvement
 
Explore threshold tuning, calibration, feature analysis, class-specific performance, and validation on independent datasets.
 
Any improvement should be assessed with a consistent evaluation strategy rather than relying solely on one test split.
 
## 21. Limitations
 
- The dataset contains only 303 observations, limiting the conclusions that can be drawn about generalization.
- The evaluation metrics depend on the dataset split and training configuration.
- The model may not generalize to populations or clinical settings that differ from the original dataset.
- Encoded features can be difficult to interpret without the dataset's original coding definitions.
- A high ROC-AUC does not guarantee that predicted probabilities are calibrated.
- The model has not been established as a clinically validated diagnostic system.
- Production-grade security, authentication, load testing, alerting, and operational monitoring require additional work. The repository provides basic API metrics and monitoring manifests, but no Grafana dashboard or deployment evidence.
- Docker and Kubernetes commands require their respective tools and a running local cluster; their configuration files alone do not establish successful deployment.
 
## 22. Disclaimer
 
This project is intended for educational, research, and software engineering purposes.
 
The model's output is a machine-learning classification based on a historical dataset. It is not a medical diagnosis, a substitute for professional clinical evaluation, or a recommendation for treatment.

 
## 23. Author
 
**Donifa Babu B M**
 
Software Developer | Cloud & Data Platform Engineer
 
The project demonstrates the integration of machine learning development, experiment tracking, model serialization, REST API development, and containerization practices in an end-to-end MLOps workflow.
 
---
 
## Acknowledgements
 
- UCI Machine Learning Repository for the Heart Disease dataset.
- scikit-learn for preprocessing, model training, and evaluation.
- MLflow for experiment tracking.
- FastAPI for API development.
- Docker for application containerization.
 
GitHub - 2025ae05820-donifa/mlops_heart_disease: MLOps Experimental Learning
MLOps Experimental Learning. Contribute to 2025ae05820-donifa/mlops_heart_disease development by creating an account on GitHub.
 
