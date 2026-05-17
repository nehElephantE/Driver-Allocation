## DRIVER ALLOCATION


## 🎯 Business Problem

Ride-hailing platforms face a critical challenge: **efficiently matching drivers with ride requests**. Poor matches result in:

| Problem | Impact |
|---------|--------|
| Driver rejects ride request | Lost revenue, poor driver experience |
| Customer waits longer | Customer churn, negative reviews |
| Reduced platform efficiency | Lower utilization, increased costs |
| Poor customer experience | Brand damage, competitor switching |

**Our Solution:** An ML-powered system that predicts driver acceptance probability before dispatching, enabling optimal matching decisions.

---

## 💡 Solution Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Raw Data  │────▶│  Feature    │────▶│   Model     │────▶│  Prediction │
│   (CSV)     │     │  Engineering│     │  Training   │     │  (Probability)│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                                       │
                           ▼                                       ▼
                  ┌─────────────────┐                    ┌─────────────────┐
                  │  Distance calc  │                    │  ACCEPT/REJECT  │
                  │  Time features  │                    │  Recommendation │
                  │  Historical     │                    │  Confidence %   │
                  └─────────────────┘                    └─────────────────┘
```

---

## 🏗️ System Architecture
![System Architecture](driver_allocation.png)

## ✨ Features

### Core Features
- **Real-time Prediction** - Get instant driver acceptance probability
- **Batch Processing** - Upload CSV for bulk predictions
- **Interactive Dashboard** - Monitor model performance metrics
- **Analytics Page** - View trends, demand patterns, and KPIs
- **Save Results** - Export predictions to CSV/JSON (local mode)

### Model Features
- **Multiple Models** - Random Forest, XGBoost, LightGBM
- **Feature Importance** - Understand what drives predictions
- **Cross-Validation** - Robust model evaluation
- **Automatic Feature Engineering** - Distance, time, historical features

### UI Features
- **Professional Design** - Gradient cards, responsive layout
- **Interactive Charts** - Plotly visualizations
- **Dark/Light Mode** - Automatic theme detection
- **Mobile Responsive** - Works on all devices

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | Streamlit, Plotly, Altair |
| **Backend** | Python 3.9+ |
| **ML Models** | Scikit-learn, XGBoost, LightGBM |
| **Data Processing** | Pandas, NumPy, PyArrow |
| **Visualization** | Matplotlib, Seaborn |
| **Containerization** | Docker, Docker Compose |
| **Deployment** | Streamlit Cloud, Azure (optional) |
| **Development** | Make, Git |


### Local Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/nehElephantE/Driver-Allocation.git
cd Driver-Allocation
```

#### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Prepare Data Directory

```bash
# Create necessary directories
mkdir -p data/raw data/processed artifacts/models artifacts/predictions artifacts/metrics logs
```

#### Step 5: Place Your Data Files

Copy your CSV files to `data/raw/`:
- `booking.csv` - Order/booking information
- `participant.csv` - Driver participation data
- `test_data.csv` - Test data for predictions

#### Step 6: Run the System

```bash
# Check system status
python main.py --mode status

# Train models
python main.py --mode train

# Make predictions
python main.py --mode predict

# Launch UI
streamlit run app.py
```

### Docker Installation

#### Build Docker Image

```bash
docker build -t ride-hailing-ml .
```

#### Run Container

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/logs:/app/logs \
  ride-hailing-ml
```

#### Using Docker Compose

```bash
# Production mode
docker-compose up ride-hailing-ml

# Development mode (hot-reload)
docker-compose up dev

# With Jupyter notebook
mkdir -p notebooks
docker-compose --profile jupyter up jupyter
```

---

## 📖 Usage Guide

### Running the Training Pipeline

```bash
# Train all models (Random Forest + XGBoost)
python main.py --mode train

# Train specific models
python scripts/train_model.py --models random_forest xgboost

# With custom parameters
python scripts/train_model.py --random-seed 123 --test-size 0.15
```

**What happens during training:**
1. Loads raw data from `data/raw/`
2. Validates and cleans data
3. Parses timestamps
4. Engineers features (distance, time, historical)
5. Splits data into train/validation sets
6. Trains Random Forest and XGBoost models
7. Evaluates with cross-validation
8. Saves models to `artifacts/models/`
9. Saves metrics to `artifacts/metrics/`

### Running Predictions

```bash
# Make predictions using best model
python main.py --mode predict

# Use specific model
python scripts/predict.py --model xgboost

# Custom threshold
python scripts/predict.py --threshold 0.6

# Custom output location
python scripts/predict.py --output my_predictions.csv
```

**What happens during prediction:**
1. Loads trained model from `artifacts/models/`
2. Loads test data from `data/raw/test_data.csv`
3. Engineers features using saved transformer
4. Predicts acceptance probabilities
5. Applies threshold (default 0.5)
6. Saves results to `artifacts/predictions/`

### Launching Streamlit UI

```bash
# Standard launch
streamlit run app.py

# With specific port
streamlit run app.py --server.port 8501

# Network accessible
streamlit run app.py --server.address 0.0.0.0
```

**UI Pages:**
| Page | Description |
|------|-------------|
| Dashboard | Key metrics, model performance, quick prediction |
| Real-Time Prediction | Single driver-order prediction |
| Batch Prediction | Upload CSV for bulk predictions |
| Analytics | Trends, demand patterns, KPIs |
| Saved Outputs | View/download past predictions |
| About | System information and documentation |

### Makefile Commands

```bash
# Show all available commands
make help

# Local development
make clean          # Clean all outputs
make train          # Train models only
make predict        # Make predictions only
make all            # Run full pipeline (train + predict)
make fresh          # Clean then run full pipeline
make streamlit      # Launch Streamlit UI
make verify         # Check project structure

# Docker commands
make docker-build   # Build Docker image
make docker-run     # Run container locally
make docker-stop    # Stop running container
```

---

## 📊 Data Preparation

### Input Data Format

#### `booking.csv` - Order Information

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| order_id | string | Unique order identifier | ORD_12345 |
| customer_id | string | Customer identifier | CUST_6789 |
| booking_status | string | COMPLETED/CANCELLED | COMPLETED |
| trip_distance | float | Expected trip distance (km) | 5.2 |
| pickup_latitude | float | Pickup location latitude | 40.7128 |
| pickup_longitude | float | Pickup location longitude | -74.0060 |
| event_timestamp | datetime | Order creation time | 2024-01-15 14:30:00 |

#### `participant.csv` - Driver Participation

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| order_id | string | Associated order ID | ORD_12345 |
| driver_id | string | Driver identifier | DRV_4567 |
| participant_status | string | ACCEPTED/REJECTED/IGNORED | ACCEPTED |
| driver_latitude | float | Driver current latitude | 40.7160 |
| driver_longitude | float | Driver current longitude | -74.0100 |
| driver_gps_accuracy | int | GPS accuracy (meters) | 10 |

#### `test_data.csv` - Test Data (Same format as booking.csv)

### Data Requirements

- **No missing values** in critical columns (order_id, coordinates)
- **Valid coordinates** between -90 to 90 (lat) and -180 to 180 (lon)
- **Timestamps** in ISO format or standard datetime
- **File encoding**: UTF-8

---

## 🤖 Model Details

### Models Used

| Model | Type | Strengths | When to Use |
|-------|------|-----------|-------------|
| **Random Forest** | Ensemble | Interpretable, robust to outliers | Baseline, explainability needed |
| **XGBoost** | Gradient Boosting | High accuracy, handles missing data | Production, maximum performance |
| **LightGBM** | Gradient Boosting | Fast training, memory efficient | Large datasets, real-time |

### Feature Engineering

| Feature Category | Features Created | Impact |
|-----------------|------------------|--------|
| **Spatial** | Haversine distance (driver to pickup) | 35% |
| **Temporal** | Hour, day of week, rush hour flag | 25% |
| **Behavioral** | Driver acceptance rate, experience | 18% |
| **Interaction** | Distance-to-trip ratio | 12% |
| **GPS** | Accuracy, signal quality | 5% |

### Performance Metrics

| Metric | Random Forest | XGBoost | LightGBM |
|--------|---------------|---------|----------|
| Accuracy | 85.0% | 86.0% | 85.0% |
| Precision | 84.0% | 85.0% | 84.0% |
| Recall | 86.0% | 87.0% | 86.0% |
| F1-Score | 85.0% | 86.0% | 85.0% |
| AUC-ROC | 0.92 | 0.93 | 0.92 |

---

## 🎨 Streamlit UI Guide

### Dashboard
- **Key Metrics**: Match rate, response time, accuracy, efficiency gain
- **Model Performance**: Comparison charts for all models
- **Quick Prediction**: Test a single prediction instantly
- **Feature Importance**: Visual breakdown of what drives predictions

### Real-Time Prediction
1. Enter order details (pickup location, trip distance, time)
2. Enter driver details (current location, experience, acceptance rate)
3. Click "Calculate Match Probability"
4. View probability gauge and recommendation
5. Save result locally (if in local mode)

### Batch Prediction
1. Upload CSV file with required columns
2. Review data preview
3. Click "Process Batch"
4. View summary statistics and detailed results
5. Download results as CSV
6. Optionally save to local disk

### Analytics
- Daily acceptance rate trends
- Hourly demand patterns
- Top pickup locations
- Performance over time

### Saved Outputs (Local Mode Only)
- List all saved prediction files
- Preview file contents
- Download saved files
- Delete all saved outputs

---
