# Smart Maintenance: Predicting Equipment Failure Using Machine Learning

## Overview
Smart Maintenance is a proactive approach leveraging advanced technologies like Machine Learning (ML) to predict equipment failures before they occur. This helps prevent unplanned downtime, reduce costs, and improve safety by addressing potential issues early. ML algorithms analyze sensor data to identify patterns and predict failures, leading to improved maintenance accuracy over time.

### Key Benefits:
- **Reduced Downtime:** Minimize disruptions in operations.
- **Cost Savings:** Avoid unexpected repair costs.
- **Enhanced Efficiency:** Improve operational reliability.

Industries such as power, manufacturing, and transportation are adopting predictive maintenance to enhance performance and reliability.

## Project Features
- **Data Cleaning and Preprocessing:**
  - Removal of duplicates.
  - Visualization of data distributions and relationships.
- **Exploratory Data Analysis:**
  - Histograms, scatter plots, and box plots to analyze features.
  - Heatmaps to study feature correlations.
- **Outlier Detection:**
  - Quantile-based outlier removal.
- **Model Development:**
  - Built using TensorFlow/Keras for a deep learning approach.
  - Multiple hidden layers with activation functions like ReLU and Tanh.
- **Evaluation:**
  - Display of training and testing accuracy.
  - Visualization of loss and accuracy trends over epochs.

## Libraries Used
- **Pandas:** Data manipulation and preparation.
- **NumPy:** Numerical computations.
- **Matplotlib/Seaborn:** Visualization.
- **Scikit-learn:** Feature scaling and dataset splitting.
- **TensorFlow/Keras:** Model development and evaluation.

## Dataset
The dataset contains sensor readings from industrial equipment, along with a target variable indicating failure status.

### Key Columns:
- `footfall`: Number of users in the facility.
- `VOC`: Volatile organic compound levels.
- `fail`: Binary target indicating equipment failure.

## Steps to Reproduce
1. **Import Necessary Libraries:**
   - Pandas, NumPy, Matplotlib, Seaborn, TensorFlow/Keras, and Scikit-learn.
2. **Load and Explore the Dataset:**
   - Use `.info()`, `.nunique()`, and visualizations to understand the data.
3. **Data Cleaning:**
   - Remove duplicates and handle outliers.
4. **Exploratory Data Analysis:**
   - Generate histograms, scatter plots, box plots, and heatmaps.
5. **Preprocessing:**
   - Standardize features and split data into training/testing sets.
6. **Model Development:**
   - Build a sequential model with:
     - Input layer with L2 regularization.
     - Multiple hidden layers using ReLU and Tanh activations.
     - Output layer with Sigmoid activation for binary classification.
7. **Evaluation:**
   - Visualize accuracy and loss trends.
   - Print training and testing accuracy.

## Model Summary
The neural network model includes:
- **Input Layer:** 9 nodes.
- **Hidden Layers:**
  - Layer 1: 500 nodes, ReLU activation.
  - Layer 2: 100 nodes, ReLU activation.
  - Layer 3: 50 nodes, Tanh activation with dropout regularization.
- **Output Layer:** 1 node, Sigmoid activation.

### Performance
- **Training Accuracy:** ~93%
- **Testing Accuracy:** ~91%

## Visualization
- **Accuracy and Loss Plots:**
  - Track model performance across epochs.
- **Correlation Heatmap:**
  - Understand feature relationships.

## Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/smart-maintenance.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python script:
   ```bash
   python AME_Project_Smart_Maintenance.py
   ```

## Future Enhancements
- Integrate real-time data from IoT sensors.
- Experiment with other ML algorithms (e.g., Random Forest, XGBoost).
- Optimize model hyperparameters.

