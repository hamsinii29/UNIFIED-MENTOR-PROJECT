# UNIFIED-MENTOR-PROJECT
# E-commerce Furniture Price Prediction & IRIS Classification

## About the Projects
These projects were completed during my internship at **Unified Mentor**, where I worked on two machine learning applications:

1. **E-commerce Furniture Price Prediction** - A regression model to predict product sales.
2. **IRIS Classification** - A classification model to categorize iris flowers based on petal and sepal dimensions.

---

## Project 1: E-commerce Furniture Price Prediction

### Objective
Predict the number of products sold based on price, product title, and shipping details.

### Dataset
E-commerce furniture dataset containing product details, prices, and sales information.

### Technologies Used
- **Python**
- **Pandas, NumPy** for data preprocessing
- **Seaborn, Matplotlib** for data visualization
- **Scikit-learn** for machine learning (RandomForestRegressor)

### Key Steps
1. **Data Preprocessing:**
   - Removed unnecessary columns
   - Converted price column to numeric format
   - Handled missing values
   - Encoded categorical features
2. **Model Training:**
   - Used **RandomForestRegressor** for prediction
   - Split data into training and testing sets
3. **Evaluation Metrics:**
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score

---

## Project 2: IRIS Flower Classification

### Objective
Classify iris flowers into three species based on petal and sepal measurements.

### Dataset
IRIS dataset containing features of different iris species.

### Technologies Used
- **Python**
- **Pandas, NumPy** for data processing
- **Matplotlib, Seaborn** for visualization
- **Scikit-learn** for model training (KNN, Decision Tree, Random Forest, SVM, Logistic Regression)

### Key Steps
1. **Data Exploration & Visualization:**
   - Pair plots, box plots, and correlation heatmaps
2. **Data Preprocessing:**
   - Checked for missing values
   - Standardized data for better model performance
3. **Model Training & Selection:**
   - Applied multiple classification models
   - Used **cross-validation** to select the best model
4. **Hyperparameter Tuning:**
   - Used **GridSearchCV** for RandomForest optimization
5. **Model Evaluation:**
   - Accuracy score
   - Confusion matrix
   - Classification report

---

## Results & Insights
- **E-commerce Project:** Successfully predicted product sales, showcasing the impact of price and shipping on sales.
- **IRIS Classification:** Achieved high accuracy with RandomForestClassifier, proving its efficiency for multi-class classification.
- **Key Takeaways:** Data preprocessing, feature engineering, and hyperparameter tuning significantly enhance model performance.

---

## How to Run the Projects
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/project-repo.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebooks for each project:
   ```sh
   jupyter notebook
   ```
4. Open the respective notebook (`ecom-furniture.ipynb` or `IRIS.ipynb`) and execute the cells.





