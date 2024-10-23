<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Credit Card Fraud Detection</h1>

<h2>Overview</h2>
<p>This project focuses on detecting fraudulent transactions in a credit card dataset using various machine learning techniques. The dataset is highly imbalanced, with fraudulent transactions being a small minority. We employ anomaly detection algorithms such as Isolation Forest, One-Class SVM, Local Outlier Factor (LOF), DBSCAN, and a custom-built Autoencoder. Additionally, we apply supervised learning models (Random Forest and XGBoost) with Synthetic Minority Over-sampling Technique (SMOTE) to improve fraud detection.</p>

<h2>Table of Contents</h2>
<ol>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#project-workflow">Project Workflow</a>
        <ol>
            <li><a href="#exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</a></li>
            <li><a href="#feature-transformation">Feature Transformation</a></li>
            <li><a href="#anomaly-detection-models">Anomaly Detection Models</a></li>
            <li><a href="#supervised-learning-models">Supervised Learning Models</a></li>
        </ol>
    </li>
    <li><a href="#model-performance">Model Performance</a></li>
    <li><a href="#how-to-run-the-code">How to Run the Code</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
</ol>

<h2 id="project-structure">Project Structure</h2>
<pre>
.
├── creditcard.csv           # Input dataset (credit card transaction data)
├── main_code.ipynb          # Main Jupyter notebook containing the code
├── README.md                # This README file
</pre>

<h2 id="dependencies">Dependencies</h2>
<p>To run this project, you need the following Python libraries:</p>
<pre>
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow imbalanced-learn xgboost
</pre>
<p>Make sure you have Jupyter Notebook or JupyterLab installed to execute the notebook.</p>

<h2 id="dataset">Dataset</h2>
<p>The dataset used for this project is the <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud">Kaggle Credit Card Fraud Detection dataset</a>. It contains 284,807 transactions, with 492 fraudulent transactions.</p>

<h3>Features:</h3>
<ul>
    <li><code>Time</code>: Time elapsed since the first transaction in seconds.</li>
    <li><code>V1</code> to <code>V28</code>: Principal components obtained via PCA (anonymized).</li>
    <li><code>Amount</code>: The transaction amount.</li>
    <li><code>Class</code>: Target variable, 0 for non-fraud and 1 for fraud.</li>
</ul>

<h2 id="project-workflow">Project Workflow</h2>

<h3 id="exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</h3>
<p>We start by analyzing the dataset to understand the distribution of the target variable (<code>Class</code>), which is heavily imbalanced. A pie chart is used to visualize the class distribution.</p>
<pre>
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', ...)
</pre>
<p>Next, histograms for each feature are plotted to observe the distributions.</p>

<h3 id="feature-transformation">Feature Transformation</h3>
<p>Given the skewness in the data, we apply log transformations to features with high skewness to make them more Gaussian-like.</p>
<pre>
def log_transform_skewed(column):
    transformed = np.where(column >= 0, np.log1p(column), -np.log1p(-column))
    return transformed
</pre>

<h3 id="anomaly-detection-models">Anomaly Detection Models</h3>
<p>We implement four different anomaly detection models to identify fraudulent transactions:</p>
<ol>
    <li><strong>Isolation Forest</strong></li>
    <li><strong>One-Class SVM</strong></li>
    <li><strong>Local Outlier Factor (LOF)</strong></li>
    <li><strong>DBSCAN</strong></li>
</ol>
<p>Each model is evaluated using the ROC AUC score and a confusion matrix is plotted for visual evaluation.</p>
<pre>
iso_forest = IsolationForest(contamination=0.05, random_state=101)
iso_preds = iso_forest.fit_predict(X_scaled)
</pre>
<p>Additionally, an <strong>Autoencoder</strong> neural network is constructed to detect anomalies, focusing on reconstructing normal transactions and flagging high reconstruction errors as fraud.</p>
<pre>
autoencoder = build_autoencoder(input_dim=X_scaled.shape[1])
</pre>

<h3 id="supervised-learning-models">Supervised Learning Models</h3>
<p>After applying anomaly detection methods, we use supervised learning techniques with <strong>SMOTE</strong> to handle class imbalance:</p>
<ol>
    <li><strong>Random Forest</strong></li>
    <li><strong>XGBoost</strong></li>
</ol>
<p>We train these models on the oversampled data and evaluate their performance using classification metrics (precision, recall, f1-score) and ROC AUC score.</p>
<pre>
rf_model = RandomForestClassifier(random_state=101)
rf_model.fit(X_train_sm, y_train_sm)
</pre>

<h2 id="model-performance">Model Performance</h2>
<h3>Anomaly Detection Models:</h3>
<table>
    <tr>
        <th>Model</th>
        <th>ROC AUC Score</th>
    </tr>
    <tr>
        <td>Isolation Forest</td>
        <td>*0.76*</td>
    </tr>
    <tr>
        <td>One-Class SVM</td>
        <td>*0.73*</td>
    </tr>
    <tr>
        <td>LOF</td>
        <td>*0.75*</td>
    </tr>
    <tr>
        <td>DBSCAN</td>
        <td>*0.50*</td>
    </tr>
    <tr>
        <td>Autoencoder</td>
        <td>*0.81*</td>
    </tr>
</table>

<h3>Supervised Learning Models (after SMOTE):</h3>
<table>
    <tr>
        <th>Model</th>
        <th>ROC AUC Score</th>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>*0.99*</td>
    </tr>
    <tr>
        <td>XGBoost</td>
        <td>*0.99*</td>
    </tr>
</table>

<h2 id="how-to-run-the-code">How to Run the Code</h2>
<ol>
    <li>Download the dataset from the Kaggle link above and place it in the project directory.</li>
    <li>Run the <code>main_code.ipynb</code> notebook in a Jupyter environment.</li>
    <li>The code will execute various anomaly detection and supervised learning models for fraud detection.</li>
</ol>

<h2 id="conclusion">Conclusion</h2>
<p>In this project, we explored both unsupervised (anomaly detection) and supervised learning techniques to detect credit card fraud. Supervised models with oversampling (SMOTE) achieved excellent performance, demonstrating the power of addressing class imbalance effectively.</p>

</body>
</html>
