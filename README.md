<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
</head>
<body>

<h1>Fake News Detection Using LSTM</h1>

<h2>Overview</h2>
<p>This project implements a deep learning model for detecting fake news using text from news articles. The model uses <strong>LSTM (Long Short-Term Memory)</strong> architecture and <strong>Word2Vec embeddings</strong> for text representation, with the goal of classifying news articles as either "Fake" or "True." The dataset used contains the URLs, headlines, and body text of news articles, along with labels indicating whether the news is fake or true.</p>

<h2>Project Structure</h2>
<ul>
    <li><strong>Data Loading and Analysis</strong>: The dataset is loaded, and an initial analysis is performed to visualize the distribution of the labels.</li>
    <li><strong>Data Preprocessing</strong>: Text data is cleaned, tokenized, and vectorized using Word2Vec for embedding.</li>
    <li><strong>Model Creation</strong>: A deep learning model with an LSTM layer is built to classify the news articles.</li>
    <li><strong>Model Training and Evaluation</strong>: The model is trained, validated, and evaluated on the test set.</li>
</ul>

<h2>Dependencies</h2>
<p>To run the project, you will need the following libraries:</p>

<pre>
<code>pip install keras nltk</code>
</pre>
<pre>
<code>pip install --upgrade tensorflow</code>
</pre>

<p>Other necessary libraries include:</p>
<ul>
    <li><code>pandas</code></li>
    <li><code>numpy</code></li>
    <li><code>gensim</code></li>
    <li><code>matplotlib</code></li>
    <li><code>seaborn</code></li>
    <li><code>sklearn</code></li>
    <li><code>tensorflow</code></li>
    <li><code>nltk</code></li>
</ul>

<h2>Dataset</h2>
<p>The dataset used in this project is from Kaggle and contains the following fields:</p>
<ul>
    <li><strong>URLs</strong>: The URLs of the news articles.</li>
    <li><strong>Label</strong>: The target label indicating whether the news is Fake (0) or True (1).</li>
    <li><strong>Headline</strong>: The headline of the news article.</li>
    <li><strong>Body</strong>: The body text of the news article.</li>
</ul>

<h2>How to Run the Project</h2>
<ol>
    <li><strong>Clone the Repository:</strong>
        <pre>
            <code>git clone https://github.com/Swandip7/Brainwave_Matrix_Intern.git</code>
        </pre>
        <pre>
            <code>cd Brainwave_Matrix_Intern</code>
        </pre>
    </li>
    <li><strong>Install Dependencies:</strong> 
        Install the required Python libraries by running:
        <pre>
            <code>pip install -r requirements.txt</code>
        </pre>
    </li>
    <li><strong>Data Preprocessing:</strong> 
        The data preprocessing step involves:
        <ul>
            <li>Combining the headline and body text into a single field.</li>
            <li>Cleaning the text by removing stopwords and special characters.</li>
            <li>Tokenizing the text and padding sequences for input into the LSTM model.</li>
        </ul>
    </li>
    <li><strong>Training the Model:</strong> 
        The LSTM model is trained with the following configurations:
        <ul>
            <li><strong>Embedding size</strong>: 200</li>
            <li><strong>Window size</strong>: 2</li>
            <li><strong>Min count</strong>: 1</li>
            <li><strong>Max length</strong>: 1000 (padding for sequences)</li>
        </ul>
        To train the model:
        <pre>
            <code>model.fit(text_train_tok_pad, y_train, validation_split=0.2, epochs=30, batch_size=32)</code>
        </pre>
    </li>
    <li><strong>Model Evaluation:</strong> 
        After training, the model is evaluated using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and classification reports are also generated.
    </li>
    <li><strong>Plotting Results:</strong> 
        The project includes visualizations for training/validation accuracy and loss across epochs.
    </li>
</ol>

<h2>Key Functions</h2>
<ul>
    <li><strong><code>clean_text()</code></strong>: Cleans the text data by removing stopwords, special characters, and unnecessary symbols.</li>
    <li><strong><code>w2v_to_keras_weights()</code></strong>: Converts Word2Vec embeddings to Keras-compatible weights.</li>
    <li><strong><code>set_model()</code></strong>: Defines the LSTM model structure.</li>
    <li><strong><code>plot_loss_epochs()</code></strong>: Plots training and validation accuracy and loss.</li>
</ul>

<h2>Example Plots</h2>
<p>The following visualizations are generated during training:</p>
<ul>
    <li><strong>Accuracy and Loss Curves</strong>: A comparison of training and validation accuracy/loss over epochs.</li>
    <li><strong>Confusion Matrix</strong>: A heatmap showing the confusion matrix on the test set.</li>
</ul>

<h2>Results</h2>
<p>The model achieves reasonable accuracy in detecting fake news. The confusion matrix and classification report provide insight into its performance.</p>

<h2>Conclusion</h2>
<p>This project demonstrates how LSTM and Word2Vec can be effectively applied to text classification tasks such as fake news detection. It also highlights the importance of data preprocessing and feature engineering in natural language processing.</p>

</body>
</html>
