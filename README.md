# Fine-Tuning DistillBERT for Sentiment Analysis

## Overview
This repository contains a Jupyter Notebook for fine-tuning DistillBERT on a sentiment analysis dataset. The model is trained using TensorFlow and Hugging Face's `transformers` library to classify tweets into sentiment categories.

## Dataset
The dataset used for training is `Tweets.csv`, which contains airline-related tweets labeled with sentiment categories (`positive`, `neutral`, `negative`).

## Steps Covered

### 1. Data Preprocessing
- Load dataset (`Tweets.csv`)
- Check for missing values and class balance
- Convert text to lowercase
- Remove unnecessary columns
- Visualize word frequency using a Word Cloud

### 2. Tokenization
- Convert text into tokenized inputs (`input_ids`, `attention_mask`)
- Use Hugging Face `DistilBertTokenizer`
- Ensure proper padding and truncation

### 3. Feature Mapping
- Map tokenized inputs to a TensorFlow dataset format
- Prepare training and testing sets

### 4. Model Training
- Load `DistilBertForSequenceClassification`
- Define loss function and optimizer
- Train model using TensorFlow/Keras

### 5. Evaluation
- Predict sentiment on test data
- Compute accuracy, precision, recall, and F1-score
- Generate a classification report

## Requirements
To run this notebook, install the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn nltk tensorflow transformers scikit-learn tqdm plotly
```

## Running the Notebook
1. Clone this repository:

```bash
git clone https://github.com/awais-124/fine-tuning-distilbert.git
cd fine-tuning-distilbert
```

2. Run the Jupyter Notebook:

```bash
jupyter notebook CODE.ipynb
```
