from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import re 
import string
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer 
import xgboost as xgb 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score 
import mlflow
import mlflow.sklearn

import dagshub
dagshub.init(repo_owner='iamprashantjain', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/mlops-mini-project.mlflow")
mlflow.set_experiment('bow vs tfidf')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
df.drop(columns=['tweet_id'], inplace=True)


# Filter only relevant sentiments (happiness and sadness)
final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

# Map 'happiness' to 1 and 'sadness' to 0
final_df['sentiment'] = final_df['sentiment'].replace(
    {
        'happiness': 1,
        'sadness': 0
    }
)

# Split the data into train and test sets
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Text preprocessing functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    return "".join([char for char in text if not char.isdigit()])

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    df.content = df.content.apply(lambda content : lower_case(content))
    df.content = df.content.apply(lambda content : remove_stop_words(content))
    df.content = df.content.apply(lambda content : removing_numbers(content))
    df.content = df.content.apply(lambda content : removing_punctuations(content))
    df.content = df.content.apply(lambda content : removing_urls(content))
    df.content = df.content.apply(lambda content : lemmatization(content))
    return df

# Apply text normalization on train and test data
train_data = normalize_text(train_data)
test_data = normalize_text(test_data)

# Extract features and labels
X_train = train_data['content'].values
y_train = train_data['sentiment'].values
X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Define vectorizers and algorithms
vectorizers = {
    'bow': CountVectorizer(),
    'tfidf': TfidfVectorizer()
}

algorithms = {
    'logistic_regression': LogisticRegression(max_iter=200),
    'multinomial_nb': MultinomialNB(),
    'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier()
}

# Start MLflow run for all combinations of algorithms and vectorizers
with mlflow.start_run(run_name="All combinations") as parent_run:
    for algo_name, algorithm in algorithms.items():
        for vec_name, vectorizer in vectorizers.items():
            with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                
                # Vectorize the training data
                X_train_transformed = vectorizer.fit_transform(X_train)
                X_test_transformed = vectorizer.transform(X_test)
                
                # Log parameters for MLflow
                mlflow.log_param("vectorizer", vec_name)
                mlflow.log_param("algorithm", algo_name)
                mlflow.log_param("test_size", 0.2)

                # Train the model
                model = algorithm
                model.fit(X_train_transformed, y_train)
                
                # Log model-specific parameters
                # Log model-specific parameters
                if algo_name == "logistic_regression":
                    mlflow.log_param("C", model.C)
                    mlflow.log_param("solver", model.solver)
                    mlflow.log_param("max_iter", model.max_iter)
                elif algo_name == "multinomial_nb":
                    mlflow.log_param("alpha", model.alpha)
                    mlflow.log_param("fit_prior", model.fit_prior)
                elif algo_name == "xgboost":
                    mlflow.log_param("learning_rate", model.learning_rate)
                    mlflow.log_param("max_depth", model.max_depth)
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("subsample", model.subsample)
                    mlflow.log_param("colsample_bytree", model.colsample_bytree)
                elif algo_name == "random_forest":
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("max_depth", model.max_depth)
                    mlflow.log_param("min_samples_split", model.min_samples_split)
                    mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
                    mlflow.log_param("max_features", model.max_features)
                elif algo_name == "gradient_boosting":
                    mlflow.log_param("learning_rate", model.learning_rate)
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("max_depth", model.max_depth)
                    mlflow.log_param("min_samples_split", model.min_samples_split)
                    mlflow.log_param("min_samples_leaf", model.min_samples_leaf)

                
                # Evaluate the model
                y_pred = model.predict(X_test_transformed)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:, 1])
                
                # Log evaluation metrics to MLflow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                # Optionally log classification report
                classification_rep = classification_report(y_test, y_pred, output_dict=True)
                mlflow.log_dict(classification_rep, "classification_report.json")
                
                # Log the model
                mlflow.sklearn.log_model(model, "model")
                
                # Print results for verification
                print(f"Algorithm: {algo_name}, Vectorizer: {vec_name}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print(f"ROC AUC: {roc_auc}")