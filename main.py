import pandas as pd
import numpy as np
import re
import json
import os
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import umap

from utils import get_sentence_embedding, load_glove_model, get_confusion_matrix, preprocess_tweet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

DOWNSAMPLE = False
CURRENT_DIR = os.getcwd()
UNKNOWN_VECTOR = np.array([ 0.05209883, -0.09711445, -0.1380765 ,  0.11075345, -0.02722748, -0.00326409,  0.03176443, -0.05076874,  0.15321693, -0.02367382, -0.0078552 ,  0.08436131, -0.08042031, -0.08836847, -0.01713637,  0.07352565, -0.16472325,  0.05473585,  0.15367231, -0.05284015, -0.16474274, -0.00894895, -0.13604094, -0.03889371, -0.09204532,  0.02874651,  0.02445944,  0.19419461, -0.03297978,  0.00509352,  0.0146906 , -0.1554301 ,  0.03542742, -0.02936257,  0.01372886, -0.0606757 ,  0.02025392, -0.14560148,  0.05823914,  0.01729455,  0.16282158,  0.18634756, -0.06337869,  0.1306742 , -0.11122588,  0.0272168 ,  0.03868013,  0.15675613,  0.01344932,  0.1942456 , -0.01218801,  0.03659216, -0.08235365, -0.24420363,  0.07523726,  0.46423653,  0.06318451,  0.0508127 , -0.38147202, -0.20739552,  0.03489431, -0.18234783,  0.09021272, -0.02504168, -0.22256528,  0.03382994, -0.13379364, -0.14375682, -0.11264054, -0.03744001,  0.06188852,  0.09650661,  0.08384212,  0.1964642 , -0.07446123,  0.00921882,  0.03034359, -0.02482695,  0.27563572,  0.02422197, -0.23416583, -0.0523523 ,  0.10200828, -0.03673672,  0.2940292 ,  0.05685116,  0.01759575,  0.07998175, -0.07554322,  0.14788596,  0.01690632,  0.07576851,  0.07596124, -0.10800065,  0.20829839, -0.07841395,  0.08663727,  0.12381283, -0.23434106, -0.00925518])
# UNKNOWN_VECTOR = np.array([0]*100)

os.makedirs('images/confusion_matrices', exist_ok=True)
os.makedirs('images/data_exploration', exist_ok=True)



############################## LOADING DATA ##############################
glove_model = load_glove_model(f"{CURRENT_DIR}/data/glove.6B.100d.txt")

train_texts = []
train_labels = []
test_texts = []
test_labels = []

with open(f"{CURRENT_DIR}/data/tweet_sentiment.train.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        train_texts.append(data['text'])
        train_labels.append(data['label'])

with open(f"{CURRENT_DIR}/data/tweet_sentiment.test.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        test_texts.append(data['text'])
        test_labels.append(data['label'])

train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

# Get the sentence embeddings for the train and test sets
train_df['embedding'] = train_df['text'].apply(lambda x: get_sentence_embedding(x, glove_model))
test_df['embedding'] = test_df['text'].apply(lambda x: get_sentence_embedding(x, glove_model))


# Our choice, whether to DOWNSAMPLE the training set or not
if DOWNSAMPLE:
  train_df = train_df.groupby('label', group_keys=False).sample(n=1890, random_state=42) # we choose 1890 because the positive label has the lowest count = 1890 in the train set


############################## DATA EXPLORATION ##############################


# Plot the class distribution
train_df['label'].value_counts().plot(kind='bar', title='Class Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
if DOWNSAMPLE:
    plt.savefig('images/data_exploration/downsampled_class_distribution.png', dpi=300)
else:
    plt.savefig('images/data_exploration/class_distribution.png', dpi=300) 
plt.close()  


# Plot the histogram of length of tweets per label
train_df['text_len'] = train_df['text'].str.len()
axes = train_df['text_len'].hist(by=train_df['label'], bins=30, figsize=(12, 6), layout=(1, 3), sharex=True, sharey=True)

for ax, label in zip(axes.flatten(), train_df['label'].unique()):
    ax.set_title(f'Label: {label}')
    ax.set_xlabel('Text Length')
    ax.set_ylabel('Frequency')

plt.tight_layout()
if DOWNSAMPLE:
    plt.savefig('images/data_exploration/downsampled_text_length_histograms_by_label.png', dpi=300)
else:
    plt.savefig('images/data_exploration/text_length_histograms_by_label.png', dpi=300)
plt.close()


## Gets the most common bigrams per label 
# vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english')
# for sentiment in train_df['label'].unique():
#     vec = vectorizer.fit_transform(train_df[train_df['label'] == sentiment]['text'])
#     sum_words = vec.sum(axis=0)
#     words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
#     top_words = sorted(words_freq, key=lambda x: x[1], reverse=True)[:10]
#     print(f'\nTop bigrams for {sentiment}:')
#     for word, freq in top_words:
#         print(f'{word}: {freq}')


# UMAP plots on the embeddings
all_embeddings = np.vstack(train_df['embedding'].values)
all_labels = train_df['label'].values

label_to_color = {
    'positive': 'green',
    'negative': 'red',
    'neutral': 'blue'
}
point_colors = [label_to_color[label] for label in all_labels]

label_to_marker = {
    'positive': 'o',
    'negative': 's',
    'neutral': '^'
}

umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(all_embeddings)

plt.figure(figsize=(8, 6))
for sentiment in label_to_color:
    idx = train_df['label'] == sentiment
    plt.scatter(
        umap_embeddings[idx, 0], umap_embeddings[idx, 1],
        c=label_to_color[sentiment],
        label=sentiment,
        alpha=0.7
    )

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projections of the train set by sentiment")
plt.legend()
plt.grid(True)
plt.tight_layout()
if DOWNSAMPLE:
    plt.savefig('images/data_exploration/umap_downsampled_train_set.png', dpi=300)
else:
    plt.savefig('images/data_exploration/umap_train_set.png', dpi=300)
plt.close()


# UMAP plots on the test set
all_embeddings = np.vstack(test_df['embedding'].values)
all_labels = test_df['label'].values

umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(all_embeddings)

plt.figure(figsize=(8, 6))
for sentiment in label_to_color:
    idx = test_df['label'] == sentiment
    plt.scatter(
        umap_embeddings[idx, 0], umap_embeddings[idx, 1],
        c=label_to_color[sentiment],
        label=sentiment,
        alpha=0.7
    )

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projections of the test set by sentiment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/data_exploration/umap_test_set.png', dpi=300)
plt.close()


############################## CLASSIFICATION ##############################
X_train = np.vstack(train_df['embedding'].values)
X_test = np.vstack(test_df['embedding'].values)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['label'])
y_test = label_encoder.transform(test_df['label'])
models = []
results = []

###### Logistic Regression ######
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Logistic Regression : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}\n\n")
temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
models.append('Logistic Regression')
get_confusion_matrix(y_test, y_pred, "logistic_regression", label_encoder, DOWNSAMPLE)


###### SVM ######
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
  svm = SVC(class_weight='balanced', kernel=kernel, random_state=42)
  svm.fit(X_train, y_train)
  y_pred = svm.predict(X_test)
  print(f"KERNEL : {kernel}\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}\n\n")
  temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
  results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
  models.append(f'SVM - {kernel} kernel')
  get_confusion_matrix(y_test, y_pred, f"svm_{kernel}", label_encoder, DOWNSAMPLE)


###### Gaussian Naive Bayes ######
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Gaussian Naive Bayes : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
get_confusion_matrix(y_test, y_pred, "gaussian_naive_bayes", label_encoder, DOWNSAMPLE)
temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
models.append(f'Gaussian Naive Bayes')


###### MLP Classifier ######
clf = MLPClassifier(hidden_layer_sizes=(), max_iter=500, random_state=42) # just a shallow MLP, input -> output
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"MLP Classifier : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
get_confusion_matrix(y_test, y_pred, "mlp_classifier", label_encoder, DOWNSAMPLE)
temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
models.append(f'MLP Classifier')

###### Decision Tree ######
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Decision Tree : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
get_confusion_matrix(y_test, y_pred, "decision_tree", label_encoder, DOWNSAMPLE)
temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
models.append(f'Decision Tree')

###### (Shallow) Random Forest ######
clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Random Forest Classifier : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
get_confusion_matrix(y_test, y_pred, "random_forest", label_encoder, DOWNSAMPLE)
temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
models.append(f'(Shallow) Random Forest')


###### Ridge Classifier ######
clf = RidgeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Ridge Classifier : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
get_confusion_matrix(y_test, y_pred, "ridge_classifier", label_encoder, DOWNSAMPLE)
temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
models.append(f'Ridge Classifier')

###### KNN Classifier ######
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"KNN Classifier : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
get_confusion_matrix(y_test, y_pred, "knn_classifier", label_encoder, DOWNSAMPLE)
temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
models.append(f'KNN Classifier')

###### Passive Aggressive Classifier ######
clf = PassiveAggressiveClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Passive Aggressive Classifier : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
get_confusion_matrix(y_test, y_pred, "passive_aggressive_classifier", label_encoder, DOWNSAMPLE)
temp = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
results.append([temp['negative']['f1-score'], temp['neutral']['f1-score'], temp['positive']['f1-score'], temp['weighted avg']['f1-score']])
models.append(f'Passive Aggressive Classifier')

# PRINTING LIKE THIS SO IT IS EASIER TO COPY-PASTE TO THE README
for i, j in zip(models, results):
    print(f"| {i} | {j[0]:.2f} | {j[1]:.2f} | {j[2]:.2f} | {j[3]:.2f} |")

raise ValueError
#### SVM HYPERPARAMETER TUNING ######
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
}

svc = SVC(kernel='rbf', class_weight='balanced', random_state=42)

grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svc = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

y_pred = best_svc.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))