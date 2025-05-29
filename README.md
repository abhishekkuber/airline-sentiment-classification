# airline-sentiment-classification

Repository for the case study for the Send AI internship.
https://colab.research.google.com/drive/1EEeQDbAkPOMNHqvn_zsoDGv54L0EOp4q?usp=sharing

The file structure should look like this: 
```main-folder/
├── data/
│   ├── glove.6B.100d.txt
│   ├── tweet_sentiment.train.jsonl
│   └── tweet_sentiment.test.jsonl
├── main.py
├── utils.py
├── unknown_vector.py
├── requirements.txt
└── README.md
```
- data: Contains the GloVe embeddings and sentiment dataset files.
- main.py: The main script to run your experiments.
- utils.py: Helper functions for preprocessing, getting embeddings, and confusion matrices.
- unknown_vector.py: Code for getting embedding for OOV words.
- requirements.txt: All required Python packages.
- README.md: Project setup and documentation.

*PS : To get an idea of the data, the design choices I made, the experiments and my interpretation of the results, continue with this README. To take a look at the plots or run the code, either go to the Colab file (easier) or follow the instructions in the next section.* 

## Setup Instructions

### 1. Clone the repository
```
git clone https://github.com/abhishekkuber/airline-sentiment-classification.git
cd airline-sentiment-classification
```
### 2. Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
### 3. Install the dependencies
```
pip install -r requirements.txt
```

### 4. Run the code
```
python main.py
```
The confusion matrices and all plots will be in a folder called images once you're done running the script.

## Task
Your team monitors social media to identify customer-experience issues in real-time. You have access to approximately 14k tweets directed at U.S. airlines, each labelled positive, neutral, or negative. Your goal is to build a simple, CPU-friendly NLP pipeline that transforms tweet texts into embeddings using GloVe vectors and trains a lightweight model to classify the sentiment of new tweets.

## Dataset
Contains two jsonl files, where each line contains a tweet and its sentiment (positive / neutral / negative).

| Set | Positive | Neutral | Negative| Total | 
| - | - | - | - | - | 
| Train | 1890 | 2479 | 7343 |  11712 | 
| Test | 1835 | 620 | 473 | 2928 |

In order to get the embeddings, I apply the following steps
- Convert emojis into their textual form (I love this movie 🔥 -> I love this movie fire) (OPTIONAL)
- Remove URLs
- Remove emojis
- Remove mentions
- Convert to lowercase
- Then tokenize and remove non alphabetical words
- Then get the word embedding
- Average the word embeddings of a sentence, and Voila ✨, you have a tweet embedding

## Data Exploration
![Class distribution per label](/assets/class_distribution.png) 
This is the class distribution for the train set. Right off the bat, we see that there are a lot of negative samples as compared to the other two classes. This might potentially skew the model results. 

Then I visualize the embedding space. I apply UMAP on the train and test set embeddings, reducing them from 100D to 2D, for better visualization.
![UMAP - Train Set](/assets/umap_train_set.png)
![UMAP - Test Set](/assets/umap_test_set.png)


The UMAP plots show significant overlap between the embeddings of all three sentiment classes (positive, negative, and neutral). This lack of clear separation in the embedding space suggests that:
1. The task is chalenging due to overlapping semantic features across the three classes
2. The model will need to rely on subtle linguistic patterns (maybe a little hard for smaller/ simpler models?)
3. High performance may require sophisticated feature extraction or architecture design

## Experiments
Before I get into the experiments, there are 3 design choices I make (not sure if it is the right word).
1. Downsampling : Since there are way more negative samples, I downsample for all classes to have equal number of training samples, and this number is equal to the smallest class in the dataset (in this case the positive class).
2. Handling OOV words : Zero vector, or we aggregate the embeddings of all tokens in the vocabulary (as per https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt/53717345#53717345).
3. Emojis to text : One thing that is common in tweets is the way people write, very informally and using emojis. Especially in the task of sentiment classification, I believe that these emojis aid the model in the task. So, I convert these emojis into their textual description. For example, "This movie is 🔥" -> "This movie is fire"


I basically ran a handful of classifiers, just off the shelf, and noted their performances. These are the performances for downsampling=false, unknown vector=average, and emojis to text=false.

| Method | F1 (Negative) | F1 (Neutral) | F1 (Positive) | Weighted F1 | 
| - | - | - | - | - | 
| Logistic Regression | **0.84** | 0.46 | 0.62 | 0.72 |
| SVM - linear kernel | 0.78 | 0.52 | 0.65 | 0.70 |
| SVM - poly kernel | 0.79 | 0.55 | **0.66** | 0.72 |
| SVM - rbf kernel | 0.80 | **0.58** | **0.66** | **0.73** |
| SVM - sigmoid kernel | 0.77 | 0.26 | 0.42 | 0.61 |
| Gaussian Naive Bayes | 0.79 | 0.40 | 0.41 | 0.65 |
| MLP Classifier | **0.84** | 0.47 | 0.62 | 0.72 |
| Decision Tree | 0.74 | 0.40 | 0.40 | 0.61 |
| (Shallow) Random Forest | 0.80 | 0.23 | 0.29 | 0.60 |
| Ridge Classifier | 0.83 | 0.39 | 0.56 | 0.69 |
| KNN Classifier | 0.81 | 0.33 | 0.43 | 0.65 |
| Passive Aggressive Classifier | 0.82 | 0.48 | 0.54 | 0.71 |

I run hyperparameter tuning on SVM RBF, which gives a better weighted F1, but reduced F1 on the neutral and negative classes. Details in the Colab file.

## Discussion

### Design Decisions
Lets start by taking a look at the effects of the design choices on the performance.
1. **OOV word vector** : For both approaches (zero vector vs. aggregated embeddings), most results remain unchanged. The only significant difference occurs in the weighted F1 scores for SVM with sigmoid kernel.
2. **emojis to text** : Converting emojis to text yields a slight increase in weighted F1 scores for some models. However, this comes at a cost: neutral class performance drops while negative class scores improve.
3. **Downsampling** : Contrary to my expectations, downsampling does not improve results. If anything, makes it a little worse.

### Confusion Matrices
- When examining the confusion matrices for each model, a clear pattern emerges: most models perform well on the negative class, likely due to it being the majority class in the training data. However, there is a consistent tendency to confuse neutral and positive examples with the negative class.
- Specifically, models like Gaussian Naive Bayes, MLP Classifier, Decision Tree, and Passive Aggressive Classifier often misclassify neutral instances as either neutral or negative, and positive instances as either positive or negative, indicating ambiguity in separating these classes. 
- In contrast, models such as the SVM with a sigmoid kernel, Random Forest, Ridge Classifier, and KNN Classifier predominantly predict the negative class regardless of the actual label, suggesting a strong bias toward the majority class.
- The SVM variants with linear, polynomial, and RBF kernels perform comparatively better, maintaining a more balanced classification across all three sentiments.

In summary, while most models reliably identify negative sentiment, distinguishing between neutral and positive remains a challenge.


### Metrics
- The SVM with RBF kernel achieved the highest overall performance (Weighted F1: 0.73), slightly outperforming other models. The top-performing classifiers were closely matched, indicating robust baselines for sentiment classification: SVM, RBF kernel (0.73); Logistic Regression (0.72); MLP Classifier (0.72); SVM, Poly kernel (0.72); Passive Aggressive (0.71).
- Neutral tweets were the most difficult to classify, with F1 scores significantly lower than positive/negative classes. SVM with a RBF kernel performed best (F1: 0.58), but all models struggled on this class,maybe due to ambiguous langauge.
- However, SVM with a RBF kernel delivered the most balanced performance across all three sentiments, making it the most reliable choice. Logistic Regression and MLP also performed well, particularly for the Positive and Negative Classes. 

I selected SVM with RBF to be my final model based on the results presented in the tables above. Ofcourse, to do so in real life, there would need to be some kind of statistical test done, to see whether the predictions between the classifiers are statistically significant (Something like McNemar's test for binary classification https://machinelearningmastery.com/mcnemars-test-for-machine-learning/ ). Ideally, my flow would be to find out which models' outputs arent statistically significant, and then do 5 fold Cross Validation and Hyperparameter Tuning on those models, which will give me THE best model, backed by experiments. However, for the case study, I skip this. 

### Quantitative Analysis of Misclassified Tweets
I conducted a qualitative analysis of SVM RBF's misclassifications. The results are interesting, and highlight a few limitations. **Please keep in mind this isn't a detailed analysis, I looked at around 10-20 examples and have made these conclusions.**

- The model relies on keywords like "thanks", "please", without understanding the underlying tone / context. A lot of the tweets express sarcasm / frustration without explicit negative words and in a non aggressive manner, and the  model misclassifies them. For example :
@AmericanAir pleaseeee find my suitcase. I want deoderant and a clean shirt for work tomorrow :( (predicted=positive, true=negative).

- Something weird, but a lot of tweets that contain words like 'flight', 'delay', 'boarding' are classified as negative (I assume due to a lot of tweets in the training set containing words like these being labeled negative)


- There are some tweets where it is obvious what the sentiment is but the model misclassifies them.
    - True: negative, Predicted: positive, Text: @JetBlue Usually I have such a great experience with you guys? Very, very unhappy with you right now.
    - True: negative, Predicted: positive, Text: @united you always surprise me with the awfulness of your airline. You guys suck. #worst


- However, I don't want to blame the model entirely, these are some examples I also found hard to classify. 
    - True: neutral, Predicted: negative, Text: @AmericanAir travel week, delays, Cancelled Flightlations, "if you want to learn more about the merger press 1", delay my connector in Chicago #deice
    - True: negative, Predicted: neutral, Text: @AmericanAir flight was 2488 out of EWR STOP AT DALLAS THEN TO LA. I need to be in la tonight! (I thought it was borderline neutral)
    - True: negative, Predicted: neutral, Text: @JetBlue nope. None to be found (No context at all, I am assuming it is a reply to a tweet, difficult to interpret on its own.)
    - True: positive, Predicted: neutral, Text: @JetBlue things happen it's ok just wish I was on the beach and not in the airport (Sentiment is unclear, or rather feels more negative.)
    - True: neutral, Predicted: positive, Text: @JetBlue I sure hope you guys get me to DC to speak tomorrow! & @JohnNosta & @United: I'm winning here with @JetBlue (Presence of "I'm winning here" likely influenced the positive prediction.)

TLDR : According to me, the model relies on superficial things like keywords to classify sentiment, and struggles with detecting tones, especially sarcasm and frustration. Notable bias towards classifying tweets containing words like flight, delay, boarding, etc. as negative. 


## Future Work
- I think the main limitation right now is using static glove embeddings, which fail to capture contextual meaning, which is VERY VERY important in a task like sentiment analysis. Switching to contextual embeddings (using CPU friendly libraries like SentenceBERT) could better handle nuances and improve performance. I also think that static embeddings is the reason why emoji to text fails.
- Currently, the final model is selected based on weighted F1, but proper statistical significance tests, cross validation, and hyperparameter tuning should be implemented for more reliable comparisons.
- Class weights might help.
- If not the constraint of being CPU friendly, I would go for slightly advanced models, for example BERTForSequenceClassification, or use Adapters (makes it less resource intensive).
- Tweets are written very informally, containing a lot of slang, and wrong spellings (for example, 'you' is spelt as 'u'). Future work should handle this.