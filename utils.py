import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix

UNKNOWN_VECTOR = np.array([ 0.05209883, -0.09711445, -0.1380765 ,  0.11075345, -0.02722748, -0.00326409,  0.03176443, -0.05076874,  0.15321693, -0.02367382, -0.0078552 ,  0.08436131, -0.08042031, -0.08836847, -0.01713637,  0.07352565, -0.16472325,  0.05473585,  0.15367231, -0.05284015, -0.16474274, -0.00894895, -0.13604094, -0.03889371, -0.09204532,  0.02874651,  0.02445944,  0.19419461, -0.03297978,  0.00509352,  0.0146906 , -0.1554301 ,  0.03542742, -0.02936257,  0.01372886, -0.0606757 ,  0.02025392, -0.14560148,  0.05823914,  0.01729455,  0.16282158,  0.18634756, -0.06337869,  0.1306742 , -0.11122588,  0.0272168 ,  0.03868013,  0.15675613,  0.01344932,  0.1942456 , -0.01218801,  0.03659216, -0.08235365, -0.24420363,  0.07523726,  0.46423653,  0.06318451,  0.0508127 , -0.38147202, -0.20739552,  0.03489431, -0.18234783,  0.09021272, -0.02504168, -0.22256528,  0.03382994, -0.13379364, -0.14375682, -0.11264054, -0.03744001,  0.06188852,  0.09650661,  0.08384212,  0.1964642 , -0.07446123,  0.00921882,  0.03034359, -0.02482695,  0.27563572,  0.02422197, -0.23416583, -0.0523523 ,  0.10200828, -0.03673672,  0.2940292 ,  0.05685116,  0.01759575,  0.07998175, -0.07554322,  0.14788596,  0.01690632,  0.07576851,  0.07596124, -0.10800065,  0.20829839, -0.07841395,  0.08663727,  0.12381283, -0.23434106, -0.00925518])
# UNKNOWN_VECTOR = np.array([0]*100)


# taken from https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def preprocess_tweet(text):
    text = re.sub(r"http[s]?\://\S+", "", text)
    text = deEmojify(text)  # remove emojis
    text = re.sub(r"@\S+", '', text) # remove mentions
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    # text = re.sub(r"#\S+", "",text) # remove hashtags
    text = text.lower()  # convert to lowercase
    return text


def get_sentence_embedding(text, glove_model):
    text = preprocess_tweet(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # remove non-alphabetic tokens
    embeddings = []
    for token in tokens:
        try:
            embeddings.append(glove_model[token])
        except:
            embeddings.append(UNKNOWN_VECTOR)

    if not embeddings:
        return UNKNOWN_VECTOR

    embeddings = np.vstack(embeddings)
    embeddings = np.array(embeddings)
    return np.mean(embeddings, axis=0).astype(np.float64)

def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def get_confusion_matrix(y_test, y_pred, model_name, label_encoder, downsample):
  cf_matrix = confusion_matrix(y_test, y_pred)
  group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

  row_percentages_values = []
  for row in cf_matrix:
      row_sum = np.sum(row)
      if row_sum == 0:
          row_percentages_values.extend([0] * len(row)) # Append 0 for percentage values
      else:
          row_percentages_values.extend([value / row_sum for value in row])

  # Create a normalized confusion matrix for the heatmap data
  cf_matrix_normalized = np.asarray(row_percentages_values).reshape(cf_matrix.shape)

  # Create percentage strings for annotation
  group_percentages_strings = ["{0:.2%}".format(value) for value in row_percentages_values]

  # Create labels for each cell combining count and percentage
  labels_list = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages_strings)]

  # Reshape the labels list into a 3x3 numpy array to match the cf_matrix shape
  labels = np.asarray(labels_list).reshape(cf_matrix.shape)

  # Get the class names from the label_encoder to use as axis labels
  class_names = label_encoder.classes_

  # Plot the heatmap
  plt.figure(figsize=(8, 6))
  sns.heatmap(cf_matrix_normalized, annot=labels, fmt="", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix (Row Normalized Percentages)")
  if downsample:
    plt.savefig(f'images/confusion_matrices/downsampled_{model_name}.png')
  else:
    plt.savefig(f'images/confusion_matrices/{model_name}.png')
  plt.close()