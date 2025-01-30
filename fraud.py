import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
from nltk.metrics import edit_distance

# 1. Data Preprocessing
def preprocess(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# 2. Autocomplete using Trie Data Structure
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._get_words(node, prefix)

    def _get_words(self, node, prefix):
        words = []
        if node.is_end_of_word:
            words.append(prefix)
        for char, child_node in node.children.items():
            words.extend(self._get_words(child_node, prefix + char))
        return words

# 3. Autocorrect using Levenshtein Distance
def autocorrect(word, dictionary):
    closest_word = min(dictionary, key=lambda dict_word: edit_distance(word, dict_word))
    return closest_word

# 4. Loading and Preprocessing the Data from 'creditcard.csv'
data = pd.read_csv('creditcard.csv')

# Check the data for basic understanding
print(data.head())

# Assuming the target column in 'creditcard.csv' is 'Class' (fraud detection label)
# We will not preprocess numerical features, only the text-based columns if any.
# For this example, we will work with any textual feature for preprocessing.

# For illustration, let’s assume there is a 'Description' column (adjust if necessary)
if 'Description' in data.columns:
    data['processed_description'] = data['Description'].apply(lambda x: preprocess(str(x)))
else:
    data['processed_description'] = ["Sample description"] * len(data)  # Fake column if it doesn't exist

# If 'Class' is the fraud detection column (binary), we will proceed with it.
X = data['processed_description'].apply(lambda x: " ".join(x))  # Join tokens back for vectorization
y = data['Class']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Logistic Regression Model for Prediction
lr_model = LogisticRegression()

# Train model
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Autocomplete Example (using Trie)
autocomplete_trie = Trie()
words = ["hello", "hi", "high", "hope", "hip", "house", "hero", "help"]
for word in words:
    autocomplete_trie.insert(word)

# Test autocomplete
prefix = "he"
predicted_words = autocomplete_trie.autocomplete(prefix)
print(f"Autocomplete suggestions for '{prefix}': {predicted_words}")

# 7. Autocorrect Example
dictionary = ["hello", "hi", "high", "hope", "hip", "house", "hero", "help"]
misspelled_word = "hipp"
corrected_word = autocorrect(misspelled_word, dictionary)
print(f"Autocorrect suggestion for '{misspelled_word}': {corrected_word}")

# 8. XGBoost Model for Prediction
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")

# 9. Plot ROC Curve
from sklearn.metrics import roc_curve, auc
y_prob = lr_model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
