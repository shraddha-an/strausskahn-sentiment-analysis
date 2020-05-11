# Sentiment Analysis of newspaper articles relating to alleged sexual assaults by former IMF Director Dominique Strauss-Kahn

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sb

import xml.etree.ElementTree as et

import re

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem.wordnet import WordNetLemmatizer

# ==============================================  Loading XML content into a Pandas DataFrame ========================================

# Parsing the XML into a Pandas dataframe
file = et.parse('nysk.xml')
root = file.getroot()
columns = ["DocID", "Source", "URL", "Title", "Summary", "Date"] # Setting the columns of the Dataframe

summary_list = []
dataset = pd.DataFrame(columns = columns)

for each_node in root:
    doc = each_node.find("docid").text
    source = each_node.find("source").text
    url = each_node.find("url").text
    title = each_node.find("title").text
    summary = each_node.find("summary").text
    date = each_node.find("date").text

    summary_list.append(summary)
    dataset = dataset.append(pd.Series([doc, source, url, title, summary, date],
                                             index = columns), ignore_index = True)
# Removing special characters
for i in range(len(dataset)):
    dataset['Title'][i] = re.sub(pattern = '[^a-zA-Z0-9]', repl = ' ', string = dataset['Title'][i])
    dataset['Summary'][i] = re.sub(pattern = '[^a-zA-Z0-9]', repl = ' ', string = dataset['Summary'][i])

# Text preprocessing to clean up data: Lemmatizing, Removing Stopwords & Punctuation
lemmar = WordNetLemmatizer()

for k in range(len(dataset)):
    main_words = (dataset['Summary'][k].lower()).split()
    main_words = [lemmar.lemmatize(p) for p in main_words if not p in set(stopwords.words('english'))]
    main_words = ' '.join(main_words)
    dataset["Summary"][k] = main_words

# Sentiment Analysis with Vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

sentiment_dict = dict()

for sentence in summary_list:
    senti = analyzer.polarity_scores(sentence)

    # Creating a dictionary of sentiment scores and their values
    # Setting up keys
    sentiment_dict.setdefault('Negative_Score', [])
    sentiment_dict.setdefault('Neutral_Score', [])
    sentiment_dict.setdefault('Positive_Score', [])
    sentiment_dict.setdefault('Compound_Score', [])

    # Appending values to the respective keys
    sentiment_dict['Negative_Score'].append(senti.get('neg'))
    sentiment_dict['Neutral_Score'].append(senti.get('neu'))
    sentiment_dict['Positive_Score'].append(senti.get('pos'))
    sentiment_dict['Compound_Score'].append(senti.get('compound'))

sentiment_df = pd.DataFrame.from_dict(sentiment_dict, orient = "columns")
sentiment_df.insert(loc = 0, column = "Article", value = [x for x in summary_list], True)

# Storing the sentiment dataframe as a CSV for easy perusal later on
sentiment_df.to_csv('sentiment_analysis.csv', index = False, header = True)

# Bonus Dataset creation
class_list = list()

for m in range(len(sentiment_df)):
    if sentiment_df['Neutral_Score'][m] > 0.5:
        class_list.append("Neutral")
    elif sentiment_df['Negative_Score'][m] > 0.5:
        class_list.append('Negative')
    else:
        class_list.append('Positive')

# Adding the Class Label as the last column to the sentiment dataframe
sentiment_df.insert(loc = 5, column = "Sentiment", value = class_list)


# =================================================   Visualizations   ===================================================
# Seaborn plot visualizations

# 1) Violin Plot
plt.figure(figsize=(8,8))
sb.set_style('darkgrid')
plt.title("Violin Plot of Sentiment Analysis")
sb.violinplot(x = sentiment_df.iloc[:, -1].values, y = sentiment_df.iloc[:, -2].values,
           data = sentiment_df, palette = sb.set_palette('magma', n_colors = 1))
plt.ylabel("Compound Scores")
plt.show()

plt.savefig('sentiment_plot_violin.png')

# 2) Boxen Plot
plt.figure(figsize=(8,8))
sb.set_style('darkgrid')
plt.title("Box Plot of Sentiment Analysis")
sb.boxenplot(x = sentiment_df.iloc[:, -1].values, y = sentiment_df.iloc[:, -2].values,
           data = sentiment_df, palette = sb.set_palette('magma', n_colors = 6))
plt.ylabel("Compound Scores")
plt.show()

plt.savefig('sentiment_plot_boxen.png')
