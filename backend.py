import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, flesch_kincaid_grade
from nltk.tokenize import sent_tokenize, word_tokenize

with open('txt files/SP1541 NA1 Original.txt', 'r', encoding='utf-8') as file:
    text1a = file.read()

with open('txt files/SP1541 NA1 Optimised (Min).txt', 'r', encoding='utf-8') as file:
    text1b = file.read()

with open('txt files/SP1541 NA1 Optimised (Max).txt', 'r', encoding='utf-8') as file:
    text1c = file.read()

with open('txt files/SP1541 NA2 Original.txt', 'r', encoding='utf-8') as file:
    text2a = file.read()

with open('txt files/SP1541 NA2 Optimised (Min).txt', 'r', encoding='utf-8') as file:
    text2b = file.read()

with open('txt files/SP1541 NA2 Optimised (Max).txt', 'r', encoding='utf-8') as file:
    text2c = file.read()

#test output
print(text1)

## Preliminary analysis - word counts, readability scores, sentiment compound scores
def analyze_text(text):
    # Word count
    word_count = len(text.split())

    # Readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    smog_index = textstat.smog_index(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    automated_readability_index = textstat.automated_readability_index(text)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)

    return {
        'word_count': word_count,
        'flesch_reading_ease': flesch_reading_ease,
        'smog_index': smog_index,
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'coleman_liau_index': coleman_liau_index,
        'automated_readability_index': automated_readability_index,
        'sentiment': sentiment
    }

texts = [text1a, text1b, text1c, text2a, text2b, text2c]

for i, text in enumerate(texts, start=1):
    analysis = analyze_text(text)
    print(f"Text {i}:")
    print("Word count:", analysis['word_count'])
    print("Flesch Reading Ease:", analysis['flesch_reading_ease'])
    print("SMOG Index:", analysis['smog_index'])
    print("Flesch-Kincaid Grade:", analysis['flesch_kincaid_grade'])
    print("Coleman-Liau Index:", analysis['coleman_liau_index'])
    print("Automated Readability Index:", analysis['automated_readability_index'])
    print("Sentiment:", analysis['sentiment'])
    print("\n")

import pandas as pd
import matplotlib.pyplot as plt

results = []

texts = [text1a, text1b, text1c, text2a, text2b, text2c]

for i, text in enumerate(texts, start=1):
    analysis = analyze_text(text)
    analysis['text_id'] = f"Text {i}"
    results.append(analysis)

df = pd.DataFrame(results)
df = df[['text_id', 'word_count', 'flesch_reading_ease', 'smog_index', 'flesch_kincaid_grade', 'coleman_liau_index', 'automated_readability_index', 'sentiment']]
df['sentiment_neg'] = df['sentiment'].apply(lambda x: x['neg'])
df['sentiment_neu'] = df['sentiment'].apply(lambda x: x['neu'])
df['sentiment_pos'] = df['sentiment'].apply(lambda x: x['pos'])
df['sentiment_compound'] = df['sentiment'].apply(lambda x: x['compound'])

# Drop the original 'sentiment' column
df = df.drop(columns=['sentiment'])

# Define a mapping of old text_id names to new text_id names
text_id_map = {
    'Text 1': 'Text 1a',
    'Text 2': 'Text 1b',
    'Text 3': 'Text 1c',
    'Text 4': 'Text 2a',
    'Text 5': 'Text 2b',
    'Text 6': 'Text 2c',
}

# Replace the text_id names using the mapping
df['text_id'] = df['text_id'].map(text_id_map)
df['series'] = df['text_id'].apply(lambda x: 'Text 1' if '1' in x else 'Text 2')
print(df)

import seaborn as sns

# Melt the DataFrame to a long format suitable for seaborn
df_long = df.melt(id_vars=['text_id', 'series'], var_name='metric', value_name='score')

# Create a FacetGrid for each series (Text 1 and Text 2)
g = sns.FacetGrid(df_long, col='series', sharey=False, height=5, aspect=1.5)

# Create bar plots for each series
g.map(sns.barplot, 'text_id', 'score', 'metric', palette='muted', order=df['text_id'].unique())

# Set titles and labels
g.set_axis_labels("Text ID", "Score")
g.set_titles(col_template="{col_name}")
g.add_legend()

plt.show()

import matplotlib.pyplot as plt
import numpy as np

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

# Plot word_count
g1 = sns.catplot(data=df, x='text_id', y='word_count', col='series', kind='bar', height=4, aspect=1.5)
g1.set_titles(col_template="{col_name}")
g1.set_xticklabels(rotation=45)
show_values_on_bars(g1.axes)
plt.show()

# Plot flesch_reading_ease
g2 = sns.catplot(data=df, x='text_id', y='flesch_reading_ease', col='series', kind='bar', height=4, aspect=1.5)
g2.set_titles(col_template="{col_name}")
g2.set_xticklabels(rotation=45)
show_values_on_bars(g2.axes)
plt.show()

# Plot sentiment_compiund
g3 = sns.catplot(data=df, x='text_id', y='sentiment_compound', col='series', kind='bar', height=4, aspect=1.5)
g3.set_titles(col_template="{col_name}")
g3.set_xticklabels(rotation=45)
show_values_on_bars(g3.axes)
plt.show()

## Word clouds
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all texts into a list
texts = [text1a, text1b, text1c, text2a, text2b, text2c]
text_ids = ["text1a", "text1b", "text1c", "text2a", "text2b", "text2c"]

# Create a function to generate and display a word cloud
def generate_word_cloud(text, text_id):
    wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=None, min_font_size=10).generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(text_id)
    plt.tight_layout(pad=0)
    plt.show()

# Generate word clouds for each text
for text, text_id in zip(texts, text_ids):
    generate_word_cloud(text, text_id)


## Identifying top 10 words within each text series 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

def clean_and_tokenize(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    cleaned_tokens = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return cleaned_tokens

texts_series_1 = [text1a, text1b, text1c]
texts_series_2 = [text2a, text2b, text2c]

tokens_series_1 = [clean_and_tokenize(text) for text in texts_series_1]
tokens_series_2 = [clean_and_tokenize(text) for text in texts_series_2]

all_tokens_series_1 = [token for tokens in tokens_series_1 for token in tokens]
all_tokens_series_2 = [token for tokens in tokens_series_2 for token in tokens]

counter_series_1 = Counter(all_tokens_series_1)
counter_series_2 = Counter(all_tokens_series_2)

common_words = set(counter_series_1.keys()) & set(counter_series_2.keys())

proportions_series_1 = {word: count / len(all_tokens_series_1) for word, count in counter_series_1.items() if word in common_words}
proportions_series_2 = {word: count / len(all_tokens_series_2) for word, count in counter_series_2.items() if word in common_words}

def common_words_within_series(series_tokens):
    common_words = set(series_tokens[0])
    for tokens in series_tokens[1:]:
        common_words &= set(tokens)
    return common_words

common_words_series_1 = common_words_within_series(tokens_series_1)
common_words_series_2 = common_words_within_series(tokens_series_2)

proportions_series_1 = {word: counter_series_1[word] / len(all_tokens_series_1) for word in common_words_series_1}
proportions_series_2 = {word: counter_series_2[word] / len(all_tokens_series_2) for word in common_words_series_2}

import matplotlib.pyplot as plt

def plot_top_proportions(proportions, title):
    top_proportions = dict(sorted(proportions.items(), key=lambda x: x[1], reverse=True)[:10])

    plt.figure(figsize=(10, 6))
    plt.bar(top_proportions.keys(), top_proportions.values())
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Proportion")
    plt.show()

plot_top_proportions(proportions_series_1, "Top 10 Words in Proportions Series 1")
plot_top_proportions(proportions_series_2, "Top 10 Words in Proportions Series 2")