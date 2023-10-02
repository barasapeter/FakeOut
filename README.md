# Fake News Detector

This project contains a machine learning model to detect fake news articles.

## Usage

There are two main functions:

- `create_news_article()` - Creates a Pandas DataFrame representing a news article from the provided title, text, subject, and date.

- `predict_fake_news()` - Takes a DataFrame from `create_news_article()` and returns a statement predicting if the article is fake or real. 

The model is trained on datasets of real and fake news articles. It uses TfidfVectorizer to extract features from the text and a Logistic Regression classifier.

To use:

```python
from news_detector import create_news_article, predict_fake_news

title = "Some Article Title" 
text = "The article text..."
subject = "politics"
date = "2021-01-01"

article = create_news_article(title, text, subject, date)

prediction = predict_fake_news(article)
print(prediction)