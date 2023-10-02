'''
This script has a function `create_news_article` that takes the following data about a news article:
title: str
text: str
subject: str
date: str
Returns: DataFrame object.

the `predict_fake_news` function takes a DataFrame object. The object is to be returned by calling `create_news_article`
function.

You get a response by calling `predict_fake_news` and give a DataFrame object that comes from calling `create_news_article`.
Happy use, I hope I've made it as readable as possible!!
'''


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def create_news_article(title, text, subject, date):
    new_article = pd.DataFrame({'title': [title],
                                'text': [text],
                                'subject': [subject],
                                'date': [date]})
    return new_article


def predict_fake_news(news_article: pd.DataFrame):
    model = LogisticRegression()
    vect = TfidfVectorizer()

    fake_df = pd.read_csv('./datasets/fake.csv')
    fake_df['label'] = 0
    true_df = pd.read_csv('./datasets/true.csv')
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df])

    X_train, X_val, y_train, y_val = train_test_split(
        df[['title', 'text', 'subject']],
        df['label'], test_size=0.2
    )

    X_train = vect.fit_transform(X_train['title'] + ' ' + X_train['text'])
    X_val = vect.transform(X_val['title'] + ' ' + X_val['text'])

    model.fit(X_train, y_train)

    new_article_vect = vect.transform(
        news_article['title'] + ' ' + news_article['text'])

    pred = model.predict(new_article_vect)
    probability = model.predict_proba(new_article_vect)[0][0]
    accuracy = model.score(X_val, y_val)

    high_threshold = 0.8
    moderate_threshold = 0.5

    if probability >= high_threshold:
        statement = f"Our analysis suggest this news article appears to be entirely fake. The prediction accuracy is {accuracy * 100:.2f}%, approach with significant skepticism."
    elif probability >= moderate_threshold:
        statement = f"This article seems to mix both real and fake elements according to our assessment. The prediction accuracy is {accuracy * 100:.2f}%, so critical evaluation is important."
    else:
        statement = f"This news article appears to be entirely true based on my prediction. The accuracy is {accuracy * 100:.2f}%. This assessment does not deem final, you may want to do further verification."
    return statement


if __name__ == '__main__':
    title = 'Scientists Discover New Species of Rainbow-Colored Butterflies'
    text = 'In a groundbreaking scientific discovery, researchers have identified a previously unknown species of butterflies that are uniquely rainbow-colored. These stunning butterflies, which have never been observed before, are believed to inhabit remote rainforests in an undisclosed location. The scientists describe them as "a mesmerizing spectacle of nature" and are excited to learn more about their behavior and habitat. The discovery has sparked interest worldwide among butterfly enthusiasts and nature lovers.'
    subject = 'scienceNews'
    date = 'September 30, 2023'

    new_article = create_news_article(title, text, subject, date)
    likelihood_statement = predict_fake_news(new_article)
    print(likelihood_statement)
