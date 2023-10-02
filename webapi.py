from flask import Flask, request, jsonify
import model

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    title = data['title']
    text = data['text']
    subject = data['subject']
    date = data['date']

    article = model.create_news_article(title, text, subject, date)
    prediction = model.predict_fake_news(article)

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
