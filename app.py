from flask import Flask, render_template, request
from sentiment import Sentiment

app = Flask(__name__)
sentiment = Sentiment()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        result = sentiment.analyze_sentiment(sentence)
        return render_template("index.html", result=result, input_sent=sentence)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

