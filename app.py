from flask import Flask, request, json

from TextSummarizer import Preprocess

app = Flask(__name__)
app.debug = True

@app.route('/summarize', methods=['POST'])
def summrize():
    text = request.json['text']
    serve = Preprocess()
    summary = serve.run_textmode(text)
    return json.dumps({'status': 'OK', 'summary': summary})




if __name__ == '__main__':
    app.run(host="0.0.0.0", use_reloader=False)