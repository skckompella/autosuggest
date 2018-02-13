from flask import Flask, request, jsonify
app = Flask(__name__)

from ngram_model import BiGramModel

@app.route("/autosuggest/")
def generate_suggestions():

    if "q" in request.args:
        q = request.args.get('q')
        model = BiGramModel()
        model.load_model()
        preds = model.predict(q)
        return jsonify(preds)
    else:
        return None
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
