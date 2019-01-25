from flask import Flask, jsonify
from flask import request
from SpamDetection.NNClassifier import NNClassifier

app = Flask(__name__)
clf = NNClassifier()

@app.route('/check', methods=['POST'])
def check():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    msg = request.get_json().get('msg')
    print(msg)
    isSpam = clf.make_predict(msg)
    return jsonify({'isSpam':str(isSpam)})

if __name__ == '__main__':
    app.run()


