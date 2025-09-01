from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbotIgesta import chatbot 

app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    user_historico = data.get("historico", "")
    resposta = chatbot(user_message, user_historico)
    return jsonify({"response": resposta})
