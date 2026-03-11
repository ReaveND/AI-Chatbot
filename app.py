"""
app.py  —  Flask web server for the Deep Learning Chatbot.

Usage:
    1. Train the model first:  python train.py
    2. Start the server:       python app.py
    3. Open browser:           http://localhost:5000
"""

import os
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

_chatbot = None


def get_chatbot():
    global _chatbot
    if _chatbot is None:
        from chatbot import Chatbot
        _chatbot = Chatbot()
    return _chatbot


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    user_message = str(data.get("message", "")).strip()
    if not user_message:
        return jsonify({"response": "Please say something!"})

    # Limit input length to prevent abuse
    if len(user_message) > 500:
        return jsonify({"response": "Please keep your message under 500 characters."})

    try:
        bot = get_chatbot()
        response = bot.respond(user_message)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception:
        return jsonify({"response": "Something went wrong on my end. Please try again."})

    return jsonify({"response": response})


@app.route("/status")
def status():
    model_dir = "saved_model"
    trained = all(
        os.path.exists(os.path.join(model_dir, f))
        for f in ("encoder.pt", "decoder.pt", "vocab.pkl")
    )
    return jsonify({"trained": trained})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Deep Learning Chatbot Server")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, port=5000)
