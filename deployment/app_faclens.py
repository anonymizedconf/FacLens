import sqlite3

from faclens import FacLens
from flask import Flask, render_template, request
from datetime import datetime
from flask_cors import CORS

faclens_model = FacLens()

app = Flask(__name__)
CORS(app)

def insert_data(question, model_name, feedback_score, time):
    conn = sqlite3.connect('feedback.sqlite')
    
    cur = conn.cursor()
    cur.execute('''
    INSERT INTO feedback (question, model_name, feedback_score, time)
    VALUES (?, ?, ?, ?)
    ''', (question, model_name, feedback_score, time))
    
    conn.commit()
    conn.close()

@app.route("/demo")
def home():
    return render_template("demo.html")

@app.route("/infer_answer")
def infer_answer():
    global faclens_model
    question = request.args.get('question')
    model_name = request.args.get('model_name')

    response = faclens_model.infer_answer(question, model_name)

    return {"answer": response}

@app.route("/infer_fact")
def infer_fact():
    global faclens_model
    question = request.args.get('question')
    model_name = request.args.get('model_name')

    response = faclens_model.infer_fact(question, model_name)

    return {"decision": response}

@app.route("/feedback")
def feedback():
    question = request.args.get('question')
    model_name = request.args.get('model_name')
    feedback_score = request.args.get('feedback_score')

    try:
        feedback_score = int(feedback_score)
    except ValueError:
        return {"status": "Error", "message": "Invalid feedback score"}, 400
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(question, model_name, feedback_score, time)
    
    insert_data(question, model_name, feedback_score, time)
    
    return {"status": "OK"}

app.run(host = "0.0.0.0", port = 10001, debug = False)