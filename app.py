
from flask import Flask, request, render_template
import pickle
import pipeline.model as model

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():

    return "Alive!"


@app.route('/industry', methods=["GET", "POST"])
def predict_industry():
    return "Welcome to the API Deployment"
