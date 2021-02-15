from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    """
    function returns string on welcome page
    parameters: GET
    return: "Alive!"
    """
    return "Alive!"


@app.route("/article", methods=["GET", "POST"])
def predict_api():
    return "Welcome to API Deployment"


if __name__ == '__main__':
    app.run(debug=True)
