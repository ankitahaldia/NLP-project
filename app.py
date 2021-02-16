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
def article_description():
    return "Welcome to Deployment"


if __name__ == '__main__':
    app.run(debug=True)
