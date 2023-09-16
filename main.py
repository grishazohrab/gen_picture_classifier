import json

from flask import Flask, request

from runner import ModelRunner

app = Flask(__name__)
runner_obj = ModelRunner()


@app.route("/generate_image", methods=['POST'])
def generate_image():
    data = json.loads(request.form.get("data"))
    res = runner_obj.run(data["conversation"])

    return {"result": res}