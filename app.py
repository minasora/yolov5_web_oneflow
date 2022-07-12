from flask import Flask, render_template, request, redirect
from PIL import Image
import argparse
import io
import os
from infer import infer
base_dir = os.environ.get("BASE_DIR", "")
app = Flask(__name__, static_url_path=base_dir + "/static")

@app.route(f"{base_dir}/", methods=["GET"])
def hello_world():
    return render_template("index.html")


@app.route(f"{base_dir}/", methods=['POST'])
def predict():
    file = request.files["file-in"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img.save('static/test.jpg')
    infer('static/test.jpg')
    return redirect('static/test.jpg')

if __name__ == '__main__':
    app.run('0.0.0.0')
