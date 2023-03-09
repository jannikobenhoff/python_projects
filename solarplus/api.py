import json
import time
from helper import getApiResponse
import pandas as pd
from flask import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home_page():
    data_set = {"Home": 1}
    response = jsonify(data_set)
    return response.headers.add("Access-Control-Allow-Origin", "*")


@app.route('/verbrauch/', methods=['GET'])
def user_page():
    strom = int(request.args.get("strom"))
    ww = int(request.args.get("ww"))
    heiz = int(request.args.get("heiz"))
    pv = str(request.args.get("pv"))
    dach = int(request.args.get("dach"))

    print(strom, ww, heiz, pv, dach)

    j = getApiResponse(strom, ww, heiz, pv, dach)

    response = jsonify(j)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(port=6777)
