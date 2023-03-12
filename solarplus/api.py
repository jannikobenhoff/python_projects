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
    """
    Dachart:
        0 - Satteldach
        1 - Pultdach
        2 - Flachdach
        3 - Walmdach
    PV:
        "crystSi" - Crystalline Silicon
        "CIS" - CIS
        "CdTe" - Cadmiumtellurid
    """
    dach = int(request.args.get("dach"))
    dachart = int(request.args.get("dachart"))
    dachneigung = int(request.args.get("dachneigung"))
    azimut = int(request.args.get("azimut"))
    strom = int(request.args.get("strom"))
    ww = int(request.args.get("ww"))
    heiz = int(request.args.get("heiz"))
    pv = str(request.args.get("pv"))

    print("Strom: {}, WW: {}, Heiz: {}, PV: {}, Dachfläche: {}, Dachart: {}, Dachneigung: {}, Azimut: {}"
          .format(strom, ww, heiz, pv, dach, dachart, dachneigung, azimut))

    j = getApiResponse(strom=strom, ww=ww, heiz=heiz, pv=pv, azimut=azimut, neigung=dachneigung, dachart=dachart, dach=dach)

    response = jsonify(j)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(port=6777)
