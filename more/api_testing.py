import json
import time

from flask import *

app = Flask(__name__)

last_time = time.time()

set = []
@app.route('/', methods=['GET'])
def home_page():
    set.append((time.time() - last_time) / 3)
    data_set = {'Data1': time.time() - last_time, 'Data2': (time.time() - last_time) / 2, 'Data3': set, 'Time:': time.time()}
    json_dump = json.dumps(data_set)

    return json_dump


@app.route('/user/', methods=['GET'])
def user_page():
    user = str(request.args.get("user"))  # /user/?user=JANNIK
    data_set = {'Page': 'Home', 'User': user, 'Time': time.time()}
    json_dump = json.dumps(data_set)
    return json_dump


if __name__ == '__main__':
    # port 7777
    app.run(port=7777)
