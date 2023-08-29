import io
import struct
import operator
import subprocess
import tempfile

from asyncify import asyncify
# from gltflib import (
#     GLTF, GLTFModel, Asset, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView, Accessor, AccessorType,
#     BufferTarget, ComponentType, GLBResource, FileResource)

from flask import Flask, jsonify, request, send_from_directory, send_file, make_response
import os
from io import BytesIO
import asyncio

import bpy
from flask_cors import CORS, cross_origin
from threed import create_3d, create_gltf_cube, create_cube_gltf

app = Flask(__name__)
CORS(app)

@app.route('/create_gltf', methods=['GET'])
@cross_origin()
def create_gltf():
    coords = request.args.get('coords')
    center = request.args.get('center')
    height = request.args.get('height')
    print("Center:", center)
    print(coords)
    cmd = ["python", "model_creator.py", "--coords", coords, "--center", center, "--height", height, "--filepath", "abc.gltf"]

    # Run the script using subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result)
    if result.returncode == 0:
        print("Success")
        with open("abc.gltf", "rb") as gltf_file:
            gltf_data = gltf_file.read()
        print(gltf_data)
        response = make_response(gltf_data)
        response.headers.set('Content-Type', 'model/gltf+json')
        response.headers.set('Content-Disposition', 'attachment', filename='abc.gltf')
        return response
        #return jsonify({"message": "Cube created successfully!"}), 200
    else:
        print("Nope")
        return jsonify({"error": result.stderr}), 500
    # if not coords:
    #     return jsonify({'error': 'coords parameter is missing'}), 400

    coords = [list(map(float, coord.split(','))) for coord in coords.split(';')][0]

    l = len(coords)
    cc = []
    for i in range(int(l/2)):
        cc.append([coords[i*2], coords[i*2+1]])


    coords = cc
    create_gltf_cube(coords, "model.gltf")
    print("data")
    #filename = create_3d(coords)
    #print("Coords:", filename)
    # gltf = create_cube_gltf()
    # gltf.save("model.gltf")

    with open("model.gltf", "rb") as gltf_file:
        gltf_data = gltf_file.read()
    print(gltf_data)
    response = make_response(gltf_data)
    response.headers.set('Content-Type', 'model/gltf+json')
    response.headers.set('Content-Disposition', 'attachment', filename='model.gltf')
    return response


    gltf = create_cube_gltf()

    # Create a temporary file to save the GLTF
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_name = temp_file.name
    gltf.save(temp_file_name)
    temp_file.close()

    # Read the file into a BytesIO buffer
    with open(temp_file_name, "rb") as f:
        buffer = BytesIO(f.read())

    # Cleanup: Remove the temporary file
    os.remove(temp_file_name)

    buffer.seek(0)  # Reset buffer's position to the beginning.
    return send_file(buffer, mimetype='model/gltf+json', as_attachment=True, download_name="cube.gltf")


@app.route('/', methods=['GET'])
@cross_origin()
def home_page():
    data = "Hello"
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
