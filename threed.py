from math import pi

import bpy
import matplotlib.pyplot as plt
# import aspose.threed as a3d
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material
import numpy as np
import base64
import asyncio


def create_gltf_cube(coords, filepath: str):
    # Clear existing mesh objects (to start fresh)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')

    # Add a cube
    bpy.ops.mesh.primitive_cube_add(size=20)

    cube = bpy.context.scene.objects.get('Cube')
    if cube:
        bpy.context.view_layer.objects.active = cube
        cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    cube.select_set(True)

    # Create a new material
    mat = bpy.data.materials.new(name="CubeMaterial")

    # Set the material's diffuse color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (0.2, 0.1, 0.1, 0.1)  # RGB and alpha
    # bsdf.inputs['Metallic'].default_value = 0.0

    # Assign material to the cube
    if len(cube.data.materials) == 0:
        cube.data.materials.append(mat)
    else:
        cube.data.materials[0] = mat

    # Export the cube as glTF
    # asyncio.sleep(1)

    override = {'active_object': cube, 'selected_objects': [cube], 'object': cube}
    bpy.ops.export_scene.gltf(override, filepath=filepath, export_format='GLTF_EMBEDDED')
    print("saved")
    # bpy.ops.object.delete()

# create_gltf_cube([1], "test.gltf")
def create_cube_gltf():
    C = bpy.context

    bpy.ops.mesh.primitive_cube_add()

    # the most recent object added to the scene is the active object. You can assign active object to a variable
    cube = C.active_object
    # another way to assign object to variable is by its name
    cube = C.scene.objects["Cube"]
    # another way to assign object to variable is by its index (camera is index 0, light is index 1 of collection)
    cube = C.scene.objects[2]

    initialFrame = 1
    # here I play with some functions that can be performed on the object variable
    # set objects values at an initial frame
    cube.location = (0, 0, 0)
    cube.scale = (1, 1, 1)
    cube.rotation_euler = (0, 0, 0)

    # keyframe the initial values
    cube.keyframe_insert("location", frame=initialFrame)
    cube.keyframe_insert("scale", frame=initialFrame)
    cube.keyframe_insert("rotation_euler", frame=initialFrame)

    finalFrame = 60

    # set objects values at an final frame
    cube.location = (0, 0, 1)
    cube.scale = (2, 2, 2)
    cube.rotation_euler = (0, 0, pi / 2)

    # keyframe the final values
    cube.keyframe_insert("location", frame=finalFrame)
    cube.keyframe_insert("scale", frame=finalFrame)
    cube.keyframe_insert("rotation_euler", frame=finalFrame)
    bpy.ops.import_scene.gltf(filepath='haus.gltf')
    bpy.ops.object.mode_set( bpy.context.scene.objects[0],mode='OBJECT')

    bpy.context.active_object = bpy.context.scene.objects[0]
    bpy.ops.export_scene.gltf(filepath='model.gltf', export_format="GLTF_EMBEDDED")
    return
    bpy.ops.preferences.addon_enable(module="io_scene_gltf2")

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Create a new cube
    #bpy.ops.scene.new()
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,0))

    bpy.ops.import_scene.gltf(filepath="haus.gltf")

    # bpy.ops.object.select_all(action='DESELECT')

    # Get the object you want to export
    obj = bpy.context.scene.objects[0]

    obj.select_set(True)  # Select the object
    bpy.context.view_layer.objects.active = obj  # Set the active object

    # Define the path to save your glTF file
    output_path = "module.gltf"

    # Export the selected object to glTF
    bpy.ops.export_scene.gltf(filepath=output_path, use_selection=False)

    print(bpy.context.scene.objects)
    # Get the context object, print the name and its reference
    obj = bpy.context.scene.objects
    print(obj)

    # Get all imported objects and print their names
    objs = bpy.context
    #cube = bpy.context.object

    # Export the cube to a temporary .glb file
    temp_file_path = "/tmp/temp_cube.glb"
    bpy.ops.export_scene.gltf(filepath=temp_file_path, export_format='GLTF_SEPARATE')

    # Read the temporary .glb file with pygltflib
    with open(temp_file_path, "rb") as f:
        gltf_content = f.read()

    gltf = GLTF2().loads_glb(gltf_content)
    print(gltf)
    return gltf
    # Cube vertices, normals and indices
    vertices = [
        -1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, 1.0, -1.0,
        -1.0, 1.0, -1.0,
        -1.0, -1.0, 1.0,
        1.0, -1.0, 1.0,
        1.0, 1.0, 1.0,
        -1.0, 1.0, 1.0,
    ]

    indices = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 4, 5, 5, 1, 0,
        1, 5, 6, 6, 2, 1,
        2, 6, 7, 7, 3, 2,
        3, 7, 4, 4, 0, 3
    ]

    # Convert data to numpy arrays
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint16)

    # Create buffer
    buffer = Buffer(byteLength=vertices.nbytes + indices.nbytes)

    # Create buffer views
    vertex_buffer_view = BufferView(buffer=0, byteOffset=0, byteLength=vertices.nbytes, target=34962)
    index_buffer_view = BufferView(buffer=0, byteOffset=vertices.nbytes, byteLength=indices.nbytes, target=34963)

    # Create accessors
    vertex_max = [float(vertices[i::3].max()) for i in range(3)]
    vertex_min = [float(vertices[i::3].min()) for i in range(3)]

    # Create accessors
    vertex_accessor = Accessor(bufferView=0, byteOffset=0, componentType=5126, count=len(vertices) // 3, type="VEC3",
                               max=vertex_max, min=vertex_min)


    index_accessor = Accessor(bufferView=1, byteOffset=0, componentType=5123, count=len(indices), type="SCALAR")

    # Create mesh
    primitive = Primitive(attributes={"POSITION": 0}, indices=1)
    mesh = Mesh(primitives=[primitive])

    # Create node
    node = Node(mesh=0)

    # Create scene
    scene = Scene(nodes=[0])

    # Create gltf object
    gltf = GLTF2(
        asset={"version": "2.0"},
        scenes=[scene],
        nodes=[node],
        meshes=[mesh],
        buffers=[buffer],
        bufferViews=[vertex_buffer_view, index_buffer_view],
        accessors=[vertex_accessor, index_accessor]
    )
    vertex_buffer_data = np.array(vertices, dtype=np.float32).tobytes()
    index_buffer_data = np.array(indices, dtype=np.uint16).tobytes()

    combined_buffer_data = vertex_buffer_data + index_buffer_data
    combined_buffer_base64 = base64.b64encode(combined_buffer_data).decode('utf-8')

    # Assign binary data to the buffer
    buffer.uri = "data:application/octet-stream;base64," + combined_buffer_base64

    # Update buffer's byteLength to the length of combined_buffer_data
    buffer.byteLength = len(combined_buffer_data)

    return gltf




def create_3d(coordinates):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Create a mesh from the footprint coordinates
    mesh = bpy.data.meshes.new(name="Cube")
    obj = bpy.data.objects.new("Cube", mesh)

    # Create vertices and faces for the cube mesh
    vertices = []
    faces = []
    for coord in coordinates[0]:
        # for vertex in coord:
        vertices.append((coord[0], coord[1], 0.0))
        face_indices = [i for i in range(len(coordinates[0]))]
        faces.append(face_indices)

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Create a cube object
    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    # Set up export options
    export_path = "cube.gltf"

    bpy.ops.export_scene.gltf(filepath=export_path)

    return "cube.gltf"

# fig, ax = plt.subplots()
#
# ax.scatter([i[0] for i in coord[0]], [i[1] for i in coord[0]])
# for i, txt in enumerate(coord[0]):
#     ax.annotate(i, (txt[0], txt[1]))
# ax.set_title(len(coord[0]))
# plt.show()

# Clear existing objects
# bpy.ops.object.select_all(action='DESELECT')
# bpy.ops.object.select_by_type(type='MESH')
# bpy.ops.object.delete()
#
# # Create a cube mesh
# bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
# cube_object = bpy.context.object
#
# # Set up the scene
# scene = bpy.context.scene
# scene.collection.objects.link(cube_object)
#
# # Set up export options
# export_path = "cube.gltf"
#
# # Export as GLTF
# bpy.ops.export_scene.gltf(filepath=export_path)


# scene = a3d.Scene.from_file("cube.glb")
# scene.save("Output.gltf")
