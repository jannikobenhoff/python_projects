import math
from math import pi
import argparse

import matplotlib.pyplot as plt
from mathutils import Vector
# import aspose.threed as a3d
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material
import numpy as np
import base64
import asyncio

import bpy
import bmesh


def create_cube_from_vertices(x, y, z, width, depth, height):
    # Define 8 vertices for the cube
    vertices = [
        (x, y, z),
        (x, y, z + height),
        (x + width, y, z),
        (x + width, y, z + height),
        (x, y + depth, z),
        (x, y + depth, z + height),
        (x + width, y + depth, z),
        (x + width, y + depth, z + height)
    ]

    # Define the 6 faces of the cube
    faces = [
        (0, 2, 6, 4),
        (1, 3, 7, 5),
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6)
    ]

    return vertices, faces


def create_gltf_block(coords, filepath: str):
    vertices = [(0, 0, 0), (0, 2, 0), (2, 2, 0), (2, 0, 0), (1, 1, 3)]
    faces = [(0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)]

    # vertices, faces = create_cube_from_vertices(0, 0,0 , 200,20,20)

    # Clear existing mesh objects (to start fresh)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Create a new mesh
    mesh = bpy.data.meshes.new(name="BlockMesh")
    obj = bpy.data.objects.new("BlockObject", mesh)

    # Link it to the scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Construct the block using bmesh
    bm = bmesh.new()
    for v in vertices:
        bm.verts.new(v)

    # Update the index table for bmesh
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for face_verts in faces:
        bm.faces.new([bm.verts[i] for i in face_verts])

    bm.to_mesh(mesh)
    bm.free()

    # Create a new material
    mat = bpy.data.materials.new(name="BlockMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)  # RGB and alpha

    # Assign material to the block
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    # Export the block as glTF
    override = {'active_object': obj, 'selected_objects': [obj], 'object': obj}
    bpy.ops.export_scene.gltf(override, filepath=filepath, export_format='GLTF_EMBEDDED')
    print("saved")


def create_gltf_cube_works(coords, center, filepath: str, height=0):
    # Clear existing mesh objects (to start fresh)
    height = float(height) * 1.6
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')

    # Add a cube
    # bpy.ops.mesh.primitive_cube_add(size=1)

    cube = bpy.context.scene.objects.get('Cube')
    print([v.co for v in cube.data.vertices])
    print(cube.data.vertices[0].co)
    # cube.data.vertices[0].co.x += 50
    # cube.data.vertices[1].co.x += 50
    # cube.data.vertices[2].co.x += 50

    for i in range(len(cube.data.vertices)):
        cube.data.vertices[i].co.z = 0  # (0,0,0)
        cube.data.vertices[i].co.x = 0

    def lonlat_to_meters(lon, lat):
        """Converts lon/lat to Web Mercator (EPSG:3857) x/y"""
        origin_shift = 2 * math.pi * 6378137 / 2.0
        x = lon * origin_shift / 180.0
        y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        y = y * origin_shift / 180.0
        return (x, y)

    def compute_centroid(coords):
        """Compute the centroid of a set of coordinates."""
        sum_lon = sum([coord[0] for coord in coords])
        sum_lat = sum([coord[1] for coord in coords])
        return sum_lon / len(coords), sum_lat / len(coords)

    def parse_footprint_coords(coors_str):
        coords_list = coors_str.split(',')
        return [(float(coords_list[i]), float(coords_list[i + 1])) for i in range(0, len(coords_list), 2)]

    # Footprint data
    coors_list = coords.split(',')
    center = center.split(",")
    center_x, center_y = lonlat_to_meters(float(center[0]), float(center[1]))

    coors = [(float(coors_list[i]), float(coors_list[i + 1])) for i in range(0, len(coors_list), 2)]

    # Convert the coordinates from lon,lat to meters
    coordinate_meters = [lonlat_to_meters(lon, lat) for lon, lat in coors]

    # Adjust each coordinate to be centered around the origin
    normalized_coords = [(x - center_x, y - center_y) for (x, y) in coordinate_meters]

    bottom_vertices = [(x, y, 0) for (x, y) in normalized_coords]
    # Top vertices
    top_vertices = [(x, y, height) for (x, y) in normalized_coords]

    vertices = bottom_vertices + top_vertices

    # Create faces
    # Bottom face
    bottom_face = [i for i in range(len(normalized_coords))]

    # Top face
    top_face = [i + len(normalized_coords) for i in range(len(normalized_coords))]

    # Side faces
    side_faces = [(i, (i + 1) % len(normalized_coords), (i + 1) % len(normalized_coords) + len(normalized_coords),
                   i + len(normalized_coords)) for i in range(len(normalized_coords))]

    faces = [bottom_face, top_face] + side_faces
    # vertices, faces = create_cube_from_vertices(0, 0, 0, 100, 100, 100)
    meshBlock = bpy.data.meshes.new(name="BlockMesh")
    obj = bpy.data.objects.new("BlockObject", meshBlock)

    # Link it to the scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()
    for v in vertices:
        bm.verts.new(v)

    # Update the index table for bmesh
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for face_verts in faces:
        bm.faces.new([bm.verts[i] for i in face_verts])

    bm.to_mesh(meshBlock)
    bm.free()

    meshFassade = bpy.data.meshes.new(name="FassadeMesh")
    objFassade = bpy.data.objects.new("FassadeObject", meshFassade)

    # Link it to the scene
    bpy.context.collection.objects.link(objFassade)
    bpy.context.view_layer.objects.active = objFassade
    objFassade.select_set(True)

    bm = bmesh.new()
    for v in vertices:
        bm.verts.new(v)

    # Update the index table for bmesh
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for face_verts in side_faces:
        bm.faces.new([bm.verts[i] for i in face_verts])

    bm.to_mesh(meshFassade)
    bm.free()

    mesh = bpy.data.meshes.new(name="WindowsMesh")
    obj = bpy.data.objects.new("WindowsObject", mesh)

    # Link it to the scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()
    # for v in vertices:
    #     if v[2] > 0:
    #         v = Vector((v[0]*1.01, v[1]*1.01, v[2]*0.5))
    #     bm.verts.new(v)
    #
    # # Update the index table for bmesh
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    #
    # for face_verts in faces:
    #     bm.faces.new([bm.verts[i] for i in face_verts])

    scale_factor = 0.25
    count = 3

    for face in meshFassade.polygons:
        center = face.center
        print("Center", center)
        left_vertices = []
        right_vertices = []

        # offset_vertices = []
        #
        # for vert_index in face.vertices:
        #     vert = meshFassade.vertices[vert_index].co
        #     direction = vert - center
        #     offset = direction * scale_factor
        #     offset_vertices.append(center + offset * 2)
        #
        # # Interpolating middle vertices for left and right rectangles
        # left_vertices = [offset_vertices[0], offset_vertices[1], (offset_vertices[1] + center) ,
        #                  (offset_vertices[0] + center) ]
        # right_vertices = [(offset_vertices[1] + center) / 2, (offset_vertices[2] + center) / 2, offset_vertices[2],
        #                   offset_vertices[3]]
        # for x in range(len(left_vertices)):
        #     left_vertices[x] = left_vertices[x] * 1.01
        #
        # for x in range(len(right_vertices)):
        #     right_vertices[x] = right_vertices[x] * 1.01
        # Create the rectangles using the offset vertices

        # bm.faces.new([bm.verts.new(vert) for vert in left_vertices])
        # bm.faces.new([bm.verts.new(vert) for vert in right_vertices])

        for vert_index in face.vertices:
            vert = meshFassade.vertices[vert_index].co
            direction = vert - center
            offset = direction * scale_factor

            # If the vertex is on the left side of the center
            if vert.x <= center.x:
                left_vertices.append(center + offset * 2)
            else:
                right_vertices.append(center + offset * 2)

        # Slightly scale them outwards
        for x in range(len(left_vertices)):
            left_vertices[x] = left_vertices[x] * 1.01

        for x in range(len(right_vertices)):
            right_vertices[x] = right_vertices[x] * 1.01

        # Create the rectangles using the offset vertices
        print(len(left_vertices))
        print(len(right_vertices))
        if len(left_vertices) == len(right_vertices):
            total_vertices = [left_vertices[1], left_vertices[0], right_vertices[0], right_vertices[1]]
            bm.faces.new([bm.verts.new(vert) for vert in total_vertices])

        # bm.faces.new([bm.verts.new(vert) for vert in right_vertices])

    bm.to_mesh(mesh)
    bm.free()


    mat = bpy.data.materials.new(name="BlockColor")

    # # Set the material's diffuse color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (
        0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)  # RGB and alpha
    bsdf.inputs['Metallic'].default_value = 0.0

    mat2 = bpy.data.materials.new(name="BlockColor")
    mat2.use_nodes = True
    bsdf = mat2.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (
        0.3, 0.3, 0.3, 1.0)  # RGB and alpha
    bsdf.inputs['Metallic'].default_value = 0.0

    mat3 = bpy.data.materials.new(name="BlockColor")
    mat3.use_nodes = True
    bsdf = mat3.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (
        0.7, 0.7, 0.7, 0.1)  # RGB and alpha
    bsdf.inputs['Metallic'].default_value = 0.0

    if len(bpy.context.scene.objects.get('BlockObject').data.materials) == 0:
        bpy.context.scene.objects.get('BlockObject').data.materials.append(mat)
    else:
        bpy.context.scene.objects.get('BlockObject').data.materials[0] = mat

    if len(bpy.context.scene.objects.get('WindowsObject').data.materials) == 0:
        bpy.context.scene.objects.get('WindowsObject').data.materials.append(mat2)
    else:
        bpy.context.scene.objects.get('WindowsObject').data.materials[0] = mat2

    if len(bpy.context.scene.objects.get('FassadeObject').data.materials) == 0:
        bpy.context.scene.objects.get('FassadeObject').data.materials.append(mat3)
    else:
        bpy.context.scene.objects.get('FassadeObject').data.materials[0] = mat3

    # Export the cube as glTF
    # asyncio.sleep(1)

    override = {'active_object': cube, 'selected_objects': [cube], 'object': cube}
    bpy.ops.export_scene.gltf(override, filepath=filepath, export_format='GLTF_EMBEDDED')
    print("saved")


def main():
    parser = argparse.ArgumentParser(description="Create a glTF cube with given coordinates and save to a file.")

    # Define command-line arguments
    parser.add_argument("--coords", type=str, default="0,0,0", help="Coordinates for the cube as x,y,z")
    parser.add_argument("--center", type=str, default="0,0,0", help="Coordinates for the cube as x,y,z")
    parser.add_argument("--filepath", type=str, required=True, help="Path to save the glTF file.")
    parser.add_argument("--height", type=str, required=True, help="Path to save the glTF file.")

    args = parser.parse_args()

    # Convert the string coords to a tuple of floats
    # coords = tuple(map(float, args.coords.split(',')))

    # Call the function with the provided arguments
    # create_gltf_block(args.coords, args.filepath)
    create_gltf_cube_works(args.coords, args.center, args.filepath, args.height)


if __name__ == "__main__":
    main()

    # coors_str = "8.679853677749634,50.104665928181475,8.679932802915573,50.10457991138384,8.679963648319244,50.1045919537448,8.679998517036438,50.10455496648359,8.680080324411392,50.104585932564675,8.680103123188019,50.10456184783666,8.680261373519897,50.10462119946595,8.680239915847778,50.10464442399652,8.68032306432724,50.1046762501868,8.680289536714554,50.10471323735433,8.680310994386673,50.10472183901709,8.680321723222733,50.10471065685522,8.680325746536255,50.10471237718798,8.680247962474823,50.10480527506584,8.68013933300972,50.10476398714235,8.680132627487183,50.10477172863074,8.679853677749634,50.104665928181475"
    # center = "8.68010587901879,50.104671856017454"
    # create_gltf_cube_works(coors_str, center, "abc.gltf", 10)
