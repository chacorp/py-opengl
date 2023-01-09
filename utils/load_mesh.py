import numpy as np
from plyfile import PlyData, PlyElement
# pip install plyfile

"""
Reference:
    https://github.com/halimacc/DenseHumanBodyCorrespondences/blob/6a6e2a7b87113fe535967bf78da80d2373f74e11/meshutil.py#L29
"""
def load_ply_mesh(mesh_path):
    data = PlyData.read(mesh_path)

    vertex_data = data['vertex'].data
    vertices = np.zeros([vertex_data.shape[0], 3], dtype=np.float32)
    for i in range(vertices.shape[0]):
        for j in range(3):
            vertices[i, j] = vertex_data[i][j]

    face_data = data['face'].data
    faces = np.zeros([face_data.shape[0], 3], dtype=np.int32)
    for i in range(faces.shape[0]):
        for j in range(3):
            faces[i, j] = face_data[i][0][j]

    return vertices, faces

def load_obj_mesh(mesh_path, vn=False):
    vertex_data = []
    vertex_normal = []
    face_data = []
    
    for line in open(mesh_path, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        if values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            vertex_normal.append(vn)
        elif values[0] == 'f':
            f = list(map(lambda x: int(x.split('/')[0]),  values[1:4]))
            face_data.append(f)
            
    vertices = np.array(vertex_data)
    vertex_normals = np.array(vertex_normal)
    faces = np.array(face_data)
    
    if vn:
        return vertices, vertex_normals, faces
    else:
        return vertices, faces

def load_mesh(mesh_path):
    if mesh_path.endswith('.ply'):
        vertices, faces = load_ply_mesh(mesh_path)
    elif mesh_path.endswith('.obj'):
        vertices, faces = load_obj_mesh(mesh_path)
    if np.min(faces) == 1:
       faces -= 1
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    return vertices, faces