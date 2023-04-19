from __future__ import print_function

import os
from os.path import exists, join, split
from glob import glob
import numpy as np

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import numpy as np
from PIL import Image
import cv2
from easydict import EasyDict


def compute_face_norm(vn, f):
    v1 = vn[f:, 0]
    v2 = vn[f:, 1]
    v3 = vn[f:, 2]
    e1 = v1 - v2
    e2 = v2 - v3

    return np.cross(e1, e2)

def LoadTexture(filename):
    pBitmap = Image.open(filename)
    pBitmap = pBitmap.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    glformat = GL_RGB # if pBitmap.mode == "RGB" else GL_RGBA
    pBitmap = pBitmap.convert('RGB') # 'RGBA
    pBitmapData = np.array(pBitmap, np.uint8)
    
    # pBitmapData = np.array(list(pBitmap.getdata()), np.int8)
    texName = glGenTextures(1)
    # import pdb; pdb.set_trace()    
    glBindTexture(GL_TEXTURE_2D, texName)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, pBitmap.size[0], pBitmap.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, pBitmapData
    )

    # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    
    ### Texture Wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    
    ### Texture Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    """        
        GL_NEAREST_MIPMAP_NEAREST: nearest mipmap to match the pixel size and uses nearest neighbor interpolation for texture sampling.
        GL_LINEAR_MIPMAP_NEAREST:  nearest mipmap level and samples that level using linear interpolation.
        GL_NEAREST_MIPMAP_LINEAR:  linearly interpolates between the two closest mipmaps & samples via nearest neighbor interpolation.
        GL_LINEAR_MIPMAP_LINEAR:   linearly interpolates between the two closest mipmaps & samples via linear interpolation.
    """
    
    # glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texName

def load_textures(filenames):
    texture = glGenTextures(len(filenames))
    for i, filename in enumerate(filenames):
        pBitmap = Image.open(filename)
        pBitmap = pBitmap.transpose(Image.Transpose.FLIP_TOP_BOTTOM) 
        # glformat = GL_RGB # if pBitmap.mode == "RGB" else GL_RGBA
        pBitmap = pBitmap.convert('RGB') # 'RGBA
        pBitmapData = np.array(pBitmap, np.uint8)
            
    
        glBindTexture(GL_TEXTURE_2D, texture[i])
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, pBitmap.size[0], pBitmap.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, pBitmapData
        )
    return texture

def load_texture(path):
    texture = glGenTextures(1)
    print(texture)
    glBindTexture(GL_TEXTURE_2D, texture)
    image = Image.open(path)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGB') # 'RGBA
    image_data = np.array(list(image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


def rotation(M, angle, x, y, z):
    angle = np.pi * angle / 180.0
    c, s = np.cos(angle), np.sin(angle)
    n = np.sqrt(x*x + y*y + z*z)
    x,y,z = x/n, y/n, z/n
    cx,cy,cz = (1-c)*x, (1-c)*y, (1-c)*z
    R = np.array([[cx*x + c,   cy*x - z*s, cz*x + y*s, 0.0],
                  [cx*y + z*s, cy*y + c,   cz*y - x*s, 0.0],
                  [cx*z - y*s, cy*z + x*s, cz*z + c,   0.0],
                  [0.0,        0.0,        0.0,        1.0]], dtype=M.dtype)

    return np.dot(M, R.T)

def y_rotation(angle):
    angle = np.pi * angle / 180.0
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[   c,       0.0,          s,        0.0],
                  [ 0.0,       1.0,        0.0,        0.0],
                  [  -s,       0.0,          c,        0.0],
                  [ 0.0,       0.0,        0.0,        1.0]], dtype=np.float32)
    return R

def z_rotation(angle):
    angle = np.pi * angle / 180.0
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[   c,        -s,        0.0,        0.0],
                  [   s,         c,        0.0,        0.0],
                  [ 0.0,       0.0,        1.0,        0.0],
                  [ 0.0,       0.0,        0.0,        1.0]], dtype=np.float32)
    return R

def normalize_v(V):
    V = (V-(V.max(0)+V.min(0))*0.5)/max(V.max(0)-V.min(0))
    
    # FLAME
    # V = V - V.mean(0)
    # V = V - V.min()
    # V = V / V.max()
    # V = (V * 2.0) - 1.0
    return V

def load_obj_mesh(mesh_path):
    mesh = EasyDict()
    vertex_data = []
    vertex_normal = []
    vertex_texture = []
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
        if values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            vertex_texture.append(vt)
        elif values[0] == 'f':
            f = list(map(lambda x: int(x.split('/')[0]),  values[1:4]))
            face_data.append(f)
    # mesh.v  = np.array(vertex_data)
    mesh.v  = normalize_v(np.array(vertex_data))
    mesh.vn = np.array(vertex_normal)
    mesh.vt = np.array(vertex_texture)
    mesh.f  = np.array(face_data) -1
    return mesh

def vertex_normal(v1, v2, v3):
    v1c = np.cross(v2 - v1, v3 - v1)
    v1n = v1c/np.linalg.norm(v1c)
    return v1n

def computeTangentBasis(vertex, uv):
    tangents = []
    tangents = np.zeros_like(vertex)
    # bitangents = []
    for idx in range(0, len(vertex)//3):
        
        # import pdb;pdb.set_trace()
        offset = idx*3
        v0 = vertex[offset]
        v1 = vertex[offset+1]
        v2 = vertex[offset+2]

        offset = idx*3
        uv0 =    uv[offset]
        uv1 =    uv[offset+1]
        uv2 =    uv[offset+2]
        #print v0,v1,v2
        deltaPos1 = np.array([v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]])
        deltaPos2 = np.array([v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]])

        deltaUV1 = np.array([uv1[0]-uv0[0], uv1[1]-uv0[1]])
        deltaUV2 = np.array([uv2[0]-uv0[0], uv2[1]-uv0[1]])

        f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0])
        tangent = (deltaPos1 * deltaUV2[1]   - deltaPos2 * deltaUV1[1]) * f
        # bitangent = (deltaPos2 * deltaUV1[0]   - deltaPos1 * deltaUV2[0]) * f

        tangents[offset]   = tangent
        tangents[offset+1] = tangent
        tangents[offset+2] = tangent
    # import pdb;pdb.set_trace()
    return tangents

def render(resolution=512):    
    # mesh = load_obj_mesh("R:\\SampleNRefine_Data\\renderpeople_obj\\rp_aaron_posed_003.obj")
    # mesh = load_obj_mesh("D:\\test\\py-opengl\\data\\obj\\0018_M201_H.obj")
    mesh = load_obj_mesh("R:\eNgine_visual_wave\engine_obj\M_012.obj")
    # scale = np.array([[1.0, 1.0, -1.0]]) * 1.90
    # trans = np.array([[0.0, 0.95, 0.0]])
    # mesh.v = mesh.v * scale + trans
    # mesh = load_obj_mesh("D:\\test\\py-opengl\\data\\FLAME\\FLAME_template.obj")
    mesh.v = mesh.v * 1.8
    
    # mesh = load_obj_mesh("D:\\Dataset\\[DigitalWardrobe_MGN]\\registered_objs\\0_registration.obj")
    # mesh.ft = mesh.f
    mesh.vn = np.zeros_like(mesh.v)
    
    # mesh.vt[:,1] = 1- mesh.vt[:,1]
    # mesh.vt[:,0] = 1- mesh.vt[:,0]
    # mesh.vt = mesh.vt * 2.0 - 1.0
    
    # import pdb;pdb.set_trace()
    
    #### if SMPL, use this
    mesh.ft = np.load(open('smpl_uvs/faces_uvs.npy', 'rb'))
    mesh.vt = np.load(open('smpl_uvs/verts_uvs.npy', 'rb'))
    
    # flame_uv = np.load(open('D:\\test\\py-opengl\\data\\FLAME\\FLAME_texture.npz')) ## FLAME PCA parameters
    

    ################## shader shader shader shader shader shader shader shader ##########################
    # image_path = "test_image.png"
    # image_path = "R:\\SampleNRefine_Data\\Textures\\renderpeople_741__512\\rp_aaron_posed_003.png"
    # image_path = "D:\\test\\py-opengl\\data\\obj\\0018_M201_H.png"
    
    # image_path = "D:\\test\\py-opengl\\vis_mask.png"
    # image_path = "D:\\test\\py-opengl\\vis_tex.png"
    # image_path = "D:\\test\\py-opengl\\vis_tex_b.png"
    # image_path = "D:\[EG_2023]\conference\T_M_012_001.jpg" ## texture map
    image_path = "D:\[EG_2023]\conference\T_M_012_001-512.jpg"
    # image_path = "D:\\[EG_2023]\\conference\\ren_output.png"
    # image_path = "D:\[EG_2023]\conference\T_M_012_001.png" ## rendered image
    # image_path = "D:\\test\\py-opengl\\output\\rendered_uv3.png" ## texture map
    
    normal_path = "D:\[EG_2023]\conference\T_M_012_001_norm.png"
    # image_path = "D:\\[EG_2023]\\conference\\tt0.png"
    # image_path = "D:\\test\\py-opengl\\data\\FLAME\\texture.png"
                
    rendered = main(mesh, resolution, image_path, normal_path, angle=0, timer=True)
    rendered = rendered[::-1, :, :]
    
    # mask = np.array(Image.open(image_path))[...,:3] / 255.0
    # vis_part = (np.array(Image.open("D:\[EG_2023]\conference\T_M_012_001.jpg").resize((1024,1024))) * mask).astype(np.uint8)
    # Image.fromarray(vis_part).save('test_vis.png')
    
    # name = image_path.split('/')[-1].split('.')[0] #[:11]

    # make directory
    savefolder = join('output')
    if not exists(savefolder):
        os.makedirs(savefolder)
    savefile = join(savefolder, 'rendered.png')

    Image.fromarray(rendered).save(savefile)
    return

def main(mesh, resolution, image_path, normal_path=None, angle=0.0, timer=False):
    if timer == True:
        import time
        start = time.time()
    
    v, f, vt, ft, vn = mesh.v, mesh.f, mesh.vt, mesh.ft, mesh.vn

    if not glfw.init():
        return

    # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    
    # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(resolution, resolution, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    
    # for face in f:
    #     v[face]
    new_v  = v[f].reshape(-1, 3)
    new_vt = vt[ft].reshape(-1,2)
    new_vt = np.concatenate((new_vt, np.zeros((new_vt.shape[0],1)) ), axis=1)
    
    # new_vn = vn[f].reshape(-1, 3)
    new_vn = np.zeros_like(new_v)
    # new_v1 = new_v[0::3]
    # new_v2 = new_v[1::3]
    # new_v3 = new_v[2::3]
    # new_vn[0::3] = vertex_normal(new_v1, new_v2, new_v3)
    
    # f_v= new_v.copy().reshape(-1, 3, 3)
    # fn = vertex_normal(f_v[:,0], f_v[:,1], f_v[:,2]) # len(new_vn) == len(f)
    # new_fn = fn[f].reshape(-1, 3)
    # # v0_i = np.where(new_v == v[0])[0]
    
    # for i in range(len(v)):
    #     v_i = np.where(new_v == v[i])[0]
    #     v_n = np.sum(new_fn[v_i], axis=0)
    #     new_vn[v_i] = v_n / np.linalg.norm(v_n)
    new_vtan = computeTangentBasis(new_v, new_vt)
    
    quad = np.concatenate( (new_v, new_vt, new_vn), axis=1)
    # quad = np.concatenate( (new_v, new_vt, new_vn, new_vtan), axis=1)
    quad = np.array(quad, dtype=np.float32)
    print(quad.shape)

    ############################################## shader ################
    vertex_shader_source   = open('shader.vs', 'r').read()
    fragment_shader_source = open('shader.fs', 'r').read()
    
    vertex_shader   = shaders.compileShader(vertex_shader_source,   GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader          = shaders.compileProgram(vertex_shader, fragment_shader)

    ############################################## buffer ################

    # VAO = glGenBuffers(1)
    # glBindVertexArray(VAO)


    # EBO = glGenBuffers(1)
    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    # glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*indices.shape[0]*indices.shape[1], indices, GL_STATIC_DRAW)
    
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 4*quad.shape[0]*quad.shape[1], quad, GL_DYNAMIC_DRAW)
    """
        GL_STREAM_DRAW:  the data is set only once and used by the GPU at most a few times.
        GL_STATIC_DRAW:  the data is set only once and used many times.
        GL_DYNAMIC_DRAW: the data is changed a lot and used many times.
    """
    
    # 4*3*3 : size of float * len(X,Y,Z) * len(pos, tex, nor)
    vertex_stride = 4 * quad.shape[1]
    
    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)
    
    texcoord = glGetAttribLocation(shader, "texcoord")
    glVertexAttribPointer(texcoord, 3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(texcoord)

    normal = glGetAttribLocation(shader, "normal")
    glVertexAttribPointer(normal,   3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(normal)
    
    # tangent = glGetAttribLocation(shader, "tangent")
    # glVertexAttribPointer(tangent,   3, GL_FLOAT, GL_FALSE, vertex_stride, ctypes.c_void_p(36))
    # glEnableVertexAttribArray(tangent)
    
    ############################################## texture map ###########
    # glEnable(GL_TEXTURE_2D)
    texture1 = load_texture(image_path)
    texture2 = load_texture(normal_path)
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1)
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture2)
        
    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, "texture1"), 0)
    glUniform1i(glGetUniformLocation(shader, "texture2"), 1)
    
    ############################################## render ################
    # glBindVertexArray(VAO)
    # glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    # glClearColor(1.0, 1.0, 1.0, 1.0)
    # glClearColor(0.5, .5, .5, 0.5)
    # glClearDepth(1.0)

    # glDepthMask(GL_TRUE)
    # glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_BLEND)
    # glEnable(GL_CULL_FACE)
    # glCullFace(GL_BACK)
    # glFrontFace(GL_CCW)
    # glShadeModel(GL_SMOOTH)
    # glDepthRange(0.0, 1.0)

    ############################################## camera ################
    # rotation_mat = rotation(np.eye(4, dtype=np.float32), angle*-18.0, 0.0, 1.0, 0.0)
    # rotation_mat = y_rotation(angle*-18.0)
    # rotation_mat = y_rotation(0)

    transform = glGetUniformLocation(shader, "transform")    
    # gltimey = glGetUniformLocation(shader, "timer_y")
    # gltimex = glGetUniformLocation(shader, "timer_x")

    i = 0
    while not glfw.window_should_close(window):
    # while not glfw.window_should_close(window) and i < 360:
        curr_time = (time.time()-start)
        # distance = 1.0
        # glUniform1f(gltimey, np.sin(curr_time) * distance)
        # glUniform1f(gltimex, np.cos(curr_time) * distance)
        
        rotation_mat = y_rotation(curr_time * -30)
        # rotation_mat = y_rotation(curr_time * 0)
        # rotation_mat = y_rotation(10)
        # rotation_mat = y_rotation(-i)
        # rotation_mat = y_rotation(348)
        
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)
        # glClearColor(0.5, .5, .5, 0.0)
        # glClearColor(0.0, .0, .0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glClear(GL_DEPTH_BUFFER_BIT)
        # glClear(GL_COLOR_BUFFER_BIT)
        
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # glActiveTexture(GL_TEXTURE0)
        # glBindTexture(GL_TEXTURE_2D, texture1)
        # glUniform1i(glGetUniformLocation(shader, "texture1"), 0)

        # glActiveTexture(GL_TEXTURE1)
        # glBindTexture(GL_TEXTURE_2D, texture2)
        # glUniform1i(glGetUniformLocation(shader, "texture2"), 1)
        
        # glDrawElements(GL_TRIANGLES, indices.shape[0]*3, GL_UNSIGNED_INT, None)
        # glDrawElements(GL_TRIANGLES, quad.shape[0], GL_UNSIGNED_INT, None)
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])

        glfw.poll_events()
        glfw.swap_buffers(window)

        glReadBuffer(GL_FRONT)
        # glReadBuffer(GL_BACK)

        pixels = glReadPixels(0, 0, resolution, resolution, GL_RGBA, GL_UNSIGNED_BYTE)
        a = np.frombuffer(pixels, dtype=np.uint8)
        a = a.reshape((resolution, resolution, 4))
        # # mask  = a[::-1, :, :3] / 255 
        # vis_part = (np.array(Image.open(image_path).resize((1024,1024))) * mask).astype(np.uint8)
        
        # Image.fromarray(a[::-1, :, :3]).save('test2/test_vis{:04}.png'.format(i))
        # i = i + 10
        
        # break

    glfw.terminate()
    return a

if __name__ == '__main__':
    render(resolution=1024)
