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

from easydict import EasyDict


def compute_face_norm(vn, f):
    v1 = vn[f:, 0]
    v2 = vn[f:, 1]
    v3 = vn[f:, 2]
    e1 = v1 - v2
    e2 = v2 - v3

    return np.cross(e1, e2)

def render():
    mesh = EasyDict({})
    # mesh.v, mesh.f, mesh.vt, mesh.ft, mesh.vn
    mesh.v = np.array([
        [ 0.5,  0.5, 0.0], # top right
        [ 0.5, -0.5, 0.0], # bottom right
        [-0.5, -0.5, 0.0], # bottom left
        [-0.5,  0.5, 0.0], # top left
    ])
    mesh.f = np.array([
        [0, 1, 3], # first triangle
        [1, 2, 3],  # second triangle
    ])
    mesh.vt = np.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])
    mesh.ft = mesh.f
    mesh.vn = np.zeros_like(mesh.v)

################## shader shader shader shader shader shader shader shader ##############################
    image_path = "test_image.png"
            
    rendered = main(mesh, 1024, image_path, angle=0, timer=True)
    rendered = rendered[::-1, :, :]

    name = image_path.split('/')[-1].split('.')[0] #[:11]

    # make directory
    savefolder = join('output')

    if not exists(savefolder):
        os.makedirs(savefolder)

    savefile = join(savefolder, '{}.png'.format(name))

    Image.fromarray(rendered).save(savefile)
    return


def LoadTexture(filename):
    texName = 0
    
    pBitmap = Image.open(filename)
    pBitmap = pBitmap.transpose(Image.Transpose.FLIP_TOP_BOTTOM) 
    glformat = GL_RGB if pBitmap.mode == "RGB" else GL_RGBA
    # pBitmap = pBitmap.convert('RGB') # 'RGBA
    pBitmapData = np.array(pBitmap, np.uint8)
    # pBitmapData = np.array(list(pBitmap.getdata()), np.int8)
    texName = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texName)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
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
    
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, pBitmap.size[0], pBitmap.size[1], 0, glformat, GL_UNSIGNED_BYTE, pBitmapData
    )
    glGenerateMipmap(GL_TEXTURE_2D)
    return texName

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

def main(mesh, resolution, image_path, angle, timer=False):
    if timer == True:
        import time
        start = time.time()
    
    quad, indices, vt, ft, vn = mesh.v, mesh.f, mesh.vt, mesh.ft, mesh.vn

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

    # print(quad.shape, indices.shape, vt.shape, ft.shape, vn.shape)
    new_v  = quad[indices].reshape(-1, 3)
    new_vt = vt[ft].reshape(-1,2)
    new_vt = np.concatenate((new_vt, np.zeros((new_vt.shape[0],1)) ), axis=1)
    new_vn = vn[indices].reshape(-1, 3)

    quad = np.concatenate( (new_v, new_vt, new_vn), axis=1)
    # quad = np.concatenate( (new_v, new_vt), axis=1)
    # quad = new_v
    #                   positions        colors
    # quad    = [-0.5,  0.5, 0.0, 1.0, 1.0, 1.0,  # Top-left
    #             0.5,  0.5, 0.0, 0.0, 1.0, 0.0,  # Top-right
    #             0.5, -0.5, 0.0, 0.0, 0.0, 1.0,  # Bottom-right
    #            -0.5, -0.5, 0.0, 1.0, 1.0, 1.0]  # Bottom-left
    # indices = [0, 1, 2,
    #             2, 3, 0]
    quad = np.array(quad, dtype=np.float32)
    # print (quad)

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
    
    ############################################## texture map ###########
    glEnable(GL_TEXTURE_2D)
    texture = LoadTexture(image_path)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture)
    
    ############################################## render ################
    glUseProgram(shader)
    # glBindVertexArray(VAO)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    # glClearDepth(1.0)

    # glDepthMask(GL_TRUE)
    # glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    # glEnable(GL_CULL_FACE)
    # glCullFace(GL_BACK)
    # glFrontFace(GL_CCW)
    # glShadeModel(GL_SMOOTH)
    # glDepthRange(0.0, 1.0)

    ############################################## camera ################
    # rotation_mat = rotation(np.eye(4, dtype=np.float32), angle*-18.0, 0.0, 1.0, 0.0)
    # rotation_mat = y_rotation(angle*-18.0)
    # rotation_mat = y_rotation(angle)

    transform = glGetUniformLocation(shader, "transform")    
    gltimey = glGetUniformLocation(shader, "timer_y")
    gltimex = glGetUniformLocation(shader, "timer_x")

    while not glfw.window_should_close(window):
        curr_time = (time.time()-start)
        distance = 0.1
        glUniform1f(gltimey, np.sin(curr_time) * distance)
        glUniform1f(gltimex, np.cos(curr_time) * distance)
        
        rotation_mat = z_rotation(curr_time * 360)
        glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glClear(GL_COLOR_BUFFER_BIT)

        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # glDrawElements(GL_TRIANGLES, indices.shape[0]*3, GL_UNSIGNED_INT, None)
        # glDrawElements(GL_TRIANGLES, quad.shape[0], GL_UNSIGNED_INT, None)
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])

        glfw.swap_buffers(window)
        glfw.poll_events()

        glReadBuffer(GL_FRONT)
        # glReadBuffer(GL_BACK)

        pixels = glReadPixels(0, 0, resolution, resolution, GL_RGBA, GL_UNSIGNED_BYTE)
        a = np.frombuffer(pixels, dtype=np.uint8)
        a = a.reshape((resolution, resolution, 4))
        #a = a.transpose(1, 0, 2)[:, ::-1, :]
        # break

    glfw.terminate()
    return a

if __name__ == '__main__':
    render()
