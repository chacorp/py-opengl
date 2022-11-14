import os
from os.path import exists, join, split
from glob import glob
import numpy as np
import cv2
from PIL import Image

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from easydict import EasyDict

def compute_face_norm(vn, f):
    v1 = vn[f:, 0]
    v2 = vn[f:, 1]
    v3 = vn[f:, 2]
    e1 = v1 - v2
    e2 = v2 - v3

    return np.cross(e1, e2)

def render():
    import time
    from easydict import EasyDict as edict
    start = time.time()
    
    mesh = edict()
    mesh.v = np.array([
        [ 0.5,  0.5, 0.0],  # top right
        [ 0.5, -0.5, 0.0],  # bottom right
        [-0.5, -0.5, 0.0],  # bottom left
        [-0.5,  0.5, 0.0]   # top left 
    ])
    mesh.f = np.array([
        [0, 1, 3],   # first triangle
        [1, 2, 3]    # second triangle
    ])
    mesh.vt = np.array([
        [ 0.5,  0.5],  # top right
        [ 0.5, -0.5],  # bottom right
        [-0.5, -0.5],  # bottom left
        [-0.5,  0.5]   # top left 
    ])
    mesh.ft = mesh.f
    mesh.vn = mesh.v

    image_path = "willbeback.jpg"
    # import pdb; pdb.set_trace()

    ### get angle from filename
    angle = 0

    rendered = main(mesh, 512, image_path, angle)
    rendered = rendered[::-1, :, :]
    # name = image_path.split('/')[-1].split('.')[0][:11]

    # make directory
    savefolder = 'output'
    
    if not exists(savefolder):
        os.makedirs(savefolder)

    savefile = join(savefolder, 'vsb_{:03}.png'.format(angle))

    print('Image rendered: {}, shape: {}'.format(savefile, rendered.shape))
    print(rendered.min(), rendered.max())
    Image.fromarray(rendered).save(savefile)

    print('Done')
    #########################################################################################################
    return

    
def LoadTexture(filename):
    texName = 0
    pBitmap = Image.open(filename)
    pBitmapData = np.array(list(pBitmap.getdata()), np.int8)
    texName = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texName)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, pBitmap.size[0], pBitmap.size[1], 0,
        GL_RGB, GL_UNSIGNED_BYTE, pBitmapData
    )
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
                  [   s,       0.0,          c,        0.0],
                  [ 0.0,       0.0,        0.0,        1.0]], dtype=np.float32)
    return R

def main(mesh, resolution, image_path, angle):
    ##### shape
    # print(quad.shape, indices.shape, vt.shape, ft.shape)
    # for i,j in zip(indices[30000:30100, :], ft[30000:30100,:]):
    #     print(i,j)
    quad, indices, vt, ft, vn = mesh.v, mesh.f, mesh.vt, mesh.ft, mesh.vn
    # quad, indices, vt, ft = mesh.v, mesh.f, mesh.vt, mesh.ft
    # import pdb; pdb.set_trace()
    # import pdb;pdb.set_trace()
    if not glfw.init():
        return
        
    # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    window = glfw.create_window(resolution, resolution, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        print('failed to open window!')
        return

    glfw.make_context_current(window)
    # add column filled with 0.0 # ????? 
    # quad = np.concatenate( (quad, np.zeros((quad.shape[0], 1))), axis=1 )

    new_v = quad[indices].reshape(-1, 3)
    new_vt = vt[ft].reshape(-1,2)
    new_vt = np.concatenate((new_vt, np.ones((new_vt.shape[0],1)) ), axis=1)
    new_vn = vn[indices].reshape(-1, 3)

    # white color : mask == 1
    # color = np.ones((quad.shape[0], 1))
    # color = np.random.rand(*quad.shape)

    # concatenate quad and color
    # quad = np.concatenate( (quad, color), axis=1)
    # quad = quad[indices].reshape(-1, 3)
    # quad = np.concatenate( (new_v, new_vt), axis=1)

    quad = np.concatenate( (new_v, new_vt, new_vn), axis=1)
    quad = np.array(quad, dtype = np.float32)

    garment_mask = np.zeros((resolution, resolution, 3), dtype=np.uint8)

    ## texture for reprojection ################
    texture = LoadTexture(image_path)

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    ##############################################

    vertex_shader_source = """
        #version 330
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 texcoord;
        layout(location = 2) in vec3 normal;
        
        //uniform mat4 transform;

        out vec4 worldPosition;
        out vec3 OutNormal;

        void main()
        {
            vec4 tempPosition = vec4(position, 1.0f);
            worldPosition = tempPosition;
            gl_Position   = tempPosition;

            OutNormal = vec3(texcoord.xy, 0.0) + normal * 0.000000001;
        }
    """

    fragment_shader_source = """
        #version 330
        in vec4 worldPosition;
        in vec3 OutNormal;

        out vec4 outColor;
        
        uniform sampler2D samplerTex; // Image_Coordinate texture

        void main()
        {          
            vec3 temp = worldPosition.xyz;
            temp = (temp + 1.0) * 0.5;

            //vec2 uv = worldPosition.xy;
            vec2 uv = OutNormal.xy;
                  
            outColor = worldPosition + texture(samplerTex, uv);
        }
    """


    # glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    # glBufferData(GL_ARRAY_BUFFER, 4*6*quad.shape[0], quad, GL_STATIC_DRAW)
    glBufferData(GL_ARRAY_BUFFER, 4*3*3*quad.shape[0], quad, GL_STATIC_DRAW)
    # glBufferData(GL_ARRAY_BUFFER, 4*3*quad.shape[0], quad, GL_STATIC_DRAW)

    vertex_shader = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    shader = shaders.compileProgram(vertex_shader, fragment_shader)

    # EBO = glGenBuffers(1)
    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    # glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*indices.shape[0]*indices.shape[1], indices, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, "position")
    # glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(0))
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 4*9, ctypes.c_void_p(0))
    # glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 4*3, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    texcoord = glGetAttribLocation(shader, "texcoord")
    # glVertexAttribPointer(texcoord, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(12))
    glVertexAttribPointer(texcoord, 3, GL_FLOAT, GL_FALSE, 4*9, ctypes.c_void_p(12))
    glEnableVertexAttribArray(texcoord)

    normal = glGetAttribLocation(shader, "normal")
    # glVertexAttribPointer(texcoord, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(12))
    glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 4*9, ctypes.c_void_p(24))
    glEnableVertexAttribArray(normal)


    glUseProgram(shader)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)

    glDepthMask(GL_TRUE)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)
    glShadeModel(GL_SMOOTH)
    glDepthRange(0.0, 1.0)

    # rotation_mat = rotation(np.eye(4, dtype=np.float32), angle*-18.0, 0.0, 1.0, 0.0)

    ### apply rotation
    # rotation_mat = y_rotation(angle*-18.0)
    rotation_mat = y_rotation(angle)
    # transform = glGetUniformLocation(shader, "transform")
    # glUniformMatrix4fv(transform, 1, GL_FALSE, rotation_mat)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, quad.shape[0])

        glfw.swap_buffers(window)

        # glReadBuffer(GL_FRONT)
        glReadBuffer(GL_BACK)

        pixels = glReadPixels(0, 0, resolution, resolution, GL_RGBA, GL_UNSIGNED_BYTE)

        a = np.frombuffer(pixels, dtype=np.uint8)
        a = a.reshape((resolution, resolution, -1))
        #a = a.transpose(1, 0, 2)[:, ::-1, :]
        garment_mask = a
        break

    glfw.terminate()
    return garment_mask

if __name__ == '__main__':
    render() # visibility render
    # print('Done')
