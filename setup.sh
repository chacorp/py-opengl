pip install easydict

apt-get update && apt-get install xvfb
xvfb-run bash

# sometimes it is downloaded in somewhere else
# ln -s /usr/local/lib/python3.6/dist-packages/easydict /usr/local/lib/python3.8/dist-packages/
# apt-get update && apt-get install xvfb
# xvfb-run bash
# MESA_GL_VERSION_OVERRIDE=3.3 python3 test_render.py
