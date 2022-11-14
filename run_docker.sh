# echo ${PWD}
xhost +local:docker
docker run --gpus all -it --rm \
  --net host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${PWD}:/render \
  -e DISPLAY=unix$DISPLAY \
  --workdir=/render \
  chacorp/opengl:latest
xhost -local:docker
