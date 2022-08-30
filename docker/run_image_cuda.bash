IMAGE=ros:noetic-ros-core-focal-cuda-415
docker run -it\
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$PWD/../../:/workspace/" \
    -v /dev:/dev \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --privileged \
    --net=host \
    ${IMAGE}

