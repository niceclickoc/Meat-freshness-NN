#!/bin/bash

# Enable access to X server from local container
echo "Allowing X11 access..."
xhost +local:docker

# Run the container
# --net=host: Share network stack (easier X11 communication on some setups)
# -v /tmp/.X11-unix:/tmp/.X11-unix: Map X11 socket
# -e DISPLAY=$DISPLAY: Pass display variable
# --device /dev/video0: Pass camera access
# -v $(pwd):/app: Optional - mount current dir to edit code live, OR rely on COPY in Dockerfile.
# We rely on the built image here.

echo "Starting Meat Freshness App in Docker..."
docker run --rm -it \
  --net=host \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --device="/dev/video0:/dev/video0" \
  meat-freshness-app
