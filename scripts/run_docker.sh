xhost local:root
docker run -d -ti \
		  -e DISPLAY=$DISPLAY -v --privileged \
		  -v /tmp/.X11-unix:/tmp\.X11-unix --net=host \
		  --shm-size 32G \
          	  -v $(pwd):/home \
			  -v /home/velibor/Data/Fuselage/crops_250x250x250/:/home/data/fuselage/crops_250x250x250 \
		  --gpus all -it --rm --name "pointcept" pointcept/pointcept:pytorch2.0.1-cuda11.7-cudnn8-devel
		  
