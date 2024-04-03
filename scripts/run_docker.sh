xhost local:root
docker run -d -ti \
		  -e DISPLAY=$DISPLAY -v --privileged \
		  -v /tmp/.X11-unix:/tmp\.X11-unix --net=host \
		  --shm-size 8G \
          	  -v $(pwd):/home \
			  -v /home/velibor/Data/scannet:/home/data/scannet \
			  -v /home/velibor/Data/AssemblyRepository/scans/:/home/data/assembly/raw \
		  --gpus all -it --rm --name "pointcept" pointcept/pointcept:pytorch2.0.1-cuda11.7-cudnn8-devel
		  
