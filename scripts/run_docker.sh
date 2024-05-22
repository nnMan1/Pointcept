xhost local:root
docker run -d -ti \
		  -e DISPLAY=$DISPLAY -v --privileged \
		  -v /tmp/.X11-unix:/tmp\.X11-unix --net=host \
		  --shm-size 8G \
          	  -v $(pwd):/home \
			  -v /home/velibor/Data/scannet:/home/data/scannet \
		  	  -v /home/velibor/Data/scannet_instance_seg:/home/data/scannet_instance_seg \
			  -v /home/velibor/Data/AssemblyRepository/chunks/:/home/data/assembly/raw \
			  -v /home/velibor/Data/AssemblyRepository/scanns/:/home/data/assembly/scanns \
			  -v /home/velibor/Data/ABCDataset/scanns/:/home/data/ABCDataset/scanns \
			  -v /home/velibor/Data/ABCDataset/chunks/:/home/data/ABCDataset/raw \
			  -v /home/velibor/Data/Cetim/:/home/data/cetim/raw \
		  --gpus all -it --rm --name "pointcept" pointcept/pointcept:pytorch2.0.1-cuda11.7-cudnn8-devel
		  
