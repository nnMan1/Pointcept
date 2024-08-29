xhost local:root
docker run -d -ti \
		  -e DISPLAY=$DISPLAY -v --privileged \
		  -v /tmp/.X11-unix:/tmp\.X11-unix --net=host \
		  --shm-size 32G \
          	  -v $(pwd):/home \
          	  	  -v /media/velibor/2842653342650742/scannet/scannet/:/home/data/raw/scannet/ \
			  -v /home/velibor/Data/scannet:/home/data/scannet \
		  	  -v /home/velibor/Data/scannet_instance_seg:/home/data/scannet_instance_seg \
			  -v /media/velibor/2842653342650742/velibor_data/ABCDataset/chunks/:/home/data/ABCDataset/raw \
			  -v /media/velibor/2842653342650742/velibor_data/ABCDataset/scanns/:/home/data/ABCDataset/scanns \
			  -v /home/velibor/Data/AssemblyRepository/chunks/:/home/data/assembly/raw \
			  -v /home/velibor/Data/AssemblyRepository/scanns/:/home/data/assembly/scanns \
			  -v /home/velibor/Data/Fuselage/:/home/data/fuselage/ \
			  -v /home/velibor/Data/Cetim/:/home/data/cetim/raw/ \
		  --gpus all -it --rm --name "pointcept" registry.gitlab.com/pmf5/pmf_ai/computer_vision/3d/3d_deep_learning_models:pointcept
		  
