xhost local:root
docker run -d -ti \
		  -e DISPLAY=$DISPLAY -v --privileged \
		  -v /tmp/.X11-unix:/tmp\.X11-unix --net=host \
		  --shm-size 32G \
          	  -v $(pwd):/home \
			  -v /home/velibor/Data/ABCDataset/chunks/:/home/data/ABCDataset/raw \
			  -v /home/velibor/Data/ABCDataset/scanns/:/home/data/ABCDataset/scanns \
			  -v /home/velibor/Data/AssemblyRepository/chunks/:/home/data/assembly/raw \
			  -v /home/velibor/Data/AssemblyRepository/scanns/:/home/data/assembly/scanns \
			  -v /home/velibor/Data/Fuselage/:/home/data/fuselage/ \
		  --gpus all -it --rm --name "pointcept" registry.gitlab.com/pmf5/pmf_ai/computer_vision/3d/3d_deep_learning_models:pointcept 
		  
