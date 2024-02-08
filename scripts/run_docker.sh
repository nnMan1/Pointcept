xhost local:root
docker run -d -ti \
		  -e DISPLAY=$DISPLAY -v --privileged \
		  -v /tmp/.X11-unix:/tmp\.X11-unix --net=host \
		  --shm-size 8G \
          	  -v $(pwd):/home \
          	  -v /media/velibor/Data/PhD_data/PhD/PartNet/:/home/data/Partnet\
		  -v /home/velibor/Projects/data/ModelNet40/:/home/data/ModelNet40 \
          	  -v /media/velibor/Data/PhD_data/STL_Data/classes/:/home/data/LEAP_synth/raw/classes \
          	  -v /media/velibor/Data/PhD_data/STL_Data/artefacts/:/home/data/LEAP_synth/raw/artefacts \
          	  -v /media/velibor/Data/PhD_data/STL_Data/contexts/:/home/data/LEAP_synth/raw/contexts \
          	  -v /media/velibor/Data/PhD_data/theoreticalPoses/:/home/data/LEAP_synth/raw/theoretical_poses \
          	  -v /media/velibor/Data/PhD_data/metadata/:/home/data/LEAP/raw \
		  --gpus all -it --rm --name "pointcept" pointcept/pointcept:pytorch2.0.1-cuda11.7-cudnn8-devel
		  
