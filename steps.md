How to reproduce HumanML3D Dataset:

Reference: [https://github.com/EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D)

1. Follow the github repo to set up the environment
	* When you set up the environment, you will likely encounter issues with body_visualizer and configer, and install==1.3.5, or some other issues. You can reference the "In the case of installation failure" part

2. Unzip the "texts.zip" inside "HumanML3D"

3. Download the **amass** datasets based on the **raw\_pose\_processing.ipynb** document, and download the two models listed in the github repo. Then, upload them to Greatlakes (they are bit large).
	* When you download the dataset, extract the directory from the unzipped file (ignore the licence.txt)
	* Make sure the directory name follows exactly what they have in the Jupyter notebook

4. run raw\_pose\_processing.ipynb:
	* Copy all codes before **Segment** section to a python file, set "cuda:0" in `torch.device()`
	* Create a slurm file and run the python file with GPU. This takes slightly less than 30 mins  
	* unzip "humanact12.zip" inside "pose_data"
	* Create "joints" directory and run the remaining code cells in **Segment** section

5. run motion_representation.ipynb:
	* Copy all codes to a python file
	* Since this one doesn't involve GPU by the given code, GPU isn't really necessary, set higher number of CPUs will make it faster. (4 CPUs ~ 25 mins)

6. run cal\_mean\_variance.ipynb:
	* Copy all codes to a python file
	* Run slurm, similar to the last one