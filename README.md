# DeepLabCut on HPC

This repository contains information on how to set up an analysis pipeline with DeepLabCut on Marvin.

# Setup instructions

## STEP 1: Getting access to Marvin GPU nodes
Adjust the SSH credential file to use the following address:
gpu.marvin.hpc.uni-bonn.de

This will allow you to log in to node login03 on Marvin, where you can use GPU acceleration for DLC.

When you now ssh into Marvin, you will see you are registered as username@login03

## STEP 2: Initialize Anaconda in your Marvin home directory
Once logged into Marvin, perform the following three commands:

module load Miniforge3 (-> this is a free Anaconda clone)

conda init (to initialize Anaconda)

source ~/.bashrc (to tell the current bash terminal it should update to use Anaconda)

You should now see the typical (base) appear on the left of the current terminal line.

## STEP 3: Install Deeplabcut in a virtual environment in your Marvin home directory
The next step is to copy the DEEPLABCUT.yaml from your machine to Marvin in order to create a DeepLabCut python environment.
1) Download the yaml from this github repo.
2) Open a new terminal and copy the yaml from your computer (here as an example from the Downloads folder) to your Marvin home directory via:

scp Downloads/DEEPLABCUT.yaml [USERNAME]@gpu.marvin.hpc.uni-bonn.de:/home/[USERNAME]

-> Replace [USERNAME] with the id you see when you log in to Marvin (the one next to @login03)

During this step, you can also already copy the files dlc_analysis.py and run_dlc_array.sh to the Marvin home directory using the same command.

scp /path/to/file [USERNAME]@gpu.marvin.hpc.uni-bonn.de:/home/[USERNAME]

Now, ssh back into Marvin, check if the yaml file (and the other files) is in your home directory (ls -all) and then create the environment via:
conda env create -f DEEPLABCUT.yaml

If there are no errors, you can now also check if the installation of the modules worked by activating DEEPLABCUT (conda activate DEEPLABCUT) and running:

python -c "import torch; print(torch.cuda.is_available())"

This should return "False", because you are checking whether CUDA (i.e. access to GPUs) is available on the login node of the cluster, which is not the case. 
Running the same command in a workspace (after "module load CUDA" would return "True").

You can also check if the installation of DeepLabCut worked (also after activating the environment of course) with:

python -c "import deeplabcut; print(deeplabcut.__version__)"

## STEP 4: Create a workspace on the cluster to store your project data in
Still on the cluster, create a workspace to put the video data that you want to analyse:

ws_allocate NAME DURATION (for example: ws_allocate DLCAnalysis 90 -> 90 means the workspace will be available for 90 days, then it will be automatically deleted. Don't worry, workspace durations can be extended and you can also just create new ones -> https://wiki.hpc.uni-bonn.de/en/marvin/workspaces)

## STEP 5: Create the project folder, labels and training dataset on your local device
Install DeepLabCut on your device and launch the GUI.

Create a DeepLabCut project folder on your local device.

! Make sure to copy all the videos that you want to work with into the project folder (i.e. select the tick box when creating the project) !

Now, it is time to:
- Extract frames
- Label the frames
- Create the training dataset

For these steps, you can simply use the Deeplabcut GUI.

## STEP 6: Copy the project folder to Marvin
Copy the entire project folder into the workspace using the command:

scp -r /path/to/local/folder [Username]@gpu.marvin.hpc.uni-bonn.de:/path/to/workspace(which you can find by running ws_list when ssh'd to Marvin)

## STEP 7: Train your DLC model!

To perform training, you need to tell the script where it should look for A) the project folder and B) the python file to execute.
Check the paths for both of these and copy them into the dlc_analysis.py file using:

nano dlc_analysis.py

Now, in order to train the DLC model, you need to run a SLURM job. The file "run_dlc_array.sh" is meant to be used twice. Once for the training, and later for the analysis. 
For the training, navigate to the home directory on Marvin and run:

sbatch run_dlc_array.sh

You should get the message that your SLURM job was submitted.

You can now check the output and error status of the job by running:

tail jobid.out/jobid.err

You can also check the status of the job on the cluster by running:

squeue --me



