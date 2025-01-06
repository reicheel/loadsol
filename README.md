# loadsol
Current project to quickly crop, processing, and analyze loadsol data. Scripts built for DVJ and single leg hop. 

Use the rename script to pull the trial info from the ascii file to rename both the pdo and ascii files so you know what file you're working with an not random numbers. This script can run in most python enrvironments. 

Standard Operating Procedure (SOP) for Using the DVJ Analysis Script
Objective
This SOP provides a step-by-step guide to set up the environment, execute the script, and modify it for specific datasets.
________________________________________
Prerequisites
1.	System Requirements:
Operating System: Windows/macOS/Linux
Python 3.9 or higher

2.	Software Requirements:
Anaconda distribution for Python
Basic familiarity with terminal commands
________________________________________
Step 1: Installing Required Software
1.	Install Anaconda:
o	Download Anaconda from the official website.
o	Follow the installation instructions for your operating system.
2.	Create a Python Environment: Open a terminal or Anaconda Prompt and run:
conda create -n dvj_analysis python=3.9
conda activate dvj_analysis
3.	Install Dependencies: Run the following commands to install the required Python packages:
pip install pandas numpy matplotlib scipy
________________________________________
Step 2: Preparing the Script
1.	Download the Script: Place the script file (dvj.py) in your working directory.
2.	Set the Base Directory: Modify the BASE_DIR variable in the script to point to the directory containing your participant data:
Helpful hint: find the folder where your data is saved, right click and ‘Copy as path’
BASE_DIR = r"your_directory_path_here"
3.	Input File Names: Update the file_names list with the names of the files you wish to process:
file_names = ["file1.txt", "file2.txt", "file3.txt"]
4.	Output Directory: The script automatically saves outputs to a subdirectory under the base directory. Ensure you have write permissions for this directory.
This script was made with the following folder structure: main study folder -> subject folder -> loadsol data -> outputs folder and .txt files for loadsol data 
________________________________________
Step 3: Running the Script
1.	Open a terminal
2.	Activate conda environment
conda activate dvj_analysis
3.	Change directory to where script is located
Cd "C:\file path"
4.	Execute the Script:
python dvj_processing.py
5.	Follow Prompts:
The script will guide you through cropping the data manually using plots.
Select regions of interest using the mouse and confirm selections in the terminal.
You have to exit out of the plot to see the prompts in the terminal. 
________________________________________
Step 4: Outputs
1.	Cropped Data:
Saved as CSV files in the output directory with the _trimmed suffix.
2.	Metrics:
Saved as drop_vertical_jump_metrics.csv in the output directory.
3.	Plots:
Phase plots and normalized waveforms are saved as .png files in the output directory.
![ACLR_SLH_Waveform_Right](https://github.com/user-attachments/assets/4cb5da22-39cf-4418-86a7-dfa98207be15)
![Avg SLH Waveforms](https://github.com/user-attachments/assets/4163b02c-cdd1-44c6-8291-1ea22e0a59ca)
![Dvj t1_phases](https://github.com/user-attachments/assets/64f493ab-d493-4560-8136-bedf1dd37be9)




