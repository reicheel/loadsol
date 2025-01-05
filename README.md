# loadsol
Current project to automate loadsol cropping, processing, and analysis.

Standard Operating Procedure (SOP) for Using the DVJ Analysis Script
Objective
This SOP provides a step-by-step guide to set up the environment, execute the script, and modify it for specific datasets.
________________________________________
Prerequisites
1.	System Requirements:
o	Operating System: Windows/macOS/Linux
o	Python 3.9 or higher
o	At least 8GB of RAM
2.	Software Requirements:
o	Anaconda distribution for Python
o	Basic familiarity with terminal commands
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
o	The script will guide you through cropping the data manually using plots.
o	Select regions of interest using the mouse and confirm selections in the terminal.
o	You have to exit out of the plot to see the prompts in the terminal. 
________________________________________
Step 4: Outputs
1.	Cropped Data:
o	Saved as CSV files in the output directory with the _trimmed suffix.
2.	Metrics:
o	Saved as drop_vertical_jump_metrics.csv in the output directory.
3.	Plots:
o	Phase plots and normalized waveforms are saved as .png files in the output directory.
