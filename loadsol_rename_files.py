import os
import shutil

# Paths to original files and destination for renamed files
original_dir = r"C:\Users\reicheel\OneDrive - University of North Carolina at Chapel Hill\__Study Data\cam_eeg_data\sub-CAM_032\loadsol\original files"
destination_dir = r"C:\Users\reicheel\OneDrive - University of North Carolina at Chapel Hill\__Study Data\cam_eeg_data\sub-CAM_032\loadsol"

# Function to rename .pdo and .ascii files
def rename_files():
    # Iterate over all .ascii files in the original directory
    for file_name in os.listdir(original_dir):
        if file_name.endswith(".txt"):  # Process only ASCII files
            ascii_file_path = os.path.join(original_dir, file_name)

            print(f"\nProcessing file: {ascii_file_path}")
            
            # Read and extract necessary details from the .ascii file
            with open(ascii_file_path, 'r') as ascii_file:
                lines = ascii_file.readlines()
                pdo_file_name = None
                comment = None

                for line in lines:
                    if line.startswith("File:"):
                        pdo_file_name = line.split(":")[1].strip()
                        print(f"Found PDO file reference: {pdo_file_name}")
                    if line.startswith("Comment:"):
                        comment = line.split(":")[1].strip()
                        print(f"Found Comment: {comment}")

            # Ensure both the .pdo file name and comment are found
            if pdo_file_name and comment:
                # Extract the study ID
                study_id = pdo_file_name.split('_', 1)[0]  # This should correctly give us 'CAM_001'
                identifier = pdo_file_name[len(study_id) + 1:].replace('.pdo', '')  # Remove .pdo extension
                
                # Construct the correct .pdo file name
                reconstructed_pdo_file_name = f"{study_id}_{identifier}.pdo"
                reconstructed_pdo_file_path = os.path.join(original_dir, reconstructed_pdo_file_name)

                print(f"Reconstructed PDO file name: {reconstructed_pdo_file_name}")

                # Check if the PDO file exists in the original directory
                if not os.path.exists(reconstructed_pdo_file_path):
                    print(f"Error: PDO file '{reconstructed_pdo_file_name}' not found.")
                    continue

                # Define new file names for .pdo and .ascii files
                new_pdo_file_name = f"{comment}.pdo"
                new_ascii_file_name = f"{comment}.txt"

                new_pdo_file_path = os.path.join(destination_dir, new_pdo_file_name)
                new_ascii_file_path = os.path.join(destination_dir, new_ascii_file_name)

                # Copy and rename the files to the destination directory
                try:
                    # Copy and rename the .pdo file
                    shutil.copy(reconstructed_pdo_file_path, new_pdo_file_path)
                    print(f"Successfully copied and renamed PDO file to: {new_pdo_file_path}")

                    # Copy and rename the .ascii file
                    shutil.copy(ascii_file_path, new_ascii_file_path)
                    print(f"Successfully copied and renamed ASCII file to: {new_ascii_file_path}")
                except Exception as e:
                    print(f"Failed to rename files. Error: {e}")
            else:
                print("Required data (PDO file name or comment) not found in .ascii file.")

# Call the function
rename_files()


