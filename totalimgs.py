import os

path = "Images/Train"

num_files = len([f for f in os.listdir(
    path) if os.path.isfile(os.path.join(path, f))])

print("Number of files in folder:", num_files)
