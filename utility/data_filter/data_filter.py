import os
import shutil

current_workspace = "utility/data_filter"
input_folder = os.path.join(current_workspace, "input_data")
output_folder = os.path.join(current_workspace, "output_data")
text_file_path = os.path.join(current_workspace, "file_list.txt")
file_list = []

with open(text_file_path, "r", encoding="utf-16") as file:
    file_list = [line.strip() for line in file]

file_set = set(file_list)
print(file_set)

for filename in os.listdir(input_folder):
    if filename in file_set:
        source_path = os.path.join(input_folder, filename)
        destination_path = os.path.join(output_folder, filename)
        shutil.move(source_path, destination_path)

print("Finished moving all files")
