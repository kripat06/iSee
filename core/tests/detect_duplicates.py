import os

path = 'c:/tensorflow1/models/research/object_detection/images/train'

files = []
x = []
# r=root, d=directories, f = files
files_map = {}
for r, d, f in os.walk(path):
    for file in f:
        name = file.split(".")[0]
        if name not in files_map:
            files_map[name]=0

        files_map[name]=files_map[name]+1

for key in files_map.keys():
    if not files_map[key] == 2:
        files.append(key)
        # file_path = f"{path}/{key}"
        # try:
        #     os.remove(f"{file_path}.jpg")
        # except:
        #     pass
        # try:
        #     os.remove(f"{file_path}.xml")
        # except:
        #     pass
print(files)
print(len(files))