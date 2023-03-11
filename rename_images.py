import os

path = "Images/Train"
prefix = 'image'
i = 1

for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        new_filename = f'{prefix}_{i}.jpg'
        os.rename(os.path.join(path, filename),
                  os.path.join(path, new_filename))
        i += 1
