import os

folder = 'faces_db'

for root, dirs, files in os.walk(folder):
    for file in files:
        base, ext = os.path.splitext(file)
        if ext.upper() in ['.JPG', '.JPEG', '.PNG']:
            new_name = base + ext.lower()
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
