import os

def set_up_images_tmp():
    path = './images_tmp/'
    os.makedirs(path, exist_ok=True)
    for item in os.listdir(path):
        os.remove(path + item)