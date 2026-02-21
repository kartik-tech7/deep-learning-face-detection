import os

print("Model exists:", os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"))
print("Config exists:", os.path.exists("deploy.prototxt"))