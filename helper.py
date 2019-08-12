import os


IMG_DIR = '/specific/disk1/home/gamir/ofer/data/unformatted_images/test1/'
RES_DIR = '/specific/disk1/home/gamir/ofer/data/object_boxes/test1/'

def to_res_dir(p):
    name = os.path.basename(p)
    name = os.path.splitext(name)[0]
    name = os.path.splitext(name)[0]
    return os.path.join(RES_DIR,name+'.pickle')

paths = os.listdir(IMG_DIR)
paths = filter(lambda p: ".pickle" in p, paths)
paths = map(lambda p: os.path.join(IMG_DIR,p), paths)
paths = list(paths)

for path in paths:
    print(path)
    res = to_res_dir(path)
    #os.rename(path, res)