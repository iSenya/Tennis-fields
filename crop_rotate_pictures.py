import os, random, string
from PIL import Image
from PIL import ImageOps
folder = r"/home/kseniia/Pictures"
courts=os.listdir(folder)
for i in courts:
    for n in range(1, 360):
        pic = Image.open(folder+"/"+i)
        border = (1240, 400, 1240, 400)
        cr_pic = ImageOps.crop(pic.rotate(n), border)
        cr_resize_pic = cr_pic.resize((224,224), Image.ANTIALIAS)
        cr_resize_pic.save("/home/kseniia/Pictures/No tennis/"+str(n)+i)
