import os, random, string
from PIL import Image
from PIL import ImageOps
#this is a path to my home directory with unprepared pictures from satellites, you can change it to your own path
folder = r"/home/kseniia/Pictures"
courts=os.listdir(folder)
for i in courts:
    for n in range(1, 360):
        pic = Image.open(folder+"/"+i)
        border = (1240, 400, 1240, 400)
        cr_pic = ImageOps.crop(pic.rotate(n), border)
        cr_resize_pic = cr_pic.resize((224,224), Image.ANTIALIAS)
# here you can create your own directory, it can be used either for creating pictures for learning dataset or for test dataset
# either for pictures with tennis courts or without
        cr_resize_pic.save("/home/kseniia/Pictures/No tennis/"+str(n)+i)
