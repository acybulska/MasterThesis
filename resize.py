import re

from PIL import Image
from imutils import paths

imagePaths = sorted(list(paths.list_images('/Users/lisa/DB/')))
base = 250
half=125
i = 1
for imagePath in imagePaths:
    img = Image.open(imagePath)
    (w, h)=img.size
    if w < h:
        wpercent = (base/float(w))
        hsize = int((float(h)*float(wpercent)))
        img = img.resize((base, hsize), Image.ANTIALIAS)
    else:
        hpercent=(base/float(h))
        wsize = int((float(w)*float(hpercent)))
        img = img.resize((wsize, base), Image.ANTIALIAS)
    (w, h) = img.size
    wx = int(w / 2)
    hx = int(h / 2)
    label = int(re.findall("0(\d+)*", imagePath)[0][0])
    label = str(label)
    sample = img.crop((wx - half, hx - half, wx + half, hx + half)).save('/Users/lisa/DB_BigFull/0' + label + 'sample'+str(i)+'.jpg')
    i = i + 1
    print(str(i))

print("done")

#    sample1 = img.crop((wx, hx, wx + 100, hx + 100)).save(
#        '/Users/lisa/DB_Pieces/0' + label + 'sample' + str(i) + '1.jpg')
#    sample2 = img.crop((wx - 100, hx - 100, wx, hx)).save(
#        '/Users/lisa/DB_Pieces/0' + label + 'sample' + str(i) + '2.jpg')
#    sample3 = img.crop((wx, hx - 100, wx + 100, hx)).save(
#        '/Users/lisa/DB_Pieces/0' + label + 'sample' + str(i) + '3.jpg')
#    sample4 = img.crop((wx - 100, hx, wx, hx + 100)).save(
#        '/Users/lisa/DB_Pieces/0' + label + 'sample' + str(i) + '4.jpg')