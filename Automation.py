import os
import cv2

dir_in = './Railgun/'
dir_out = './Railgun-Frames/'
dir_face = './Railgun-Faces/'

def detect(filename, filenum, cascade_file = "./Face-Extractor/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)   
    print(filename)
    try:
        cascade = cv2.CascadeClassifier(cascade_file)
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (128, 128))
    except:
        return
    facenum = 1
    for (x, y, w, h) in faces:
        width, height, _ = image.shape
        add = w // 4
        if w != h or w < 128:
            continue
        x_new = x - add
        y_new = y - add
        w_new = w + (2*add)
        h_new = h + (2*add)
        if x_new > width or y_new > height or x_new < 0 or y_new < 0:
            face = image[y:y+h, x:x+w]
        else:
            face = image[y_new:y_new+h_new, x_new:x_new+w_new]
        face = cv2.resize(face, (128, 128))
        cv2.imwrite(dir_face + 'face-%d-%d.jpg'%(filenum, facenum), face)
        facenum += 1

counter = 1
for filename in os.listdir(dir_in):
    script = 'ffmpeg -i ' + dir_in + filename + ' -r 1 ' + dir_out + 'image%d-%%04d.jpg'%(counter)
    counter += 1
    os.system(script)

filenum = 1
for filename in os.listdir(dir_out):
    detect(dir_out + filename, filenum)
    filenum += 1