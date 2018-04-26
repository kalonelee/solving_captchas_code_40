import numpy as np
from PIL import Image
import pytesseract
from imutils import paths
import os
import os.path
openfrom = "generated_captcha_images_1/"
saveto = "generated_captcha_images/"

def numberToLongerString(innum,length):
    numStr=""
    tempNum=innum
    innumlen=0
    while int(tempNum) > 0:
        tempNum /= 10
        innumlen += 1
    #print(innumlen)
    if innum == 0:
        innumlen = 1
    for i in range(innumlen, length):
        numStr += "0"
    numStr += str(innum)
    return numStr

#imgID = '0000'
def removeLine(imgid):
    im = Image.open(openfrom+imgid+".bmp")
    w_arr = [([]) for i in range(0, 24)]
    x = 0
    for i in np.array(im):
        x += 1
        for j in i:
            #print(j)
            if j == 0:
                w_arr[x-1].append(1)
            if j == 255:
                w_arr[x-1].append(' ')
    tempStrs = []
    tempShowRow = []
    for i in range(0, 24):
        tempStr=""
        t = ""
        rml = 0

        for j in range(0, 96):
            tempStr += str(w_arr[i][j])
            if w_arr[i][j] == 1:
                rml += 1
                if tempShowRow.__len__() < 96:
                    tempShowRow.append(0)
                else:
                    tempShowRow[j] += 1
            else:
                if tempShowRow.__len__() < 96:
                    tempShowRow.append(0)
            t += "0"

        if tempStr != t and rml < 36:
            tempStrs.append(tempStr)
    strImg=""
    y = 0
    for i in range(0,tempStrs.__len__()):
        tempStr = ""
        x=0
        for j in range(0,48):
            #if tempShowRow[j] > 0:
            #    tempStr += tempStrs[i][j]
            #    if tempStrs[i][j] == '1':
            #        x += 1
            tempStr += tempStrs[i][j]
        x=24
    #    print(x)
        if y < 9 and x !=0:
            y += 1
            print(str(y)+"  "+tempStr)
        elif x != 0:
            y += 1
            print(str(y)+" " +tempStr)
        if x != 0:
            for i1 in range(0,len(tempStr)):
                if tempStr[i1]==' ':
                    strImg += '255'
                else:
                    strImg += '0'
                if i1 < len(tempStr) - 1:
                    strImg += ' '
            strImg += ';\n'
    x = 24
    y = 96
    #print(str(len(tempStr))+" Column")
    #print(strImg[0:-2])
    matrixImg=np.matrix(strImg[0:-2])
    #data=np.reshape(matrixImg,(y,len(tempStr)))
    newImg=Image.fromarray(matrixImg)
    #newImg.show()
    #newImg.convert('RGB').save('im.png')
    newImg.convert('RGB').save(saveto + imgid + '.bmp')
    #newImg.convert('RGB').save('imgCode/' + captcha_correct_text + '.png')
    #print(str.encode(strImg))
    #newimg = Image.frombytes("1",(len(tempStr),y),str.encode(strImg))
    #newimg.save('im.png')


def saveImg(nub):
    image = Image.open('im.png')
    #vcode = pytesseract.image_to_string(image,lang="eng",config="-psm 7 4wn").replace(' ','')
    #print(vcode)
    #if vcode.__len__!=0:
    #    image.save('imgCode/' + str(vcode) + '0.png')
    #else:
    image.save('imgCode/' + nub + '.png')


def saveImgTif(number):
    image = Image.open('im.png')
    image.save('tif/' + number + '.tif')

def changeName(number):
    image = Image.open('tif/' + number + '.tif')
    image.save('tif1/captcha40.normal.exp' + number + '.tif')
	

gg=0
captcha_image_files = list(paths.list_images(openfrom))
for (i,img) in enumerate(captcha_image_files):
#for i in range(0, 100):
    #print(numberToLongerString(gg, 4))
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
    #gg = gg + 1
    filename = os.path.basename(img)
    captcha_correct_text = os.path.splitext(filename)[0]
    removeLine(captcha_correct_text)
    #saveImg(numberToLongerString(i, 4))
    #saveImgTif(numberToLongerString(i, 4))
    #changeName(numberToLongerString(i, 4))
