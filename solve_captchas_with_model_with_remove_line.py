from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
from PIL import Image
import imutils
import cv2
import pickle
import os
import os.path

def removeLine(imgid):
    #im = Image.open(openfrom+imgid+".bmp")
    #print(str(imgid))
    #im = Image.open(str(imgid))
    im = Image.fromarray(imgid)
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
    newImg=Image.fromarray(np.uint8(matrixImg))
    return np.array(newImg.convert('RGB'))
    #newImg.show()
    #newImg.convert('RGB').save('im.png')
    #newImg.convert('RGB').save(saveto + imgid + '.bmp')
    #newImg.convert('RGB').save('imgCode/' + captcha_correct_text + '.png')
    #print(str.encode(strImg))
    #newimg = Image.frombytes("1",(len(tempStr),y),str.encode(strImg))
    #newimg.save('im.png')

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
#CAPTCHA_IMAGE_FOLDER = "img1"
#OUTPUT_FOLDER = "gci"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
#captcha_image_files = np.random.choice(captcha_image_files, size=(20,), replace=False)
#captcha_image_files = np.random.choice(captcha_image_files, replace=False)
# loop over the image paths
counter = 0
for (i,image_file) in enumerate(captcha_image_files):
    filename = os.path.basename(image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = removeLine(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Add some extra padding around the image
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        #img = image.copy()
        #cv2.drawContours(img,contour,-1,(0,0,255),3)
        #cv2.imshow("img",img ) 
        #cv2.waitKey(0)  	
        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if h > 3:
            if w / h > 1.5:
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
            # This is a normal letter by itself
                letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 4:
        #continue
        print("cant process")
    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        #print(str(x)+"."+str(y)+"."+str(w)+"."+str(h))
        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    #print("CAPTCHA text is: {}".format(captcha_text))
    #image = cv2.imread(image_file)
	# Get the folder to save the image in
    #save_path = os.path.join(OUTPUT_FOLDER)

    # if the output directory does not exist, create it
    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)

    # write the letter image to a file

    #p = os.path.join(save_path, "{}.bmp".format(captcha_text))
    #cv2.imwrite(p, image)
	
    # Show the annotated image
    #cv2.imshow("Output", output)
    #cv2.waitKey()
    if captcha_text == str(captcha_correct_text):
        counter += 1
        print("same")
    else:
        print ( "CAPTCHA text is: {}  origin:{}".format(captcha_text, captcha_correct_text) )
    print("[INFO] processing image {}/{} {}%".format(i + 1, len(captcha_image_files), counter/(i+1)*100 ))
