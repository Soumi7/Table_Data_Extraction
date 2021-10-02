# Table Detection and Text Extraction — OpenCV and Pytesseract

![Image from https://nanonets.com/blog/ocr-with-tesseract/](https://user-images.githubusercontent.com/51290447/135706980-370be025-4c9d-4ce8-bbdc-e9356ec89656.png)

Given a image including random text and a table, extracting data from only the table is the objective. This is what worked out for me after trying out several different approaches from the docs as well as articles, on a set of images.
* Pytesseract and tesseract-ocr are used for image to text conversion.
* First we need to identify the part of the image which has the table. We will use openCV for this.
* Start with downloading an image with a table in it. This image was downloaded from here.

![http://softlect.in/index.php/html-table-tags/](https://user-images.githubusercontent.com/51290447/135707002-e4923955-ca4c-44dd-9361-fca2c1b28de9.png)

First the image has to be converted to binary, i.e. if the pixel value is greater than a certain value, it is assigned one value, and if it is less, then the other value. Here different parameters can be specified for different styles of thresholding.

**cv2.threshold()** : First argument is the source image, which should be a grayscale image. Second argument is the threshold value which is used to classify the pixel values. Third argument is the maxVal which represents the value to be given if pixel value is more than the threshold value. OpenCV provides different styles of thresholding and it is decided by the fourth parameter of the function.
**Global thresh holding** : In global thresholding, an arbitrary value is used as threshold value. Global, because the same value is applied as a threshold for all pixels.

```
thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
img_bin = 255-img_bin
plotting = plt.imshow(img_bin,cmap='gray')
plt.title("Inverted Image with global thresh holding")
plt.show()
```

![Image by author, generated using colaboratory](https://user-images.githubusercontent.com/51290447/135707020-cdb568ec-e52e-41bd-b616-52400a36957c.png)

**Otsu thresholding** : If it is required to automate the process of the selection of the threshold value, otsu can be tried. Otsu will work well for a bimodal image, i.e the histogram of all the pixel values will have two peaks. For that image, otsu chooses a value approximately in the middle of the two peaks as a threshold. So, Otsu works well for bimodal images.

```
img_bin1 = 255-img
thresh1,img_bin1_otsu = cv2.threshold(img_bin1,128,255,cv2.THRESH_OTSU)
plotting = plt.imshow(img_bin1_otsu,cmap='gray')
plt.title("Inverted Image with otsu thresh holding")
plt.show()
```

![Image generated using colaboratory](https://user-images.githubusercontent.com/51290447/135707045-d27278ff-4d31-4c98-a79d-04f7fec9a16b.png)

If both **cv2.THRESH_BINARY** and **cv2.THRESH_OTSU** are passed in the fourth parameter, the function performs both global and otsu thresholding.

```
img_bin2 = 255-img
thresh1,img_bin_otsu = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plotting = plt.imshow(img_bin_otsu,cmap='gray')
plt.title("Inverted Image with otsu thresh holding")
plt.show()
```

Here is what the image looks like after :

![Image generated using colaboratory](https://user-images.githubusercontent.com/51290447/135707054-44338758-272e-4946-bfb9-b1c3e23abf2f.png)

Morphological operations are performed on images based on their shapes. It takes the image and a structuring element or kernel.
**cv2.getStructuringElement()** : Here the shape and size of the kernel can be passed as parameters and accordingly a matrix is generated.
This is the format of use, as this may be tricky to remember:

```
cv2.getStructuringElement(shape,(num_of_columns,num_of_rows))
```

* The first argument specifies the shape of the kernel that you want, can be rectangular, circular or even elliptical.
* The second argument is tuple denoting the shape of the required kernel, the width and height.

```
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
```

**np.array(image).shape** is used to get the image dimensions. So first, the rectangular vertical kernel is defined with 1 row and columns as length of the numpy array of the image divided by 150. It will look something like this:
[1
1
1
1 ]

Here we will perform erosion on the binary image with the vertical kernel. It will remove all horizontal lines.

## Extracting vertical lines

The vertical kernel consists of a rectangular matrix consisting of one row and columns equal to number of columns in original image pixel array divided by 150.

### Erosion

What happens when erosion is performed on this image?

The vertical kernel which moves on the image, a pixel will be considered only if all pixels are 1 under the vertical kernel. So in this way, the horizontal lines get eroded, as only the pixels in each column remain.

```
import numpy
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//150))
eroded_image = cv2.erode(binary_image, vertical_kernel, iterations=5)
```

### Dilation

Now, lets perform dilation on the image. **Dilation** will make the pixel 1, if at least one of the pixels under the kernel is 1. This makes the vertical lines more prominent.

```
vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=5)
```

Erosion and dilation are used to remove noise.

After performing erosion and dilation, the image looks like this:

![Image generated using colaboratory](https://user-images.githubusercontent.com/51290447/135707144-28efea40-429d-4dd1-b991-0261143d3f90.png)

## Extracting horizontal Lines

### Erosion

The horizontal kernel which moves on the image, a pixel will be considered only if all pixels are 1 under the horizontal kernel. So in this way, the horizontal lines get eroded, as only the pixels in each column remain.
The horizontal kernel would look like this:
[1 1 1 1 1 1 1 ]

### Dilation

This makes the horizontal lines more prominent.

```
image_2 = cv2.erode(img_bin, hor_kernel, iterations=5)
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//150, 1))
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)
```

![Image generated using colaboratory](https://user-images.githubusercontent.com/51290447/135707158-fdb9e521-a5d6-4d5b-9d72-453a7294a990.png)

The horizontal and vertical lines are added, with equal weights to create a blended image.
weighted image = weight of first image * first image + weight of second image + gamma (which is an arbitrary constant).

```
vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
```

**Thresholding** is applied on the image containing vertical and horizontal lines.

```
thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
b_image = cv2.bitwise_not(cv2.bitwise_xor(img,vertical_horizontal_lines))
plotting = plt.imshow(b_image,cmap='gray')
plt.show()
```

![Image generated using colaboratory](https://user-images.githubusercontent.com/51290447/135707173-6112dbd6-343e-42bc-b3d4-a5a5418e2f8a.png)

Install pytesseract and import it in colab :

```
!pip install pytesseract
!sudo apt install tesseract-ocr
import pytesseract
```

## Contours

Contours on an image join pixels with same intensity. This works better for binary images.
Thresholding was performed on the image with horizontal and vertical lines.
Next, the contours are identified with **cv2.findContours()**.

```
contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

![Image generated using colaboratory](https://user-images.githubusercontent.com/51290447/135707207-482407ad-5afd-4085-8e0c-890acebe7d5a.png)

## Bounding Boxes

Bounding boxes are created for respective contours.

```
boundingBoxes = [cv2.boundingRect(c) for c in cnts]
(contours, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
key=lambda x:x[1][1]))
```

Lists of heights of all bounding boxes are obtained for each cell in the table and mean of heights.
The contours on the image are drawn and stored in list boxes. It is stored as a **list (x,y,w,h) : x and y** beind coordinates of top left corner and w and h being width and height of the box respectively.

```
boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if (w<1000 and h<500):
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        boxes.append([x,y,w,h])
plotting = plt.imshow(image,cmap='gray')
plt.title("Identified contours")
plt.show()
```

To store rows and columns :

* Now rows and columns lists are initalised as empty.
* The mean height of all boxes is calculated.
* Initially, the first box is appended to the columns list.
* The columns list is essentially a temporary list.
* Previous box is assigned to the first box as well.
* Loop through the remaining boundingBoxes list.
* At each iteration, it is checked if the y coordinate of the top left corner of the current box is less than the y coordinate of the previous box added with half the mean of all heights.

If yes :

* The current box is appended to columns list.
* The current box is assigned to previous box.
* Next it is checked if we are at the last index. If yes :
  * The entire column is appended to rows list.

If no :

* The columns list is appended to rows.
* The columns list is assigned to empty as this will start a new empty columns list.
* The current box is assigned to the previous box variable.
* The current box is appended to the empty column list we just created.

```
rows=[]
columns=[]
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
mean = np.mean(heights)
print(mean)
columns.append(boxes[0])
previous=boxes[0]
for i in range(1,len(boxes)):
    if(boxes[i][1]<=previous[1]+mean/2):
        columns.append(boxes[i])
        previous=boxes[i]
        if(i==len(boxes)-1):
            rows.append(columns)
    else:
        rows.append(columns)
        columns=[]
        previous = boxes[i]
        columns.append(boxes[i])
print("Rows")
for row in rows:
    print(row)
```

Lets get the total cells in each row :

```
total_cells=0
for i in range(len(row)):
    if len(row[i]) > total_cells:
        total_cells = len(row[i])
print(total_cells)
```

The width of cell to left bottom x coordinate is added to the the x coordinate of centre of cell.

```
center = [int(rows[i][j][0]+rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
print(center)
center=np.array(center)
center.sort()
print(center)
```

Now we create a list of the coordinates of the boxes :

```
boxes_list = []
for i in range(len(rows)):
    l=[]
    for k in range(total_cells):
        l.append([])
    for j in range(len(rows[i])):
        diff = abs(center-(rows[i][j][0]+rows[i][j][2]/4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        l[indexing].append(rows[i][j])
    boxes_list.append(l)
for box in boxes_list:
    print(box)
```

## Extracting text from cells in image using Pytesseract

Extract the region of interest(ROI) from the image.
The cell is resized, then morphological operations are performed on the extracted cell area to remove noise.
Finally, pytesseract is used to convert the image to a string.
The strings are appended to each row first to temporary string s with spaces, and then we append this temporary string to the final dataframe.

```
dataframe_final=[]
for i in range(len(boxes_list)):
    for j in range(len(boxes_list[i])):
    s=''
    if(len(boxes_list[i][j])==0):
        dataframe_final.append(' ')
    else:
        for k in range(len(boxes_list[i][j])):
            y,x,w,h = boxes_list[i][j][k][0],boxes_list[i][j][k][1], boxes_list[i][j][k][2],boxes_list[i][j][k][3]
            roi = bitnot[x:x+h, y:y+w]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            border = cv2.copyMakeBorder(roi,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
            resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            dilation = cv2.dilate(resizing, kernel,iterations=1)
            erosion = cv2.erode(dilation, kernel,iterations=2)
            out = pytesseract.image_to_string(erosion)
            if(len(out)==0):
                out = pytesseract.image_to_string(erosion)
            s = s +" "+ out
        dataframe_final.append(s)
print(dataframe_final)
```

Creating a numpy array from generated dataframe

```
arr = np.array(dataframe_final)
arr
```

Creating a dataframe from array

The array is reshaped into a dataframe with the number of rows and columns.

Print out the columns and check!

```
import pandas as pd
dataframe = pd.DataFrame(arr.reshape(len(rows), total_cells))
data = dataframe.style.set_properties(align="left")
#print(data)
#print(dataframe)
d=[]
for i in range(0,len(rows)):
    for j in range(0,total_cells):
        print(dataframe[i][j],end=" ")
print()
```

Final task is to save this data into a csv format for further uses.
A **output.csv** file is generated in google colab, which can be downloaded.

```
dataframe.to_csv("output.csv")
```

This is what the csv looks like!

![Image generated using colaboratory, output.csv](https://user-images.githubusercontent.com/51290447/135707370-84ea1c2a-9b32-4be2-b200-987833a48938.png)

Please leave a star on my github if you find this useful!

Here is a video for the next part, creating a web application with Flask :

[Invoice Data Extraction | Table Data Extraction | Image to Table API | Flask — YouTube](https://www.youtube.com/watch?v=flZiBg4wd5k&t=6s)

Here is the tutorial for this : [Host a Table Text Extraction API](https://medium.datadriveninvestor.com/host-a-table-text-extraction-api-with-heroku-tesserract-f4f73c24c94d)

## References
* [OpenCV: OpenCV modules](docs.opencv.org)
* [pytesseract](pypi.org)
