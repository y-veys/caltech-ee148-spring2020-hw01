import os
import numpy as np
import json
from PIL import Image, ImageDraw

def flatten_normalize(I):
    I = np.ndarray.flatten(I)
    norm = np.linalg.norm(I)

    if norm == 0:
        return I 
    else: 
        I = I/np.linalg.norm(I)
        return I 

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    '''
    box_height = 8
    box_width = 6
    
    num_boxes = np.random.randint(1,5) 
    
    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)
        
        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width
        
        bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    '''
    
    '''
    END YOUR CODE
    '''
    #large_kernel = np.asarray(Image.open('../data/kernels/large_kernel.jpg'))
    # read image using PIL:
    med_kernel = np.asarray(Image.open('../data/kernels/med_kernel.jpg').convert("HSV"))

    (n_rows,n_cols,n_channels) = np.shape(I)
    (kernel_width, kernel_height, _) = np.shape(med_kernel)

    maximum = -1
    threshold = 0.8

    kernel_R = flatten_normalize(med_kernel[:,:,0])
    #kernel_G = flatten_normalize(med_kernel[:,:,1])
    #kernel_B = flatten_normalize(med_kernel[:,:,2])
    
    for i in range(n_rows//2 - kernel_width):
        for j in range(n_cols - kernel_height):
            tl_row = i 
            tl_col = j 
            br_row = i + kernel_width
            br_col = j + kernel_height

            patch_R = flatten_normalize(I[tl_row:br_row, tl_col:br_col,0])
            #patch_G = flatten_normalize(I[tl_row:br_row, tl_col:br_col,1])
            #patch_B = flatten_normalize(I[tl_row:br_row, tl_col:br_col,2])

            inner_product_R = np.dot(patch_R, kernel_R)
            #inner_product_G = np.dot(patch_G, kernel_G)
            #inner_product_B = np.dot(patch_B, kernel_B)

            if inner_product_R > threshold:# and inner_product_B > threshold and inner_product_G > threshold: 
                bounding_boxes.append([tl_row,tl_col,br_row,br_col])  
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(1):#len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    Img = I.convert("HSV")
    
    # convert to numpy array:
    Img = np.asarray(Img)
    
    bounding_boxes = detect_red_light(Img)
    preds[file_names[i]] = bounding_boxes

    for box in bounding_boxes:
        draw = ImageDraw.Draw(I)  
        draw.rectangle([box[1],box[0],box[3],box[2]], outline ="green")
        I.save("output_H.jpg", "JPEG")

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
