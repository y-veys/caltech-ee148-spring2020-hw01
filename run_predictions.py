import os
import numpy as np
import json
from PIL import Image

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
    # set the kernel path 
    kernel_path = '../data/kernels'
    # get sorted list of files 
    kernel_names = sorted(os.listdir(kernel_path)) 
    # remove any non-JPEG files: 
    kernel_names = [f for f in kernel_names if '.jpg' in f] 
    # read image using PIL:
    med_kernel = Image.open(os.path.join(kernel_path,kernel_names[1]))
    
    # convert to numpy array:
    med_kernel = np.asarray(med_kernel)

    (n_rows,n_cols,n_channels) = np.shape(I)
    (kernel_width, kernel_height, _) = np.shape(med_kernel)

    for i in range(n_rows//2 - kernel_width):
        for j in range(n_cols//2 - kernel_height):
            patch = I[i:i+kernel_width, j:j+kernel_height,0]
            kernel_R = med_kernel[:,:,0]

            patch = np.ndarray.flatten(patch)
            patch = (patch-128)/256

            kernel_R = np.ndarray.flatten(kernel_R)
            kernel_R = (kernel_R-128)/256

            print(kernel_R)
            print(patch)
            print(np.dot(patch,kernel_R))
            exit()

    
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
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
