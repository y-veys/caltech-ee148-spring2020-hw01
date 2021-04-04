import os
import numpy as np
import json
from PIL import Image, ImageDraw

def flatten_normalize(I):
    I = np.ndarray.flatten(I)
    I = I/np.linalg.norm(I)
    return I 

def check_for_red(I_HSV):
    H = I_HSV[:,:,0]
    S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    (n_rows,n_cols) = np.shape(H)
    count = 0 

    for i in range(n_rows):
        for j in range(n_cols):
            check_1 =(H[i,j] < 40 or H[i,j] > 240) and (S[i,j] > 150) and (V[i,j] > 150)
            if check_1:
                count += 1
    
    return count > 0.1*n_rows*n_cols

def filter_red(I, I_HSV):
    H = I_HSV[:,:,0]
    S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    (n_rows,n_cols) = np.shape(H)

    for i in range(n_rows//2):
        for j in range(n_cols):
            check_1 =(H[i,j] < 40 or H[i,j] > 240) and (S[i,j] > 150) and (V[i,j] > 150)
            
            #check_2 = (H[i,j] < 30 or H[i,j] > 240) and (S[i,j] < 80) and (V[i,j] > 220)
            #check_3 = (H[i,j] < 30 or H[i,j] > 240) and (S[i,j] > 220) and (V[i,j] < 170)
            if check_1:# or check_2:# or check_3:
                I[i,j] = [255,255,255]
            else:
                I[i,j] = [0,0,0]
    
    return I[0:n_rows//2,:] 
    
def detect_red_light(I, I_HSV):
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
    
    large_kernel = np.asarray(Image.open('../data/kernels/large_kernel.jpg'))
    # read image using PIL:
    med_kernel = np.asarray(Image.open('../data/kernels/small_kernel.jpg'))

    (n_rows,n_cols,n_channels) = np.shape(I)
    (med_kernel_width, med_kernel_height, _) = np.shape(med_kernel)
    (large_kernel_width, large_kernel_height, _) = np.shape(large_kernel)

    maximum = -1
    threshold = 0.85

    med_kernel_R = flatten_normalize(med_kernel[:,:,0])
    med_kernel_G = flatten_normalize(med_kernel[:,:,1])
    med_kernel_B = flatten_normalize(med_kernel[:,:,2])

    large_kernel_R = flatten_normalize(large_kernel[:,:,0])
    large_kernel_G = flatten_normalize(large_kernel[:,:,1])
    large_kernel_B = flatten_normalize(large_kernel[:,:,2])
    
    for i in range(n_rows//2 - med_kernel_width):
        for j in range(n_cols - med_kernel_height):
            tl_row = i 
            tl_col = j 
            br_row = i + med_kernel_width
            br_col = j + med_kernel_height

            #patch = I[tl_row:br_row,tl_col:br_col,:]
            patch_HSV = I_HSV[tl_row:br_row,tl_col:br_col,:]

            patch_R = flatten_normalize(I[tl_row:br_row, tl_col:br_col,0])
            patch_G = flatten_normalize(I[tl_row:br_row, tl_col:br_col,1])
            patch_B = flatten_normalize(I[tl_row:br_row, tl_col:br_col,2])

            inner_product_R = np.dot(patch_R, med_kernel_R)
            inner_product_G = np.dot(patch_G, med_kernel_G)
            inner_product_B = np.dot(patch_B, med_kernel_B)

            if inner_product_R > threshold and inner_product_B > threshold and inner_product_G > threshold: 
                if check_for_red(patch_HSV):
                    bounding_boxes.append([tl_row,tl_col,br_row,br_col])  

    for i in range(n_rows//2 - large_kernel_width):
        for j in range(n_cols - large_kernel_height):
            tl_row = i 
            tl_col = j 
            br_row = i + large_kernel_width
            br_col = j + large_kernel_height

            #patch = I[tl_row:br_row,tl_col:br_col,:]
            patch_HSV = I_HSV[tl_row:br_row,tl_col:br_col,:]

            patch_R = flatten_normalize(I[tl_row:br_row, tl_col:br_col,0])
            patch_G = flatten_normalize(I[tl_row:br_row, tl_col:br_col,1])
            patch_B = flatten_normalize(I[tl_row:br_row, tl_col:br_col,2])

            inner_product_R = np.dot(patch_R, large_kernel_R)
            inner_product_G = np.dot(patch_G, large_kernel_G)
            inner_product_B = np.dot(patch_B, large_kernel_B)

            if inner_product_R > threshold and inner_product_B > threshold and inner_product_G > threshold: 
                #if check_for_red(patch_HSV):
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
for i in [0]:#len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    Img = np.array(I)
    Img_HSV = I.convert("HSV")
    Img_HSV = np.array(Img_HSV)
    
    #red_filter = Image.fromarray(filter_red(Img, Img_HSV))
    #red_filter.show()
    #filtered_img.save("filtered_red_2.jpg", "JPEG",quality=95)

    bounding_boxes = detect_red_light(Img, Img_HSV)
    preds[file_names[i]] = bounding_boxes
    I.save("output.jpg", "JPEG")

    for box in bounding_boxes:
        draw = ImageDraw.Draw(I)  
        draw.rectangle([box[1],box[0],box[3],box[2]], fill=None, outline=None, width=1)
        I.save("output.jpg", "JPEG")

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
