import os
import numpy as np
import json
from PIL import Image, ImageDraw
from time import sleep

def flatten_normalize(I):
    I = np.ndarray.flatten(I)
    I = I/np.linalg.norm(I)
    return I 

def check_for_black(I_HSV):
    H = I_HSV[:,:,0]
    S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    (n_rows,n_cols) = np.shape(H)
    count = 0 

    for i in range(n_rows):
        for j in range(n_cols):
            if V[i,j] < 55: count += 1
    
    return count > 0.15*n_rows*n_cols

def check_for_red(I_HSV):
    H = I_HSV[:,:,0]
    S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    (n_rows,n_cols) = np.shape(H)
    count = 0 

    for i in range(n_rows):
        for j in range(n_cols):
            if (H[i,j] < 25 or H[i,j] > 240) and (S[i,j] > 150) and (V[i,j] > 150): count += 1
    
    return count > 0.05*n_rows*n_cols

def mask(I, I_HSV):
    H = I_HSV[:,:,0]
    S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    (n_rows,n_cols) = np.shape(H)

    for i in range(n_rows):
        for j in range(n_cols):
            if V[i,j] < 55: 
                I[i,j] = [255,255,255]
            elif (H[i,j] < 25 or H[i,j] > 240) and (S[i,j] > 150) and (V[i,j] > 150): 
                I[i,j]= [255,0,0]
            else:
                I[i,j] = [0,0,0]
    
    return I[0:n_rows,:] 
    
def convolve(kernel, bounding_boxes, inner_products, I, I_HSV):
    response_R = []
    response_G = []
    response_B = []

    (n_rows,n_cols,n_channels) = np.shape(I)
    (kernel_width, kernel_height, _) = np.shape(kernel)

    kernel_R = flatten_normalize(kernel[:,:,0])
    kernel_G = flatten_normalize(kernel[:,:,1])
    kernel_B = flatten_normalize(kernel[:,:,2])

    threshold = 0.88
    
    for i in range(2*n_rows//3 - kernel_width):
        for j in range(n_cols - kernel_height):
            tl_row = i 
            tl_col = j 
            br_row = i + kernel_width
            br_col = j + kernel_height

            patch_HSV = I_HSV[tl_row:br_row,tl_col:br_col,:]

            patch_R = flatten_normalize(I[tl_row:br_row, tl_col:br_col,0])
            patch_G = flatten_normalize(I[tl_row:br_row, tl_col:br_col,1])
            patch_B = flatten_normalize(I[tl_row:br_row, tl_col:br_col,2])

            inner_product_R = np.dot(patch_R, kernel_R)
            inner_product_G = np.dot(patch_G, kernel_G)
            inner_product_B = np.dot(patch_B, kernel_B)

            if inner_product_R > threshold and inner_product_B > threshold and inner_product_G > threshold: 
                if check_for_red(patch_HSV) and check_for_black(patch_HSV):
                    bounding_boxes.append([tl_row,tl_col,br_row,br_col])  
                    inner_products.append([inner_product_R, inner_product_G, inner_product_B])

    return bounding_boxes, inner_products

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
    inner_products = [] 
    
    # read image using PIL:
    avg_large_kernel = np.asarray(Image.open('../data/large_kernels/aggregated_light.jpg'))
    avg_med_kernel = np.asarray(Image.open('../data/medium_kernels/aggregated_light.jpg'))
    avg_small_kernel = np.asarray(Image.open('../data/small_kernels/aggregated_light.jpg'))

    bounding_boxes, inner_products = convolve(avg_large_kernel, bounding_boxes, inner_products, I, I_HSV)
    bounding_boxes, inner_products = convolve(avg_med_kernel, bounding_boxes, inner_products, I, I_HSV)
    bounding_boxes, inner_products = convolve(avg_small_kernel, bounding_boxes, inner_products, I, I_HSV)

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes, inner_products

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
for i in range(len(file_names)):
    print(i+1)
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    Img = np.array(I)
    Img_HSV = I.convert("HSV")
    Img_HSV = np.array(Img_HSV)
    
    # mask = Image.fromarray(mask(Img, Img_HSV))
    # mask.show()

    bounding_boxes, inner_products = detect_red_light(Img, Img_HSV)

    bounding_boxes = list(bounding_boxes)
    inner_products = list(inner_products)

    zipped_lists = zip(bounding_boxes, inner_products)
    sorted_zipped_lists = sorted(zipped_lists)

    inner_products = [element for _, element in sorted_zipped_lists]
    bounding_boxes.sort()

    j=0
    while j < len(bounding_boxes)-1: 
        k = j + 1 
        while k < len(bounding_boxes):
            diff = np.array(bounding_boxes[k])-np.array(bounding_boxes[j])
            diff = np.abs(diff)
            close = max(diff) < 20

            if close: 
                if np.linalg.norm(inner_products[j+1]) > np.linalg.norm(inner_products[j]): 
                    bounding_boxes.remove(bounding_boxes[j+1])
                    inner_products.remove(inner_products[j+1])
                else: 
                    bounding_boxes.remove(bounding_boxes[j])
                    inner_products.remove(inner_products[j])
                j -=1
                break;
            k += 1
        j += 1

    preds[file_names[i]] = bounding_boxes

    #for box in bounding_boxes:
    #    draw = ImageDraw.Draw(I)  
    #    draw.rectangle([box[1],box[0],box[3],box[2]], fill=None, outline=None, width=1)
    #save_name = "results_2/output_" + file_names[i] 
    #I.save(save_name, "JPEG", quality=85)

#save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_3.json'),'w') as f:
    json.dump(preds,f)

