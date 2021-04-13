import os, numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def process_img(dir_path):
    """[converts images into numpy array to load a dataset]

    Args:
        dir_path ([type]): [Path to image folder]

    Returns:
        [numpy.ndarray]: [list of images]
    """
    img_list = []
    for images in os.listdir(dir_path):
        if '.png' in images:
            #load image
            img = load_img(os.path.join(dir_path,images))
            
            print("original",type(img))

            #convert to numpy array
            img_array = img_to_array(img)

            print("typeee:", type(img_array))

            print("type:",img_array.dtype)
            print("shape:",img_array.shape)
            img_list.append(img_array)
    img_list = numpy.asarray(img_list)
    
    return img_list

dir_path = '/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/val'
process_img(dir_path)
