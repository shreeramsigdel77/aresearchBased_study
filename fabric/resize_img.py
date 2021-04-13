import os
import cv2

inference_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/custom_created_testdata"

inference_dir_output = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/custom_created_testdata"



for i in (os.listdir(inference_dir)):
    print(i)
    img_path = os.path.join(inference_dir,i)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(inference_dir_output,i),img)
