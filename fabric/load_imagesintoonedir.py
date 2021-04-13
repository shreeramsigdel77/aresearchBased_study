import os

#Corduroy Blended Cotton
path = "/home/pasonatech/workspace/deepNN/Datasets/Fabrics/Corduroy"

output_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/dataset"
for i in os.listdir(path):
    print(i)
    base_path = os.path.join(path,i)
    print(base_path)
    print(os.path.join(path,i))
    for items in os.listdir(base_path):
       print(items)
       if '.png' in items:

            print(i+items)
            os.rename(os.path.join(base_path,items),os.path.join(output_dir,i+items))
            