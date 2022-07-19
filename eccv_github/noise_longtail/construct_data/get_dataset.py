
import os,sys
import json
import requests
from tqdm import *


DATA_PATH_1 = '.dataset_no_images/mini-imagenet-annotations.json'
DATA_PATH_2 = './dataset_no_images/stanford-cars-annotations.json'
create_data_path_1 ='/data/clean_web/'
create_data_path_2 ='/data/long_tail_noise/stanford_cars/'



def create_folder(number,data_path):
    i = 0
    for i in range(number): 
        i=i+1
        file_name = data_path + str(i)
        os.mkdir(file_name)



def read_data(data_path,read_path):
    noise_dict={}
    damaged_list = []
    with open(data_path,'r',encoding='utf8')as fp:
        dataset = json.load(fp)
    dataset = dataset['data']
    length = len(dataset)
    for i in tqdm(range(length)):
        image_dict = dataset[i]
        image_info = image_dict[0]
        image_uri = image_info['image/uri']
        image_id = image_info['image/id']
        image_label =  image_info['image/class/label']
        image_flag = image_info['image/class/label/is_clean']
        if image_flag == 1:
            #download
            try:
                r = requests.request('get',image_uri,timeout = 4)
                with open(os.path.join(read_path,str(image_label),image_id)+'.jpg','wb') as f :
                    f.write(r.content)
                f.close()
                
                noise_dict[image_id] = image_flag 
            except:
                damaged_list.append(image_id)
                pass

            i = i + 1
    
    return noise_dict,damaged_list



#create folder
create_folder(100,create_data_path_1)
#create_folder(196,create_data_path_2)


# download img 
noise_dict_1,damaged_list_1 = read_data(DATA_PATH_1,create_data_path_1)
#noise_dict_2 ,damaged_list_2 = read_data(DATA_PATH_2,create_data_path_2)

# save noise dict
mimg_dict = json.dump(noise_dict_1)
#car_dict = json.dump(noise_dict_2)
mimg_dam = json.dump(damaged_list_1)
#car_dam = json.dump(damaged_list_2)

f1 = open('./noise_longtail/json/mimg_dict.json', 'w')
f1.write(mimg_dict)

#f2 = open('./noise_longtail/json/car_dict.json', 'w')
#f2.write(car_dict)

f3 = open('./noise_longtail/json/mimg_dam.json', 'w')
f3.write(mimg_dam)

#f4 = open('./noise_longtail/json/car_dam.json', 'w')
#f4.write(car_dam)


