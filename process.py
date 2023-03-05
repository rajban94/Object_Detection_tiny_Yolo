import ast
from utils import *
import numpy as np
import copy
import cv2
import configparser

config = configparser.ConfigParser()
config.read("./config.ini")

box_overlap_score=float(config['NMS']['overlap_score'])

def postprocess(data,img_processed_path: str,image_path):

    raw_prediction_dict=copy.deepcopy(data)
    image_bbox_dtls=copy.deepcopy(data)
    # image_cv = cv2.imread(image_path)

    car_list = ast.literal_eval(data["bbox_data"]["car_bbox"])
    truck_list = ast.literal_eval(data["bbox_data"]["truck_bbox"])
    van_list = ast.literal_eval(data["bbox_data"]["van_bbox"])

    #image_res = ast.literal_eval(data["page_res"])
    if car_list:
        car_list = non_max_suppression(np.asarray(car_list), overlapThresh=box_overlap_score).tolist()
    if truck_list:
        truck_list = non_max_suppression(np.asarray(truck_list), overlapThresh=box_overlap_score).tolist()
    if van_list:
        van_list = non_max_suppression(np.asarray(van_list), overlapThresh=box_overlap_score).tolist()

    image_bbox_dtls["bbox_data"]['car_bbox'] = str(car_list)
    image_bbox_dtls["bbox_data"]['truck_bbox'] = str(truck_list)
    image_bbox_dtls["bbox_data"]['van_bbox'] = str(van_list)

    
    draw_bounding_boxes(image_path,image_bbox_dtls,img_processed_path)
    return image_bbox_dtls