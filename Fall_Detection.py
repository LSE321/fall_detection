import argparse
import logging
import time

import cv2
import numpy as np

import os
import math
import socket

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--save_video', type=bool, default=False,
                        help='To write output video. Default name output.avi')

    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.video)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    y1 = [0, 0]
    frame = 0
    last_time=time.time()
    fcount = 0
    fall_state=False
    fall_count=0
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
                                  10, (image.shape[1], image.shape[0]))
   
    while True:
        ret_val, image = cam.read()
        if image is None:
            print("NULL")
            break
        
        i_h, i_w=image.shape[:2]
        i_h=480*i_h//i_w
        i_w=480
        image=cv2.resize(image, (i_w, i_h), interpolation=cv2.INTER_AREA)
        
        fcount+=1
        if fcount%15 != 0:
            continue
        
        img_y=cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_planes=cv2.split(img_y)
        
        if(np.mean(ycrcb_planes[0])<70):
            ycrcb_ss=list(ycrcb_planes)
            ycrcb_ss[0]=cv2.equalizeHist(ycrcb_ss[0])
            dst_y=cv2.merge(ycrcb_ss)
            image=cv2.cvtColor(dst_y, cv2.COLOR_YCrCb2BGR)
        
        
        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #logger.debug('show+')
        no_people = len(humans)
        print("No. of people: ", no_people)

        if (no_people==1):
            for human in humans:
                try:
                    if 0 in human.body_parts:
                        a = human.body_parts[0]  # head shot
                    else:
                        a = human.body_parts[1]
                    head_x = a.x*image.shape[1]
                    head_y = a.y*image.shape[0]
                    y1.append(head_y)
                    cv2.putText(image,
                    "head: %d, %d" % (head_x, head_y),
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
                except:
                    pass
                
                if fall_state:
                    fall_count+=1
                    if fall_count<=5:
                        if (y1[-(fall_count+1)]-head_y)>20:
                            bbox_x=[]
                            bbox_y=[]
                            for i in human.body_parts:
                                bbox_x.append(human.body_parts[i].x)
                                bbox_y.append(human.body_parts[i].y)
                            min_x=int(min(bbox_x)*image.shape[1])
                            min_y=int(min(bbox_y)*image.shape[0])
                            max_x=int(max(bbox_x)*image.shape[1])
                            max_y=int(max(bbox_y)*image.shape[0])
                            if (max_y-min_y)/(max_x-min_x)>1:
                                print("stand")
                                fall_state=False
                    else:
                        print("Fall Detected")
                        cv2.putText(image,
                                "FALL DETECTED",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, 11)
                        fall_state=False
                    
                    
                else:
                    if head_y - y1[-2] > 60: #머리 낙하 속도 확인
                        #중심선 양끝 점 구하기
                        if 8 in human.body_parts:
                            key_r=human.body_parts[8]
                            key_l=human.body_parts[11]
                        elif 9 in human.body_parts:
                            key_r=human.body_parts[9]
                            key_l=human.body_parts[12]
                        elif 10 in human.body_parts:
                            key_r=human.body_parts[10]
                            key_l=human.body_parts[13]
                        else:
                            pass
                        
                        if key_r: 
                            key_cen_x=(key_r.x+key_l.x)/2
                            key_cen_y=(key_r.y+key_l.y)/2
                            #각도 계산
                            theta=math.degrees(math.atan(abs(head_y-key_cen_y)/abs(head_x-key_cen_x)))
                            if theta<45:
                                #bounding box
                                bbox_x=[]
                                bbox_y=[]
                                for i in human.body_parts:
                                    bbox_x.append(human.body_parts[i].x)
                                    bbox_y.append(human.body_parts[i].y)
                                min_x=int(min(bbox_x)*image.shape[1])
                                min_y=int(min(bbox_y)*image.shape[0])
                                max_x=int(max(bbox_x)*image.shape[1])
                                max_y=int(max(bbox_y)*image.shape[0])
                                image=cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255))
                                #세로/가로 <1이면
                                if (max_y-min_y)/(max_x-min_x)<1:
                                    print("fall state")
                                    fall_state=True
                        else:
                            pass
                
                """
                if (len(y1)>3) and ((head_y - y1[-2]) > (image.shape[0]/8)) and ((time.time()-last_time)<30):
                    print("Fall Detected")
                    cv2.putText(image,
                                "FALL DETECTED",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, 11)
                last_time=time.time()
        """
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(image,
                    "No. of People: %d" % (no_people),
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if(frame == 0) and (args.save_video):
            
            out.write(image)
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')

    cv2.destroyAllWindows()
