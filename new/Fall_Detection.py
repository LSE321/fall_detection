import argparse
import logging
import time

import cv2
import numpy as np

import os
import math
import socket, threading, base64

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



def sendMsg(soc):
    while True:
        msg = input('')
        soc.sendall(msg.encode(encoding='utf-8'))
        if msg == '/stop':
            break
    print('클라이언트 메시지 입력 쓰레드 종료')

def sendImg(soc, img):
    #while True:
    msg = 'gps'
    soc.sendall(msg.encode(encoding='utf-8'))
    image=img
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', image, encode_param)
    data = np.array(imgencode)
    stringData = base64.b64encode(data)
    length = str(len(stringData))
    soc.sendall(length.encode('utf-8').ljust(64))
    soc.send(stringData)
    msg='/stop'
    soc.sendall(msg.encode(encoding='utf-8'))
    
    
def recvMsg(soc):
    while True:
        data = soc.recv(1024)
        msg = data.decode()
        print(msg)
        if msg == '/stop':
            break
    soc.close()
    print('클라이언트 리시브 쓰레드 종료')

class Client:
    ip = '43.201.85.174'
    port = 56027

    def __init__(self, img):
        self.client_soc = None
        self.img=img

    def conn(self):
        self.client_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_soc.connect((Client.ip, Client.port))

    def run(self):
        self.conn()
        t = threading.Thread(target=sendImg, args=(self.client_soc,self.img,))
        t.start()
        t2 = threading.Thread(target=recvMsg, args=(self.client_soc,))
        t2.start()



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
        
        fcount+=1
        if fcount%15 != 0:
            continue
        
        i_h, i_w=image.shape[:2]
        i_h=480*i_h//i_w
        i_w=480
        image=cv2.resize(image, (i_w, i_h), interpolation=cv2.INTER_AREA)
        
        filename1="./images/aa/input"+str(fcount)+".png"
        cv2.imwrite(filename1, image)
        
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
                    if fall_count<=10:
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
                            image=cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255))
                            if (max_y-min_y)/(max_x-min_x)>1:
                                print("stand")
                                cv2.putText(image,
                                "STAND",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, 11)
                                fall_state=False
                                fall_count=0
                    else:
                        print("Fall Detected")
                        cv2.putText(image,
                                "FALL DETECTED",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, 11)
                        c = Client(image)
                        c.run()
                        fall_state=False
                        fall_count=0
                    
                    
                else:
                    if head_y - y1[-2] >= 40: #머리 낙하 속도 확인
                        # 중심선 각도 체크해서 상황에 따라 bbox 비율 다르게 정/측면 기준 달라야지 
                        
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
                        if (max_y-min_y)/(max_x-min_x)<1.5:
                            print("fall state")
                            cv2.putText(image,
                                "FALL state",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, 11)
                            fall_state=True
                        #else:
                           # pass
                
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
        filename="./images/aa/output"+str(fcount)+".png"
        cv2.imwrite(filename, image)
        if(frame == 0) and (args.save_video):
            
            out.write(image)
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')

    cv2.destroyAllWindows()