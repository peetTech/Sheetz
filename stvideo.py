import streamlit as st
import io
import time
import os
import pandas as pd
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
from argparse import Namespace
from threading import Timer
import time
import glob
import streamlit as st
from PIL import Image
from PIL import ImageFont
from google.cloud import storage
import gcsfs
import os
gcs = storage.Client()
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch.backends.cudnn as cudnn
from utils import google_utils
from utils.datasets import *
from utils.utils import *
import matplotlib.pyplot as plt
import numpy as np
from detecto.utils import read_image
from detecto.core import Model, Dataset, DataLoader
from detecto.visualize import show_labeled_image
import subprocess

import requests

def headerblue(url): 
    st.markdown(f'<p style="color:#0f29f2;font-size:48px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)
    
def headerred(url): 
    st.markdown(f'<p style="color:#55eb34;font-size:48px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)

def api():
    try:
        url = 'http://35.230.11.154:8000/api/plan'
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        return e



def videos_selector():
    storage_client = storage.Client()
    bucket_name = 'can_detection_data'
    bucket = storage_client.get_bucket(bucket_name)
    prefix='videos/'
    iterator = bucket.list_blobs(delimiter='/', prefix=prefix)
    response = iterator._get_next_page_response()
    data=[]
    for i in response['items']:
        z='gs://'+bucket_name+'/'+i['name']
        data.append(z)
    data=data[1:]
    return data 

opt = Namespace(
        weights = "last_yolov5s_results.pt",
        output = "inference/output",
        img_size = 640,
        conf_thres = 0.6,
        iou_thres = 0.3,
        fourcc = 'mp4v',
        device = '',
        view_img = False,
        save_txt = True,
        agnostic_nms = False,
        augment = False,
        classes = None
    )


def Compare_grid(plan, Grid):
    products = set([])
    
    for i in plan:
        for j in i:
            products.add(j)
    
    filled = {}
    empty = {}
    
    for p in products:
        filled[p] = []
        empty[p] = []
    
    for i,_ in enumerate(plan):
        for j,_ in enumerate(plan[i]):
            if len(Grid[i]) < j+1:
                empty[plan[i][j]].append([i,j])
                plan[i][j] = "empty"
            else:
                if Grid[i][j] == 0:
                    empty[plan[i][j]].append([i,j])
                    plan[i][j] = "empty"
                else:
                    filled[plan[i][j]].append([i,j])
                    plan[i][j] = "filled"
    
    return plan,filled,empty


def get_planogram():
    #### Get the Plano gram From FrontEnd
    plan =[["can1","can1","can1","can1","can1","can1","can2","can2","can2","can2"],
           ["can1","can1","can1","can1","can1","can1","can2","can2","can2","can2"],
           ["can1","can1","can1","can1","can1","can1","can2","can2","can2","can2"],
           ["can3","can3","can3","can3","can3","can3","can2","can2","can2","can2"],
           ["can3","can3","can3","can3","can3","can3","can2","can2","can2","can2"],
           ["can3","can3","can3","can3","can3","can3","can2","can2","can2","can2"],
           ["can4","can4","can4","can4","can4","can4","can4","can4","can4","can4"],
           ["can4","can4","can4","can4","can4","can4","can4","can4","can4","can4"]]
    return plan

def detect(source,exclude = [-1,-1,-1,-1],save_img=False,croped = None):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  
    os.makedirs(out) 
    half = device.type != 'cpu'  
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  
    
    
    model.to(device).eval()
    imgsz = check_img_size(imgsz, s=model.model[-1].stride.max()) 
    if half:
        model.half()  

    save_img = True
    dataset = LoadImages(source, img_size=imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    co_ordinates = []
    
    def outside(one_co_ordinate):
        cen_x = (one_co_ordinate[0]+one_co_ordinate[2])/2
        cen_y = (one_co_ordinate[1]+one_co_ordinate[3])/2
        return exclude[0] > cen_x or exclude[1] > cen_y or exclude[2] < cen_x or exclude[3] < cen_y
    
    for path, img, im0s,_ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        one_co_ordinate = [i.item() for i in xyxy] 
                        if outside(one_co_ordinate) and exclude[0] != -1:
                            continue
                        one_co_ordinate.append("Filled")
                        
                        co_ordinates.append(one_co_ordinate)
                        
                        plot_one_box(xyxy, im0, label="", color=[49, 207, 31], line_thickness=2)
                
                Image.fromarray(im0).save("output.jpg")
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
    
    label=['Empty']
    model = Model.load('Empty.pth',label)

    image = read_image(save_path)

    labels, boxes, scores = model.predict(image)

    im_h,im_w,im_c = image.shape

    image_white = np.zeros([im_h, im_w, 3], dtype=np.uint8)

    image_white[:,:] = [235, 245, 239]
    def condition(x): 
            return x > con_emp_thresh

    output = [idx for idx, element in enumerate(scores) if condition(element)]            
    labels=[labels[i] for i in output]
    boxes=boxes[output,:]
    scores=scores[output]

    for i in boxes :
        one_co_ordinates = [float(int(j.item())) for j in i]
        if outside(one_co_ordinates) and exclude[0] != -1:
            continue
        one_co_ordinates.append("Empty")
        plot_one_box(one_co_ordinates, im0, label="", color=[245, 5, 21], line_thickness=2)
        co_ordinates.append(one_co_ordinates)
#     st.image(im0)
    #print(co_ordinates)
    for xyxy in co_ordinates:
        if xyxy[-1] == "Empty":
            color = [235, 38, 23]
        else:
            color = [49, 207, 31] 

        plot_one_box(xyxy, image, label="", color=color, line_thickness=2)
        plot_one_box(xyxy, image_white, label="", color=color, line_thickness=2)
    #     plot_one_box([(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,(xyxy[0]+xyxy[2])/2+1,(xyxy[1]+xyxy[3])/2+1], image_white, label="", color=[255,0,0], line_thickness=2)
    files = os.listdir("./inference/output")
    if(files[0][-3:] != "txt"):
        img_file = files[0]
    else:
        img_file = files[1]
    Image.fromarray(image).save("./inference/output/{}".format(img_file))
#     Image.fromarray(image_white).save("./inference/output/White.jpg")
    #Realogram--------------------------
    
    def sort_key(l):
        return l[1]
    
    # Filled Co-ordinates
    Filled_co_ordinates = [i[:4] for i in co_ordinates if i[4] == "Filled"]
    
    #average Height and Width
    avg_height = 100
    avg_width = 50
    if len(Filled_co_ordinates):
        avg_height = (sum([(i[3]-i[1]) for i in Filled_co_ordinates]))/len(Filled_co_ordinates)
        avg_width = (sum([(i[2]-i[0]) for i in Filled_co_ordinates]))/len(Filled_co_ordinates)
    
#   sorting Both Co_ordinates
    Filled_co_ordinates = sorted(Filled_co_ordinates, key  = sort_key)
    sorted_co_ordinates = sorted(co_ordinates, key  = sort_key)

#     image = Image.open('./inference/output/{}'.format(image_name))
    im_h,im_w = Image.fromarray(image).size
    image_white = np.zeros([im_w, im_h, 3], dtype=np.uint8)
    image_white = Image.fromarray(image_white)
    
    cen = (Filled_co_ordinates[0][1]+Filled_co_ordinates[0][3])/2
    now_y_min = Filled_co_ordinates[0][1]
    idx = 1
    row = 1
    RowWiseList = [[]]
    cans = 0
    
    overlapping_thresh = 0.2
    
    for xyxy in sorted_co_ordinates:
        if xyxy[4] == "Empty":
            color = "#e82020"
        else :
            cans += 1
            color = "#33f016"
        idx += 1
        if cen < xyxy[1]:
            idx = 1
            cen = (xyxy[1]+xyxy[3])/2
            now_y_min = xyxy[1]
            row += 1
            RowWiseList.append([])
        cen = cen*(idx-1)+(xyxy[1]+xyxy[3])/2
        cen = cen/idx
        RowWiseList[len(RowWiseList)-1].append([xyxy[0],now_y_min,xyxy[0]+avg_width,now_y_min+avg_height,xyxy[4]])
    def x_sort(l):
        return l[0]
    for i in RowWiseList:
        i.sort(key = x_sort)
    def isOverlapping(prev, now):
        return ((avg_width)*(overlapping_thresh)) < (prev[2] - now[0])

    for row in RowWiseList:
        for index,can in enumerate(row):
            if can[4] == "Filled":
                continue
            if index == 0:
                continue
            if row[index-1][4] == "Empty" and isOverlapping(row[index-1],can):
                row.remove(row[index-1])

    cans = 0

    Grid = [[]]
    image_white2 = np.full([im_w, im_h, 3], 255,dtype=np.uint8)
    image_white2 = Image.fromarray(image_white2)

    for row in RowWiseList:
        for xyxy in row:
            if xyxy[4] == "Empty":
                color = "#e82020"
                Grid[len(Grid)-1].append(0)
            else :
                cans += 1
                color = "#33f016"
                Grid[len(Grid)-1].append(1)
            cen = cen*(idx-1)+(xyxy[1]+xyxy[3])/2
            cen = cen/idx
            img1 = ImageDraw.Draw(image_white2)
            img1.rectangle([(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3])], fill = color, outline ="black")
        Grid.append([])
    Grid.remove([])
    
    # END OF MODIFIED CODE
    print('Done. (%.3fs)' % (time.time() - t0))
    return Grid,image_white2,Image.fromarray(image)
    
    
    return Image.fromarray(image)

if __name__ == "__main__":
    col1, col2, col3 = st.beta_columns([2,6,1])

    with col1:
        st.write("")

    with col2:
        st.image("./Sheetz_logo.jpg")

    with col3:
        st.write("")
    
    st.sidebar.write("Poewered By")
    st.sidebar.image("./techl.png")
    st.sidebar.write(" ")
    uploaded_file = st.file_uploader("Upload Video Files",type=['mp4'])
#     stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
#     stroke_color = st.sidebar.color_picker("Stroke color hex: ")
#     bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    
    
    if uploaded_file:
        path = './upload_video/{name}'.format(name=uploaded_file.name)
        new_path = path
            
        g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
        temporary_location = path
            
        with open(temporary_location, 'wb') as vid:  ## Open temporary file as bytes
            vid.write(g.read())  ## Read bytes into file
        vid.close()
        gcs.get_bucket('can_detection_data').blob('videos/{name1}'.format(name1= uploaded_file.name)).upload_from_filename('./upload_video/{name}'.format(name=uploaded_file.name))
        
    filenames = videos_selector()
    filenames.append("-")
    filenames = filenames[::-1]
    file_path = st.selectbox("choose an image",filenames)
    len_of_files = len(filenames)
        
    opt.conf_thres = st.sidebar.slider("Object Detection Threshold", 0.3,0.7,0.4)
    con_emp_thresh = st.sidebar.slider("Empty Space Detection Threshold", 0.1, 0.9, 0.15)
        
        ####CV2 Code and Detection
    if file_path != "-":
    
        video = cv2.VideoCapture("./upload_video/{}".format(file_path.split("/")[-1]))
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        image_holder = st.empty()
        void = Image.fromarray(np.full((10, 10, 3),
                        255, dtype = np.uint8))
        frame_to_be_used = None
        
        
        stop = st.checkbox("Use current frame for calibration")
        
        resize_frac =  300/frame_width
        
        while(not stop):
            succ, frame = video.read()
            if succ == True:
                image = Image.fromarray(frame)
                image = image.resize((int(image.width*resize_frac),int(image.height*resize_frac)))
                image.save("./frame/frame.jpg")
                _,plano_image,image = detect("./frame/frame.jpg")
#                 out.write(np.array(image))
                
                image_holder.image(image)
                    
                time.sleep(1)
            else:
                break
        image_holder.image(void)
        video.release()
#         out.release()
        frame_to_be_used = Image.open("./inference/output/frame.jpg")
    
        st.write("")
        
        st.header("Select Part of the frame using the cursor")
        st.write("")
        
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.15)",  # Fixed fill color with some opacity
            stroke_width=2,
            stroke_color="#171614",
            background_color="#eee",
            background_image=frame_to_be_used ,
            update_streamlit=True,
            height=frame_to_be_used.height,
            width=frame_to_be_used.width,
            drawing_mode="rect",
            key="canvas",
        )
        
        if canvas_result.json_data is not None:
            data = canvas_result.json_data
            if len(data["objects"]) > 0:
                left = data["objects"][0]["left"]/resize_frac
                top = data["objects"][0]["top"]/resize_frac
                right = left+data["objects"][0]["width"]/resize_frac
                bottom = top+data["objects"][0]["height"]/resize_frac
                video = cv2.VideoCapture("./upload_video/{}".format(file_path.split("/")[-1]))
                frame_width = int(video.get(3))
                frame_height = int(video.get(4))

                output_file = "./out.mp4"
                
                fps = 20

                count = 0
                out_width = int(right-left)
                out_height = int(bottom-top)
                resize_frac = 300/out_width
                
                #out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (int(out_width*0.25)*2, int(out_height*0.25)*2))
                
                out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (out_width*2, out_height))
#                 out_plano = cv2.VideoWriter("./out_plano.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, (int(out_width), int(out_height)))
                video = cv2.VideoCapture("./upload_video/{}".format(file_path.split("/")[-1]))
    
                Grid = None

                while(True):
                    succ, frame = video.read()
                    
                    count += 1
                    
                    print(count)

                    if succ :
                        image = Image.fromarray(frame)
                        image.save("./frame/frame.jpg")
                        Grid,plano_image,image = detect("./frame/frame.jpg",[left,top,right,bottom])
                        
#                         if Grid == None:
#                             Grid = grid
                        
                        image = image.crop((left, top, right, bottom))
                        plano_image  = plano_image.crop((left, top, right, bottom))
#                         image = image.resize((int(out_width*0.25), int(out_height*0.25)))
#                         plano_image = plano_image.resize((int(out_width*0.25), int(out_height*0.25)))
                        image = cv2.hconcat([np.array(image),np.array(plano_image)])
                        image = Image.fromarray(image).resize((out_width*2, out_height))
                        out.write(np.array(image))
#                         out_plano.write(np.array(plano_image))
                    else:
                        break
                
#                 data = api()
#                 planogram = data['data'][1]['colorArray']
#                 categories = data['data'][1]['finalSelect']
#                 for obj in categories:
#                     for x,y in obj['points']:
#                         planogram[x][y] = obj['name']

                planogram = get_planogram()
                _,filled,empty = Compare_grid(planogram,Grid)
                
                video.release()
                out.release()
#                 out_plano.release()
                output_file = "./out.mp4"
                output_file_plano = "./out_plano.mp4"
                output_file_streamlit = "./frame/out_st.mp4"
                output_file_streamlit_plano = "./frame/out_st_plano.mp4"
                os.system('ffmpeg -y -i {} -vcodec libx264 {}'.format(output_file,output_file_streamlit))
#                 os.system('ffmpeg -y -i {} -vcodec libx264 {}'.format(output_file_plano,output_file_streamlit_plano))
                video_file = open('./frame/out_st.mp4','rb')
                video_bytes = video_file.read()
        
                headerred("Object")
                headerblue("Empty")
                
                st.video(video_bytes)
                
                st.write("Empty Co-ordinates")
                print("Missing Items")
                print(empty)
                st.write(empty)
                st.write("Filled Co-ordinates")
                print("Items Present")
                print(filled)
                st.write(filled)
            
                
#                 video_file = open('./frame/out_st_plano.mp4','rb')
#                 video_bytes = video_file.read()
#                 st.video(video_bytes)
#                 data = api()
#                 planogram = data['data'][1]['colorArray']
#                 categories = data['data'][1]['finalSelect']
#                 for obj in categories:
#                     for x,y in obj['points']:
#                         planogram[x][y] = obj['name']

                print(planogram)