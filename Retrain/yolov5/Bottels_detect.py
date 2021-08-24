import pandas as pd
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
from argparse import Namespace

import streamlit as st
from PIL import Image
from PIL import ImageFont
from google.cloud import storage
import gcsfs
import os
gcs = storage.Client()
import argparse
import matplotlib.pyplot as plt
from detecto.utils import read_image
from detecto.core import Model, Dataset, DataLoader
from detecto.visualize import show_labeled_image
import numpy as np
from PIL import Image, ImageDraw
import torch.backends.cudnn as cudnn
from utils import google_utils
from utils.datasets import *
from utils.utils import *
import matplotlib.pyplot as plt
from detecto.utils import read_image
from detecto.core import Model, Dataset, DataLoader
from detecto.visualize import show_labeled_image
import numpy as np

def image_selector():
    storage_client = storage.Client()
    bucket_name = 'can_detection_data'
    bucket = storage_client.get_bucket(bucket_name)
    prefix='images/'
    iterator = bucket.list_blobs(delimiter='/', prefix=prefix)
    response = iterator._get_next_page_response()
    data=[]
    for i in response['items']:
        z='gs://'+bucket_name+'/'+i['name']
        data.append(z)
    data=data[1:]
    return data 

def detect(source,save_img=False,croped = None):
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
            return x > 0.60

    output = [idx for idx, element in enumerate(scores) if condition(element)]            
    labels=[labels[i] for i in output]
    boxes=boxes[output,:]
    scores=scores[output]

    for i in boxes :
        one_co_ordinates = [float(int(j.item())) for j in i]
        one_co_ordinates.append("Empty")
        plot_one_box(one_co_ordinates, im0, label="", color=[245, 5, 21], line_thickness=2)
        co_ordinates.append(one_co_ordinates)
#     st.image(im0)
    print(co_ordinates)
    for xyxy in co_ordinates:
        if xyxy[-1] == "Empty":
            color = [235, 38, 23]
        else:
            color = [49, 207, 31] 

        plot_one_box(xyxy, image, label="", color=color, line_thickness=2)
        plot_one_box(xyxy, image_white, label="", color=color, line_thickness=2)
    #     plot_one_box([(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,(xyxy[0]+xyxy[2])/2+1,(xyxy[1]+xyxy[3])/2+1], image_white, label="", color=[255,0,0], line_thickness=2)
    files = os.listdir("/home/jupyter/peet/Retail-Store-Item-Detection-using-YOLOv5/inference/output")
    if(files[0][-3:] != "txt"):
        img_file = files[0]
    else:
        img_file = files[1]
    Image.fromarray(image).save("./inference/output/{}".format(img_file))
#     Image.fromarray(image_white).save("./inference/output/White.jpg")
    
    
    # MODIFIED CODE
    def sort_key(l):
        return l[1]
    
    # Filled Co-ordinates
    Filled_co_ordinates = [i[:4] for i in co_ordinates if i[4] == "Filled"]
    
    #average Height and Width
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
        return ((avg_width)*(0.25)) < (prev[2] - now[0])

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
    image_white2 = np.zeros([im_w, im_h, 3], dtype=np.uint8)
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
    return Grid,image_white2,im0

def header1(url): 
    st.markdown(f'<p style="color:#2C8C42;font-size:48px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)

def count(Grid):
    total = sum([len(i) for i in Grid])
    cans = sum(list(map(sum, Grid)))
    empties = total-cans
    return cans,empties

def save_and_train(image,f_name):
    image.save("./data/train/{}.jpg".format(f_name.split(".")[0]))

def show(bg_image, f_name):
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=bg_image ,
        update_streamlit=True,
        height=bg_image.height,
        width=bg_image.width,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Do something interesting with the image data and paths
#     if canvas_result.image_data is not None:
#         st.image(canvas_result.image_data)
#     bg_image.convert("RGB").save("./CroppedOut/cropped.jpg")
#     _,image_white = detect("./CroppedOut/cropped.jpg")

    
#     if show_for_full :
#         bg_image.convert("RGB").save("./CroppedOut/original.jpg")
#         Grid,image_white,im0 = detect("./CroppedOut/original.jpg")
#         im0 = Image.fromarray(im0)
#         itext = ImageDraw.Draw(im0)
#         myFont = ImageFont.truetype('PlayfairDisplay-Black.ttf', 15)
#         itext.text((10, 10), "Object", font = myFont,fill=(0, 255, 0))
#         itext.text((10, 25), "Empty", font = myFont,fill=(255, 0, 0))
#         col1, col2 = st.beta_columns(2)
# #             show_image = Image.open("./inference/output/cropped.jpg")
#         col1.header("Original")
#         col1.image(im0, use_column_width=True)
#         col2.header("Row Optimized")
#         col2.image(image_white, use_column_width=True)
#         filled,empties = count(Grid)
        
#         if filled == 1:
#             a = str(filled) + " can detected"
#         else:
#             a = str(filled) + " cans detected"
            
#         header1(a)
        
        
#     if canvas_result.json_data is not None:
#         data = canvas_result.json_data
#         # a dictionary
        
#         for i in range(len(data["objects"])):
#             left = data["objects"][i]["left"]
#             top = data["objects"][i]["top"]
#             right = left+data["objects"][i]["width"]
#             bottom = top+data["objects"][i]["height"]
#             #####---------------------
            
#             masked_image = np.zeros([bg_image.height, bg_image.width, 3], dtype=np.uint8)
#             masked_image[:,:] = [255,255,255]
#             masked_image = Image.fromarray(masked_image)
#             croped_part = bg_image.crop((left,top,right,bottom))
#             masked_image.paste(croped_part, (left,top,right,bottom))
#             masked_image.convert("RGB").save("./CroppedOut/cropped{}.jpg".format(i))
            
            
#             #####---------------------
#             Grid,image_white,im0 = detect("./CroppedOut/cropped{}.jpg".format(i))
            
#             filled,empties = count(Grid)
            
#             col1, col2 = st.beta_columns(2)
# #             show_image = Image.open("./inference/output/cropped.jpg")
#             col1.header("Cropped----{}".format(i+1))
#             col1.image(Image.fromarray(im0).crop((left,top,right,bottom)), use_column_width=True)
#             col2.header("Row Optimized")
#             col2.image(image_white.crop((left,top,right,bottom)), use_column_width=True)
            
#             if filled == 1:
#                 a = str(filled) + " can detected"
#             else:
#                 a = str(filled) + " cans detected"
            
#             header1(a)
    
    annote = []
    st.write(str(len(annote)))
    if canvas_result.json_data is not None:
        data = canvas_result.json_data
        if len(data["objects"]) != 0:
            for i in range(len(annote),len(data["objects"])):
                left = data["objects"][i]["left"]
                top = data["objects"][i]["top"]
                right = left+data["objects"][i]["width"]
                bottom = top+data["objects"][i]["height"]
                w = bg_image.width
                h = bg_image.height
                annote.append([Object_type,[left/w,top/h,right/w,bottom/h]])
            
    file1 = open("./stream_annotation/{}.txt".format(f_name.split(".")[0]), "a")  # append mode
    for i in annote:
        i = list(map(str, i[1]))
        file1.write("0 ")
        file1.write("{} {} {} {}\n".format(i[0], i[1], i[2], i[3]))
    
    
    file1.close()
    annote
    if st.button("hello"):
        save_and_train(bg_image,f_name)
    
        
        
      
if __name__ == "__main__":
    opt = Namespace(
        weights = "last_yolov5s_results.pt",
        output = "inference/output",
        img_size = 640,
        conf_thres = 0.4,
        iou_thres = 0.3,
        fourcc = 'mp4v',
        device = '',
        view_img = False,
        save_txt = True,
        agnostic_nms = False,
        augment = False,
        classes = None
    )

    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.file_uploader("Background image:", type=["png", "jpg"])
    if bg_image:
        image = Image.open(bg_image)
        uploaded_file = bg_image
        path = '/home/jupyter/peet/Retail-Store-Item-Detection-using-YOLOv5/images/{name}'.format(name=uploaded_file.name)
        image.save(path)
        gcs.get_bucket('can_detection_data').blob('images/{name1}'.format(name1= uploaded_file.name)).upload_from_filename('/home/jupyter/peet/Retail-Store-Item-Detection-using-YOLOv5/images/{name}'.format(name=uploaded_file.name))
    
    filenames = image_selector()
    filenames.append("-")
    filenames = filenames[::-1]
    file_path = st.selectbox("choose an image",filenames)
    
    
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("rect", "transform")
    )
    Object_type = st.sidebar.selectbox(
        "Type:", ("Object", "Empty")
    )
    show_for_full = st.sidebar.checkbox("show for full image", True)
    use_masking = st.sidebar.checkbox("use masking for better processing",True)
#     opt.conf_thres = st.sidebar.slider("Conf Threshold", 0.3,0.7,0.5)
#     opt.iou_thres = opt.conf_thres-0.1
    if file_path != "-":
            path = 'images/{name}'.format(name=file_path.split("/")[-1])
            f_name = file_path.split("/")[-1]
            show(Image.open(path),f_name)
