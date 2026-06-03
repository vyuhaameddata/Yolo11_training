import os, sys
import openslide
import xml.etree.ElementTree as ET
import xml.dom.minidom                    
import numpy as np
import torch
from tqdm import tqdm
import time
from multiprocessing import Pool
import concurrent.futures

from models.experimental import attempt_load
from utils.datasets import create_dataloader_custom, letterbox
from utils.general import check_img_size, non_max_suppression, xyxy2xywh, set_logging, scale_coords
# from utils.metrics import ap_per_class, ConfusionMatrix
# from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, TracedModel

from ultralytics import YOLO

class GlobalVars:
    def __init__(self):
        self.folder = r'D:\Shreyan\Test'
        self.wsi = ''
        self.weights = r'.\best_abnormal_updated.pt'
        self.source = ''
        self.batch_size = 32
        self.img_size = 640
        self.conf_thres = 0.5
        self.single_cls = False
        self.iou_thres = 0.5
        self.device = ''
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.no_trace = True
        self.cat_weights = r'.\best_cat.pt'
    def __init__(self, fldr):
        self.folder = fldr
        self.wsi = ''
        self.weights = r'.\abnormal11.pt'
        self.source = ''
        self.batch_size = 16
        self.img_size = 640
        self.conf_thres = 0.5
        self.single_cls = False
        self.iou_thres = 0.5
        self.device = ''
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.no_trace = True
        self.cat_weights = r'.\best_cat.pt'




def write_ndpa(tile_size=1024,overlap=128):
    # print("WRITE_NDPA CALLED")
    start = time.time()
    ndpi_folder = globalVars.folder
    device = select_device(globalVars.device)
    # half = (globalVars.device != 'cpu')  # half precision only supported on CUDA        
    # model = attempt_load(globalVars.weights,map_location=device)

    #model loading function for yolo11
    model = YOLO(globalVars.weights).to(device)

    # print('model_loaded')
    # if not globalVars.no_trace:
    #     model = TracedModel(model, device, globalVars.img_size)

    # not suported for yolo11  gives RuntimeError: expected m1 and m2 to have the same dtype, but got: struct c10::Half != float  
    # if half:
    #     model.half()  # to FP16
    count = 0
    #wsi = globalVars.wsi           
    # getting the x and y reference from the wholeslide info 
    for file in os.listdir(ndpi_folder):
        if file.endswith('.ndpi'):
            print(file)
            # try:
            if True:
                wsi = file.split('.')[0]
                wsi_path = os.path.join(ndpi_folder,wsi+".ndpi")         
                
                X_Reference,Y_Reference,nm_p = get_referance(wsi_path)
                
                # results_folder = os.path.join(ndpi_folder,'results')
                results_folder = ndpi_folder
                if not os.path.isdir(results_folder):
                    os.mkdir(results_folder)                
                xml_path= os.path.join(results_folder, wsi + '.ndpi.ndpa')    

                annotations = ET.Element('annotations')
                
                # update id for writing into the existing file.
                start_id =  update_annote_id(annotations) 
                
                print("current :",start_id)        
                
                end = time.time()
                print("the time of loading:",  (end - start) * 10**3, "ms")
                
                read_start = time.time()

                #first run
                anote_list = run_predict_wsi_multithread(wsi_path,model,overlap,tile_size=tile_size,batch_size = globalVars.batch_size)
                print('anote_list_created')
                anote_final = prune_list(anote_list) 

                #second run after prune
                anote_rerun = rerun_predict_pool(wsi_path,model,anote_final,batch_size=globalVars.batch_size)
                pruned_anote_rerun =prune_list(anote_rerun)                

                # third run for categorization
                # cat_model = get_cat_model()
                # cat_anote = rerun_predict_pool(wsi_path,cat_model,pruned_anote_rerun,batch_size=globalVars.batch_size)
                # pruned_cat_anote =prune_list(cat_anote)                

                # anote_xml = write_xml(annotations,start_id,anote_final,X_Reference,Y_Reference,nm_p)      
                anote_xml = write_xml(annotations,start_id,pruned_anote_rerun,X_Reference,Y_Reference,nm_p)     
                # anote_xml = write_xml(annotations,start_id,pruned_cat_anote,X_Reference,Y_Reference,nm_p)     
                with open(xml_path, "w") as f:
                    f.write(anote_xml)
                f.close()                
                end = time.time()    

                """uncomment if you need to generate result log file, GTp is path to ground truth annotations"""
                # GTp= os.path.join(results_folder, 'GTA')
                # GT_xml_path= os.path.join(GTp, wsi + '.ndpi.ndpa')
                # print(GT_xml_path)  # Check output
                # if os.path.isfile(GT_xml_path):
                #     dump_results(GT_xml_path, xml_path)                


                torch.cuda.empty_cache()
                print("The total of processing:",  (end-read_start) * 10**3, "ms")
                count +=1
                if count > 100 : break
            # except Exception as e:
            #      # Print the exception message
            #     print(f"An error occurred: {e}")
            #     # Prompt the user to press a key before exiting
            #     if sys.platform.startswith('win'):
            #         import msvcrt
            #         print("\nPress any key to exit...")
            #         msvcrt.getch()
            #     else:
            #         input("\nPress Enter to exit...")
                    
            # break #- run just one file
    # print("WRITE_NDPA RETURNED")



def get_referance(wsi_path):
    # print("GET_REFERENCE CALLED")
    slide = openslide.open_slide(wsi_path)    
    
    w = int(slide.properties.get('openslide.level[0].width'))
    h = int(slide.properties.get('openslide.level[0].height'))
    nm_p = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])*1000
        
    ImageCenter_X = (w/2)*nm_p
    ImageCenter_Y = (h/2)*nm_p
    
    OffSet_From_Image_Center_X = slide.properties.get('hamamatsu.XOffsetFromSlideCentre')
    OffSet_From_Image_Center_Y = slide.properties.get('hamamatsu.YOffsetFromSlideCentre')
    
    print("offset from Img center units?", OffSet_From_Image_Center_X,OffSet_From_Image_Center_Y)
    
    X_Ref = float(ImageCenter_X) - float(OffSet_From_Image_Center_X)
    Y_Ref = float(ImageCenter_Y) - float(OffSet_From_Image_Center_Y)
    slide.close()
    #print(ImageCenter_X,ImageCenter_Y)    
    #print(X_Reference,Y_Reference)
    print("resolution in nano meters per pixel", nm_p)
    return X_Ref,Y_Ref, nm_p

def update_annote_id(annotations):
    # print("UPDATE_ANOTE_ID CALLED")
    max_id = 0
    for elem in annotations.iter():
        #print(elem.tag)
        if elem.tag == 'ndpviewstate':
            _id_ = elem.attrib.get('id')                        
            if int(_id_) > max_id :
                max_id = int(_id_)
    # print("UPDATE_ANOTE_ID RETURNED")
    return max_id





def run_predict_wsi_multithread(path, model, overlap, tile_size=1024, batch_size=12):
    """Run inference on WSI using YOLOv11 with multi-threading."""
    anote = []
    set_logging()
    
    device, imgsz = globalVars.device, globalVars.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    # half = device.type != 'cpu'  # Half precision only supported on CUDA
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    
    print(f'Running inference with batch size: {batch_size}')
    
    dataloader = create_dataloader_custom(path, imgsz, stride, globalVars, batch_size, tile_size, overlap)[0]
    
    for batch_i, (img, shapes, coords) in enumerate(tqdm(dataloader)):
        # img = torch.from_numpy(np.ascontiguousarray(img)).to(device)
        # img = img.half() if half else img.float()  # Convert to float precision
        # img /= 255.0  # Normalize to [0,1]

        img = img.to(device).float()
        # img = img.half()
        img /= 255.0  # Normalize to [0,1]
        # if img.ndim == 3:
        #     img = img.unsqueeze(0)  # Ensure batch dim
        
        # Inference
        # with torch.no_grad():
        results = model.predict(img, conf=0.5, iou=0.5, verbose=False, device=device)
        # results = model(img, verbose=False)  # YOLOv11 inference
        # print(type(results))
        
        for i, result in enumerate(results):
            det = result.boxes.data  # Extract bounding boxes
            coord = coords[i][0]
            
            if len(det):
                # print("Before scale_coords:")
                # print("det dtype:", det.dtype)
                # print("det requires_grad:", det.requires_grad)
                # print("img shape:", img.shape)
                # print("shapes[i]:", shapes[i])
                # print("Is det a JIT Tensor?:", det.is_leaf)  # Should be True for normal tensors
                # print("Is det a view?:", det._base is not None)


                # Check if det is inference tensor
                # print("Inference Mode:", torch.is_inference_mode_enabled())
                
                # Even though requires_grad: False, the tensor is marked as an inference tensor
                # inplace update to det tensor is not allowed
                # det = det.clone().detach() detach is unecessary bc model is not in inference mode but it is still restrictive so clone is needed because tensor is in inference mode
                # clone and detach for error occurs because scale_coords() is modifying a tensor that is in inference mode (torch.inference_mode() or torch.no_grad()), which does not allow in-place operations.
                det = det.clone()  
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (shapes[i])).round()
                
                for *xyxy, conf, cls in reversed(det):
                    x1 = coord[0] + float(xyxy[0])
                    y1 = coord[1] + float(xyxy[1])
                    x2 = coord[0] + float(xyxy[2])
                    y2 = coord[1] + float(xyxy[3])
                    
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    anote.append([x1, y1, x2, y2, label, round(conf.item(), 2), path])
        
        del img
        torch.cuda.empty_cache()
    
    del dataloader
    return anote



def prune_list(box_list):    
    print('before prune:',len(box_list))
    bool_list =[True for i in range(len(box_list))]        
    for i in range(len(box_list)-1):
        
        if bool_list[i] == True:            
            for j in range(i+1, len(box_list)):                
                iou =bb_intersection_over_union(box_list[i],box_list[j])                                     
                if iou > 0.01:                       
                    # replace rectlist{j] with largest of rectlist{i] and rectlist{j]                    
                    x1,y1,x2,y2,cls,conf,path = replace_bigger_box(box_list[i],box_list[j])
                    box_list[j] = (x1,y1,x2,y2,cls,conf,path) 
                    bool_list[i] = False
    				# loop continue = true
                    break
    # collect all with true
    annote_final =[]
    for i in range(len(box_list)):        
        # print('check change',i,bool_list[i],box_list[i])        
        if bool_list[i] == True:
            annote_final.append(box_list[i])
    print('after prune :',len(annote_final))
    # print("PRUNE_LIST RETURNED")
    return annote_final

def replace_bigger_box(boxA,boxB):            
    x1 = min(boxA[0],boxB[0])
    y1 = min(boxA[1],boxB[1])
    x2 = max(boxA[2],boxB[2])
    y2 = max(boxA[3],boxB[3])
    conf = (float(boxA[5])+ float(boxB[5]))/2    
    return x1,y1,x2,y2,boxA[4],conf,boxA[6]

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

 

def rerun_predict_pool(wsi_path,model,pred_list,tile_size=1024,batch_size=12):
    anote =[]
    set_logging()
    device = select_device(globalVars.device)    
    # half = (globalVars.device != 'cpu')  # half precision only supported on CUDA        
    # pool = Pool(processes=5)
    # # images = pool.map(process,pred_list)        
    # print('through pooled reading')
    names = model.module.names if hasattr(model, 'module') else model.names    
    # for image in divide_chunks(images,batch_size):
    for pred_batch in divide_chunks(pred_list,batch_size):        
        # image_batch = pool.map(process,pred_batch)  
        image_batch = read_tiles_parallel(wsi_path,pred_batch,tile_size)
        # print(len(image_batch))
        lefts, tops, img = fetch_batch(image_batch)        
        # print(len(lefts),len(tops),img.shape)        
        img = torch.from_numpy(img).to(device).float()
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        # #img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)        
        #Inference                
        # with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        #     pred = model(img, augment=False)[0]        
        # Apply NMS
        # pred = non_max_suppression(pred,globalVars.conf_thres,globalVars.iou_thres, globalVars.classes,False)   
        # img = img.to(device).float()
        # img /= 255.0  # Normalize to [0,1]
        results = model.predict(img, conf=0.5, iou=0.5, verbose=False, device=device)
                   
        for i, result in enumerate(results):
            det = result.boxes.data  # Extract bounding boxes                            
            left, top = lefts[i],tops[i]
            #gn = torch.tensor((shapes[i][1],shapes[i][2]))[[1, 0, 1, 0]]  # normalization gain whwh
            gn = torch.tensor((tile_size,tile_size))[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords((img.shape[2:], det[:, :4],(shapes[i][1],shapes[i][2])).round()
                det = det.clone()  
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],(tile_size,tile_size)).round()
                #print(img.shape[2:],shapes[i],coords[i])    
                for *xyxy, conf, cls in reversed(det): 					                    
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh                        
                    line = ([*xywh])  #label format                                        
                    label = f'{names[int(cls)]} {conf:.2f}'                                 
                    x1 = left + (float(line[0]) - float(line[2])/2)*(tile_size)
                    y1 = top + (float(line[1]) - float(line[3])/2)*(tile_size)
                    x2 = x1 + float(line[2])*tile_size
                    y2 = y1 + float(line[3])*tile_size
                    anote.append([x1,y1,x2,y2,label,round(float(conf),3),wsi_path])        
            det = None
            gn = None
        pred = None
        del(image_batch)
        img = None
        torch.cuda.empty_cache()    
    # del(model)
    return anote    



def divide_chunks(Anotelist, batch_size):      
    # looping till length l
    for i in range(0, len(Anotelist), batch_size): 
        yield Anotelist[i:i + batch_size]

def read_tiles_parallel(slide_path, pred_list, tile_size):
    tiles = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit the read_tile function for each tile position
            future_to_tile = {executor.submit(process, slide_path, pred, tile_size): pred for pred in pred_list}
            for future in concurrent.futures.as_completed(future_to_tile):
                # tile = future.result()
                tiles.append(future.result())
            # print(executor._max_workers)
            executor.shutdown()
    except:
        print('something is wrong in threadpool reading')
        
    return tiles

def process(wsi_path,pred,tile_size):
    # print(pred)
    x1,y1,x2,y2,label,conf,path = pred #wsi_path = pred
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    #breath in pixels
    breath = abs(x2-x1)
    length = abs(y2-y1)
    #centering the Groundtruth
    xc, yc = tile_size/2, tile_size/2
    left = int(cx -xc- breath/2)
    top = int(cy -yc- length/2)
    slide = openslide.open_slide(wsi_path)
    level = slide.get_best_level_for_downsample(1.0 / 40)
    tile = slide.read_region((left,top),level,(tile_size,tile_size))
    im = tile.convert('RGB')
    img0 = np.array(im)
    img, ratio, __ = letterbox(img0,auto=False)
    img = img.transpose(2,0,1)
    img = np.expand_dims(img, 0)
    slide.close()
    return left,top,ratio,img

def fetch_batch(image):  
    lefts,tops,imgs= [],[],[]
    # print(len(image))
    for left,top,ratio,img in image:
        lefts.append(left)
        tops.append(top)        
        imgs.append(img)            
    imges = np.concatenate(imgs.copy()) 
    return lefts.copy(),tops.copy(),imges            
    # return lefts,tops,imges            


def get_cat_model():
    device = select_device(globalVars.device)    
    cat_model = attempt_load(globalVars.cat_weights,map_location=device)  # 
    half = (globalVars.device != 'cpu')  # half precision only supported on CUDA        
    if not globalVars.no_trace:
        cat_model = TracedModel(cat_model, device, globalVars.img_size)        
    if half:
        cat_model.half()  # to FP16            
    return cat_model


        
def write_xml(annotations,start_id,anote_list,X_Reference,Y_Reference,nm_p)     :    
    # print("WRITE_XML CALLED")
    id_ = start_id
    for line in anote_list : 
        # print(line)
        write_annotation(annotations,id_,line[0],line[1],line[2],line[3],line[4],line[5],X_Reference,Y_Reference,nm_p=nm_p)   
        id_ +=1
    # Convert the ElementTree object to a raw XML string
    raw_xml = ET.tostring(annotations, encoding='utf-8')  # Ensure UTF-8 encoding
    
    # Pretty-print the XML using minidom
    parsed_xml = xml.dom.minidom.parseString(raw_xml)  # Parse the raw XML
    formatted_xml = parsed_xml.toprettyxml(indent="  ")  # Indent with two spaces
    
    # print("WRITE_XML RETURNED")   
    return formatted_xml
        
def write_annotation(annotations,_id,x1,y1,x2,y2,cls,conf,X_Reference,Y_Reference,nm_p=221):
    # print("WRITE_ANNOTATION CALLED")
    sub_elem  = ET.SubElement(annotations,'ndpviewstate')
    sub_elem.set('id',str(_id))
    sub_elem1 = ET.SubElement(sub_elem,'title')
    sub_elem1.text = str(cls) + str(_id) # str(conf)

    # Inserting empty <details>
    details = ET.Element('details')
    sub_elem.insert(1, details)
    

    sub_elem2 = ET.SubElement(sub_elem,'coordformat')    
    sub_elem2.text = 'nanometers'
    sub_elem3 = ET.SubElement(sub_elem,'lens')
    sub_elem3.text = '40.0' 
    sub_elem4 = ET.SubElement(sub_elem,'cat')
    sub_elem4.text = str(cls) 
    sub_elem5 = ET.SubElement(sub_elem,'predict')
    sub_elem5.text = str(cls) 
    sub_elemX,sub_elemY, sub_elemZ = ET.SubElement(sub_elem,'x'), ET.SubElement(sub_elem,'y'),ET.SubElement(sub_elem,'z')
    sub_show = ET.SubElement(sub_elem,'showtitle')
    sub_show.text = str(1)
    sub_show = ET.SubElement(sub_elem,'conf')
    sub_show.text = str(conf)    

    sub_show = ET.SubElement(sub_elem,'showhistogram')
    sub_show.text = str(0)
    sub_show = ET.SubElement(sub_elem,'showlineprofile')
    sub_show.text = str(0) 
    sub_elemX.text,sub_elemY.text,sub_elemZ.text = str(int((x1+x2)*nm_p/2 -X_Reference)), str(int((y1+y2)*nm_p/2 -Y_Reference)),  '0' 
    #print(sub_elemX.text,sub_elemY.text,sub_elemZ.text)
      
    anote = ET.SubElement(sub_elem,'annotation')
    anote.set('type',"freehand")
    anote.set('displayname',"AnnotateRectangle")
    color = '#90EE90'
    if conf >= 0.5 and conf < 0.7 : 
        color    = "#9acd32"        
    elif conf >= 0.7 and conf < 0.9 :
        color = '#FFA500'
    elif conf >=0.9: 
        color = '#FFFF00'     
    anote.set('color', color)
    measure_type =ET.SubElement(anote,'measuretype')
    measure_type.text = str(3)
    Pointlist = ET.SubElement(anote, 'pointlist')
    point1 = ET.SubElement(Pointlist,'point')
    ndpa_x1 = ET.SubElement(point1,'x')
    ndpa_y1 = ET.SubElement(point1,'y')
    
    ndpa_x1.text = str(int(x1*nm_p-X_Reference)) 
    ndpa_y1.text = str(int(y1*nm_p-Y_Reference))
    
    
    #print(ndpa_x1.text,ndpa_y1.text)    
    point2 = ET.SubElement(Pointlist,'point')
    
    ndpa_x2 = ET.SubElement(point2,'x')
    ndpa_y2 = ET.SubElement(point2,'y')     
    
    ndpa_x2.text = ndpa_x1.text
    ndpa_y2.text = str(int(y2*nm_p-Y_Reference))
    
    point3 = ET.SubElement(Pointlist,'point')
    ndpa_x3 = ET.SubElement(point3,'x')
    ndpa_y3 = ET.SubElement(point3,'y')
    ndpa_x3.text = str(int(x2*nm_p -X_Reference))
    
    ndpa_y3.text = ndpa_y2.text
        
    point4 = ET.SubElement(Pointlist,'point')
      
    ndpa_x4 = ET.SubElement(point4,'x')
    ndpa_y4 = ET.SubElement(point4,'y')                            
    ndpa_x4.text = ndpa_x3.text
    ndpa_y4.text = ndpa_y1.text
        
    anote_type =ET.SubElement(anote,'specialtype')
    anote_type.text = 'rectangle'
    anote_type =ET.SubElement(anote,'closed')
    anote_type.text = '1'     
    # print("WRITE_ANNOTATION RETURNED")



def dump_results(gt_xml_path,predicts_xml_path):  
    # print("DUMP_RESULTS CALLED")  
    gt_box_list = get_box_list(gt_xml_path)
    predict_box_list = get_box_list(predicts_xml_path)    
    tp_count=0       
    #write =[]
        
    for gt_box in gt_box_list:
        
        for p_box in predict_box_list:
            iou = bb_intersection_over_union(gt_box,p_box)
            if iou > 0.3:
                tp_count +=1
            #print(,iou)
            #total_predicts = len(predict_box_list)        
    
    fp_count =(len(predict_box_list)-tp_count)
    tp_rate = tp_count/len(gt_box_list)
    fp_rate = tp_count/len(predict_box_list)
    
    log_file=r"D:\Shreyan\Development\Test\GT\results_log.txt"
    # Log the results into a text file
    with open(log_file, "a") as f:  # "a" for append mode
        f.write(f"GT XML: {gt_xml_path}\n")
        f.write(f"groundtruth: {len(gt_box_list)}\n")
        f.write(f"predictions: {len(predict_box_list)}\n")
        f.write(f"TP Count: {tp_count}\n")
        f.write(f"FP Count: {fp_count}\n")
        f.write(f"Recall (TP Rate): {tp_rate:.4f}\n")
        f.write(f"Precision (FP Rate): {fp_rate:.4f}\n")
        f.write("-" * 50 + "\n")  # Separator for readability

    # print('tp_count',tp_count,'recall or tp rate :', tp_rate)                
    # print('fp_count',fp_count,'precsion or fp rate',fp_rate)     
    # print("DUMP_RESULTS RETURNED")           
     
def get_box_list(xml_path,nm_p=221):    
    # print("GET_BOX_LIST CALLED")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x1,y1,x2,y2 = 0,0,0,0    
    box_list =[]
    for elem in root.iter():
        #print(elem.tag)
        # if elem.tag == 'ndpviewstate':
            #_id = elem.attrib.get('id')        
        x = []
        y = []
        if elem.tag == 'pointlist':
            for sub in elem.iter(tag='point'):
                x.append(int(sub.find('x').text))                    
                y.append(int(sub.find('y').text))                    
            x1=int(min(x)/nm_p)
            x2=int(max(x)/nm_p)
            y1=int(min(y)/nm_p)
            y2=int(max(y)/nm_p)
            #breath = abs(x2-x1)
            #length = abs(y2-y1)            
            row = (x1,y1,x2,y2)                
            box_list.append(row)    
    # print("GET_BOX_LIST RETURNED")         
    return box_list




if __name__ == '__main__':

    # folder_path = sys.argv[1]
    
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument('--folder', type=str, default='D:\Marked_cytology', help='source') 
    # parser.add_argument('--wsi', type=str, default='', help='source')     
    # parser.add_argument('--weights', nargs='+', type=str, default='.\best.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    # parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')        
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--no-trace', action='store_true', help='don`t trace model')


    # input the folder containing all .ndpi files you want to process here
    folder_path = r'D:\Shreyan\Development\Test\dump'
    # folder_path = r'D:\Shreyan\Development\Test\GT'
    globalVars = GlobalVars(folder_path)
    print(globalVars)
    #check_requirements(exclude=('pycocotools', 'thop'))   
    start = time.time()
    
    # try:
    write_ndpa(tile_size = 1024, overlap=128)

    # except Exception as e:
    #     # Print the exception message
    #     print(f"An error occurred: {e}")
    #     import msvcrt
    #     print("\nPress any key to exit...")
    #     msvcrt.getch()
    #     # Prompt the user to press a key before exiting
    
    end = time.time()          
    print("The time of execution of one whole slide:", (end-start) * 10**3, "ms")