import os
import xml.etree.ElementTree as ET
import xml.dom.minidom 
import openslide

def Check_pred_with_gt(pred_list,gt_list):    
    tps = []
    fps = []
    for boxA in pred_list: 
        flag = False
        for boxB in gt_list:
            iou = bb_intersection_over_union(boxA, boxB)
            if iou > 0.1:
                tps.append(boxA)
                flag = True                
                break                
        if not flag:            
            fps.append(boxA)
    return tps, fps

def update_fp_tp_anotes(pred_folder, gt_folder, result_folder):

    os.makedirs(result_folder, exist_ok=True)
    tp_folder = os.path.join(result_folder, "tp")
    fp_folder = os.path.join(result_folder, "fp")
    os.makedirs(tp_folder, exist_ok=True)
    os.makedirs(fp_folder, exist_ok=True)

    for file in os.listdir(pred_folder):
        if file.endswith('.ndpi'):
            filename = file.split('.')[0]
            print(filename)
            wsi_path = os.path.join(pred_folder,filename +'.ndpi')        
            
            pred = os.path.join(pred_folder,filename +'.ndpi.ndpa')
            pred_list = get_box_list(wsi_path, pred)

            gt = os.path.join(gt_folder,filename +'.ndpi.ndpa')
            gt_list = get_box_list(wsi_path, gt)
            print(f"groundtruth: {len(gt_list)}")
            # print('ground truth',len(gt_list))                
            if os.path.isfile(pred):
                tps, fps = Check_pred_with_gt(pred_list, gt_list)
                print(f"TP Count: {len(tps)}")
                print(f"FP Count: {len(fps)}\n")
                
                annotations = ET.Element('annotations')
                
                x_ref, y_ref = get_referance(wsi_path)
                
                tp_xml = write_xml_with_title(annotations, 0, "tp", tps, x_ref, y_ref)         
                tp_xml_path = os.path.join(tp_folder, filename +'.ndpi.ndpa')

                with open(tp_xml_path, "w") as f:
                      f.write(tp_xml)
                f.close()

                fp_xml = write_xml_with_title(annotations, 0, "fp", fps, x_ref, y_ref)         
                fp_xml_path = os.path.join(fp_folder, filename +'.ndpi.ndpa')

                with open(fp_xml_path, "w") as f:
                    f.write(fp_xml)
                f.close()
            # break


def get_box_list(wsi_path,xml_path,nm_p =221):    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x1,y1,x2,y2 = 0,0,0,0    
    box_list =[]
    X_Reference, Y_Reference = get_referance(wsi_path, nm_p)    
    for elem in root.iter():
        #print(elem.tag)
        if elem.tag == 'ndpviewstate':
            conf = elem.find('title').text        
        x = []
        y = []
        if elem.tag == 'pointlist':
            for sub in elem.iter(tag='point'):
                x.append(int(sub.find('x').text))                    
                y.append(int(sub.find('y').text))                    
            #float
            x1=float((min(x) + X_Reference)/nm_p)
            x2=float((max(x) + X_Reference)/nm_p)
            y1=float((min(y) + Y_Reference)/nm_p)
            y2=float((max(y) + Y_Reference)/nm_p)                         
            row = (x1,y1,x2,y2,conf)                          
            box_list.append(row)             
    return box_list

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

def get_referance(wsi_path, nm_p=221):
    
    slide = openslide.open_slide(wsi_path)

    w = int(slide.properties.get('openslide.level[0].width'))
    h = int(slide.properties.get('openslide.level[0].height'))

    ImageCenter_X = (w/2)*nm_p
    ImageCenter_Y = (h/2)*nm_p

    OffSet_From_Image_Center_X = slide.properties.get(
        'hamamatsu.XOffsetFromSlideCentre')
    OffSet_From_Image_Center_Y = slide.properties.get(
        'hamamatsu.YOffsetFromSlideCentre')

    # print("offset from Img center units?", OffSet_From_Image_Center_X,OffSet_From_Image_Center_Y)

    X_Ref = float(ImageCenter_X) - float(OffSet_From_Image_Center_X)
    Y_Ref = float(ImageCenter_Y) - float(OffSet_From_Image_Center_Y)

    
    return X_Ref, Y_Ref

def write_xml_with_title(annotations,start_id,title,anote_list,X_Reference,Y_Reference):    
    id_ = start_id
    for line in anote_list : 
        write_annotation(annotations,id_,line[0],line[1],line[2],line[3],title,X_Reference,Y_Reference)   
        id_ +=1
    # Convert the ElementTree object to a raw XML string
    raw_xml = ET.tostring(annotations, encoding='utf-8')  # Ensure UTF-8 encoding
    
    # Pretty-print the XML using minidom
    parsed_xml = xml.dom.minidom.parseString(raw_xml)  # Parse the raw XML
    formatted_xml = parsed_xml.toprettyxml(indent="  ")  # Indent with two spaces
    
    # print("WRITE_XML RETURNED")   
    return formatted_xml

def write_annotation(annotations,_id,x1,y1,x2,y2,title,X_Reference,Y_Reference,nm_p=221):
    # print("WRITE_ANNOTATION CALLED")
    sub_elem  = ET.SubElement(annotations,'ndpviewstate')
    sub_elem.set('id',str(_id))
    sub_elem1 = ET.SubElement(sub_elem,'title')
    sub_elem1.text = title

    # Inserting empty <details>
    details = ET.Element('details')
    sub_elem.insert(1, details)
    

    sub_elem2 = ET.SubElement(sub_elem,'coordformat')    
    sub_elem2.text = 'nanometers'
    sub_elem3 = ET.SubElement(sub_elem,'lens')
    sub_elem3.text = '40.0'
    sub_elemX,sub_elemY, sub_elemZ = ET.SubElement(sub_elem,'x'), ET.SubElement(sub_elem,'y'),ET.SubElement(sub_elem,'z')
    sub_show = ET.SubElement(sub_elem,'showtitle')
    sub_show.text = str(1)  

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
    if "v7" in title: 
        color    = "#00FF00	"        
    elif "v11" in title:
        color = '#0000FF'
    elif "common" in title: 
        color = '#FF00FF' 
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


pred_folder = r'D:\Shreyan\Development\Test\GT'
gt_folder = r'D:\Shreyan\Development\Test\GT\GTA'
result_folder = r'D:\Shreyan\Development\Test\GT\TPnFP'

# update_fp_tp_anotes(pred_folder, gt_folder, result_folder)





"""
Created on 27 feb 2025

@author: Shreyan

function : to merge 2 annotations with different colour boxes for comparision
example: comparing yolov7 and yolov11 or comparing TP and GT annotations
"""
def merge_GT_TP(GT_folder, TP_folder, result_folder):
    os.makedirs(result_folder, exist_ok=True)
    for file in os.listdir(TP_folder):
        if file.endswith('.ndpa'):
            filename = file.split('.')[0]
            print(filename)
            TP_xml = os.path.join(TP_folder,filename +'.ndpi.ndpa')
            GT_xml = os.path.join(GT_folder,filename +'.ndpi.ndpa')
            r_xml = os.path.join(result_folder,filename +'.ndpi.ndpa')

            if os.path.isfile(TP_xml) and os.path.isfile(GT_xml):
                tree1 = change_color(GT_xml, "green")
                tree2 = change_color(TP_xml, "blue")

                root1 = tree1.getroot()
                root2 = tree2.getroot()

                # Create a new root <annotations>
                new_root = ET.Element("annotations")

                # Get all <ndpviewstate> elements from both files
                all_ndpviewstates = root1.findall("ndpviewstate") + root2.findall("ndpviewstate")

                # Assign new sequential IDs starting from 0
                for new_id, ndp in enumerate(all_ndpviewstates):
                    ndp.set("id", str(new_id))  # Update the id
                    new_root.append(ndp)  # Add to new root

                # Create a new tree and save it
                new_tree = ET.ElementTree(new_root)
                new_tree.write(r_xml, encoding="utf-8", xml_declaration=True)
                print(f"Merged .ndpa file created as {r_xml}")
            else:
                print(f'abnormal file: {filename}')


def change_color(xml_path, color):
    colors = {
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF"
    }
    choice = colors[color]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    for elem in root.iter():
        if elem.tag == 'annotation':
            elem.set('color', choice)
    return tree


"""
Created on 14 march 2025

@author: Shreyan

function : comparing yolov7 and yolov11
"""
def merge_yolo(ann1_folder, ann2_folder, wsi_folder, result_folder):
    os.makedirs(result_folder, exist_ok=True)
    for file in os.listdir(ann1_folder):
        if file.endswith('.ndpa'):
            filename = file.split('.')[0]
            print(filename)
            ann1_xml = os.path.join(ann1_folder,filename +'.ndpi.ndpa')
            ann2_xml = os.path.join(ann2_folder,filename +'.ndpi.ndpa')
            r_xml = os.path.join(result_folder,filename +'.ndpi.ndpa')
            wsi_path = os.path.join(wsi_folder,filename +'.ndpi') 

            if os.path.isfile(ann1_xml) and os.path.isfile(ann2_xml):
                tree1 = change_color(ann1_xml, "green")
                tree2 = change_color(ann2_xml, "blue")

                root1 = update_title(tree1.getroot()," v7")
                root2 = update_title(tree2.getroot()," v11")
                

                # Create a new root <annotations>
                new_root = ET.Element("annotations")

                # Get all <ndpviewstate> elements from both files
                all_ndpviewstates = root1.findall("ndpviewstate") + root2.findall("ndpviewstate")

                # Assign new sequential IDs starting from 0
                for new_id, ndp in enumerate(all_ndpviewstates):
                    ndp.set("id", str(new_id))  # Update the id
                    new_root.append(ndp)  # Add to new root

                # Create a new tree and save it
                new_tree = ET.ElementTree(new_root)

                new_tree.write(r_xml, encoding="utf-8", xml_declaration=True)
                print(f"Merged .ndpa file created as {r_xml}")

                box_list = get_box_list(wsi_path, r_xml)
                p_list = prune_list(box_list)
                annotations = ET.Element("annotations")
                x_ref, y_ref = get_referance(wsi_path)
                new_tree = write_xml(annotations, 0, p_list, x_ref, y_ref) 
                with open(r_xml, "w") as f:
                      f.write(new_tree)
                f.close()
                print(f"merged common preds")

            else:
                print(f'abnormal file: {filename}')

def update_title(root, label):
    for elem in root.iter():
        if elem.tag == 'title':
            elem.text = elem.text + label
    return root

def prune_list(box_list):    
    print('before prune:',len(box_list))
    bool_list =[True for i in range(len(box_list))]        
    for i in range(len(box_list)-1):
        if bool_list[i] == True:            
            for j in range(i+1, len(box_list)):                
                iou =bb_intersection_over_union(box_list[i],box_list[j])                                     
                if iou > 0.01:                       
                    # replace rectlist{j] with largest of rectlist{i] and rectlist{j]                    
                    x1,y1,x2,y2,conf = replace_bigger_box(box_list[i],box_list[j])
                    box_list[j] = [x1,y1,x2,y2,conf]
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
    conf = "common" 
    return x1,y1,x2,y2,conf

def write_xml(annotations,start_id,anote_list,X_Reference,Y_Reference):    
    id_ = start_id
    for line in anote_list : 
        write_annotation(annotations,id_,line[0],line[1],line[2],line[3],line[4],X_Reference,Y_Reference)   
        id_ +=1
    # Convert the ElementTree object to a raw XML string
    raw_xml = ET.tostring(annotations, encoding='utf-8')  # Ensure UTF-8 encoding
    
    # Pretty-print the XML using minidom
    parsed_xml = xml.dom.minidom.parseString(raw_xml)  # Parse the raw XML
    formatted_xml = parsed_xml.toprettyxml(indent="  ")  # Indent with two spaces
    
    # print("WRITE_XML RETURNED")   
    return formatted_xml

f1 = r'E:\archive_Shreyan\positive cases\yolov7'
f2 = r'E:\archive_Shreyan\positive cases\yolo11_small'
wsi_folder = r'E:\archive_Shreyan\positive cases'
result_folder = r'E:\archive_Shreyan\positive cases\merge_yolo11vs7'
merge_yolo(f1, f2, wsi_folder, result_folder)

