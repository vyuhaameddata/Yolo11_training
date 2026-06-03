# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:04:18 2022 by Krishna

@author: lucid

Docmumented on 20 feb 2024 by shreyan
dump_annotation_tiles : function to call for generating tiles from annotations 
get_lnb : takes ndpa file and to create list for each annotation in the file :  [(id, length breadth, center coords, title)

edited by shreyan on 21 feb 2025: to adjust to naming convention for new training data
"""

import os
from tifffile import imsave
import random
import xml.etree.ElementTree as ET
import numpy as np
import openslide 

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

def get_lnb(xml_path,wsi_path,nm_p=221):    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x1,y1,x2,y2 = 0,0,0,0    
    lnb =[]
    X_Reference, Y_Reference = get_referance(wsi_path, nm_p)
    for elem in root.iter():
        #print(elem.tag)
        if elem.tag == 'ndpviewstate':
            _id = elem.attrib.get('id')        
            title = elem.find('title').text    
        x = []
        y = []
        if elem.tag == 'pointlist':
            for sub in elem.iter(tag='point'):
                x.append(int(sub.find('x').text))                    
                y.append(int(sub.find('y').text))                    
            x1=float((min(x) + X_Reference)/nm_p)
            x2=float((max(x) + X_Reference)/nm_p)
            y1=float((min(y) + Y_Reference)/nm_p)
            y2=float((max(y) + Y_Reference)/nm_p)                         

            breath = abs(x2-x1)
            length = abs(y2-y1)          
            cx=(x1+x2)/2
            cy=(y1+y2)/2
            row = (int(_id),breath,length,cx,cy,title)                
            lnb.append(row)             
    return lnb

            
# def handle_border_tiles(left,top,tile_size,x1,y1,breath,length):                
#     if (left + tile_size) >= dims[0]:                      
#           left  = dims[0]-tile_size - breath/2          
#     if left < 0 :
#           left = 0                                            
#     if(top +tile_size)>= dims[1]:
#           top = dims[1]-tile_size - length/2
#     if (top) < 0:
#           top = 0
#     return left,top

def tile_intersection(boxA, boxB,limit=5):	
    xA = max(boxA[0], boxB[0]) 
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])	
    yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    return (interArea > limit)

def update_SameTile_annotations_new(tile,anote,lnb):
    for line in lnb:
        if line[5] != "":
            breath = line[1]
            length = line[2]
            cx = int(line[3])
            cy = int(line[4])                
            anote_box = (cx-breath/2,cy-length/2,cx+breath/2,cy+length/2)
            if (tile_intersection(tile, anote_box)):            
                x1 = anote_box[0]-tile[0]
                y1 = anote_box[1]-tile[1]
                box = (int(x1),int(y1),int(x1+breath),int(y1+length),line[5])
                #print(box/1024)
                anote.append(box)            
    return anote

def xy2yolo(x1,y1,x2,y2,tile_size):    
    
    b_center_x = (x1 + x2)/ 2 
    b_center_y = (y1 + y2) / 2
    b_width    = (x2 - x1)
    b_height   = (y2 - y1)
    
    # Normalise the co-ordinates by the dimensions of the image
    
    b_center_x /= tile_size
    b_center_y /= tile_size
    b_width    /= tile_size
    b_height   /= tile_size
    t =b_center_x, b_center_y, b_width, b_height
    t = tuple(0 if i < 0 else 1 if i > 1 else i for i in t)
    return t

"""
this uses just ndpa/xml file for all info on the annotations.
thereby we take only rectangles marked.
select class dict for required setting

"""
# class dict for False Negative data 
class_dict = {"LSIL": 0 ,"HSIL": 1 ,"ASC-H": 2 , "ASCUS": 3 , "AGUS": 4, "SQ": 5}

# class dict for False positive data
class_dict = {'OUT OF FOCUS': 0, 'unknown': 1, 'Bare Nucleui': 2, 'OVER LAP': 3, 'cell clusters': 4, 'metaplastic cells': 5, 'WBC': 6, 'folded cell': 7, 'MUCK/DURT/STAIN DEPOSIT': 8, 'ATROPHY': 9, 'ARTIFACTS': 10, 'BLANK ARTIFACTS': 11, 'neutrophils overlap on squamous cells': 12, 'HYPERSTAIN': 13, 'Pseudo kolilocytes': 14, 'Apoptosis': 15, 'hyperstaning parabasal cells': 16, 'Reactive degeretal endocervical cells': 17, 'Reactive intermediate cells': 18}

# class dict for abnormal data
class_dict = {'abnormal': 0, 'FP':0, 'FN':0, 'cell':0}

def Get_Defect_tiles_new(fname,lnb,dump_folder,slide,defect,tile_size=1024):     
    print('tile_size',tile_size)
    count, count_bg, count_bgcell = 0 , 0 , 0
    images_folder = os.path.join(dump_folder,'images')
    labels_folder = os.path.join(dump_folder,'labels')
    if not os.path.isdir(images_folder) : os.mkdir(images_folder)
    if not os.path.isdir(labels_folder) : os.mkdir(labels_folder)
    
    for line in lnb:
        tile_anote =[]                    
        print_buffer =[]
        title = line[5]
        # if line[5] != None:
        #     for s in check:
        #         if s in line[5]:
        #             title = s
        # if line[5] != None:
        #     title = line[5].split(",")[1] if "," in line[5] else ""
        #     ab = title.split(' ')[0] if " " in title  else ""
        
        if title != '':
            randx = random.randint(int(tile_size*0.25), int(tile_size*0.75))
            randy = random.randint(150, 450)                                                                  
            x1 = randx
            y1 = randy                                
            breath = line[1]
            length = line[2]
            cx = int(line[3])
            cy = int(line[4])
            left = int(cx -randx -breath/2)
            top =  int(cy -randy -length/2)

            tile_box = (left ,top, left + tile_size,top + tile_size)
            box = (int(x1), int(y1), int(x1+breath), int(y1+length), line[5]) 
            #tile_anote.append(box)
            
            tile_anote = update_SameTile_annotations_new(tile_box, tile_anote, lnb)        

            level = slide.get_best_level_for_downsample(1.0 / 40)        
            im_roi = slide.read_region((left, top), level, (tile_size, tile_size))
            file = im_roi.convert('RGB')
            np_img = np.array(file)
            filename = r'{0}_{1}.tif'.format(fname,count)             
            # filename = r'{0}_{1}_{2}.tif'.format(fname,title,count)                
            if np_img.shape[0]!=tile_size or np_img.shape[1]!=tile_size:
                print('In',fname,'tile:',filename, 'is abnormal')
            elif len(tile_anote)!=0:                  
                filepath = os.path.join(images_folder,filename)
                # writer = Writer(filepath, tile_size, tile_size)            
                imsave(filepath,np_img)                 
                count +=1
                if defect:            
                    for k in range(len(tile_anote)):
                        b_center_x, b_center_y, b_width, b_height = xy2yolo(tile_anote[k][0],tile_anote[k][1], tile_anote[k][2], tile_anote[k][3],tile_size)            
                        title = str(tile_anote[k][4])
                        if title != None:
                            for s in class_dict:
                                if s in line[5]:
                                    title = s
                            try:
                                class_id = class_dict[title]  
                                print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
                                # print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(0, b_center_x, b_center_y, b_width, b_height))   
                            except Exception as e : print("exception",e)

                     
                    yolo_name = os.path.join(labels_folder,filename.replace("tif","txt"))
                    # print("\n".join(print_buffer), file= open(yolo_name, "w"))
                    print("".join(print_buffer), file= open(yolo_name, "w"))
                else :
                    yolo_name = os.path.join(labels_folder,filename.replace("tif","txt"))
                    print("\n".join(print_buffer), file= open(yolo_name, "w"))                            
            # if count > 50 : break
        else:
            if title == "BG" : 
                count_bg +=1
            else:
                count_bgcell +=1 
            # print('title',title)    
    print(fname,"cell, bg_markings,cell_bg", count,count_bg,count_bgcell)    
    
    """
    this pieces of code incase we want to compare on label image.
    can uncomment and use it just incase.
    """
    # if defect:
    #     tile_ref_folder = os.path.join(dump_folder,'tile_refs')
    #     save_ndpi_ref_file = os.path.join(tile_ref_folder,fname+'.csv')                
    #     ndpi_df = pd.DataFrame(print_buffer, columns=['filename','left','top'])
    #     ndpi_df.to_csv(save_ndpi_ref_file)          

""" 
uses two inputs 
1) from the csv which takes care of the reference issues,
2) taken from the ndpa/xml file which is what the ndpi viewer uses.

"""
 

def dump_annotation_tiles(folder,tile_size,nm_p):
    dump_folder = os.path.join(folder,"check_data")
    if not os.path.isdir(dump_folder):   
        os.mkdir(dump_folder)       
    
    for files in os.listdir(folder):    
        if files.endswith('.ndpa'):
            filename = files.split('.')[0]  
            print('start =>',filename)        
            csv_name = filename + ".csv"
            csv_path =  os.path.join(folder, csv_name)        
            #print("csv path :", csv_path)
            #df = pd.read_csv(csv_path)
            xml_name = filename + ".ndpi.ndpa"
            xml_path = os.path.join(folder,xml_name)
            wsi = filename + ".ndpi"
            wsi_path = os.path.join(folder,wsi)
            slide = openslide.open_slide(wsi_path)        
            lnb = get_lnb(xml_path,wsi_path,nm_p)
            print('tile dumping starts')
            # Get_Defect_tiles(filename,lnb,csv_path,dump_folder,slide,ts,dims,defect=False,tile_size=tile_size)    
            Get_Defect_tiles_new(filename,lnb,dump_folder,slide,defect=True,tile_size=tile_size)    
            # Get_fp_tiles(filename,lnb,csv_path,dump_folder,ts,dims,defect=True,tile_size=tile_size)                
            print('done =>',filename)
            print('\n')
            # break