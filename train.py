from ultralytics import YOLO

if __name__ == "__main__":
    
    #object detection training

    # model = YOLO(r'w\abnormals.pt')
    # model = YOLO(r'dump\yolo11l.pt')
    # model.train(data='cfg\cell.yaml', epochs=1000, imgsz=640, batch=16, workers=4, device="cuda", project=r'runs\detect\cell', name='cellmar20', patience=100)

    # model = YOLO(r'D:\Shreyan\Development\yolo11\runs\detect\cell\cellmar20\weights\best.pt')
    # model = YOLO(r'D:\Shreyan\Development\yolo11\runs\detect\abnormal\abnormalmar20\weights\best.pt')
    # model.train(data=r'cfg\abnormal.yaml', epochs=2000, imgsz=640, batch=16, workers=4, device="cuda", project=r'runs\detect\abnormal', name='abnormalmar20', patience=200)

    # model = YOLO(r'D:\Shreyan\Development\yolo11\runs\detect\abnormal\abnormalmar202\weights\best.pt')
    # model.train(data=r'cfg\cat.yaml', epochs=1000, imgsz=640, batch=16, workers=4, device="cuda", project=r'runs\detect\cat', name='catmar20', patience=100)

    # After training, evaluate on the test set
    # model.val(data="cat.yaml", split="test")  # Evaluate on the test set



    # segemtation training

    model = YOLO('yolo11s-seg.pt')
    model.train(data=r'cfg\segment.yaml', epochs=1000, imgsz=640, batch=16, workers=4, device="cuda", project=r'runs\train\segment', name='segapr4', patience=100)
    # model.predict(r"D:\Shreyan\Development\Train\Data\cell\yolo11_cell\images\val", save=True, imgsz=640, conf=0.2)





    # classification training

    # model = YOLO('best.pt')
    # model.train(data=r'D:\Shreyan\Development\Train\Data\classification\classify_split_fp', epochs=1000, batch=16, workers=4, device="cuda", project=r'runs\train\classify', name='apr1', patience=100)
    # model.predict(r"D:\Shreyan\Development\Train\Data\cell\yolo11_cell\images\val", save=True, imgsz=640, conf=0.2)



    