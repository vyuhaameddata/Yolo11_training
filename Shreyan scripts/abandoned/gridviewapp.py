import sys
import os
import cv2
import glob
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage

class YOLOGridViewer(QWidget):
    def __init__(self, image_folder, label_folder, grid_size=(3, 3)):
        super().__init__()
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.grid_size = grid_size
        self.image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))  # Change extension if needed
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("YOLO Dataset Grid Viewer")
        layout = QGridLayout()
        
        for i, image_path in enumerate(self.image_paths[:self.grid_size[0] * self.grid_size[1]]):
            row, col = divmod(i, self.grid_size[1])
            label = QLabel(self)
            pixmap = self.loadImageWithBBoxes(image_path)
            label.setPixmap(pixmap)
            label.mousePressEvent = lambda event, img=image_path: self.openEditor(img)
            layout.addWidget(label, row, col)
        
        self.setLayout(layout)
    
    def loadImageWithBBoxes(self, image_path):
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        label_path = os.path.join(self.label_folder, os.path.basename(image_path).replace(".jpg", ".txt"))
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    cls, x, y, w, h = map(float, line.strip().split())
                    x, y, w, h = int(x * width), int(y * height), int(w * width), int(h * height)
                    cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimage).scaled(200, 200)
    
    def openEditor(self, image_path):
        os.system(f"labelImg {image_path}")  # Open LabelImg for editing

if __name__ == "__main__":
    app = QApplication(sys.argv)
    image_folder = QFileDialog.getExistingDirectory(None, "Select Image Folder")
    label_folder = QFileDialog.getExistingDirectory(None, "Select Label Folder")
    viewer = YOLOGridViewer(image_folder, label_folder)
    viewer.show()
    sys.exit(app.exec_())
