import torch
import numpy as np
import cv2
import sys

from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QLabel, QWidget, QTextBrowser


class tehseencode(QDialog):

    def __init__(self):
        super(tehseencode, self).__init__()
        loadUi('tes.ui', self)
        self.logic = 0
        self.btnRekam.clicked.connect(self.btnRekamClicked)
        self.TEXT.setText('tekan "Rekam" untuk deteksi muka')
        self.btnStop.clicked.connect(self.btnStopClicked)
    

    # @pyqtSlot()
    def btnRekamClicked(self):     
        self.logic=1
        model = torch.hub.load('yolov5/', 'custom', source = 'local', path='yolov5s.pt', force_reload=True)
        cap = cv2.VideoCapture(1)
        while (cap.isOpened()):
            global detec
            ret, frame = cap.read()
            results = model(frame)
            if ret == True:   
                coba = results.pandas().xyxy[0]
                coba = coba.values[:, 6:]
                no = len(coba)
                self.TEXT.setText(f"terdeteksi {no} objek")          
                self.displayImage(np.squeeze(results.render()), 1)
                
                cv2.waitKey()
                if (self.logic == 0):
                    self.TEXT.setText('Deteksi Berhenti')
                    break
            else:
                print ('return not found')
        cap.release()
        cv2.destroyAllWindows()
        self.logic = 0 

    def btnStopClicked(self):
        self.logic = 0

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2])==4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.cameraOutput.setPixmap(QPixmap.fromImage(img))

app=QApplication(sys.argv)
window=tehseencode()
window.show()

try:
    sys.exit(app.exec_())
except:
    print('exiting')