from PyQt5 import QtWidgets,QtGui,QtCore
from mainwindow import Ui_MainWindow
import numpy as np 
import sys
import cv2
import matplotlib.pyplot as plt

from function.sfsnet import main as sfs
from function.light_extraction import main as lightExtract
from function.race_change import main as race_change

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.image_res = 256

        #self.ui.colorPicker.clicked.connect(self.color_picker)
        self.ui.browseFacebtn.clicked.connect(self.browseFile)
        #self.ui.browseFaceSfsbtn.clicked.connect(self.browseFile)
        self.ui.extractLightbtn.clicked.connect(lambda:self.extractLight(self.ui.browseLine.text()))
        self.ui.saveOriLightbtn.clicked.connect(lambda:self.saveFile(self.ui.oriDisplay))
        self.ui.saveResultLightbtn.clicked.connect(lambda:self.saveFile(self.ui.resultDisplay))
        # self.ui.saveAlbedobtn.clicked.connect(lambda:self.saveFile(self.ui.albedoDisplay))
        # self.ui.saveNormalbtn.clicked.connect(lambda:self.saveFile(self.ui.normalDisplay))
        # self.ui.saveShadingbtn.clicked.connect(lambda:self.saveFile(self.ui.shadingDisplay))
        # self.ui.saveRecontbtn.clicked.connect(lambda:self.saveFile(self.ui.reconDisplay))
        # self.ui.saveAlbedoHarmobtn.clicked.connect(lambda:self.saveFile(self.ui.albedoHarmodisplay))
        # self.ui.saveNormalHarmobtn.clicked.connect(lambda:self.saveFile(self.ui.normalHarmodisplay))
        # self.ui.saveShadingHarmobtn.clicked.connect(lambda:self.saveFile(self.ui.shadingHarmodisplay))
        # self.ui.saveReconHarmobtn.clicked.connect(lambda:self.saveFile(self.ui.reconHarmodisplay))
        # self.ui.reconBtn.clicked.connect(lambda:self.reconSFS())
        # self.ui.harmonizeBtn.clicked.connect(lambda:self.reconSFS())

    # def color_picker(self):
    #     self.color = QtWidgets.QColorDialog.getColor()

    #     if self.color.isValid():
    #         self.ui.colorDisplay.setStyleSheet("QWidget { background-color: %s ;color: %s}" % (self.color.name(),self.color.name()))

    def display2label (self, image, labelname):
        image[image<0] = 0
        image[image>1] = 1
        image *= 255
        image = image.astype(np.uint8)
        height, width, channel = image.shape
        # plt.imshow(image)
        # plt.show()
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        self.image = QtGui.QPixmap(pixmap01)
        labelname.setPixmap(self.image.scaled(self.image_res,self.image_res,QtCore.Qt.KeepAspectRatio))

    def browseFile(self):
        self.btnName = self.sender().objectName()
        self.openFileNameDialog(self.btnName)
    
    def openFileNameDialog(self,btnName):
        dialog_style = QtWidgets.QFileDialog.DontUseNativeDialog
        dialog_style |= QtWidgets.QFileDialog.DontUseCustomDirectoryIcons
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
        "PNG (*.PNG *.png);; GIF (*.GIF *.gif)", options=dialog_style)
        if self.fileName:
            if(self.btnName == 'browseFacebtn'):
                self.ui.browseLine.setText(self.fileName)
            elif(self.btnName == 'browseFaceSfsbtn'):
                self.ui.browseSfsLine.setText(self.fileName)
        else :
            pass

    # def changeBrowse(self):
    #     if self.ui.checkLight.isChecked() == False:
    #         self.ui.browseSfsLine.setEnabled(True)
    #         self.ui.browseSfsLine.setReadOnly(True)
    #         self.ui.browseFaceSfsbtn.setEnabled(True)
    #     else :
    #         self.ui.browseSfsLine.setEnabled(False)
    #         self.ui.browseFaceSfsbtn.setEnabled(False)
    #         self.ui.browseSfsLine.setText("")

    def extractLight(self,filePath):
        #function to do extraction using model
        if filePath != "":
            # self.oriImage = QtGui.QPixmap(filePath)
            # self.oriResImage = self.oriImage.scaled(self.image_res, self.image_res, QtCore.Qt.KeepAspectRatio)
            # self.ui.oriDisplay.setPixmap(self.oriResImage)
            cut_face,out_face = lightExtract(filePath)
            extracted_face=out_face[0].permute(1,2,0)
            extracted_face=extracted_face.detach().cpu().numpy()
            self.display2label(extracted_face,self.ui.resultDisplay)

            extracted_face=cut_face[0].permute(1,2,0)
            extracted_face=extracted_face.detach().cpu().numpy()
            self.display2label(extracted_face,self.ui.oriDisplay)

    def saveFile(self,image=None):
        self.saveFileNameDialog(image)

    def saveFileNameDialog(self,image):
        if image is None :
            pass
        else :
            dialog_style = QtWidgets.QFileDialog.DontUseNativeDialog
            dialog_style |= QtWidgets.QFileDialog.DontUseCustomDirectoryIcons
            fileName, extension = QtWidgets.QFileDialog.getSaveFileName(self, "choose save file name","whatever",
            "PNG;; JPEG", options=dialog_style)
            if fileName:
                image.pixmap().save(fileName+".png", extension)
            else :
                pass


    # def reconSFS(self):
    #     self.filePath=self.ui.browseSfsLine.text()
    #     self.btnName = self.sender().objectName()
    #     if(self.filePath == ""):
    #         print("Please select PNG file")
    #     #function for doing SFS
    #     if (self.btnName == "reconBtn"):
    #         #sfs
    #         mask = False
    #         if self.ui.mask.isChecked():
    #             mask = True
    #         image,normal,albedo,shading,recon = sfs(self.filePath, mask)

    #         self.display2label(normal,self.ui.normalDisplay)
    #         self.display2label(albedo,self.ui.albedoDisplay)
    #         self.display2label(shading,self.ui.shadingDisplay)
    #         self.display2label(recon,self.ui.reconDisplay)

    #     elif(self.btnName == "harmonizeBtn"):
    #         #sfs+harmonize+display
    #         if (self.ui.raceBox.currentText() != 'None'):
    #             template = self.ui.templateBox.currentText()
    #             if template =="None":
    #                 template = None
    #                 degree = None
    #             else :
    #                 degree = self.color.hue()
    #             skin = self.ui.raceBox.currentText()
    #             if skin == "None":
    #                 skin = None
    #             image = race_change(self.filePath, choice = skin)
    #             image,normal,albedo,shading,recon = sfs(image, template = template, degree =degree  ,skin = skin )
    #             self.display2label(normal,self.ui.normalHarmoDisplay)
    #             self.display2label(albedo,self.ui.albedoHarmodisplay)
    #             self.display2label(shading,self.ui.shadingHarmoDisplay)
    #             self.display2label(recon,self.ui.reconHarmoDisplay)
    #         else :
    #             print("Please Select template / race")


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()