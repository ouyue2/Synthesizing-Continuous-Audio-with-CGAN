import sys,os,time

import subprocess

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QFileDialog, QLineEdit, QComboBox

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import numpy as np
from loader import decode_audio


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        
        self.setWindowTitle("Synthesizing Continuous Audio")
        
        #Set Widgets
        self.label_ml = QLabel("Choose model your audio generated from")
        self.label_ml.setFixedWidth(480)
        
        self.label_ml_ml = QLabel("Model:")
        self.label_ml_ml.setFixedWidth(48)
        
        self.model_list=["Bird Song","Ocean Wave","Piano","Railway Train","Customized"]
        self.combo_ml = QComboBox()
        self.combo_ml.addItem(" ")
        self.combo_ml.addItems(self.model_list)
        
        self.combo_ml.currentIndexChanged.connect(self.model_sel)
        
        self.label_ml_path = QLabel("Path:")
        self.label_ml_path.setFixedWidth(48)
        self.path_ml = QLabel("Model path")
        self.path_ml.setFrameStyle(QFrame.Panel|QFrame.Sunken)
        self.button_ml_path = QPushButton("Choose")
        self.button_ml_path.clicked.connect(self.open_ml)
        self.button_ml_path.setEnabled(False)
        self.button_ml_path.setFixedWidth(60)
        
        self.label_out = QLabel("Choose path for your output audio")
        self.label_out.setFixedWidth(480)
        
        self.label_out_path = QLabel("Path:")
        self.label_out_path.setFixedWidth(48)
        self.path_out = QLabel("Output path")
        self.path_out.setFrameStyle(QFrame.Panel|QFrame.Sunken)
        self.button_out_path = QPushButton("Choose")
        self.button_out_path.clicked.connect(self.open_out)
        self.button_out_path.setFixedWidth(60)
        
        self.label_otime = QLabel("Output Time (in seconds):")
        self.edit_otime = QLineEdit("15")
        self.edit_otime.setFixedWidth(90)
        
        self.button_run = QPushButton("Run")
        self.button_run.clicked.connect(self.run)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.stop)
        
        self.canvas_out = FigureCanvas(Figure())
        self.toolbar_out = NavigationToolbar(self.canvas_out, self)
        self.ax_out = self.canvas_out.figure.subplots()
        self.timer_out = self.canvas_out.new_timer(2000, [(self.update, (), {})])
        
        #Set Layout        
        self.hbox_ml_ml = QHBoxLayout()
        self.hbox_ml_ml.addWidget(self.label_ml_ml)
        self.hbox_ml_ml.addWidget(self.combo_ml)
        
        self.hbox_ml_path = QHBoxLayout()
        self.hbox_ml_path.addWidget(self.label_ml_path)
        self.hbox_ml_path.addWidget(self.path_ml)
        self.hbox_ml_path.addWidget(self.button_ml_path)
        
        self.hbox_out_path = QHBoxLayout()
        self.hbox_out_path.addWidget(self.label_out_path)
        self.hbox_out_path.addWidget(self.path_out)
        self.hbox_out_path.addWidget(self.button_out_path)
        
        self.hbox_otime = QHBoxLayout()
        self.hbox_otime.addWidget(self.label_otime)
        self.hbox_otime.addWidget(self.edit_otime)
        
        self.hbox_ctrl = QHBoxLayout()
        self.hbox_ctrl.addWidget(self.button_run)
        self.hbox_ctrl.addWidget(self.button_stop)
        
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label_ml)
        self.vbox.addLayout(self.hbox_ml_ml)
        self.vbox.addLayout(self.hbox_ml_path)
        self.vbox.addWidget(self.label_out)
        self.vbox.addLayout(self.hbox_out_path)
        self.vbox.addLayout(self.hbox_otime)
        self.vbox.addLayout(self.hbox_ctrl)
        self.vbox.addWidget(self.toolbar_out)
        self.vbox.addWidget(self.canvas_out)
        
        self.setLayout(self.vbox)
    
    def open_ml(self):
        self.path_model,self.filetype_path_model=QFileDialog.getOpenFileName(self,
                                                                      "Choose Model File",
                                                                      "./",
                                                                      "Checkpoint(model.ckpt-*)")
        if self.path_model is None:
            self.path_ml.setText("Model path")
        else:
            self.path_ml.setText(self.path_model)
    def open_out(self):
        self.path_output,self.filetype_path_output=QFileDialog.getSaveFileName(self,
                                                                      "Choose Output File",
                                                                      "./",
                                                                      "Wave(*.wav)")
        if self.path_output is None:
            self.path_ml.setText("Model path")
        else:
            self.path_out.setText(self.path_output)
    
    def model_sel(self,index):
        if self.combo_ml.itemText(0) == " ":
            self.combo_ml.removeItem(0)
        self.path_dict={"Bird Song":"./bird/model.ckpt-0","Ocean Wave":"./ocean/model.ckpt-0","Piano":"./piano/model.ckpt-0","Railway Train":"./rwtrain/model.ckpt-0","Customized":None}
        self.path_model=self.path_dict[self.combo_ml.currentText()]
        if self.combo_ml.currentText() == "Customized":
            self.button_ml_path.setEnabled(True)
            self.path_ml.setText("Model path")
        else: 
            self.button_ml_path.setEnabled(False)
            self.path_ml.setText(self.path_model)
    def run(self):
        path_format = int(self.path_model.find('.',self.path_model.find('model.ckpt')+11))
        
        if path_format>0:
            self.path_model_1 = self.path_model[0:path_format]
        else:
            self.path_model_1 = self.path_model
        
        self.run_cmd = "python","generate.py","--ckpt_path",self.path_model_1,"--wav_out_path",self.path_output
        
        self.timer_out.start()
        
        # self.run_cmd = ['python',self.run_args]
        self.gen = subprocess.Popen(self.run_cmd)
        print("Start process with pid %s" % self.gen.pid)
    
    def update(self):
        self.ax_out.clear()
        while os.path.isfile(self.path_output) is False:
            time.sleep(1)
        while True:
            try:
                signal_out_r = decode_audio(self.path_output)
                print("File opened!")
                break
            except NotImplementedError:
                time.sleep(2)
                print("Trying reopening...")
        
        signal_out=signal_out_r[:,0]
        # Shift the sinusoid as a function of time.
        self.ax_out.plot(signal_out)
        self.ax_out.figure.canvas.draw()
    
    def stop(self):
        self.timer_out.stop()
        print("Killing process with pid %s" % self.gen.pid)
        try:
            self.gen.kill()
        except Exception as error:
            print(error)
            

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
