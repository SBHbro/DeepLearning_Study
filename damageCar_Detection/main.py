import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
from PIL import Image
import os

import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
from werkzeug import secure_filename
from keras import backend as K

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import custom 
import datetime as pydatetime
import shutil

def get_now():
    """
        현재 시스템 시간을 datetime형으로 반환
    """
    return pydatetime.datetime.now()

def get_now_timestamp():
    """
        현재 시스템 시간을 POSIX timestamp float형으로 반환
    """
    return get_now().timestamp()

PEOPLE_FOLDER =os.path.join('static', 'image')

app = Flask(__name__)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

custom_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_damage_0010.h5")  # TODO: update this path

config = custom.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, "static")
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER




# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Load weights
class_names = ['BG', 'Damage']


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        
        # 업로드 파일 처리 분기
        K.clear_session()
        file = request.files['image']
        if not file: return render_template('index.html', label="No Files")
        
        # timestamp 생성
        ts = get_now_timestamp()
        ts = pydatetime.datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d %H-%M-%S')
        
        firstImage = ts + 'damage.jpg'     
        firstImageDir = os.path.join(app.config['UPLOAD_FOLDER'], '원본/')
        #firstImageDir = os.path.join(firstImageDir, firstImage)
        
        # 이미지 픽셀 정보 읽기 & 원본 저장
        file.save(firstImageDir + secure_filename(firstImage))
        image = misc.imread(file)
        
        

        
        # 신경망 모델 불러오기
        
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
        model.load_weights(custom_WEIGHTS_PATH, by_name=True)
        results = model.detect([image], verbose=1)
        
        # 사진 검출
        ax = get_ax(1)
        r = results[0]
        frame = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'], ax=ax,
                                    title="Predictions")
        #img = Image.fromarray(frame.astype(np.uint8), 'RGB')              
        #?t=12234234
        
        ts = ts + 'damage.jpg'
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], ts)
        ax.figure.savefig(fname=full_filename, bbox_inches='tight', pad_inches=0)
        
        #캐시 초기화를 위한 타임스탬프 생성
        ts = get_now_timestamp()
        full_filename = os.path.join(full_filename, '?t=')
        ts = str(ts)
        full_filename = os.path.join(full_filename, ts)
        
        # 결과 리턴
        return render_template('index.html', label=full_filename)

@app.route('/scoreto', methods=['POST'])
def movefile():
    if request.method == 'POST':
        
        #값 받아오기
        value = request.form['score']
        img = request.form['img']
        
        #이미지 경로 추출
        a = img.partition('\\?t')
        b = a[0]
        b = a[0].partition('ge\\')
        path = app.config['UPLOAD_FOLDER'] + '/'
        
        #점수별 이미지 경로 변환
        if value == '1':
            movepath = os.path.join(app.config['UPLOAD_FOLDER'],'1')
            movepath = movepath + '/'
            shutil.move(path+b[2],movepath + b[2])
        
        if value == '2':
            movepath = os.path.join(app.config['UPLOAD_FOLDER'],'2')
            movepath = movepath + '/'
            shutil.move(path+b[2],movepath + b[2])
            
        if value == '3':
            movepath = os.path.join(app.config['UPLOAD_FOLDER'],'3')
            movepath = movepath + '/'
            shutil.move(path+b[2],movepath + b[2])
            
        if value == '4':
            movepath = os.path.join(app.config['UPLOAD_FOLDER'],'4')
            movepath = movepath + '/'
            shutil.move(path+b[2],movepath + b[2])
            
        if value == '5':
            movepath = os.path.join(app.config['UPLOAD_FOLDER'],'5')
            movepath = movepath + '/'
            shutil.move(path+b[2],movepath + b[2])
        
        return render_template('index.html')
    
    
if __name__ == '__main__':

        
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
