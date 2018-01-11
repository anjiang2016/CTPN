#
# The codes are used for implementing CTPN for scene text detection, described in: 
#
# Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
# Connectionist Text Proposal Network, ECCV, 2016.
#
# Online demo is available at: textdet.com
# 
# These demo codes (with our trained model) are for text-line detection (without 
# side-refiement part).  
#
#
# ====== Copyright by Zhi Tian, Weilin Huang, Tong He, Pan He and Yu Qiao==========

#            Email: zhi.tian@siat.ac.cn; wl.huang@siat.ac.cn
# 
#   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
#
#

from cfg import Config as cfg
from other import draw_boxes_zmm, resize_im, CaffeModel
import cv2, os, caffe, sys
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer

DEMO_IMAGE_DIR="/data1/mingmingzhao/CTPN/demo_images/"
NET_DEF_FILE="/data1/mingmingzhao/CTPN/models/deploy.prototxt"
MODEL_FILE="/data1/mingmingzhao/CTPN/models/ctpn_trained_model.caffemodel"

if len(sys.argv)>1 and sys.argv[1]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

# text_detect.text_detec(img_url)
demo_imnames=os.listdir(DEMO_IMAGE_DIR)
timer=Timer()

for im_name in demo_imnames:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%im_name

    im_file=osp.join(DEMO_IMAGE_DIR, im_name)
    im=cv2.imread(im_file)

    timer.tic()

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)

    print "Number of the detected text lines: %s"%len(text_lines)
    print "Time: %f"%timer.toc()

    im_with_text_lines=draw_boxes_zmm(im, text_lines, caption=im_name, wait=False)

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "Thank you for trying our demo. Press any key to exit..."
def text_detec(img_url):
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

    # initialize the detectors
    text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
    text_detector=TextDetector(text_proposals_detector)
    im=cv2.imread(img_url)
    timer.tic()
    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)
    obj_num=len(text_lines)
    print "Number of the detected text lines: %s"%len(text_lines)
    print "Time: %f"%timer.toc()
    
    boxstr=u'';
    
    count=0
    #http://192.168.7.37:8393/static/jz66f1d49d97d048fe9e4a62004199d0b2_1_for_trail.jpg
    print text_lines 
    for bbox in text_lines:
        print bbox
        count+=1
        boxstr+="text[%d]:[%f,%f,%f,%f]<br/>"%(count,bbox[0],bbox[1],bbox[2],bbox[3])
    im_name=img_url.split('/')[-1]
    im_name.replace("?",'_')
    im_name.replace("%",'_')
    im_name.replace("&",'_')
    im_name.replace("=",'_')
    local_url=img_url 
    write_path="/data1/mingmingzhao/data_sets/test/text_detect/text_detect_%s" %(local_url.split('/')[-1])
    print "write_path:"+write_path
    im_with_text_lines=draw_boxes_zmm(im, text_lines, caption=write_path, wait=False)
    server_url="http://192.168.7.37:8393/static/text_detect/%s"%(write_path.split('/')[-1])
    print "server_url:"+server_url
    return boxstr,server_url,count
    #return boxstr,img_server_url,obj_num
