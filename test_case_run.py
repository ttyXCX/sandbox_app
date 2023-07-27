import rospy
import requests
import random
import time
import json
from utils.convert import ImageConverter

from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
cv_bridge = CvBridge()

if __name__ == "__main__":
    img_cvt = ImageConverter()

    '''
    retreive image & convert
    '''
    rospy.init_node("img_node")
    print("waiting for message")

    rgb_Image = rospy.wait_for_message("/camera/rgb/image_raw", Image, timeout=1)
    height, width = rgb_Image.height, rgb_Image.width
    rgb_data = rgb_Image.data
    
    print("rgb image: height={}, width={}, data length={}"\
          .format(height, width, len(rgb_data)))

    # reorganize rgb img data
    cv_img = cv_bridge.imgmsg_to_cv2(rgb_Image, desired_encoding="passthrough") # RGB
    print(type(cv_img), cv_img.shape)
    # cv_img = img_cvt.compress_image(cv_img, ratio=0.5)

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/home/agilex/sandbox_app/recieved_rgb_img.png", cv_img) # cv2: BGR saved as RGB


    ## depth_Image = rospy.wait_for_message("/camera/depth/image", Image, timeout=1)
    ## height, width = depth_Image.height, depth_Image.width
    ## depth_data = depth_Image.data

    ## print("depth image: height={}, width={}, data length={}"\
    ##       .format(height, width, len(depth_data)))

    '''
    test color detection
    '''
    # EPOCHS = 10
    # URL = "http://47.101.169.122:8762/vision/detect_traffic_light_color"
    # truth_colors = ["red", "green", "yellow", "red_yellow"]
    
    # time_start = time.time()
    # for i in range(EPOCHS):
    #     tclr = random.choice(truth_colors)
    #     path = "/home/agilex/sandbox_app/images/light_{}.png".format(tclr)

    #     # path = "/home/agilex/sandbox_app/images/recieved_rgb_img.png".format(tclr)
    #     enc_img = img_cvt.encode_from_path(path)
    #     rsp = requests.post(url=URL, data={"image": enc_img})
    #     rsp = json.loads(rsp.text)

    #     print("{}/{}, time consumed:{:.2f}s, true-detect: {}-{}, size: {}".format(
    #            (i + 1), EPOCHS, 
    #            (time.time() - time_start) / (i + 1), 
    #            tclr, rsp["data"]["color"],
    #            rsp["data"]["size"]))

    # time_end = time.time()

    # avg_time = (time_end - time_start) / EPOCHS
    # avg_fps = 1 / avg_time
    # print("\naverage time: {:.2f}s, {:.2f} fps".format(avg_time, avg_fps))