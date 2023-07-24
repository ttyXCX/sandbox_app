import rospy
from sensor_msgs.msg import Image
import cv2
import requests

import json
import uuid
import os
import traceback

# from ..utils.convert import ImageConverter
from convert import ImageConverter
from cv_bridge import CvBridge

URL = "http://47.101.169.122:8762/vision/detect_traffic_light_color"
IMG_PATH = "/home/agilex/sandbox_app/images/"

img_cvt = ImageConverter()
cv_bridge = CvBridge()

rospy.init_node("image_retriever", anonymous=True)

COLOR = None
SIZE = None
SIZE_THRESHOLD = 3200


def __retieve_iamge(compression=None):
    rgb_Image = rospy.wait_for_message("/camera/rgb/image_raw", Image, timeout=1)
    # height, width, rgb_data = rgb_Image.height, rgb_Image.width, rgb_Image.data
    ## reorganize rgb img data
    cv_img = cv_bridge.imgmsg_to_cv2(rgb_Image, desired_encoding="passthrough") # RGB

    if compression is not None:
        img_name = uuid.uuid1().hex
        cv_img = img_cvt.compress_image(cv_img, ratio=compression)

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    img_path = "{}.{}".format(IMG_PATH + img_name, "png")
    cv2.imwrite(img_path, cv_img) # cv2: BGR saved as RGB

    return img_path


def __query_image(img_path):
    enc_img = img_cvt.encode_from_path(img_path)
    rsp = requests.post(url=URL, data={"image": enc_img})
    rsp = json.loads(rsp.text)
    return rsp


def __output_variables():
    print("color={}, size={}, threshold={}".format(COLOR, SIZE, SIZE_THRESHOLD))


def __delete_image(img_path):
    os.remove(img_path)


def scan():
    img_path = __retieve_iamge(compression=0.5)
    rsp = __query_image(img_path)
    __delete_image(img_path)

    try:
        global COLOR
        global SIZE

        data = rsp["data"]
        COLOR = data["color"]
        SIZE = data["size"]
    except Exception as e:
        print(e.args)
        print(traceback.format_exc())


def is_exist_traffic_light():
    return COLOR is not None


def is_traffic_light_green():
    return COLOR == "green"


def is_safe_distance():
    return SIZE is not None and SIZE <= SIZE_THRESHOLD
    


if __name__ == "__main__":
    while (True):
        op = raw_input("input operation:")

        if op == "exit":
            break
        elif op == "scan":
            scan()
        elif op == "light?":
            print(is_exist_traffic_light())
        elif op == "green?":
            print(is_traffic_light_green())
        elif op == "safe?":
            print(is_safe_distance())
        elif op == "o":
            __output_variables()
        else:
            print("{}Invalid operation '{}'{}".format("\033[0;31;47m", op, "\033[0m"))