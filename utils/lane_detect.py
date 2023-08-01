import numpy as np
import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

SRC = np.float32([(0.05, 0.7), (0.95, 0.7), (0.02, 1), (1, 1)])
DST = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
DST_SIZE = (640, 480)

LEFT_TOKEN = "left"
RIGHT_TOKEN = "right"


class LaneDetector(object):
    def __init__(self,
                 left_token, right_token,
                 s_thresh=(100, 255), sx_thresh=(15, 255), h_thresh=(0, 50),
                 dst_size=DST_SIZE, perspec_src=SRC, perspec_dst=DST,
                 nwindows=15, win_margin=60, win_minpix=250,
                 lr_ratio=1.8, x_ratio=1.2):
        self.LEFT_TOKEN = left_token
        self.RIGHT_TOKEN = right_token

        # Lane filter
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh
        self.h_thresh = h_thresh

        # Perspective warp
        self.dst_size = dst_size
        self.perspec_src = perspec_src
        self.perspec_dst = perspec_dst

        # Sliding windows
        self.nwindows = nwindows
        self.win_margin = win_margin
        self.win_minpix = win_minpix
        self.LEFT_PIXEL_COLOR = [255, 0, 100]
        self.RIGHT_PIXEL_COLOR = [0, 100, 255]

        # Correction
        self.lr_ratio = lr_ratio
        self.x_ratio = x_ratio


    @staticmethod
    def __rgb2hls(img, sep_channel=False):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        if sep_channel:
            return hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
        else:
            return hls


    @staticmethod
    def __sobel_scaled_abs(img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 1)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        return scaled_sobel


    @staticmethod
    def __binary_in_range(var, thresh):
        r_binary = np.zeros_like(var)
        r_binary[(var >= thresh[0]) & (var <= thresh[1])] = 1
        return r_binary


    def filter_pipeline(self, img):
        # Convert to HLS color space and separate the V channel
        h_channel, l_channel, s_channel = self.__rgb2hls(img, sep_channel=True)

        # Sobel x
        scaled_sobel = self.__sobel_scaled_abs(l_channel)

        # Threshold x gradient
        sx_binary = self.__binary_in_range(scaled_sobel, self.sx_thresh)
        # Threshold color channel
        s_binary = self.__binary_in_range(s_channel, self.s_thresh)
        # Threshold lane color
        h_binary = self.__binary_in_range(h_channel, self.h_thresh)

        # color_binary = np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary)) * 255

        combined_binary = np.zeros_like(sx_binary)
        combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1
        combined_binary[h_binary == 0] = 0

        return combined_binary


    def perspective_warp(self, img, inverse=False):
        img_size = np.float32([(img.shape[1], img.shape[0])])
        if inverse:
            src = self.perspec_dst * img_size
            dst = self.perspec_src * np.float32(self.dst_size)
        else:
            src = self.perspec_src * img_size
            dst = self.perspec_dst * np.float32(self.dst_size)

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, self.dst_size)
        return warped


    @staticmethod
    def __get_hist(img):
        hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
        return hist


    def __step_through_window(self, x_base, img,
                              out_img=None, draw_windows=False, draw_pixel_color=[0, 100, 255]):
        img_height = img.shape[0]
        x_current = x_base
        if draw_windows and out_img is None:
            out_img = np.dstack((img, img, img)) * 255

        # Set height of windows
        win_height = np.int(img_height / self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        for win in range(self.nwindows):
            # Identify window boundaries in x and y
            win_y_low = img_height - (win + 1) * win_height
            win_y_high = img_height - win * win_height

            win_x_low = x_current - self.win_margin
            win_x_high = x_current + self.win_margin

            # Identify the nonzero pixels in x and y within the window
            indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # If you found > minpix pixels, recenter next window on their mean position
            if len(indices) > self.win_minpix:
                x_current = np.int(np.mean(nonzerox[indices]))
            else:
                break

            # Draw the windows on the visualization image
            if draw_windows:
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high),
                              (100, 255, 255), 3)

            # Append these indices to the lists
            lane_inds.append(indices)

        pixel_x, pixel_y = None, None
        if len(lane_inds) > 0:
            # Concatenate the arrays of indices
            lane_inds = np.concatenate(lane_inds) if len(lane_inds) > 0 else None
            # Extract left and right line pixel positions
            pixel_x = nonzerox[lane_inds]
            pixel_y = nonzeroy[lane_inds]
            # Draw pixels within windows
            if draw_windows:
                out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = draw_pixel_color

        return out_img, pixel_x, pixel_y


    def apply_sliding_windows(self, img, draw_windows=False):
        hist = self.__get_hist(img)

        # find peaks of left and right halves
        midpoint = int(hist.shape[0] / 2)
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint

        out_img, leftx, lefty = self.__step_through_window(leftx_base, img,
                                                           draw_windows=draw_windows,
                                                           draw_pixel_color=self.LEFT_PIXEL_COLOR)
        out_img, rightx, righty = self.__step_through_window(rightx_base, img,
                                                             out_img=out_img,
                                                             draw_windows=draw_windows,
                                                             draw_pixel_color=self.RIGHT_PIXEL_COLOR)
        return out_img, leftx, lefty, rightx, righty


    @staticmethod
    def fit_lane_curve(fit_range, x, y):
        x_low, x_high, y_low, y_high = fit_range
        ploty = np.linspace(y_low, y_high - 1, y_high)

        f = np.polyfit(y, x, 2)
        fit_x = f[0] * ploty ** 2 + f[1] * ploty + f[2]

        return fit_x, ploty


    @staticmethod
    def __filter_curve(filter_range, x, y):
        x_low, x_high, _, _h = filter_range
        binary = ((x_low <= x) & (x < x_high))

        filter_x = x[binary]
        filter_y = y[binary]
        return filter_x, filter_y


    def __decide_correction(self, leftx, rightx, width):

        def __two_side_correction(leftx, rightx, lr_ratio):
            pl = max(leftx) - min(leftx)
            pr = max(rightx) - min(rightx)

            if pl > pr:
                ra = pl / pr
                # print("ra =", ra)
                if ra > lr_ratio: # towards left now
                    return self.RIGHT_TOKEN
            else:
                ra = pr / pl
                # print("ra =", ra)
                if ra > lr_ratio: # towards right now
                    return self.LEFT_TOKEN
            return None

        def __one_side_correction(x, point_range,
                                  left_mode=True):
            l, r = point_range
            midpoint = (l + r) // 2

            if left_mode:
                leftx = x[((l <= x) & (x < midpoint))]
            else: # right mode
                leftx = x[x < midpoint]
            if len(leftx) == 0:
                return self.RIGHT_TOKEN

            if left_mode:
                rightx = x[midpoint <= x]
            else: # right mode
                rightx = x[((midpoint <= x) & (x < r))]
            if len(rightx) == 0:
                return self.LEFT_TOKEN

            return __two_side_correction(rightx, leftx, self.lr_ratio)


        corr = None
        if leftx is not None and rightx is not None:
            # print("two side")
            corr = __two_side_correction(leftx, rightx, self.lr_ratio)
        elif leftx is not None:
            # print("left side")
            prange = (0, width // 2)
            corr = __one_side_correction(leftx, prange,
                                         left_mode=True)
        elif rightx is not None:
            # print("right size")
            prange = (width // 2, width)
            corr = __one_side_correction(rightx, prange,
                                         left_mode=False)
        return corr


    def lane_correction(self, img,
                        plot_img=False):
        def __plot():
            # Visualize warp
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(feat_img, cmap='gray')
            ax2.set_title('Warped Image', fontsize=30)
            plt.show()
            # Visualize sliding windows
            f, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 10))
            ax2.imshow(feat_img)
            ax2.set_title('Filter+Perspective Tform', fontsize=30)

            ax3.imshow(out_img)
            if left_filter_x is not None:
                ax3.plot(left_filter_x, left_filter_y, color='yellow', linewidth=10)
            if right_filter_x is not None:
                ax3.plot(right_filter_x, right_filter_y, color='yellow', linewidth=10)
            ax3.set_title('Sliding window+Curve Fit', fontsize=30)

            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()
            # Visualize lane
            img_ = lane_dt.draw_lanes(img, left_fitx, right_fitx)
            plt.imshow(img_, cmap='hsv')
            plt.show()

        h, w, _ = img.shape
        midpoint = w // 2
        left_range = (0, midpoint, 0, h)
        right_range = (midpoint, w, 0, h)
        img_range = (0, w, 0, h)

        # preprocess
        feat_img = self.filter_pipeline(img)
        feat_img = self.perspective_warp(feat_img)

        # sliding window
        out_img, leftx, lefty, rightx, righty = self.apply_sliding_windows(img=feat_img,
                                                                           draw_windows=plot_img)
        # lane curve detection
        left_fitx, right_fitx, left_filter_x, right_filter_x = None, None, None, None
        if leftx is not None:
            left_fitx, ploty = self.fit_lane_curve(fit_range=left_range, x=leftx, y=lefty)
            left_filter_x, left_filter_y = self.__filter_curve(img_range, left_fitx, ploty)
        if rightx is not None:
            right_fitx, ploty = self.fit_lane_curve(fit_range=right_range, x=rightx, y=righty)
            right_filter_x, right_filter_y = self.__filter_curve(img_range, right_fitx, ploty)

        # visualization
        if plot_img:
            __plot()

        # make correction decision
        return self.__decide_correction(leftx=left_filter_x, rightx=right_filter_x, width=w)


    @staticmethod
    def get_curve(img, leftx, rightx, dst_size1=480):  # 720
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y_eval = np.max(ploty)
        ym_per_pix = 30.5 / dst_size1  # meters per pixel in y dimension
        xm_per_pix = 3.7 / dst_size1  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        car_pos = img.shape[1] / 2
        l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        # Now our radius of curvature is in meters
        return (left_curverad, right_curverad, center)


    def draw_lanes(self, img, left_fit, right_fit):
        h, w, _ = img.shape
        if right_fit is None:
            right_fit = np.ones(h) * (w - 1)
        if left_fit is None:
            left_fit = np.zeros(h)

        ploty = np.linspace(0, h - 1, h)
        color_img = np.zeros_like(img)

        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))

        cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))
        # inv_perspective = self.inv_perspective_warp(color_img)
        inv_perspective = self.perspective_warp(color_img, inverse=True)
        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
        return inv_perspective


if __name__ == "__main__":
    img_path = "./images/lane_right.png"
    # img_path = "./lane_true3.jpg"
    lane_dt = LaneDetector(left_token=LEFT_TOKEN, right_token=RIGHT_TOKEN)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    corr = lane_dt.lane_correction(img, plot_img=True)
    print(corr)

    # dst = lane_dt.filter_pipeline(img)
    # # cv2.imwrite("./images/pipeline.png", dst)
    #
    # dst = lane_dt.perspective_warp(dst)
    # # cv2.imwrite("./images/perspective_warp.png", dst)
    #
    # # Visualize undistortion
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(dst, cmap='gray')
    # ax2.set_title('Warped Image', fontsize=30)
    # plt.show()
    #
    # h, w, _ = img.shape
    #
    # out_img, leftx, lefty, rightx, righty = lane_dt.apply_sliding_windows(dst, True)
    # left_fitx, ploty = lane_dt.fit_lane_curve((0, w // 2, 0, h), leftx, lefty)
    # right_fitx, ploty = lane_dt.fit_lane_curve((w // 2, h, 0, h), rightx, righty)
    #
    # img_ = lane_dt.draw_lanes(img, left_fitx, right_fitx)
    # plt.imshow(img_, cmap='hsv')
    # plt.show()
    #
    # f, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 10))
    # ax2.imshow(dst)
    # ax2.set_title('Filter+Perspective Tform', fontsize=30)
    #
    # ax3.imshow(out_img)
    # ax3.plot(left_fitx, ploty, color='yellow', linewidth=10)
    # ax3.plot(right_fitx, ploty, color='yellow', linewidth=10)
    # ax3.set_title('Sliding window+Curve Fit', fontsize=30)
    #
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()
