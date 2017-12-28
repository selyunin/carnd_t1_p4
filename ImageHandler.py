'''
Created on Dec 27, 2017
@author: selyunin
'''
import cv2
import numpy as np
from Camera import Camera
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks_cwt
from Line import Line


class ImageHandler():
    def __init__(self, *args, **kwargs):
        self.left_line = Line()  
        self.right_line = Line()
        self.camera = Camera()
    
    def process_image(self, img):
        undist = self.camera.undistort(img)
        binary_thresh = np.uint8(self.apply_thresholds(undist))
        binary_masked = self.region_of_interest(binary_thresh)
        bin_img = self.change_perspective(binary_masked)
        out_img, right_lane_indices = self.get_right_lane(bin_img)
        out_img[right_lane_indices] = [0, 0, 255]
        _, left_lane_indices = self.get_left_lane(bin_img)
        out_img[left_lane_indices] = [255, 0, 0]
        left_poly_fit, left_poly_y, left_poly_x = self.get_left_lane_poly_fit(bin_img)
        right_poly_fit, right_poly_y, right_poly_x = self.get_right_lane_poly_fit(bin_img)
        left_rad = self.radius_of_curvature(left_poly_y, left_poly_x)
        right_rad = self.radius_of_curvature(right_poly_y, right_poly_x)
        left_offset = self.get_lane_offset(left_poly_fit, left_poly_y)
        right_offset = self.get_lane_offset(right_poly_fit, right_poly_y)
        self.left_line.set_current_poly_fit(left_poly_fit, left_poly_y, left_poly_x)
        self.right_line.set_current_poly_fit(right_poly_fit, right_poly_y, right_poly_x)
        self.left_line.set_current_curvature(left_rad)
        self.right_line.set_current_curvature(right_rad)
        self.left_line.set_current_offset(left_offset)
        self.right_line.set_current_offset(right_offset)
        avg_left_offset = self.left_line.get_offset()
        avg_right_offset = self.right_line.get_offset()
        avg_left_rad = self.left_line.get_curvature()
        avg_right_rad = self.right_line.get_curvature()
        offset_from_center = avg_left_offset + avg_right_offset
        newwarp = self.visualize_final(bin_img)   
        result = cv2.addWeighted(undist, 1, newwarp, 0.2, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        msg_rad = "Radius of Curvature = {:.1f} m".format( (avg_left_rad + avg_right_rad)/2 )
        msg_off = "Vehicle is {:.2f} m from center".format( offset_from_center )
        cv2.putText(result, msg_rad, (30,50),  font, 1.6,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(result, msg_off, (30,100), font, 1.6,(255,255,255),2,cv2.LINE_AA)
        return result
    
    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient: x or y 
        # Apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary
    
    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude (l2-norm)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return mag_binary
    
    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction -- arctan(y/x)
        # Apply threshold
        # 1) Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobel_x = np.absolute(sobel_x)
        abs_sobel_y = np.absolute(sobel_y)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
        # 5) Create a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(grad_dir)
        # 6) Return this mask as your binary_output image
        dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
        return dir_binary
    
    def threshold_image(self, image, x_kernel=3, x_thresh=(30, 110), y_kernel=3, y_thresh=(40,110),
                               m_kernel=3, m_thresh=(30, 80),  d_kernel=7, d_thresh=(0.8,1.3)):
        gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=x_kernel, thresh=x_thresh)
        grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=y_kernel, thresh=y_thresh)
        mag_binary = self.mag_thresh(image, sobel_kernel=m_kernel, mag_thresh=m_thresh)
        dir_binary = self.dir_threshold(image, sobel_kernel=d_kernel, thresh=d_thresh)
        
        combined = np.zeros_like(dir_binary)
        #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        combined[((gradx == 1)) | ((mag_binary == 1)) & (dir_binary == 1)] = 1
    
        return combined
    
    def hls_threshold(self, img, s_thresh=(130, 255), l_thresh=(90,255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
        return s_binary, l_binary
    
    def apply_thresholds(self, img):
        s_binary, l_binary = self.hls_threshold(img)
        out_img = self.threshold_image(img)
        combined_binary = np.zeros_like(out_img)
        combined_binary[ (l_binary == 1) & (s_binary == 1) | (out_img == 1)] = 1
        return combined_binary
    
    def get_color_binary(self, img):
        s_binary, l_binary = self.hls_threshold(img)
        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.uint8(np.dstack(( np.zeros_like(s_binary), l_binary, s_binary)) * 255)
        return color_binary
    
    def vertices_img_pipeline(self, img):
        """Define source and destination vertices for perspective transform
        """
        w = img.shape[1]
        h = img.shape[0]
        d_w = w/25
        d_h = 95
        bottom_w_offset = 425
        bottom_h_offset = 15
        w_offset = 0
        src_v1 = [w/2 + w_offset - bottom_w_offset, h - bottom_h_offset]
        src_v2 = [w/2 + w_offset - d_w, h/2 + d_h ]
        src_v3 = [w/2 + w_offset + d_w, h/2 + d_h ]
        src_v4 = [w/2 + w_offset + bottom_w_offset, h - bottom_h_offset]
        src_vertices = np.array( [[src_v1, src_v2, src_v3, src_v4]], dtype=np.float32 )
        dst_offset = 110
        dst_v1 = [src_vertices[0,0,0] + dst_offset, 720]
        dst_v2 = [src_vertices[0,0,0] + dst_offset, 0]
        dst_v3 = [src_vertices[0,3,0] - dst_offset, 0]
        dst_v4 = [src_vertices[0,3,0] - dst_offset, 720]
        dst_vertices = np.array( [[dst_v1, dst_v2, dst_v3, dst_v4]], dtype=np.float32 )
        return src_vertices, dst_vertices
    
    def region_of_interest(self, img):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        src_vertices, _ = self.vertices_img_pipeline(img)
    
        src_v1 = src_vertices[0,0]
        src_v2 = src_vertices[0,1]
        src_v3 = src_vertices[0,2]
        src_v4 = src_vertices[0,3]
        mask_w_bottom_offset = 100
        mask_w_top_offset = 5
        mask_h_top_offset = 10
        mask_v1 = [src_v1[0] - mask_w_bottom_offset, src_v1[1]]
        mask_v2 = [src_v2[0] - 5, src_v2[1] - mask_h_top_offset]
        mask_v3 = [src_v3[0] + 30, src_v3[1] - mask_h_top_offset]
        mask_v4 = [src_v4[0] + mask_w_bottom_offset, src_v4[1]]
        vertices_mask = np.array([[mask_v1, mask_v2, mask_v3, mask_v4]], dtype=np.int32)
        
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices_mask, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    def change_perspective(self, img, inv=False):
        src, dst = self.vertices_img_pipeline(img)
        if not inv:
            M = cv2.getPerspectiveTransform(src, dst)
        else:
            M = cv2.getPerspectiveTransform(dst, src)
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped
    
    def get_lane_peaks(self, img):
        img_height_half = int(img.shape[0]/2)
        histogram = np.sum(img[img_height_half:,:], axis=0)
        histogram = gaussian_filter1d(histogram, 40)
        peaks = find_peaks_cwt(histogram, np.arange(90,300))
        return histogram, peaks
    
    def scan_lane_initial(self, img, peak, nwindows=9):
        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        out_img = np.dstack((img, img, img))*255
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        current_center = peak
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            good_inds, out_img, current_center = self.get_lane_segment(img, 
                                                                  out_img, 
                                                                  window, 
                                                                  current_center, 
                                                                  nonzerox, 
                                                                  nonzeroy)
            lane_inds.append(good_inds)
        return out_img, lane_inds
    
    def get_lane_segment(self, img, out_img, window, current_center, nonzerox, nonzeroy):
        # Set the width of the windows +/- margin
        margin = 140
        # Set minimum number of pixels found to recenter window
        minpix = 50
        nwindows=9
        window_height = np.int(img.shape[0]/nwindows)
        # Identify window boundaries in x and y (and right and left)
        win_y_low  = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_x_low  = current_center - margin
        win_x_high = current_center + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            current_center = np.int(np.mean(nonzerox[good_inds]))
        return good_inds, out_img, current_center
    
    def get_lane(self, img, peak_idx):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        histogram, peaks = self.get_lane_peaks(img)
        out_img, lane_inds = self.scan_lane_initial(img, peaks[peak_idx])
        lane = np.concatenate(lane_inds)
        lane_indices = nonzeroy[lane], nonzerox[lane]
        return out_img, lane_indices
    
    def get_left_lane(self, img):
        return self.get_lane(img, 0)
    
    def get_right_lane(self, img):
        return self.get_lane(img, 1)
    
    def get_lane_poly_fit(self, img, peak_idx):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        histogram, peaks = self.get_lane_peaks(img)
        _, lane_inds = self.scan_lane_initial(img, peaks[peak_idx])
        lane = np.concatenate(lane_inds)
        x = nonzerox[lane]
        y = nonzeroy[lane]
        poly_fit = np.polyfit(y, x, 2)
        poly_y = np.linspace(0, img.shape[0]-1, img.shape[0] )
        poly_x = poly_fit[0]*poly_y**2 + poly_fit[1]*poly_y + poly_fit[2]
        return poly_fit, poly_y, poly_x
    
    def get_left_lane_poly_fit(self, img):
        return self.get_lane_poly_fit(img, 0)
    
    def get_right_lane_poly_fit(self, img):
        return self.get_lane_poly_fit(img, 1)
    
    def get_successive_poly_fit(self, img, poly_fit):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 130
        lane_inds = ((nonzerox > (poly_fit[0]*(nonzeroy**2) + 
                                  poly_fit[1]*nonzeroy + 
                                  poly_fit[2] - margin)) & 
                     (nonzerox < (poly_fit[0]*(nonzeroy**2) + 
                                  poly_fit[1]*nonzeroy + 
                                  poly_fit[2] + margin)))
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 
        new_poly_fit = np.polyfit(y, x, 2)
        poly_y = np.linspace(0, img.shape[0]-1, img.shape[0] )
        poly_x = new_poly_fit[0]*poly_y**2 + new_poly_fit[1]*poly_y + new_poly_fit[2]
        return new_poly_fit, poly_y, poly_x
    
    def radius_of_curvature(self, poly_y, poly_x, ym_per_pix=30/720, xm_per_pix=3.7/700):
        y_eval = np.max(poly_y)
        fit_cr = np.polyfit(poly_y*ym_per_pix, poly_x*xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) /  \
                        np.absolute(2*fit_cr[0])
        return curverad
    
    def get_lane_offset(self, poly_fit, poly_y):
        mid_point = 640
        xm_per_pix=3.7/690
        y_eval = max(poly_y)
        line_x = poly_fit[0]*y_eval**2 \
               + poly_fit[1]*y_eval \
               + poly_fit[2]
        offset_from_center = (line_x - mid_point)*xm_per_pix
        return offset_from_center
    
    def visualize_final(self, img):
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        _, right_lane_indices = self.get_right_lane(img)
        _, left_lane_indices = self.get_left_lane(img)
        color_warp[right_lane_indices] = [255, 0, 0]
        color_warp[left_lane_indices] = [255, 0, 0]
        left_poly_fit, left_poly_y, left_poly_x = self.get_left_lane_poly_fit(img)
        right_poly_fit, right_poly_y, right_poly_x = self.get_right_lane_poly_fit(img)
        pts_left = np.array([np.transpose(np.vstack([left_poly_x, left_poly_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_poly_x, right_poly_y])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # color_warp[right_lane_indices] = [255,0, 0]
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        new_warp = self.change_perspective(color_warp, inv=True)
        return new_warp