import numpy as np
import glob
import os
import cv2
import pandas as pd
from pdf2image import convert_from_path

# scale an image, and return
def scaleCropTopBottom_old(img,crop_height_percent=28.3,crop_width_percent=53.3,
                       top_offset=35,bot_offset=35,left_offset=30,right_offset=30,show=True):
    # resize
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    #img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    crop_height = int(img.shape[0] * (crop_height_percent / 100)) # get the top 28% of the image
    crop_width = int(img.shape[1] * (crop_width_percent / 100)) # get the left 53% of the image
    img_top = img[0+top_offset:int(crop_height/2)-bot_offset, 0+left_offset:crop_width-right_offset]
    img_bot = img[int(crop_height/2)+top_offset:crop_height-bot_offset, 0+left_offset:crop_width-right_offset]

    _showImages(show, images=[img_top,img_bot],titles=["Top","Bottom"])

    return(img_top,img_bot)


def scaleCropTopBottom(img, crop_height_percent=28.3, crop_width_percent=53.3,
                       top_offset=35, bot_offset=35, left_offset=30, right_offset=30,
                       show=True, manual=True):
    if manual:
        # Manually select bounding box for top
        print("Select the top part of the image")
        roi_top = cv2.selectROI("Select Top", img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Top")

        # Manually select bounding box for bottom
        print("Select the bottom part of the image")
        roi_bot = cv2.selectROI("Select Bottom", img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Bottom")

        # Crop based on manually selected regions
        img_top = img[int(roi_top[1]):int(roi_top[1] + roi_top[3]), int(roi_top[0]):int(roi_top[0] + roi_top[2])]
        img_bot = img[int(roi_bot[1]):int(roi_bot[1] + roi_bot[3]), int(roi_bot[0]):int(roi_bot[0] + roi_bot[2])]

    else:
        # Automatic cropping based on percentages
        crop_height = int(img.shape[0] * (crop_height_percent / 100))  # get the top 28% of the image
        crop_width = int(img.shape[1] * (crop_width_percent / 100))  # get the left 53% of the image
        img_top = img[0 + top_offset:int(crop_height / 2) - bot_offset, 0 + left_offset:crop_width - right_offset]
        img_bot = img[int(crop_height / 2) + top_offset:crop_height - bot_offset,
                  0 + left_offset:crop_width - right_offset]

    # Show cropped images if requested
    _showImages(show, images=[img_top, img_bot], titles=["Top", "Bottom"])

    return img_top, img_bot

def findWingContour(img, use_len_thru_white=True, use_pixel_area=True, cm_per_pixel=0.0108, dilate_kernel_size=5, thresh_c=5, show=False):

    start_img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur image
    #img = cv2.blur(img, ksize=(5,5))

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 21, thresh_c)

    # invert mask so shapes are white on black background
    thresh = cv2.bitwise_not(thresh)

    # remove islands
    # Find all connected components (blobs) in the image
    #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    # Create an output image that will hold the results
    #mask = np.zeros_like(thresh)
    # Go through all found components
    #for label in range(1, num_labels):  # label 0 is the background
        # If the component size is greater than or equal to 5, keep it
    #    if stats[label, cv2.CC_STAT_AREA] >= 1500: #1000
    #        mask[labels == label] = 255

    # new remove islands
    # Find all connected components (blobs) in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    # Create an output image that will hold the results
    mask = np.zeros_like(thresh)
    # Find the largest component's label, excluding the background (label 0)
    if num_labels > 1:  # Ensure there is at least one component apart from the background
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Find the index of the largest component
        mask[labels == largest_label] = 255  # Assign 255 only to the largest component

    # dilate
    # Define the kernel size.
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    # Dilate the image
    mask = cv2.dilate(mask, kernel, iterations=1)

    # add black border before flood fill
    mask = cv2.copyMakeBorder(mask, top=5, bottom=5, left=5, right=5, borderType=cv2.BORDER_CONSTANT,
                                        value=0)

    # Fill in inside of wings using flood fill, invert, then flood fill
    to_fill = np.copy(mask)
    # Note: The size needs to be 2 pixels more than the image size
    h, w = to_fill.shape[:2]
    extended = np.zeros((h + 2, w + 2), np.uint8)
    # Flood fill from the corner with white (now black after inversion)
    cv2.floodFill(to_fill, extended, (0, 0), 255)
    # Invert the colors back to original
    filled = cv2.bitwise_not(to_fill)
    # Display the result
    #cv2.imshow('Filled Image', filled)
    #cv2.waitKey(0)
    mask_filled = cv2.bitwise_or(mask,filled)
    #cv2.imshow('Filled Plus Mask', mask_filled)
    #cv2.waitKey(0)


    # Fit an ellipse to the white wing pixels
    # Find contours of the white regions
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Merge all contours into a single one for the purpose of fitting an ellipse
    all_contours = np.vstack([contours[i] for i in range(len(contours))])

    # visualize
    contour_img = mask_filled.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
    #cv2.imshow("Contours for Ellipse",contour_img)
    #cv2.waitKey(0)
    # Fit an ellipse to the merged contour if there are enough points
    if all_contours.shape[0] >= 5:
        ellipse = cv2.fitEllipse(all_contours)
        # Draw the ellipse on a copy of the original image (or on a blank image if preferred)
        image_with_ellipse = mask_filled.copy()
        image_with_ellipse = cv2.cvtColor(image_with_ellipse, cv2.COLOR_GRAY2RGB)
        cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)  # Green ellipse with thickness of 2

        # draw axis lines too
        center, axes, angle = ellipse
        # Calculate the endpoints of the major and minor axes
        major_axis_end1 = (
        int(center[0] + axes[0] * np.cos(np.radians(angle))), int(center[1] + axes[0] * np.sin(np.radians(angle))))
        major_axis_end2 = (
        int(center[0] - axes[0] * np.cos(np.radians(angle))), int(center[1] - axes[0] * np.sin(np.radians(angle))))
        minor_axis_end1 = (int(center[0] + axes[1] * np.cos(np.radians(angle + 90))),
                           int(center[1] + axes[1] * np.sin(np.radians(angle + 90))))
        minor_axis_end2 = (int(center[0] - axes[1] * np.cos(np.radians(angle + 90))),
                           int(center[1] - axes[1] * np.sin(np.radians(angle + 90))))
        # Draw the major axis in red
        cv2.line(image_with_ellipse, major_axis_end1, major_axis_end2, (0, 0, 255), 2)
        # Draw the minor axis in blue
        cv2.line(image_with_ellipse, minor_axis_end1, minor_axis_end2, (255, 0, 0), 2)
    else:
        print("Not enough points to fit an ellipse.")

    def get_len_thru_white(mask,line_end1,line_end2):
        # get major axis length that passes thru white
        # Create an empty (black) image with the same dimensions as the mask
        mask_shape = mask.shape
        line_image = np.zeros(mask_shape, dtype=np.uint8)
        # Draw the major axis line on the empty image
        cv2.line(line_image, line_end1, line_end2, color=(255, 255, 255), thickness=1)
        # Perform a bitwise AND between the mask and the line image
        result_image = cv2.bitwise_and(line_image, mask)
        # Count the white pixels in the result_image
        length = np.sum(result_image == 255)
        return(length)

    #mask_filled2 = np.copy(mask_filled)
    #_, binary_image = cv2.threshold(mask_filled2, 127, 255, cv2.THRESH_BINARY)
    #n_white = np.sum(binary_image == 255)
    #print(n_white)
    #print(np.shape(binary_image))
    #n_white = np.sum(np.all(mask_filled == [255, 255, 255], axis=-1))
    #print(n_white)

    n_white = cv2.countNonZero(mask_filled)
    area = (cm_per_pixel*cm_per_pixel) * n_white
    #print(area)

    major_axis_length, minor_axis_length = ellipse[1]
    width = major_axis_length * cm_per_pixel
    length = minor_axis_length * cm_per_pixel

    if use_len_thru_white:
        width = get_len_thru_white(mask_filled,major_axis_end1,major_axis_end2) * cm_per_pixel
        length = get_len_thru_white(mask_filled, minor_axis_end1, minor_axis_end2) * cm_per_pixel
    if not use_pixel_area:
        area = width * length
    #print(length_adj)
    #print(width_adj)

    measures_str = "(cm) Length: " + str(round(length,1)) + ", Width: " + str(round(width,1)) + ", Area: " + str(round(area,1))
    _showImages(show,maintitle=measures_str,images=[thresh,mask,mask_filled,contour_img,image_with_ellipse],
                titles=["Thresholded","Islands Removed","Flood Filled","Contours","Mask with Ellipse"],save_folder="D:/new_dragonfly_wings/plots")

    return(area,length,width)

def scaleCropFindWingContour(img,img_id,cm_per_pixel=0.0108, top_offset=35, bot_offset=35,
                             crop_height_percent=28.3,crop_width_percent=53.3, thresh_c=5, dilate_kernel_size=5, manual=False, show=False):
    print(img_id)
    top_bot = scaleCropTopBottom(img,show=show,top_offset=top_offset,bot_offset=bot_offset,crop_height_percent=crop_height_percent,crop_width_percent=crop_width_percent,manual=manual)
    top = findWingContour(top_bot[0],cm_per_pixel=cm_per_pixel,thresh_c=thresh_c,dilate_kernel_size=dilate_kernel_size,show=show)
    bot = findWingContour(top_bot[1],cm_per_pixel=cm_per_pixel,thresh_c=thresh_c,dilate_kernel_size=dilate_kernel_size,show=show)
    return(top,bot)

def tableImg(img):
    # Check if the image is grayscale or RGB
    if len(img.shape) == 2:  # Grayscale image
        # Reshape the grayscale image to a 2D array where each row is a pixel
        pixels = img.reshape(-1, 1)
    else:  # RGB image
        # Reshape the RGB image to a 2D array where each row represents a pixel's RGB values
        pixels = img.reshape(-1, img.shape[-1])

    # Find unique colors/intensities and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # Display the unique colors/intensities and their counts
    unique_colors_with_counts = dict(zip([tuple(color) for color in unique_colors], counts))
    print(unique_colors_with_counts)

def processFolder(folder_path,
                  cm_per_pixel=0.0108,
                  define_scalebar=False,  # <- NEW
                  top_offset=35,
                  bot_offset=35,
                  thresh_c=3,
                  crop_height_percent=28.3,
                  crop_width_percent=53.3,
                  dilate_kernel_size=5,
                  manual=False,
                  resize_if_huge=False,
                  show=False):
    """
    Process all images in a folder, optionally letting the user manually define the scalebar for each image.
    """

    imgs = []
    file_names = []

    # Grab all jpg, jpeg, png, etc. that match
    for file_path in glob.glob(os.path.join(folder_path, "*.[pj][np]g")):
        img = cv2.imread(file_path)

        # If the image is huge and we want to resize it
        if img.shape[0] + img.shape[1] > 1500 and resize_if_huge:
            original_width = img.shape[1]
            original_height = img.shape[0]
            new_width = int(original_width / 4)
            new_height = int(original_height / 4)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        imgs.append(img)

        # Extract just the file name (no extension)
        base_name = os.path.basename(file_path)
        name_without_extension, _ = os.path.splitext(base_name)
        file_names.append(name_without_extension)

    # Process each image
    metrics = []
    for img, img_id in zip(imgs, file_names):

        # If define_scalebar is True, measure the scale bar *for this image*
        if define_scalebar:
            # This overwrites the passed-in cm_per_pixel, using user-drawn line
            cm_per_pixel_for_this_image = measureScaleBar(img)
        else:
            # Use the user-provided cm_per_pixel
            cm_per_pixel_for_this_image = cm_per_pixel

        # Now pass that to the wing contour pipeline
        result_top, result_bot = scaleCropFindWingContour(
            img,
            img_id,
            cm_per_pixel=cm_per_pixel_for_this_image,
            top_offset=top_offset,
            bot_offset=bot_offset,
            crop_height_percent=crop_height_percent,
            crop_width_percent=crop_width_percent,
            thresh_c=thresh_c,
            dilate_kernel_size=dilate_kernel_size,
            manual=manual,
            show=show
        )
        # Collect the metrics
        metrics.append((result_top, result_bot))

    # After processing all images, build the dataframe
    df = pd.DataFrame({
        'fore_area_cm2': [m[0][0] for m in metrics],
        'fore_length_cm': [m[0][1] for m in metrics],
        'fore_width_cm': [m[0][2] for m in metrics],
        'hind_area_cm2': [m[1][0] for m in metrics],
        'hind_length_cm': [m[1][1] for m in metrics],
        'hind_width_cm': [m[1][2] for m in metrics],
    })

    df['id'] = file_names
    csv_file_path = 'metrics.csv'
    df.to_csv(csv_file_path, index=False)

    return df


import pandas as pd
import glob

def loadCombineSaveCSV(file_paths, output_file):
    # Load all CSV files into a list of DataFrames
    dataframes = [pd.read_csv(file) for file in file_paths]

    # Combine all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a new CSV
    combined_df.to_csv(output_file, index=False)


def measureScaleBar(image):
    """
    Let the user draw a straight line over the scalebar by clicking two points.
    Returns cm_per_pixel for that image, assuming the scalebar is 1 cm.
    """
    clone = image.copy()
    points = []

    # Mouse callback
    def draw_line(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Once we have two points, we can close the window
            if len(points) == 2:
                cv2.destroyWindow("Define Scale Bar")

    cv2.namedWindow("Define Scale Bar")
    cv2.setMouseCallback("Define Scale Bar", draw_line)

    # Keep showing the image until we have two clicks
    while True:
        cv2.imshow("Define Scale Bar", clone)
        key = cv2.waitKey(1) & 0xFF

        # Escape key to quit if user wants to cancel
        if key == 27:  # ESC
            break

        # Once two points are clicked, break out
        if len(points) == 2:
            break

    cv2.destroyAllWindows()

    if len(points) < 2:
        # User didn't define scale properly; return None or default
        print("Scale bar was not defined. Using default cm_per_pixel=0.01 for safety.")
        return 0.01

    # Calculate pixel distance between the two clicked points
    (x1, y1), (x2, y2) = points
    dist_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # The scale bar is always 1 cm, so:
    cm_per_pixel = 1.0 / dist_pixels
    print(f"Scale bar length in pixels: {dist_pixels:.2f}.  ->  cm_per_pixel = {cm_per_pixel:.5f}")
    return cm_per_pixel

# new version for Judicael with threshold
def processFolder_threshold(folder_path,
                            lightness_threshold=128,  # <- NEW: threshold for darkness analysis
                            cm_per_pixel=0.0108,
                            define_scalebar=False,
                            top_offset=35,
                            bot_offset=35,
                            thresh_c=3,
                            crop_height_percent=28.3,
                            crop_width_percent=53.3,
                            dilate_kernel_size=5,
                            manual=False,
                            resize_if_huge=False,
                            show=False):
    """
    Process all images in a folder, optionally letting the user manually define the scalebar for each image.
    Additionally calculates the percentage of wing area that is below the lightness threshold (darker regions).

    Parameters:
    -----------
    lightness_threshold : int
        Grayscale threshold value (0-255). Pixels below this value are considered "dark".
        Lower values = only very dark pixels counted, Higher values = more pixels counted as dark.
    """

    def calculate_darkness_percentage(wing_img, mask_filled, lightness_threshold):
        """
        Calculate the percentage of the wing area that is below the lightness threshold.

        Parameters:
        -----------
        wing_img : numpy array
            The cropped wing image (BGR or grayscale)
        mask_filled : numpy array
            The binary mask of the wing (255 for wing, 0 for background)
        lightness_threshold : int
            The threshold value for determining darkness

        Returns:
        --------
        float : Percentage of wing pixels below the threshold
        """
        # Convert to grayscale if needed
        if len(wing_img.shape) == 3:
            gray_img = cv2.cvtColor(wing_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = wing_img

        # Apply the mask to only consider pixels within the wing
        wing_pixels = gray_img[mask_filled == 255]

        # Count pixels below the threshold
        if len(wing_pixels) > 0:
            dark_pixels = np.sum(wing_pixels < lightness_threshold)
            total_pixels = len(wing_pixels)
            darkness_percentage = (dark_pixels / total_pixels) * 100
        else:
            darkness_percentage = 0.0

        return darkness_percentage

    def findWingContourWithThreshold(img, lightness_threshold, use_len_thru_white=True,
                                     use_pixel_area=True, cm_per_pixel=0.0108,
                                     dilate_kernel_size=5, thresh_c=5, show=False):
        """
        Modified version of findWingContour that also returns the mask and darkness percentage.
        """
        start_img = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # adaptive threshold
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 21, thresh_c)

        # invert mask so shapes are white on black background
        thresh = cv2.bitwise_not(thresh)

        # Find largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        mask = np.zeros_like(thresh)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask[labels == largest_label] = 255

        # dilate
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # add black border before flood fill
        mask = cv2.copyMakeBorder(mask, top=5, bottom=5, left=5, right=5,
                                  borderType=cv2.BORDER_CONSTANT, value=0)

        # Fill in inside of wings using flood fill
        to_fill = np.copy(mask)
        h, w = to_fill.shape[:2]
        extended = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(to_fill, extended, (0, 0), 255)
        filled = cv2.bitwise_not(to_fill)
        mask_filled = cv2.bitwise_or(mask, filled)

        # Calculate darkness percentage using the original image and mask
        # Need to account for the border we added
        start_img_bordered = cv2.copyMakeBorder(start_img, top=5, bottom=5, left=5, right=5,
                                                borderType=cv2.BORDER_CONSTANT, value=0)
        darkness_pct = calculate_darkness_percentage(start_img_bordered, mask_filled, lightness_threshold)

        # Fit an ellipse to the white wing pixels
        contours, _ = cv2.findContours(mask_filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        all_contours = np.vstack([contours[i] for i in range(len(contours))])



        if all_contours.shape[0] >= 5:
            ellipse = cv2.fitEllipse(all_contours)

            # Calculate ellipse axes endpoints
            center, axes, angle = ellipse
            major_axis_end1 = (int(center[0] + axes[0] * np.cos(np.radians(angle))),
                               int(center[1] + axes[0] * np.sin(np.radians(angle))))
            major_axis_end2 = (int(center[0] - axes[0] * np.cos(np.radians(angle))),
                               int(center[1] - axes[0] * np.sin(np.radians(angle))))
            minor_axis_end1 = (int(center[0] + axes[1] * np.cos(np.radians(angle + 90))),
                               int(center[1] + axes[1] * np.sin(np.radians(angle + 90))))
            minor_axis_end2 = (int(center[0] - axes[1] * np.cos(np.radians(angle + 90))),
                               int(center[1] - axes[1] * np.sin(np.radians(angle + 90))))

            def get_len_thru_white(mask, line_end1, line_end2):
                mask_shape = mask.shape
                line_image = np.zeros(mask_shape, dtype=np.uint8)
                cv2.line(line_image, line_end1, line_end2, color=(255, 255, 255), thickness=1)
                result_image = cv2.bitwise_and(line_image, mask)
                length = np.sum(result_image == 255)
                return length

            n_white = cv2.countNonZero(mask_filled)
            area = (cm_per_pixel * cm_per_pixel) * n_white

            major_axis_length, minor_axis_length = ellipse[1]
            width = major_axis_length * cm_per_pixel
            length = minor_axis_length * cm_per_pixel

            if use_len_thru_white:
                width = get_len_thru_white(mask_filled, major_axis_end1, major_axis_end2) * cm_per_pixel
                length = get_len_thru_white(mask_filled, minor_axis_end1, minor_axis_end2) * cm_per_pixel
            if not use_pixel_area:
                area = width * length

            if show:
                measures_str = f"(cm) L: {length:.1f}, W: {width:.1f}, A: {area:.1f}, Dark%: {darkness_pct:.1f}"
                # Create visualization if needed
                image_with_ellipse = mask_filled.copy()
                image_with_ellipse = cv2.cvtColor(image_with_ellipse, cv2.COLOR_GRAY2RGB)
                cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)
                cv2.line(image_with_ellipse, major_axis_end1, major_axis_end2, (0, 0, 255), 2)
                cv2.line(image_with_ellipse, minor_axis_end1, minor_axis_end2, (255, 0, 0), 2)

                # Create darkness visualization
                darkness_viz = start_img_bordered.copy()
                if len(darkness_viz.shape) == 2:
                    darkness_viz = cv2.cvtColor(darkness_viz, cv2.COLOR_GRAY2BGR)
                # Highlight dark areas in red
                dark_mask = (cv2.cvtColor(start_img_bordered, cv2.COLOR_BGR2GRAY) < lightness_threshold) & (
                            mask_filled == 255)
                darkness_viz[dark_mask] = [0, 0, 255]  # Red for dark areas

                _showImages(show, maintitle=measures_str,
                            images=[thresh, mask, mask_filled, image_with_ellipse, darkness_viz],
                            titles=["Thresholded", "Islands Removed", "Flood Filled",
                                    "Mask with Ellipse", f"Dark Areas (< {lightness_threshold})"])

            return area, length, width, darkness_pct
        else:
            print("Not enough points to fit an ellipse.")
            return 0, 0, 0, 0

    def scaleCropFindWingContourWithThreshold(img, img_id, lightness_threshold, cm_per_pixel=0.0108,
                                              top_offset=35, bot_offset=35, crop_height_percent=28.3,
                                              crop_width_percent=53.3, thresh_c=5, dilate_kernel_size=5,
                                              manual=False, show=False):
        print(img_id)
        top_bot = scaleCropTopBottom(img, show=show, top_offset=top_offset, bot_offset=bot_offset,
                                     crop_height_percent=crop_height_percent,
                                     crop_width_percent=crop_width_percent, manual=manual)
        top = findWingContourWithThreshold(top_bot[0], lightness_threshold, cm_per_pixel=cm_per_pixel,
                                           thresh_c=thresh_c, dilate_kernel_size=dilate_kernel_size, show=show)
        bot = findWingContourWithThreshold(top_bot[1], lightness_threshold, cm_per_pixel=cm_per_pixel,
                                           thresh_c=thresh_c, dilate_kernel_size=dilate_kernel_size, show=show)
        return top, bot

    # Main processing logic
    imgs = []
    file_names = []

    # Grab all jpg, jpeg, png, etc. that match
    for file_path in glob.glob(os.path.join(folder_path, "*.[pj][np]g")):
        img = cv2.imread(file_path)

        # If the image is huge and we want to resize it
        if img.shape[0] + img.shape[1] > 1500 and resize_if_huge:
            original_width = img.shape[1]
            original_height = img.shape[0]
            new_width = int(original_width / 4)
            new_height = int(original_height / 4)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        imgs.append(img)

        # Extract just the file name (no extension)
        base_name = os.path.basename(file_path)
        name_without_extension, _ = os.path.splitext(base_name)
        file_names.append(name_without_extension)

    # Process each image
    metrics = []
    for img, img_id in zip(imgs, file_names):

        # If define_scalebar is True, measure the scale bar *for this image*
        if define_scalebar:
            cm_per_pixel_for_this_image = measureScaleBar(img)
        else:
            cm_per_pixel_for_this_image = cm_per_pixel

        # Now pass that to the wing contour pipeline with threshold analysis
        result_top, result_bot = scaleCropFindWingContourWithThreshold(
            img,
            img_id,
            lightness_threshold=lightness_threshold,
            cm_per_pixel=cm_per_pixel_for_this_image,
            top_offset=top_offset,
            bot_offset=bot_offset,
            crop_height_percent=crop_height_percent,
            crop_width_percent=crop_width_percent,
            thresh_c=thresh_c,
            dilate_kernel_size=dilate_kernel_size,
            manual=manual,
            show=show
        )
        # Collect the metrics (now includes darkness percentage)
        metrics.append((result_top, result_bot))

    # After processing all images, build the dataframe with new darkness columns
    df = pd.DataFrame({
        'id': file_names,
        'fore_area_cm2': [m[0][0] for m in metrics],
        'fore_length_cm': [m[0][1] for m in metrics],
        'fore_width_cm': [m[0][2] for m in metrics],
        'fore_darkness_pct': [m[0][3] for m in metrics],  # NEW
        'hind_area_cm2': [m[1][0] for m in metrics],
        'hind_length_cm': [m[1][1] for m in metrics],
        'hind_width_cm': [m[1][2] for m in metrics],
        'hind_darkness_pct': [m[1][3] for m in metrics],  # NEW
        'lightness_threshold': [lightness_threshold] * len(metrics)  # Record threshold used
    })

    # Save with a filename that includes the threshold
    csv_file_path = f'metrics_threshold_{lightness_threshold}.csv'
    df.to_csv(csv_file_path, index=False)

    print(f"\nProcessing complete with lightness threshold: {lightness_threshold}")
    print(f"Average forewing darkness: {df['fore_darkness_pct'].mean():.2f}%")
    print(f"Average hindwing darkness: {df['hind_darkness_pct'].mean():.2f}%")

    return df

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def _showImages(show, images, titles=None, maintitle=None, list_cmaps=None, grid=False,
                num_cols=3, figsize=(10, 10), title_fontsize='auto', sample_n=None,
                save_folder=None, title_wrap=True, title_max_chars=30):
    '''
    Shows a grid of images with automatically adjusted titles to prevent overlap.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int or 'auto'
        If 'auto', automatically calculate based on subplot size.
        Otherwise, value to be passed to set_title().
    sample_n: int or None
        If specified, randomly sample n images to display
    save_folder: str or None
        Folder path to save the figure
    title_wrap: boolean
        If True, wrap long titles to multiple lines
    title_max_chars: int
        Maximum characters per line when wrapping titles
    '''

    if show:
        # Convert to lists if needed
        if not isinstance(images, list):
            images = [images]
        if titles is not None and not isinstance(titles, list):
            titles = [titles]

        list_images = images
        list_titles = titles

        # Sample n images if specified
        if sample_n is not None and sample_n < len(list_images):
            indices = random.sample(range(len(list_images)), sample_n)
            list_images = [list_images[i] for i in indices]
            if list_titles is not None:
                list_titles = [list_titles[i] for i in indices]

        # Convert images to RGB
        for index, img in enumerate(list_images):
            if len(img.shape) == 4:
                img = img[:, :, :3]
                img = img.astype(np.uint8)
            list_images[index] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Assertions
        assert isinstance(list_images, list)
        assert len(list_images) > 0
        assert isinstance(list_images[0], np.ndarray)

        if list_titles is not None:
            assert isinstance(list_titles, list)
            assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

        if list_cmaps is not None:
            assert isinstance(list_cmaps, list)
            assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

        num_images = len(list_images)
        num_cols = min(num_images, num_cols)
        num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

        # Create figure and subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

        # Create list of axes
        if isinstance(axes, np.ndarray):
            list_axes = list(axes.flat)
        else:
            list_axes = [axes]

        # Calculate automatic font size if needed
        if title_fontsize == 'auto':
            # Base font size on subplot dimensions
            subplot_width = figsize[0] / num_cols
            subplot_height = figsize[1] / num_rows

            # Use smaller dimension for font size calculation
            subplot_size = min(subplot_width, subplot_height)

            # Calculate font size (adjust multiplier as needed)
            # Smaller subplots get smaller fonts
            calculated_fontsize = max(8, min(16, int(subplot_size * 2.5)))
        else:
            calculated_fontsize = title_fontsize

        # Display images
        for i in range(num_images):
            img = list_images[i]
            title = list_titles[i] if list_titles is not None else None
            cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

            list_axes[i].imshow(img, cmap=cmap)

            if title is not None:
                # Wrap long titles if enabled
                if title_wrap and len(title) > title_max_chars:
                    wrapped_title = wrap_title(title, title_max_chars)
                else:
                    wrapped_title = title

                # Set title with calculated font size
                list_axes[i].set_title(wrapped_title, fontsize=calculated_fontsize,
                                       pad=calculated_fontsize * 0.5)  # Add padding based on font size

            list_axes[i].grid(grid)

        # Hide unused subplots
        for i in range(num_images, len(list_axes)):
            list_axes[i].set_visible(False)

        # Remove ticks
        for i in range(len(list_axes)):
            list_axes[i].set_xticks([])
            list_axes[i].set_yticks([])

        # Adjust layout with better spacing
        if maintitle is not None:
            # Make room for main title
            fig.suptitle(maintitle, fontsize=calculated_fontsize * 1.5 if title_fontsize == 'auto' else 30)
            # Adjust spacing to prevent overlap
            plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98,
                                hspace=0.15, wspace=0.05)
        else:
            # No main title, can use more space
            plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98,
                                hspace=0.10, wspace=0.05)

        # Save if specified
        if save_folder is not None:
            filename = generate_unique_filename(save_folder, "plot", ".jpg")
            plt.savefig(os.path.join(save_folder, filename), bbox_inches='tight', dpi=100)

        plt.show()


def wrap_title(title, max_chars):
    """Wrap title text to multiple lines."""
    words = title.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) > max_chars:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                # Single word longer than max_chars
                lines.append(word)
                current_line = []
                current_length = 0
        else:
            current_line.append(word)
            current_length += len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)


def img_is_color(img):
    """Check if an image is color or grayscale."""
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True
    return False


def generate_unique_filename(directory, base_filename, extension):
    """Generate a unique filename in the specified directory."""
    i = 1
    unique_filename = f"{base_filename}{extension}"
    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base_filename}_{i}{extension}"
        i += 1
    return unique_filename


# Alternative: Use constrained_layout for automatic spacing (matplotlib >= 3.0)
def _showImages_constrained(show, images, titles=None, maintitle=None, list_cmaps=None,
                            grid=False, num_cols=3, figsize=(10, 10), sample_n=None,
                            save_folder=None):
    '''
    Alternative version using constrained_layout for better automatic spacing.
    Requires matplotlib >= 3.0
    '''

    if show:
        # [Same preprocessing code as above until fig creation]
        if not isinstance(images, list):
            images = [images]
        if titles is not None and not isinstance(titles, list):
            titles = [titles]

        list_images = images
        list_titles = titles

        if sample_n is not None and sample_n < len(list_images):
            indices = random.sample(range(len(list_images)), sample_n)
            list_images = [list_images[i] for i in indices]
            if list_titles is not None:
                list_titles = [list_titles[i] for i in indices]

        for index, img in enumerate(list_images):
            if len(img.shape) == 4:
                img = img[:, :, :3]
                img = img.astype(np.uint8)
            list_images[index] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_images = len(list_images)
        num_cols = min(num_images, num_cols)
        num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

        # Use constrained_layout for automatic spacing
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 constrained_layout=True)

        if isinstance(axes, np.ndarray):
            list_axes = list(axes.flat)
        else:
            list_axes = [axes]

        # Auto-calculate font size
        subplot_size = min(figsize[0] / num_cols, figsize[1] / num_rows)
        fontsize = max(8, min(16, int(subplot_size * 2.5)))

        for i in range(num_images):
            img = list_images[i]
            title = list_titles[i] if list_titles is not None else None
            cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

            list_axes[i].imshow(img, cmap=cmap)
            if title is not None:
                list_axes[i].set_title(title, fontsize=fontsize)
            list_axes[i].grid(grid)
            list_axes[i].set_xticks([])
            list_axes[i].set_yticks([])

        for i in range(num_images, len(list_axes)):
            list_axes[i].set_visible(False)

        if maintitle is not None:
            fig.suptitle(maintitle, fontsize=fontsize * 1.5)

        if save_folder is not None:
            filename = generate_unique_filename(save_folder, "plot", ".jpg")
            plt.savefig(os.path.join(save_folder, filename), bbox_inches='tight')

        plt.show()


import numpy as np
import cv2


def selectROIbyPolyline(window_name, img):
    """
    Allow user to click points to define a polygon region.

    Instructions:
    - Left click to add points
    - Press 'r' to reset/clear all points
    - Press 'Enter' or 'Space' to finish selection
    - Press 'Esc' to cancel

    Returns:
    --------
    mask : numpy array
        Binary mask where selected region is 255, rest is 0
    bounding_rect : tuple
        (x, y, w, h) bounding rectangle of the selected region
    """
    clone = img.copy()
    points = []
    finished = False
    cancelled = False

    def draw_polygon(image, pts):
        """Draw the current polygon on the image."""
        overlay = image.copy()
        if len(pts) > 0:
            # Draw lines between points
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i + 1], (0, 255, 0), 2)
            # Draw line from last point to first (if more than 2 points)
            if len(pts) > 2:
                cv2.line(overlay, pts[-1], pts[0], (0, 255, 0), 2)
            # Draw points
            for pt in pts:
                cv2.circle(overlay, pt, 5, (0, 0, 255), -1)
        return overlay

    def mouse_callback(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n{window_name} - Instructions:")
    print("  - Left click to add points")
    print("  - Press 'r' to reset/clear all points")
    print("  - Press 'Enter' or 'Space' to finish")
    print("  - Press 'Esc' to cancel")

    while not finished and not cancelled:
        display_img = draw_polygon(clone.copy(), points)

        # Add text instructions on the image
        cv2.putText(display_img, f"Points: {len(points)} | 'r':reset | Enter:finish | Esc:cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            cancelled = True
        elif key == 13 or key == 32:  # Enter or Space
            if len(points) >= 3:  # Need at least 3 points for a polygon
                finished = True
            else:
                print(f"Need at least 3 points to finish. Current: {len(points)}")
        elif key == ord('r') or key == ord('R'):  # Reset
            points = []
            print("Points reset")

    cv2.destroyWindow(window_name)

    if cancelled or len(points) < 3:
        print("Selection cancelled or invalid")
        return None, None

    # Create mask from polygon
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(points_array)

    return mask, (x, y, w, h)


def scaleCropTopBottom_polyline(img, show=True):
    """
    Select top and bottom wing regions by drawing polygons with connected line segments.

    Parameters:
    -----------
    img : numpy array
        Input image
    show : bool
        Whether to display the cropped results

    Returns:
    --------
    tuple : (img_top, img_bot)
        Cropped top and bottom wing images
    """
    print("\n=== Select TOP wing region ===")
    mask_top, bbox_top = selectROIbyPolyline("Select Top Wing", img)

    if mask_top is None:
        raise ValueError("Top wing selection was cancelled")

    print("\n=== Select BOTTOM wing region ===")
    mask_bot, bbox_bot = selectROIbyPolyline("Select Bottom Wing", img)

    if mask_bot is None:
        raise ValueError("Bottom wing selection was cancelled")

    # Extract regions using bounding boxes
    x_top, y_top, w_top, h_top = bbox_top
    x_bot, y_bot, w_bot, h_bot = bbox_bot

    # Crop to bounding box and apply mask
    img_top_cropped = img[y_top:y_top + h_top, x_top:x_top + w_top].copy()
    img_bot_cropped = img[y_bot:y_bot + h_bot, x_bot:x_bot + w_bot].copy()

    # Crop the masks to match
    mask_top_cropped = mask_top[y_top:y_top + h_top, x_top:x_top + w_top]
    mask_bot_cropped = mask_bot[y_bot:y_bot + h_bot, x_bot:x_bot + w_bot]

    # Apply masks to set outside regions to white (or black, depending on preference)
    img_top_masked = img_top_cropped.copy()
    img_bot_masked = img_bot_cropped.copy()
    img_top_masked[mask_top_cropped == 0] = 255  # White background
    img_bot_masked[mask_bot_cropped == 0] = 255  # White background

    if show:
        _showImages(show, images=[img_top_masked, img_bot_masked], titles=["Top", "Bottom"])

    return img_top_masked, img_bot_masked


def processFolder_threshold_lines(folder_path,
                                  lightness_threshold=128,
                                  cm_per_pixel=0.0108,
                                  define_scalebar=False,
                                  thresh_c=3,
                                  dilate_kernel_size=5,
                                  resize_if_huge=False,
                                  show=False):
    """
    Process all images in a folder using polyline selection for wing regions.
    User clicks to draw connected line segments defining the wing boundaries.

    Parameters:
    -----------
    folder_path : str
        Path to folder containing images
    lightness_threshold : int
        Grayscale threshold value (0-255) for darkness analysis
    cm_per_pixel : float
        Conversion factor from pixels to centimeters
    define_scalebar : bool
        If True, user manually defines scale bar for each image
    thresh_c : int
        Constant for adaptive threshold
    dilate_kernel_size : int
        Kernel size for dilation operation
    resize_if_huge : bool
        If True, resize large images
    show : bool
        If True, display intermediate processing steps

    Returns:
    --------
    pandas.DataFrame
        DataFrame with wing measurements including darkness percentage
    """

    # Helper function to process with polyline selection
    def scaleCropFindWingContourWithPolyline(img, img_id, lightness_threshold,
                                             cm_per_pixel=0.0108, thresh_c=5,
                                             dilate_kernel_size=5, show=False):
        print(f"\nProcessing: {img_id}")

        # Use polyline selection instead of rectangle
        top_bot = scaleCropTopBottom_polyline(img, show=show)

        # Process each wing
        top = findWingContourWithThreshold(
            top_bot[0], lightness_threshold,
            cm_per_pixel=cm_per_pixel,
            thresh_c=thresh_c,
            dilate_kernel_size=dilate_kernel_size,
            show=show
        )

        bot = findWingContourWithThreshold(
            top_bot[1], lightness_threshold,
            cm_per_pixel=cm_per_pixel,
            thresh_c=thresh_c,
            dilate_kernel_size=dilate_kernel_size,
            show=show
        )

        return top, bot

    # Load images
    imgs = []
    file_names = []

    for file_path in glob.glob(os.path.join(folder_path, "*.[pj][np]g")):
        img = cv2.imread(file_path)

        if img is None:
            print(f"Warning: Could not read {file_path}")
            continue

        # Resize if needed
        if img.shape[0] + img.shape[1] > 1500 and resize_if_huge:
            original_width = img.shape[1]
            original_height = img.shape[0]
            new_width = int(original_width / 4)
            new_height = int(original_height / 4)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        imgs.append(img)

        # Extract file name without extension
        base_name = os.path.basename(file_path)
        name_without_extension, _ = os.path.splitext(base_name)
        file_names.append(name_without_extension)

    if len(imgs) == 0:
        print(f"No images found in {folder_path}")
        return None

    print(f"\nFound {len(imgs)} images to process")

    # Process each image
    metrics = []
    for img, img_id in zip(imgs, file_names):
        try:
            # Define scale bar if requested
            if define_scalebar:
                cm_per_pixel_for_this_image = measureScaleBar(img)
            else:
                cm_per_pixel_for_this_image = cm_per_pixel

            # Process with polyline selection
            result_top, result_bot = scaleCropFindWingContourWithPolyline(
                img,
                img_id,
                lightness_threshold=lightness_threshold,
                cm_per_pixel=cm_per_pixel_for_this_image,
                thresh_c=thresh_c,
                dilate_kernel_size=dilate_kernel_size,
                show=show
            )

            metrics.append((result_top, result_bot))

        except ValueError as e:
            print(f"Skipping {img_id}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            continue

    if len(metrics) == 0:
        print("No images were successfully processed")
        return None

    # Build DataFrame
    df = pd.DataFrame({
        'id': file_names[:len(metrics)],
        'fore_area_cm2': [m[0][0] for m in metrics],
        'fore_length_cm': [m[0][1] for m in metrics],
        'fore_width_cm': [m[0][2] for m in metrics],
        'fore_darkness_pct': [m[0][3] for m in metrics],
        'hind_area_cm2': [m[1][0] for m in metrics],
        'hind_length_cm': [m[1][1] for m in metrics],
        'hind_width_cm': [m[1][2] for m in metrics],
        'hind_darkness_pct': [m[1][3] for m in metrics],
        'lightness_threshold': [lightness_threshold] * len(metrics)
    })

    # Save results
    csv_file_path = f'metrics_threshold_lines_{lightness_threshold}.csv'
    df.to_csv(csv_file_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Processing complete with lightness threshold: {lightness_threshold}")
    print(f"Processed {len(metrics)} images successfully")
    print(f"Results saved to: {csv_file_path}")
    print(f"Average forewing darkness: {df['fore_darkness_pct'].mean():.2f}%")
    print(f"Average hindwing darkness: {df['hind_darkness_pct'].mean():.2f}%")
    print(f"{'=' * 60}")

    return df

def findWingContourWithThreshold(img, lightness_threshold, use_len_thru_white=True,
                                 use_pixel_area=True, cm_per_pixel=0.0108,
                                 dilate_kernel_size=5, thresh_c=5, show=False):
    """
    Modified version of findWingContour that also returns the mask and darkness percentage.
    """
    start_img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 21, thresh_c)

    # invert mask so shapes are white on black background
    thresh = cv2.bitwise_not(thresh)

    # Find largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    mask = np.zeros_like(thresh)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask[labels == largest_label] = 255

    # dilate
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # add black border before flood fill
    mask = cv2.copyMakeBorder(mask, top=5, bottom=5, left=5, right=5,
                              borderType=cv2.BORDER_CONSTANT, value=0)

    # Fill in inside of wings using flood fill
    to_fill = np.copy(mask)
    h, w = to_fill.shape[:2]
    extended = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(to_fill, extended, (0, 0), 255)
    filled = cv2.bitwise_not(to_fill)
    mask_filled = cv2.bitwise_or(mask, filled)

    # Calculate darkness percentage using the original image and mask
    # Need to account for the border we added
    start_img_bordered = cv2.copyMakeBorder(start_img, top=5, bottom=5, left=5, right=5,
                                            borderType=cv2.BORDER_CONSTANT, value=0)
    darkness_pct = calculate_darkness_percentage(start_img_bordered, mask_filled, lightness_threshold)

    # Fit an ellipse to the white wing pixels
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    all_contours = np.vstack([contours[i] for i in range(len(contours))])



    if all_contours.shape[0] >= 5:
        ellipse = cv2.fitEllipse(all_contours)

        # Calculate ellipse axes endpoints
        center, axes, angle = ellipse
        major_axis_end1 = (int(center[0] + axes[0] * np.cos(np.radians(angle))),
                           int(center[1] + axes[0] * np.sin(np.radians(angle))))
        major_axis_end2 = (int(center[0] - axes[0] * np.cos(np.radians(angle))),
                           int(center[1] - axes[0] * np.sin(np.radians(angle))))
        minor_axis_end1 = (int(center[0] + axes[1] * np.cos(np.radians(angle + 90))),
                           int(center[1] + axes[1] * np.sin(np.radians(angle + 90))))
        minor_axis_end2 = (int(center[0] - axes[1] * np.cos(np.radians(angle + 90))),
                           int(center[1] - axes[1] * np.sin(np.radians(angle + 90))))

        def get_len_thru_white(mask, line_end1, line_end2):
            mask_shape = mask.shape
            line_image = np.zeros(mask_shape, dtype=np.uint8)
            cv2.line(line_image, line_end1, line_end2, color=(255, 255, 255), thickness=1)
            result_image = cv2.bitwise_and(line_image, mask)
            length = np.sum(result_image == 255)
            return length

        n_white = cv2.countNonZero(mask_filled)
        area = (cm_per_pixel * cm_per_pixel) * n_white

        major_axis_length, minor_axis_length = ellipse[1]
        width = major_axis_length * cm_per_pixel
        length = minor_axis_length * cm_per_pixel

        if use_len_thru_white:
            width = get_len_thru_white(mask_filled, major_axis_end1, major_axis_end2) * cm_per_pixel
            length = get_len_thru_white(mask_filled, minor_axis_end1, minor_axis_end2) * cm_per_pixel
        if not use_pixel_area:
            area = width * length

        if show:
            measures_str = f"(cm) L: {length:.1f}, W: {width:.1f}, A: {area:.1f}, Dark%: {darkness_pct:.1f}"
            # Create visualization if needed
            image_with_ellipse = mask_filled.copy()
            image_with_ellipse = cv2.cvtColor(image_with_ellipse, cv2.COLOR_GRAY2RGB)
            cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)
            cv2.line(image_with_ellipse, major_axis_end1, major_axis_end2, (0, 0, 255), 2)
            cv2.line(image_with_ellipse, minor_axis_end1, minor_axis_end2, (255, 0, 0), 2)

            # Create darkness visualization
            darkness_viz = start_img_bordered.copy()
            if len(darkness_viz.shape) == 2:
                darkness_viz = cv2.cvtColor(darkness_viz, cv2.COLOR_GRAY2BGR)
            # Highlight dark areas in red
            dark_mask = (cv2.cvtColor(start_img_bordered, cv2.COLOR_BGR2GRAY) < lightness_threshold) & (
                        mask_filled == 255)
            darkness_viz[dark_mask] = [0, 0, 255]  # Red for dark areas

            _showImages(show, maintitle=measures_str,
                        images=[thresh, mask, mask_filled, image_with_ellipse, darkness_viz],
                        titles=["Thresholded", "Islands Removed", "Flood Filled",
                                "Mask with Ellipse", f"Dark Areas (< {lightness_threshold})"])

        return area, length, width, darkness_pct
    else:
        print("Not enough points to fit an ellipse.")
        return 0, 0, 0, 0

def calculate_darkness_percentage(wing_img, mask_filled, lightness_threshold):
    """
    Calculate the percentage of the wing area that is below the lightness threshold.

    Parameters:
    -----------
    wing_img : numpy array
        The cropped wing image (BGR or grayscale)
    mask_filled : numpy array
        The binary mask of the wing (255 for wing, 0 for background)
    lightness_threshold : int
        The threshold value for determining darkness

    Returns:
    --------
    float : Percentage of wing pixels below the threshold
    """
    # Convert to grayscale if needed
    if len(wing_img.shape) == 3:
        gray_img = cv2.cvtColor(wing_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = wing_img

    # Apply the mask to only consider pixels within the wing
    wing_pixels = gray_img[mask_filled == 255]

    # Count pixels below the threshold
    if len(wing_pixels) > 0:
        dark_pixels = np.sum(wing_pixels < lightness_threshold)
        total_pixels = len(wing_pixels)
        darkness_percentage = (dark_pixels / total_pixels) * 100
    else:
        darkness_percentage = 0.0

    return darkness_percentage