import numpy as np
from bigcrittercolor.helpers import _showImages
import glob
import os
import cv2
import pandas as pd

# scale an image, and return
def scaleCropTopBottom(img,crop_height_percent=28.3,crop_width_percent=53.3,
                       top_offset=35,bot_offset=35,left_offset=30,right_offset=30,show=False):
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
    all_contours = np.vstack(contours[i] for i in range(len(contours)))
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
                             crop_height_percent=28.3,crop_width_percent=53.3, thresh_c=5, dilate_kernel_size=5, show=False):
    print(img_id)
    top_bot = scaleCropTopBottom(img,show=show,top_offset=top_offset,bot_offset=bot_offset,crop_height_percent=crop_height_percent,crop_width_percent=crop_width_percent)
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

def processFolder(folder_path, cm_per_pixel,top_offset=35,bot_offset=35,thresh_c=3,
                  crop_height_percent=28.3,crop_width_percent=53.3, dilate_kernel_size=5, show=False):
    # Create a list of images and a list of filenames without the extension
    imgs = []
    file_names = []

    for file_path in glob.glob(os.path.join(folder_path, "*.png")):
        # Read and append the image to the list
        imgs.append(cv2.imread(file_path))
        # Extract the basename (e.g., "example.png") and then split the extension
        base_name = os.path.basename(file_path)
        name_without_extension, _ = os.path.splitext(base_name)
        # Append the name without extension to the list
        file_names.append(name_without_extension)

    metrics = [scaleCropFindWingContour(img,img_id,cm_per_pixel=cm_per_pixel,top_offset=top_offset,bot_offset=bot_offset, dilate_kernel_size=dilate_kernel_size,
                                        thresh_c=thresh_c,crop_height_percent=crop_height_percent,crop_width_percent=crop_width_percent,show=show) for img,img_id in zip(imgs,file_names)]

    # Unpack each tuple and create a DataFrame
    df = pd.DataFrame({
        'fore_area_cm2': [item[0][0] for item in metrics],
        'fore_length_cm': [item[0][1] for item in metrics],
        'fore_width_cm': [item[0][2] for item in metrics],
        'hind_area_cm2': [item[1][0] for item in metrics],
        'hind_length_cm': [item[1][1] for item in metrics],
        'hind_width_cm': [item[1][2] for item in metrics],
    })

    df['id'] = file_names  # Add the file names as another column

    csv_file_path = 'metrics.csv'
    df.to_csv(csv_file_path, index=False)

    return df

# steps I did:
# 1. removed PDFs in Seth outlines
# 2. replaced spaces with underscores in folder names
# 3. moved 3 images from bisque to amnh/lacie because they were amnh and not in bisque format
# 4. cropped Seth outlines to remove top dragonfly part
# 5. changed zygoptera tifs to pngs using #convertTIFs('D:/new_dragonfly_wings/zygoptera','D:/new_dragonfly_wings/zygoptera_pngs', 5)
# 6. resized 2 oversized images in tests

# what I moved for manual:
# amnh - moved one trouble pigmented image for manual
# fsca - moved 5 imgs with either too big wings or no wings in frame for manual
# tests - moved one with none in frame for manual
# zygoptera - moved a few with wings too low or high one that has weird frame

show=False

# expected format
df1 = processFolder("D:/new_dragonfly_wings/amnh/lacie", cm_per_pixel=0.01149,thresh_c=3,dilate_kernel_size=5,show=show) # pixel multiplier, top wing topleft, top wing botright, bot_wingtopleft...
df2 = processFolder("D:/new_dragonfly_wings/fsca", cm_per_pixel=0.01149,thresh_c=3,dilate_kernel_size=5,show=show)
df3 = processFolder("D:/new_dragonfly_wings/fsca/lacie", cm_per_pixel=0.01149,thresh_c=3,dilate_kernel_size=5,show=show)
df4 = processFolder("D:/new_dragonfly_wings/tests",cm_per_pixel=0.01149,thresh_c=3,dilate_kernel_size=5,show=show)
df5 = processFolder("D:/new_dragonfly_wings/zygoptera_pngs",show=show,cm_per_pixel=0.01449,thresh_c=3,
                   top_offset=50)

# bisque images
df6_1 = processFolder("D:/new_dragonfly_wings/bisque/1",cm_per_pixel=0.00200, crop_width_percent=100,crop_height_percent=60, show=show,
                   top_offset=0,bot_offset=0)
df6_2 = processFolder("D:/new_dragonfly_wings/bisque/2",cm_per_pixel=0.00195, crop_width_percent=100,crop_height_percent=60, show=show,
                   top_offset=0,bot_offset=0)
df6_3 = processFolder("D:/new_dragonfly_wings/bisque/3",cm_per_pixel=0.0022, crop_width_percent=100,crop_height_percent=60, show=show,
                   top_offset=0,bot_offset=0)
df6_4 = processFolder("D:/new_dragonfly_wings/bisque/4",cm_per_pixel=0.00175, crop_width_percent=100,crop_height_percent=50, show=show,
                   top_offset=0,bot_offset=0)
df6_5 = processFolder("D:/new_dragonfly_wings/bisque/5",cm_per_pixel=0.0035, crop_width_percent=100,crop_height_percent=50, show=show,
                   top_offset=0,bot_offset=0)
df6_6 = processFolder("D:/new_dragonfly_wings/bisque/6",cm_per_pixel=0.0020, crop_width_percent=100,crop_height_percent=55, show=show,
                    top_offset=0,bot_offset=0)
df6_7 = processFolder("D:/new_dragonfly_wings/bisque/7",cm_per_pixel=0.00246, crop_width_percent=100,crop_height_percent=55, show=show,
                    top_offset=0,bot_offset=0)
df6_8 = processFolder("D:/new_dragonfly_wings/bisque/8",cm_per_pixel=0.00233, crop_width_percent=100,crop_height_percent=55, show=show,
                    top_offset=0,bot_offset=0)
df6_9 = processFolder("D:/new_dragonfly_wings/bisque/9",cm_per_pixel=0.00209, crop_width_percent=100,crop_height_percent=55, show=show,
                    top_offset=0,bot_offset=0)
df6_10 = processFolder("D:/new_dragonfly_wings/bisque/10",cm_per_pixel=0.00182, crop_width_percent=100,crop_height_percent=55, show=show,
                    top_offset=0,bot_offset=0)

df_list = [df1,df2,df3,df4,df5,df6_1,df6_2,df6_3,df6_4,df6_5,df6_6,df6_7,df6_8,df6_9,df6_10]
df = pd.concat(df_list, ignore_index=True)
df.to_csv("D:/new_dragonfly_wings/metrics.csv", index=False)