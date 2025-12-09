import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def select_points_and_calculate_area(image_folder, csv_path='wing_areas.csv', show=False):
    # Create or read the CSV file
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
    else:
        results_df = pd.DataFrame(columns=['image_name', 'hindwing_area_cm2', 'forewing_area_cm2'])

    # Iterate through images in the folder
    for image_name in os.listdir(image_folder):
        if image_name in results_df['image_name'].values:
            print(f"Skipping {image_name}, already processed.")
            continue

        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load {image_name}. Skipping.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image for the user to select points
        plt.imshow(image_rgb)
        plt.title(f"Select points for hindwing contour, forewing contour, and a 1 cm scale bar for {image_name}")
        forewing_points = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close()

        # Display the image for the user to select points
        plt.imshow(image_rgb)
        plt.title(f"Select points for hindwing contour, forewing contour, and a 1 cm scale bar for {image_name}")
        hindwing_points = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close()

        # Display the image for the user to select points
        plt.imshow(image_rgb)
        plt.title(f"Select points for hindwing contour, forewing contour, and a 1 cm scale bar for {image_name}")
        scale_points = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close()

        # remove the first point it is ALWAYS the zoom point
        forewing_points = np.array(forewing_points[1:])
        hindwing_points = np.array(hindwing_points[1:])
        scale_points = np.array(scale_points[1:])

        if len(forewing_points) <= 2:
            print(f"Not enough points selected for {image_name}. Skipping.")
            continue

        # Calculate the length of the scale bar in pixels
        scale_length_pixels = np.linalg.norm(scale_points[1] - scale_points[0])

        print("Scale bar is " + str(scale_length_pixels) + " pixels")
        # Calculate area using the selected points (using cv2 contourArea)
        hindwing_area_pixels = cv2.contourArea(hindwing_points.astype(np.float32))
        forewing_area_pixels = cv2.contourArea(forewing_points.astype(np.float32))


        print("Hindwing has " + str(hindwing_area_pixels) + " pixels")
        print("Forewing has " + str(forewing_area_pixels) + " pixels")

        # Scale factor in pixels per cm
        pixels_per_cm = scale_length_pixels
        pixels_per_cm2 = pixels_per_cm ** 2

        print("Pixels per cm2: " + str(pixels_per_cm2))

        # Convert areas to cm²
        hindwing_area_cm2 = hindwing_area_pixels / pixels_per_cm2
        forewing_area_cm2 = forewing_area_pixels / pixels_per_cm2

        # Show outlined parts if `show` is True
        if show:
            outlined_image = image_rgb.copy()
            cv2.polylines(outlined_image, [hindwing_points.astype(np.int32)], isClosed=True, color=(255, 0, 0),
                          thickness=2)
            cv2.polylines(outlined_image, [forewing_points.astype(np.int32)], isClosed=True, color=(0, 255, 0),
                          thickness=2)
            cv2.line(outlined_image, tuple(scale_points[0].astype(int)), tuple(scale_points[1].astype(int)),
                     color=(0, 0, 255), thickness=2)

            plt.imshow(outlined_image)
            plt.title(f"Outlined parts for {image_name}")
            plt.show()

        # Save result to DataFrame
        new_row = {
            'image_name': image_name,
            'hindwing_area_cm2': hindwing_area_cm2,
            'forewing_area_cm2': forewing_area_cm2
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # Save to CSV
        results_df.to_csv(csv_path, index=False)
        print(f"Processed and saved results for {image_name}.")


# Example usage
select_points_and_calculate_area("C:/Users/hiest/Desktop/Absolutely Key Papers/new_wings_downloads/p longipennis", 'wing_areas.csv',show=True)