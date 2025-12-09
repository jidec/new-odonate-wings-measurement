import pandas as pd
from module import processFolder, processFolder_threshold, processFolder_threshold_lines

show=True
df = processFolder_threshold_lines("D:/threshold_test", lightness_threshold=128, cm_per_pixel=0.013888,thresh_c=2,dilate_kernel_size=5,show=show)
