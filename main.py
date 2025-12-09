import pandas as pd
from module import processFolder

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

show=True

# expected format
# 61
df1 = processFolder("D:/new_dragonfly_wings/amnh/lacie", cm_per_pixel=0.01149,thresh_c=3,dilate_kernel_size=5,show=show) # pixel multiplier, top wing topleft, top wing botright, bot_wingtopleft...
# 376
df2 = processFolder("D:/new_dragonfly_wings/fsca", cm_per_pixel=0.01149,thresh_c=3,dilate_kernel_size=5,show=show)
# 207
df3 = processFolder("D:/new_dragonfly_wings/fsca/lacie", cm_per_pixel=0.01149,thresh_c=3,dilate_kernel_size=5,show=show)
# 24
df4 = processFolder("D:/new_dragonfly_wings/tests",cm_per_pixel=0.01149,thresh_c=3,dilate_kernel_size=5,show=show)
# 219
df5 = processFolder("D:/new_dragonfly_wings/zygoptera_pngs",show=show,cm_per_pixel=0.01449,thresh_c=3,top_offset=50)

# 30ish
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