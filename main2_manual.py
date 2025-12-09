from module import processFolder
import pandas as pd

# 2nd pass adding new species that were missed the first time around
# steps I did:
# 1. converted PDFs and TIFs to PNGs
# 2. for converted PDFs, put those that are scaled in separate folders
# 3. for converted TIFs, split into folders one per scale
# 4. after processing, removed hind wings column values from those images without hind wings

show=True

# manual traced
# 62
df = processFolder("D:/new_wings_downloads/traced",cm_per_pixel=1,thresh_c=3,dilate_kernel_size=5,crop_width_percent=100,crop_height_percent=65, top_offset=0,bot_offset=0,show=show)
df.to_csv("D:/new_wings_downloads/traced_metrics.csv", index=False)

# manual pdf to png set 1
#
#df = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs1",
#                    cm_per_pixel=1,thresh_c=3,dilate_kernel_size=5,
#                    crop_width_percent=100,crop_height_percent=65, top_offset=0,bot_offset=0,
#                    show=show)
#df.to_csv("D:/new_wings_downloads/pdfs_to_pngs1_metrics.csv", index=False)

#  manual pdf to png set 2
#df = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs2",
#                    cm_per_pixel=1,thresh_c=3,dilate_kernel_size=5,
#                    crop_width_percent=100,crop_height_percent=65, top_offset=0,bot_offset=0,
#                    show=show)
#df.to_csv("D:/new_wings_downloads/pdfs_to_pngs2_metrics.csv", index=False)

# fore only - so remember to remove hind from df
# Arabicnemis.png
# Arrhenocnemis.png
# Disparocypha.png
# Metacnemis.png

# tifs
# df1 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_tifs_to_pngs/1",
#                     cm_per_pixel=0.0106,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
#
# df2 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_tifs_to_pngs/2",
#                     cm_per_pixel=0.01449,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
#
# df3 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_tifs_to_pngs/3",
#                     cm_per_pixel=0.01111,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
#
# df_list = [df1,df2,df3]
# df = pd.concat(df_list, ignore_index=True)
# df.to_csv("D:/new_wings_downloads/tifs_metrics.csv", index=False)

# df1 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs_scaled/1",
#                     cm_per_pixel=0.002906,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
# df2 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs_scaled/2",
#                     cm_per_pixel=0.003663,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
# df3 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs_scaled/3",
#                     cm_per_pixel=0.002906,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
# df4 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs_scaled/4",
#                     cm_per_pixel=0.003745,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
# df5 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs_scaled/5",
#                     cm_per_pixel=0.005988,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
# df6 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs_scaled/6",
#                     cm_per_pixel=0.0098,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
# df7 = processFolder("D:/new_wings_downloads/TO DO OCTOBER 2024/new_pdfs_to_pngs_scaled/7",
#                     cm_per_pixel=0.0033,thresh_c=3,dilate_kernel_size=5, manual=True, show=show)
#
# df_list = [df1,df2,df3,df4,df5,df6,df7]
# df = pd.concat(df_list, ignore_index=True)
# df.to_csv("D:/new_wings_downloads/pdfs_scaled_metrics.csv", index=False)

from module import loadCombineSaveCSV

loadCombineSaveCSV(file_paths=["D:/new_wings_downloads/pdfs_to_pngs1_metrics.csv",
                              "D:/new_wings_downloads/pdfs_to_pngs2_metrics.csv",
                              "D:/new_wings_downloads/traced_metrics.csv",
                              "D:/new_wings_downloads/tifs_metrics.csv",
                              "D:/new_wings_downloads/pdfs_scaled_metrics.csv"
                              ], output_file="D:/new_wings_downloads/combined_metrics.csv")