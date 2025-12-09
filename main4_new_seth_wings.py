import pandas as pd
from module import processFolder

# steps I did:
# 1. converted TIFs to X * 2000 PNGs with Py script
# 2. converted pdfs to X * 2000 PNGs with pdfconverter
# 3, organized into 2 folders - topleft, and manual

# what I moved for manual:
# amnh - moved one trouble pigmented image for manual
# fsca - moved 5 imgs with either too big wings or no wings in frame for manual
# tests - moved one with none in frame for manual
# zygoptera - moved a few with wings too low or high one that has weird frame

# 1 cm is 72 pixels horizontally
show=True

# topleft: expected format - 1 cm is 72 pixels horizontally
#df = processFolder("D:/new_seth_wings/topleft", cm_per_pixel=0.013888,thresh_c=5,top_offset=50,dilate_kernel_size=5,show=False)
#df.to_csv("D:/new_seth_wings/topleft.csv", index=False)

# topleft_manual: manual ones for expected format - where method fails slightly so need to do manual - libellago_dorsocyana_f has fore only just changed
# libellago_dorsocyana_f has fore only
#df = processFolder("D:/new_seth_wings/topleft_manual", cm_per_pixel=0.013888,thresh_c=2,top_offset=50,dilate_kernel_size=5,show=show,manual=True)
#df.to_csv("D:/new_seth_wings/topleft_manual.csv", index=False)

# for Arabicnemis - no hindwing, only fore - just edited this in the .csv
#df = processFolder("D:/new_seth_wings/manual_unscaled", cm_per_pixel=1,resize_if_huge=True,thresh_c=3,dilate_kernel_size=5,show=show,manual=True)
#df.to_csv("D:/new_seth_wings/manual_unscaled.csv", index=False)

# manual images where scale bar must be defined becauses differs in every image
#df = processFolder("D:/new_seth_wings/manual_pngs_jpgs", resize_if_huge=True, define_scalebar=True,thresh_c=3,dilate_kernel_size=5,show=show,manual=True)
#df.to_csv("D:/new_seth_wings/manual_pngs_jpgs.csv", index=False)

# topleft_manual2
#df = processFolder("D:/new_seth_wings/topleft_manual2", cm_per_pixel=0.013888,thresh_c=2,top_offset=50,dilate_kernel_size=5,show=show,manual=True)
#df.to_csv("D:/new_seth_wings/topleft_manual2.csv", index=False)

# manual unscaled - the new 12ish images on 1/23/25
df = processFolder("D:/new_seth_wings/more_manual_unscaled", resize_if_huge=True, cm_per_pixel=1,thresh_c=2,top_offset=50,dilate_kernel_size=5,show=show,manual=True)
df.to_csv("D:/new_seth_wings/more_manual_unscaled.csv", index=False)

# List all your CSV files
file_paths = [
    "D:/new_seth_wings/topleft.csv",
    "D:/new_seth_wings/topleft_manual.csv",
    "D:/new_seth_wings/manual_unscaled.csv",
    "D:/new_seth_wings/manual_pngs_jpgs.csv",
    "D:/new_seth_wings/topleft_manual2.csv",
    "D:/new_seth_wings/more_manual_unscaled.csv"
]

# Read each CSV and store DataFrames in a list
df_list = []
for path in file_paths:
    df_temp = pd.read_csv(path)
    df_list.append(df_temp)

# Concatenate all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined DataFrame
output_csv = "D:/new_seth_wings/all_combined.csv"
combined_df.to_csv(output_csv, index=False)

