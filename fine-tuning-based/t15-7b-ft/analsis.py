import json 
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image

workbook = openpyxl.Workbook()
worksheet = workbook.active

test_df = pd.read_csv("../../data/twitter/twitter2017/test.tsv", sep="\t")
test_df = test_df.rename({"index": "sentiment", "#1 ImageID": "image_id", "#2 String": "tweet_content", "#2 String.1": "target"}, axis=1)
test_dataset = test_df.values
predictions = json.load(open("./new_bsz_4_lr_2e-4_twitter2017_llava1.5_7b[2]/predict_record_8.json", "r", encoding="utf-8"))
idx = 0
for p, td in zip(predictions, test_dataset):
  aspect = td[3]
  tweet = td[2].replace("$T$", aspect)
  image = Image(f"../../data/twitter/twitter2017_images/{td[1]}")
  image.width = 100
  image.height = 100
  pred = p["pred"]
  gt = p["gt"]
  if pred != gt:
    idx += 1
    # A: image, B: tweet, C: aspect, D: gt, E: pred
    worksheet.add_image(image, f"A{idx}")
    worksheet[f"B{idx}"] = tweet
    worksheet[f"C{idx}"] = aspect
    worksheet[f"D{idx}"] = gt 
    worksheet[f"E{idx}"] = pred
    worksheet.row_dimensions[idx].height = 100

workbook.save("./saved_models/error_results.xlsx")
workbook.close()