import json 

five_star_res = json.load(open("./5-star/results/zero-shot-results.json", "r", encoding="utf-8"))["results"]

loss = 0
for item in five_star_res:
  if item["pred"] not in item["generation"]:
    loss += 1

import pdb; pdb.set_trace()