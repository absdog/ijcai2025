import json 

gsm = json.load(open("./gsm8k_test.json", "r", encoding="utf-8"))

new_gsm = []
idx = 0
for d in gsm:
    # import pdb; pdb.set_trace()
    res = d["answer"].split("###")[1].replace(" ", "")
    if res in d["cot"]:
        continue
    else:
        d.update({
            "id": idx,
        })
        new_gsm.append(d)
        idx += 1

json.dump(new_gsm, open("./new_gsm8k_test.json", "w", encoding="utf-8"), ensure_ascii=False)