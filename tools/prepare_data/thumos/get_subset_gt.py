import json
from collections import defaultdict
category_videos = defaultdict(int)
with open('exps/thumos/anno/thumos_14_anno.json') as f:
    origin_anno = json.load(f)

new_anno = {}
new_anno["database"] = {}
for video_id, video_info in origin_anno["database"].items():
    if origin_anno["database"][video_id]['subset'] == 'validation':
        for annotation in video_info["annotations"]:
            label = annotation["label"]
            if category_videos[label] < 2:
                category_videos[label] += 1
                new_anno["database"][video_id] = origin_anno["database"][video_id]
                break

with open('exps/thumos/anno/thumos_14_subtest_anno.json', 'w') as f:
    json.dump(new_anno, f, indent=4)
