from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import numpy as np
import cv2
# Load model
config_file = 'cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = 'ICDAR.19.Track.B2.Modern.table.structure.recognition.v2.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')

# Test a single image 
# img = "demo.png"
img = "mmexport1607322541739.pic_hd.jpg"
# img = "mmexport1607322847441.pic_hd.jpg"

# Run Inference
result = inference_detector(model, img)
# print('result:', result)

if isinstance(result, tuple):
    bbox_result, segm_result = result
    if isinstance(segm_result, tuple):
        segm_result = segm_result[0]  # ms rcnn
else:
    bbox_result, segm_result = result, None
bboxes = np.vstack(bbox_result)
labels = [
    np.full(bbox.shape[0], i, dtype=np.int32)
    for i, bbox in enumerate(bbox_result)
]
labels = np.concatenate(labels)
# print('bboxes:', bboxes)
# print('labels:', labels)

image = cv2.imread(img)
score_thr=0.85
for i in range(len(bboxes)):
    if labels[i] == 2 and bboxes[i][-1] >= score_thr:
        cropped = image[int(bboxes[i][1])-10:int(bboxes[i][3])+10, int(bboxes[i][0])-10:int(bboxes[i][2])+10]
        cv2.imwrite(img[:-4]+'_cropped.jpg', cropped)


# Visualization results
# show_result_pyplot(model, img, result, score_thr=0.85)
