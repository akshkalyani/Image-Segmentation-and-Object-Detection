import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import os

synthetic_image_folder = 'Input'
synthetic_output_folder = 'Output'

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
class_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

for i, image_file in enumerate(os.listdir(synthetic_image_folder)):
    plt.clf()
    image = Image.open(os.path.join(
        synthetic_image_folder, image_file)).convert('RGB')
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    with torch.no_grad():
        predictions = model([image_tensor])
    plt.imshow(image)
    ax = plt.gca()
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            class_name = class_names[label]
            patch = plt.Rectangle((x1, y1), w, h, fill=False,
                                  edgecolor='green', linewidth=2.0)
            ax.add_patch(patch)
            label_text = f'{class_name}: {score:.2f}'
            ax.text(x1, y1, label_text, fontsize=12, color='white',
                    bbox=dict(facecolor='green', alpha=0.5, edgecolor='none'))
    plt.savefig(os.path.join(synthetic_output_folder, f'image_{i}.jpg'))
