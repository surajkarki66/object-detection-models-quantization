# Human Presence Detection Dataset
The Human Presence Detection Dataset is a combination of VOC2012_Person and private collection of images annotated with bounding boxes for detecting people. This dataset is designed for training and evaluating object detection models, mainly YOLO.

## Dataset Details
- Content: Images of people in various settings (e.g., indoor, outdoor, different poses, lighting conditions).
- Sources:
  - Public: PASCAL VOC 2012 Person dataset.
  - Private: Custom collection of images from Steinel GmbH.

- Annotations: Bounding boxes in YOLO format, with class ID 0 representing "person".
- Image Size: Images are typically 640x640 pixels (adjust based on your dataset).
- Annotation Format: Text files (.txt) with one line per bounding box in the format: class_id x_center y_center width height (normalized to [0, 1]).

Happy Coding
