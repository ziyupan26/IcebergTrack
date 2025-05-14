# Steps in tracking icebergs in Greenland's fjord

1. Data preparation: search and downloaded hh band SAR images of Sentinel 1 from planatary computer. Mask it with land shapefiles to help image detection model recognize icebergs.
2. Iceberg detection and tracking:
   1. Morphology: Input tif files. Use superpixel method to detect all possible icebergs and apply a classification model to detect real icebergs. Record each iceberg's shape and use this shape dataset to track icebergs over a long time and large area.
   2. CNN: Input jpg files. Label all large icebergs on the sentinel 1 images and finetune YOLOv8 model. Apply YOLOv8 model to detect and DeepSORT to track icebergs. 
