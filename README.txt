Training and testing:

1.Training datasets
    Step 1: Create your dataset using LabelImg. Place the original images in VOCdevkit/VOC2007/JPEGImages and the corresponding annotation files in VOCdevkit/VOC2007/Annotations. Run voc_annotation.py to split the dataset into training and test sets. Finally, execute train_yolov7.py to train the YOLOv7 model.
    Step 2: Execute crop.py and use the Batch Crop Images function in yolo.py to crop all detected cells from a folder using the trained YOLOv7 model.
    Step 3: Label the cropped cells with LabelMe and place the results in the datasets/before folder. Run json_to_dataset.py to convert JSON annotations into PNG format. Store the original images in datasets/JPEGImages and the labels in datasets/SegmentationClass. Transfer these to VOCdevkit_unet/VOC2007, then run voc_annotation_unet.py to partition the training and validation sets. Train the U-Net model with train_unet.py.

2.AFM automatic detection
    Run predict.py.

Additional Notes:
    1.The program runs in Visual Studio Code.
    2.Trained models are saved in the logs/ directory. Ensure you update the model paths in configuration files during testing
    3.The integration of U-Net into YOLOv7 is implemented in yolo.py.
    4.The combination of template matching and network predictions is handled in yolo.py.
    5.Connect the local computer to the JPK microscope control computer via an Ethernet switch, ensuring both devices are on the same local area network (LAN).
    6.Copy the code from auto_jpk.txt into JPK NanoWizard software and execute it.


Reference:
       •https://github.com/bubbliiiing/yolov7-pytorch
       •https://github.com/bubbliiiing/yolov7-tiny-pytorch
       •https://github.com/bubbliiiing/unet-pytorch/tree/bilibili
       •https://github.com/WongKinYiu/yolov7
