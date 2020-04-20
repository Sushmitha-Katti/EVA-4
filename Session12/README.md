# Object Localisation - YOLO

Till now we had concentrated on object classification. This session was about Object Localisation.

Object Localisation is a very hard task to do because we should not only predict the object but also we need to detect the exact position of it.

### **Detection Approaches**
**Simplify Object Detection problem by:**

1. ignoring lower prediction values
2. predicting bounding boxes instead of the exact object mask mixing different receptive field layers but this is easier said than done.
 
**There are two main approaches to driving detection algorithms, namely:**

1. YOLO-like approach, where k-means extracted anchor boxes are used, and
2. SSD-like approach, where a fixed number of predefined bounding boxes are used.

# *Assignment*

## **Assignment A - Tiny ImageNet**[Code](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/S12-AssignmentA/FinalCode.ipynb)

Should train ResNet18  on **Tiny ImageNetData Set** and reach test accuracy of 50%

### **Implementation**

1. Wrote a Code to download, mix train and test , split and convert to the dataset format.
2. Used One Cycle Policy as Scheduler. It yielded better and fast results than others
3. Reached the target accuracy

### **Parameters**

1. Agumentations - Horizontal flip, Padding , Random Crop, Normalisation, Cutout
2. Batch Size - 256
3. Model - Resnet 18 with 200 classes
4. Optimiser - SGD(momentum - 0.9 , weight_decay - 0.0001)
5. Scheduler - One Cycle (  max_lr = 0.02, epochs=30,  pct_start=1/3, anneal_strategy='linear', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=10.0,final_div_factor =10)

### **Results**

1. Best train Accuracy - 74%
2. Best test Accuracy - 57.69%
4. Accuracy Change Graph

![Accuracy Change](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/Assets/Accuracy%20Change.png)

5. GradCam on MisClassified Images (All 4 layers)
![MisClassified](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/Assets/misclassified.png)


## **Assignment B - Find out the best total numbers of clusters** [Code](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/S12-AssignmentB/Clustering%20Dogs%20Bounding%20Boxes.ipynb)
   
 To download 50 dog images, annotate with vgg annotator, find out the best no of clusters
 
 **Visualisation of data**
 
 ![data](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/Assets/data.png)
 
 **Elbow method to find out K**
 
 ![elbow](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/Assets/elbow.png)
 
 
 **Mean Iou Method to find out k**
 
 ![Mean IOU](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/Assets/IOU.png)
 
 **K means with k = 3**
 
 ![K=3](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/Assets/K%3D3.png)
 
 **K Means with k =4**
 
 ![k=4](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/Assets/k%3D4.png)


