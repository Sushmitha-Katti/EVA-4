# Training and Learning Rates
This session largely covered on the Learning rate Algorithms. Mainly there was 2 **Constant Learning Rate Algorithms** and **Adaptive Learning Rate Algorithms** discussed their pros and cons and when to use them.

Discussed various versions of SGD like **SGD, Mini batch SGD, SGD with momentum(normal momentum and nesterov momentum)**.
Introduced to Adam and RMSProps algorithmns as well.

## Assignment
* Implement LR Finder to find best LR
* Implement ReduceLROnPlateau
* Use SGD with mometum
* Train For 50 epochs
* Target accuracy is 88%
* Run GradCAM on the any 25 misclassified images

## Implementation 

* Implented LR Finders from (https://github.com/davidtvs/pytorch-lr-finder)
* Experimented with the parameters(num_iteration,End_Lr,threshold,,start_Lr etc..) of LR finder and got the best LR.
* Implemented Save and Reload of best accuracy model.
* Used ReduceLROnPleateau as scheduler.
* Included the plot_curve function to plot Accuracy and Loss Graphs.
* Improved GradCam Code Such that it takes misclassified images and plot the given layers. Improved the way it plots.

## Parameteres
* **Architecture** - ResNet
* **Image agumentation** - HorizontalFlip, Rotate, RGBShift, Normalise, Cutout.
* **Dropout** - 0
* **Best Lr got from LR Finder** - 0.02417315480804103
* **No Of Epochs** - 50

## Results
* Best Traing Accuracy - 97.30%(50th Epoch)
* Best Test Accuracy - 93.80%(42nd Epoch)
* LR Finder Curve

![LR Finder Curve](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session10/Assets/Lr_finder_plot.png "Lr finder Curve")
* Accuracy Change Graph

![Accuracy Change Graph](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session10/Assets/AccuracyChange.png "Accuracy Change Graph")
* MisClassified Images
![Misclassified with GradCAm](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session10/Assets/Misclassified_images_withGradCam.png "Misclassifed images with GradCam")

## Observations
* Model is slightly overfitting. Need to add more images agumentation techniques.
* Could have used ReduceLrOnPleateau in better way, explored more parameters. Since there is no much change in accuracy at the end. It may be stucked at local minima or pleateau.
 
