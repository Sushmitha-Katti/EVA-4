# Super Convergence
This session was about history and evolution of Learning rates. Strategies adopted in using learning rate. 
* **Learning rate Annealing** - Selecting a good starting learning rate is merely the first step. In order to efficiently train a robust model, we will need to gradually decrease the learning rate during training. If the learning rate remains unchanged during the course of training, it might be too large to converge and cause the loss function to fluctuate around the local minimum
In this type We have **PIECEWISE CONSTANT**, **REDUCE_LR_ON_PLATEAU**
* **WarmUp Strategeis**

  * Constant Warmup
  * Gradual Warmup
 Both Cyclic Learning rate and One Cycle Policy was introduced by **LESLIE SMITH**
### Cyclic Learning Rate 
  The essence of the learning rate policy comes from the observation that increasing the LR might have a short term negative effect
and yet achieve a longer-term beneficial effect. This observation leads to the idea of letting the LR vary within a range of values
rather than adopting a stepwise fixed or exponentially decreasing value

### One Cycle Policy 
Similar to Cyclic Learning Rate, but here we have only one Cycle. The correct combination of momemtum, weight decay, Learning rate, batch size does magic.
One Cycle Policy will not increase, but the reasons to use it are

* It reduces the time it takes to reach "near" to your accuracy. 
* It allows us to know if we are going right early on. 
* It let us know what kind of accuracies we can target with given model.
* It reduces the cost of training. 
* It reduces the time to deploy

Both Cyclic Learning rate and One Cycle Policy was introduced by **LESLIE SMITH**
<hr>

## **Assignment**
* To write code by cyclic Lr grpah
* Implement given new Resnet Architecture
* Batch Size = 512, Target Accuracy - 90%
* To use One Cyclic LR such that the max lr should be at 5th epoch. The total no of epochs to use is 24.

## **Implementation**

* Implemented new Resnet Architecture and used Specified Agumentaion
* Used One Cyclic Lr and reached such that it reached max lr at 5th epoch.
* Implemented Range test to find Max Lr.
* Moldularised the training code, and simplified the plot function to take axis labels, x-axis, title.

## **Parameters**
* Batch Size - 512
* Transforms - 
  * Padding(4,4)
  * Random Crop (32,32)
  * Flip Lr
  * Cutout(8,8)
* Model Total Parameters - 6,573,120
* Range Test
  * Max Lr - 0.02
  * Min Lr - 0.001
  * Epochs - 24
* Loss function - NLLloss()
* Optimiser - SGD
  * Weight Decay - 0.05
  * Mometum - 0.9
* Scheduler - One Cycl2 Lr
  * epoch - 24
  * Max  Lr - 0.0125
  * no of steps - 98
  * pct start - 0.0125
  * cyclic momentum -False

## **Results**
* Best Train Accuracy - **95.65%**
* Best Test Accuracy - **91.73%**
* Epoch v/s Lr

![](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session11/Assets/Lr.png)

* Accuracy Graph

![](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session11/Assets/Accuracy_graph.png)

* Misclassified images

![](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session11/Assets/misclassified_images.png)

## **Observations**

* Some overfitting can be seen.
* Can use better combination of momentum, weight decay, Lr to get high accuracy.


  


  
