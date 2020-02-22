## Code1 - Setup
**Target**
. Set up the skeleton with dropout, batch Norm etc

**Results** 
. Parameters - 8064
. Best training accuracy- 98.13
. Best test accuracy - 99.17

**Analysis**
. Model is under-fitting
. The gap between test and train is high
. Capacity can be increased

## Code2
**Target**
. Added 1 more convolution layer and Gap Layer
. Changed the number of kernels in some conv layers
. Increased the capacity

**Results**
. Parameters - 9634
. Best training accuracy- 98.67
. Best test accuracy - 99.4

**Analysis**
. Target is reached, but only once.
. Model is still under-fitting

## Code 3

**Target**
. Increased the dropout from 0.1 to 0.12

**Results**
. Parameters - 9872
. Best training accuracy- 98.64
. Best test accuracy - 99.32

**Analysis**
. Model is still under-fitting
. The accuracy of both is reduced by increasing dropout

## Code 4

**Target**
. Decreased the dropout from 0.12 to 0.05

**Results**
. Parameters - 9872
. Best training accuracy- 98.91
. Best test accuracy - 99.53

**Analysis**
. Reached the desired accuracy at 8th epoch
. Should see how it works introducing LR and reducing no of parameters

## Code 5

**Target**
. Decreased the dropout from 0.05 to 0.02

**Results**
Parameters - 8016
Best training accuracy- 98.93
Best test accuracy - 99.42

**Analysis**
. Reached the desired accuracy at 11th epoch with less parameters
