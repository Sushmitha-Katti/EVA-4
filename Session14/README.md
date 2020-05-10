# Table of Contents
1. [Introduction](#monocular-depth-estimation-and-segmentation)
2. [Steps followed - Brief](#how-did-we-do-it-thinking)
3. [Steps followed - In detail](#but-was-it-that-simple-as-above-not-at-all-that-was-just-a-brief-lets-look-each-steps-in-detailman_technologist)
4. [Data Statistcis](#data-statistics-bar_chart)
5. [Conclusion](#Conclusion)
6. [References](#references-books)
<br/>
# **Monocular Depth Estimation and Segmentation** 

   !["Depth estimation"](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Extras/image.png)

 Data is the :heart: of AI/ML, without data AI/ML is nothing.So data preparation plays a important role in research. Without a correct data
 AI/ML is just **GIGO**(Garbage In Garbage Out).:wastebasket: :arrow_right: :wastebasket:
 <br/>
 First we need to collect data, pre-process it and bring it to the form that our algorithm understands.

 ### We have to prepared the dataset for **Monocular Depth Estimation and Segmentation**
 
  We are not doing this from any company/Organisation, we have to do it from our own normal systems only. This is challenging and 
  super intersting problem to work with.
  <br/>
  
  ## **How did we do it :thinking:?**
  
  1. [Collected 100 background images.](#small_orange_diamond-collecting-100-background-images)
  2. [Collected 100 foreground images, removed it's background and made it transparent.](#small_orange_diamond-collecting-100-foreground-images)
  3. [Created masks for 100 foreground images.](#small_orange_diamond-creating-mask-for-foreground-images)
  3. [For each background images, overlapped each foreground images on 20 random positions.](#small_orange_diamond-overlapping-foreground-and-background-and-creating-mask)
  4. [Got the mask of the overlapped images](#small_orange_diamond-overlapping-foreground-and-background-and-creating-mask)
  5. [Got the depth for the overlapped foreground-background images.](#small_orange_diamond-generating-depth-images-for-fg-bg-images)
 
  
  <br/>
    
   That's all:hugs: We got around  400k fg-bg, 400k fg-bg-mask, 400k depth images:innocent:
    
   ## But was it that simple as above? Not at all. That was just a brief. Lets look each steps in detail:man_technologist:

  
   ### **:small_orange_diamond: Collecting 100 Background images** 
    
   * We chose **Living Room** as a background image theme.
   * Collected 100 different types living rooms.
   * Resized all to **224x224**. Size can be anything between 150 to 250. But choosing the large size image can increase the storage
             so better stick to small images.
   * Each background image size is around 10-20 kb.
             
  ### **:small_orange_diamond: Collecting 100 Foreground images**
            
   * we chose **Humans** as foreground image theme.
   * Collected 100 different types of humans, with single, multiple humans etc.
   * Removed their background using **PPT**(ever thought you can do that?:astonished: It possible!) and saved as **png**(Only png images give transparent images, since they have 4 channels-4th channel represents transparency )
   * We kept their size aroung 100-120 px as height. We maintained aspect ratio. Else humans would look wierd:sweat_smile:
   * Each foregound image size is around 5-10 kb approx. Some are more also.
   
   ### **:small_orange_diamond: Creating mask for foreground images**
   
   * We used **GIMP** to create mask.
   * Select the transparent foreground image, go to colors and select invert colors. It will give white image with black background.
   * Then select threshold, vary the threshold values till you get complete white mask.
   * Then save it as jpg.
   
   ### **:small_orange_diamond: Overlapping foreground and background and creating mask**
   
   * By this we had 100 background, 100 foreground and mask images.
   * Since we were 5 in a team we decided to create 80k images each.
   * [This](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Overlap_and_mask.ipynb) is the code we followed.
   * For each background images, we took 20 foreground images,their flips, mask and flips masks images and overlapped 20 times at 20 random positions
   * For flips we used PIL modules **Image.FLIP_LEFT_RIGHT** function.
   * To generate masks for overlayed images, first we generated black background of size same as background image and overlayed the foreground mask on top of it.
   * For overlaying, we used **bg.paste(fg, (r1,r2),fg)**, where bg is background image, fg is foreground image, (r1,r2) are the top-left overlayed position. We generated those from random numbers.
   * We optimised those images, reduce it quality to 30, to save storage space.
   * So at a time we are saving 4 images at once, 1 fg-bg, 1fg-bg-flip, 1 fg-bg-mask, 1 fg-bg-flip-mask in 2 folders - 1 for mask ,and other for original
   * It took around **10 minutes** for 80k images. 
   * Then we zipped both the folders. 
   * At the end we had 5 zip folder named as data_Part1, data_Part2,...data_Part5. Each are arounf **580MB**.
   
   ### **:small_orange_diamond: Generating depth images for fg-bg images**
   
   * Since we don't have any depth camera, we had to lie on some pre-trained model to generate depth images.
   * we referred [this](https://github.com/ialhashim/DenseDepth) repository to generate our depth images. It is trained on **KITTY and NYU** dataset.
   * We used **NYU trained model** to generate images, because our images were similar to **NYU** dataset, so we get good results.
   * We modified some portion of the code to get the results we want.
   * In **layers.py** we modified **resize_images as resize**, since the syntax was changed from tf>2.0
   * In **utils.py** 
     * **Load Images** - added resize(448,448), since 224 is very small size,and anyway the result is half the size of output. So we doubled the size.
     * **display Images** - we converted the images to grayscale and saved the image.
   * In **test.py** 
     * we removed the glob and passed the direct images path.
     * we processed the images as **batches**. 200 images per batch.
     * So we ran for 80k/200 times
     * In between colab used to crash, we changed the start number according to number of images already processed and run again for remaining images.
   * Zipped the entire folder 
   * Each depth image size is 2-3kb. Each of 5 zip files took around 260-290 MB of storage. It took around 3-4 hours to generate depth for 80k images.
   
   ### **:small_orange_diamond: Merging the coresponding fg-bg and depth images**
   
   * Since we have 5 fg-bg zip, 5 and depth zip, we thought of merging each of 1 fg-bg zip with corresponding depth zip.
   * It took around 20 min and finally we have 5 zips of approx 800MB. Used [this](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Merge_files.ipynb) code.
   
   ### **:small_orange_diamond: Calculation Of Mean and Standard Deviation and generating labels**
   
   * Used [this code](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Mean_and_Std_of_Dataset.ipynb) to calculate mean and std of image. Mean and std are important to apply transformation, to normalise the data etc..
   * Used [this code](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Generate_Labels.ipynb) to generate the path of all images. It is in a order **FG BG FG-BG FG-BG-Mask Depth**
   
   
   ## Data Statistics :bar_chart:
   
   ### **:pushpin: Folder Structure**
   
            ├── Foreground
            |   └───── fg1.png
            |   └───── fg2.png
            |   └───── .... 
            |   └───── fg100.png
            |
            |── Background
            |   └───── bg1.jpg
            |   └───── bg2. jpg
            |   └───── ....
            |   └───── bg100.jpg
            |
            |── Mask
            |   └───── mask1.jpg
            |   └───── mask2. jpg
            |   └───── ....
            |   └───── mask100.jpg
            |
            |── Dataset
            |   └───── data_part1.zip
            |   |      └───── data_1
            |   |       |      └───── Fg-Bg
            |   |       |      |     └───── fg-bg <1-80k>.jpg
            |   |       |      └──── Fg-Bg-Mask
            |   |       |      |     └───── fg-bg-mask<1-80k>.jpg
            |   |       |      └──── Depth
            |   |       |      |     └────── depth<1-80k>.jpg\
            |   └───── data_part2.zip
            |   |      └───── data_2
            |   |       |      └───── Fg-Bg
            |   |       |      |     └───── fg-bg <80k-160k>.jpg
            |   |       |      └──── Fg-Bg-Mask
            |   |       |      |     └───── fg-bg-mask<80k-160k>.jpg
            |   |       |      └──── Depth
            |   |       |      |     └────── depth<80k-160k>.jpg
            |   └────── .....
            |
            |   └───── data_part5.zip
            |   |      └───── data_5
            |   |       |      └───── Fg-Bg
            |   |       |      |     └───── fg-bg <320k-400k>.jpg
            |   |       |      └──── Fg-Bg-Mask
            |   |       |      |     └───── fg-bg-mask<320k-400k>.jpg
            |   |       |      └──── Depth
            |   |       |      |     └────── depth<320-400k>.jpg
            |
            |────── labels.txt
            
  ### **:pushpin: Size and Storage**
   * Total - 3.98GB
     * Background Images - 1.2MB
     * Foreground Images - 1.2MB
     * Mask - 333KB
     * Dataset - 3.92GB
     * Labels - 52MB
     <br/>
   (Above values are generated from **du** command in colab)
   
   ### **:pushpin: Sample Images**
   * **Background Images**
   
   ![Background](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Images/background.png)
   
   * **Foreground Images**
    
   ![Foreground](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Images/Foreground.png)
   
 
   * **Foreground Masks**
   
   ![Mask](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Images/Mask.png)
   
   * **Foreground Background Images**
   
   ![Fg-Bg](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Images/fg-bg.png)
   
  * **Foreground Background Mask Images**
  
  ![Fg-Bg-Mask](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Images/fg-bg-mask.png)
  
  * **Depth Images**
  
  ![Depth](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Images/depth.png)
   
   
   ### **:pushpin: Mean and Standard Deviation**
   
   |        Type          |                Mean                  |                 STD                   |
   | ---------------------| -------------------------------------| --------------------------------------|
   | Fg-Bg                | [0.65830478, 0.61511271, 0.5740604 ] |  [0.24408717, 0.2542491, 0.26870159]  |
   | Fg-Bg-Mask           | [0.04608837, 0.04608837, 0.04608837] |  [0.20544916, 0.20544916, 0.20544916] |
   | Depth-Images         | [0.50911522, 0.50911522, 0.50911522] |  [0.28174302, 0.28174302, 0.28174302] |
   
        
        
## **Conclusion**

   This is how we created our dataset. There is lot of improvement can be done though. In depth images ram crashes after 3-4k images, we need to restart the program. This can be improvised by using garbage collector and deleting the variables which takes huge chunk of data.
   
   **Some tips**
   
   * Always use 3 channels to store images, since they consume less memory
   * Only for foreground images use png images. we need to maintain transparency.
   * Always use square images as background and use small image to overcome storage issues.
   * Depth model is trained on kitti(city,roads etc..) and nyu(indoor) dataset. Try to use similar kind of data to get good depth images, since they act as ground truth for our model.
   * Store the depth model and masks in gray scale since they require only 1 channels.
   
   
## **References :books:**
* [Overlap and masking](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Overlap_and_mask.ipynb)
* [Referred Depth model](https://github.com/ialhashim/DenseDepth)
* [Modified Depth model](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Final_Depth_Model.ipynb)
* [Merge all 10 zips to 5](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Merge_files.ipynb)
* [Generate labels](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/Generate_Labels.ipynb)
* [Data Statistics](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session14/data_statistics.ipynb)
* [Drive Link for Data](https://drive.google.com/open?id=1RbJHVxo91jhekv3_E9GZvujUDNDaxFQu)
            
    
   
   
   
   
   
   
   
   
   
    
    
    
  



 
 
 
 
