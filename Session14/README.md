# **Monocular Depth Estimation and Segmentation Data Preparation** 

 Data is the :heart: of AI/ML, without data AI/ML is nothing.So data preparation plays a important role in research. Without a correct data
 AI/ML is just **GIGO**(Garbage In Garbage Out).:wastebasket: :arrow_right: :wastebasket:
 <br/>
 First we need to collect data, pre-process it and bring it to the form that our algorithm understands.

 ### We have to prepared the dataset for **Monocular Depth Estimation and Segmentation**
 
  We are not doing this from any company/Organisation, we have to do it from our own normal systems only. This is challenging and 
  super intersting problem to work with.
  <br/>
  
  ## **How did we do it :thinking:?**
  
  1. Collected 100 background images.
  2. Collected 100 foreground images, removed it's background and made it transparent.
  3. Created masks for 100 foreground images.
  3. For each background images, overlapped each foreground images on 20 random positions.
  4. Got the mask of the overlapped images
  5. Got the depth for the overlapped foreground-background images.
  
  <br/>
    
   That's all:hugs:We got around  400k fg-bg, 400k fg-bg-mask, 400k depth images:innocent:
    
   #### But was it that simple as above? Not at all. That was just a brief. Lets look each steps in detail

  <hr/>
  
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
   * Each depth image size is 2-3kb. Each of 5 zip files took around 260-290 MB of storage.
   
   
   
   
   
    
    
    
  



 
 
 
 
