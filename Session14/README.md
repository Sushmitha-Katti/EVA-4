# **Monocular Depth Estimation and Segmentation Data Preparation** 

 Data is the :heart: of AI/ML, without data AI/ML is nothing.So data preparation plays a impotant role in research. Without a correct data
 AI/ML is just **GIGO**(Garbage In Garbage Out).
 <br/>
 First we need to collect data, pre-process it and bring it to the form that our algorithm understands.
 
 <br/>
 <br/>
 
 ## We have to prepared the dataset for **Monocular Depth Estimation and Segmentation**
 
  We are not doing this from any company/Organisation, we have to do it from our own normal systems only. This is challenging and 
  super intersting problem to work with.
  <br/>
  <br/>
  
  ## **How did we do it :thinking:?**
  
  1. Collected 100 background images.
  2. Collected 100 foreground images, removed it's background and made it transparent.
  3. Created masks for 100 foreground images.
  3. For each background images, overlapped each foreground images on 20 random positions.
  4. Got the mask of the overlapped images
  5. Got the depth for the overlapped foreground-background images.
  
  <br/>
  <br/>
    
   That's all :innocent: We got around  400k fg-bg, 400k fg-bg-mask, 400k depth images :hugs:
    
   #### But was it that simple as above? Not at all. That was just a brief. Lets look each steps in detail
    
   ### **:small_orange_diamond: Collecting 100 Background images** 
    
   * We chose **Living Room** as a background image theme.
   * Collected 100 different types living rooms.
   * Resized all to **224x224**. Size can be anything between 150 to 250. But choosing the large size image can increase the storage
             so better stick to small images.
             
  ### **:small_orange_diamond:  Collecting 100 Foreground images**
            
   * we chose **Humans** as foreground image theme.
   * Collected 100 different types of humans, with single, multiple humans etc.
   * Removed their background using **PPT**(ever thought you can do that? It possible!) and saved as **png**(Only png images give transparent images, since they have 4 channels-4th channel represents transparency )
   * We kept their size aroung 100-120 px as height. We maintained aspect ratio. Else humans would look wierd:sweat_smile:
    
  



 
 
 
 
