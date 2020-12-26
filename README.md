# CAR_Traffic_Sign_Classification
 Traffic Sign Classification
 
 ## Project: Build a Traffic Sign Recognition Program
 Overview
 ---
This project, using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is to train and validate a model so it can classify traffic sign images. After the model is trained, new German traffic signs on the web are classified.
 
 To meet specifications, the project will require submitting three files:
 * the Ipython notebook with the code
 * a writeup report either as a markdown or pdf file
 
 Load Data Set
 ---
 Read data using pickle
 Read classified sign maes along with label from csv and store in classID_signnames
 signnames_df = pd.read_csv(signnames_file)
 classID_signames = list(signnames_df['SignName'])
 
 The Project
 ---
 The goals / steps of this project are the following:
 * preprocess data (normalization)
 * Explore, summarize and visualize the data set (planned to add gray scale but currently not applied)
 * Design, train and test a model architecture - described in Model Architecure in detail
 * Use the model to make predictions on new images : downloaded 8 images (32x32x3) and uploaded in downloads folder
 * Analyze the softmax probabilities of the new images
 * Summarize the results with a written report
 
 ### Model Architecture
 ### Solution:Implement LeNet-5
 Implement the LeNet-5(http://yann.lecun.com/exdb/lenet/) neural network architecture
 data is 32*32*3 (RGB) not a gray scale so that I modified MNIST LeNet5 to apply RGB scale image data
 LeNet-5 Architecture will be discussed in detail in Model Architecture
 With hyper parater tuning (BATCHsize =128, EPOCH=100) I obtained 93~95% training accuracy
 #### Input
 The LeNet architecture accepts a 32X32XC as input, where C is the number of color channels(Gray scale C is in, RGB color C is 3).
 #### Architecture
 ##### Layer1 Convolution: The output shape should be 28X28X6
 Activation: choice of activation function is ReLu.
 Pooling: The output shape should be 14X14X6.
 #####  Layer2 Convolutional: The output shape should be 10X10X16
 Activation: ReLu again,
 Pooling: The output shape should be 5X5X16.
 *Flatten: Flatten the output shape of the final pooling layer such that its 1D instead of 3D, The easiest way to do is by using tensorflow.contrib.layers.flatten, which is imported, the output is 400.
 #####  Layer3: Fully Connected: This should have 120 outputs
 Activation: ReLu again.
 #####  Layer4: Fully Connected , this should be 84 outputs
 Activation: ReLu gain.
 #####  Layer5: Fully Connected(Logits) This should have 43 outputs
 #### Output
 Return the resell of 2nd fully connected layer.
 
 #### Model Evaluation
 Evaluate how well the loss and accuracy of the model for a given dataset.
 (EPOCHS, BATCH_SIZE) --> (100,128):93.9%, previous expriment (20,32):82% (20,64):90.9%,(20,128):89.9%
 For the validation set , accuracy goes up to 93.9% to 95%.
 for the test accuracy was similar to accuracy of validation set around 93%
 #### New Data test & Analysis
 Applied training model to newly downloaded 8 images under downloads folder. Got 87.5% accuracy
 Applied top 5 Softmax predicted signs and listed each probabilities along with predected sign name in the bar chart
 One failed example, 80speed limit showed 30 speed limit with 100% probabilities. Outstading issues are row resolution images.I only used data normalization by divining 255. However, I could experiment gray scale and apply gausian filter or other filters to extract more distintive features before applying CNN in LeNet-5 architecture.
 ### Summary
 Digested LeNet-5 Architecture , Hyperparameter tuning by exploreing BATCH_SIZE and EPOCH , and achieved 93.9% ~ 95% validation set accuracy and around 93% testing accuracy. That was pretty good since two accuracies are balanced. However new data set was not predicted well (87.5% accuracy). Need to apply more image data preprossing(RGB 2 gray, applying gausian filter or other filters, etx) Even for normazliation, I could find min and max or RGB data and normalize data by dividing the difference between min-max instead of divining by 255
 
