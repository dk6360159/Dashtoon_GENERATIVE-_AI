# Dashtoon_GENERATIVE-_AI
“  If you want to run the code file first you have to download the images have have to put the respective path of content_image_path and style_image_path “

An explanation of the code what each cell is doing is written at the top of each in the colab notebook 
In the same folder, I have uploaded some content images and style images, 

For feature calculation of the image, I  used a pre-trained VGG19 network because it requires much GPU and data images to train the model for feature extraction.

All the things are also mentioned in the code file.

For Style transfer, took a generated_image, initialised it with content image, ran 3000 epochs for this tensor and updated this generated image, 

The loss function for this model is  content_loss and style_loss,
Content_loss is the MSE loss between original_image and generated_image features till ith epoch 

and style loss is calculated by style_image and generated_image features by calculating Gram_matrix of style_image and generated_image feature matrix

Style_loss is MSE loss of gram_matrix of style_image and generated_image
Gram_marix=W*(W transpose)
content_loss=    torch.mean((gen_feature   - orig_feature)**2)
style_loss=        torch.mean((Gram(gen_feature)-   Gram(orig_feature))**2)


Total loss =alpha * content_loss + beta* Style_loss

The value of alpha and beta is hyper Parameter tried many different values of alpha and beta
Found alpha= 1 and beta=0.01 are giving good results, still can try different values of alpha and beta 

Plotted content_loss and Style_loss and total_loss for each 200 epochs
 
Content_loss should increase with epoch because we initialised the generated image with content_image

Style_loss should decrease with the epoch we are transferring style to the generated image
Total_loss also decreases with epoch




