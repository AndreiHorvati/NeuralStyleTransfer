# NeuralStyleTransfer

My Code - ImageLoader class
        - CustomStyleLoss class
        - Menu class
        - StyleTransfer class - get_model method
        - StyleTransferSettings class
        - StyleTransferUtils class
       
Internet Code - ContentLoss class
              - StyleLoss class
              - Normalization class
              - StyleTransfer class - style_transfer method
             
My Contribution - created a custom style loss (CustomStyleLoss class)
                - searched for other two models besides VGG-19 (ResNet-50 and Inception-v3)
                - for all three models and combined multiple layers in order to get the best visual results
                - used Huber Loss instead of Mean Squared Erros Loss for content
                - created a menu for the application where you can choose the model and how many iterations do you want to
                use for gradient descent
