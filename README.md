# Attention Based Image Compression Post-Processing Convolutional Neural Network

This repo contains the implementation of <a href= "https://arxiv.org/abs/1905.11045"> this paper</a>. 

This paper aims at developing a novel approach for image compression at low bitrates by leveraging the power of Convolutional Neural Networks (CNNs). Traditionally, image compressors such as BPG and H.266 have achieved impressive compression quality. However, to further enhance the compression results, this study proposes an attention-based CNN architecture designed to post-process the output of traditional image compression decoders.

By incorporating attention mechanisms into the CNN, the proposed method focuses on enhancing specific regions of the reconstructed image to improve its overall quality. The attention-based post-processing module is trained using Mean Absolute Error (MAE) and Multi-Scale Structural Similarity Index (MS-SSIM) loss functions, which encourage the network to generate visually appealing images while maintaining compression efficiency.

Through extensive experimentation on validation sets, the proposed approach demonstrates remarkable performance. At a bit-rate of 0.15, the post-processing module achieves an average Peak Signal-to-Noise Ratio (PSNR) of 32.10, surpassing other competing methods. This signifies the effectiveness of the attention-based CNN in significantly improving image compression quality at low bitrates.

Overall, this research introduces a promising solution that combines traditional image compression techniques with the capabilities of CNNs. By leveraging attention mechanisms and training the post-processing module with appropriate loss functions, the proposed approach achieves impressive compression results, enhancing the overall visual quality of compressed images at low bitrates.

This repo hosts the resnet architecture code, channel and spatial attention functions and a high level code of the pre-processing residual neural network. 

The visualization of model architecture can be seen <a href="https://drive.google.com/file/d/1U4Klx0N9o6WFJqjsj5sB4VIroCgchU7c/view?usp=sharing">here</a>. 