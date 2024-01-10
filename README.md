# VisionFuseNet

The latest iteration in medical segmentation analysis by the state-of-the-art Vision Lab aims to elevate performance while markedly reducing model parameters. Initially, the lab refined the Atrous Spatial Pyramid Pooling (ASPP) module, which notably enhanced segmentation accuracy but demanded extensive computational resources. The VisionFuseNet breakthrough reconfigures the ASPP module into a U-Net structured framework. This novel approach progressively diminishes image dimensions at each step, resulting in a substantial reduction in parameters employed within the multihead attention component. Consequently, this innovation significantly enhances efficiency, making it a considerably superior approach. 

<h2 style="color: Green;">Architecture</h2>

![VisionFuseNet Arch](https://github.com/Bhavjot-Singh03/VisionFuseNet/assets/131793243/040e83b9-39b5-4168-abfb-bbe7cd54e48e)
MHSAR Block
![VisionFuseNet Block](https://github.com/Bhavjot-Singh03/VisionFuseNet/assets/131793243/b1ec59ae-cede-412c-88de-178114b188ba)

<h2>Performance comparison</h2>

<h3>1. Kvasir Instrument</h3>

Backbone : DenseNet121

| Models               | Dice Score | mIoU   | Accuracy | Recall | Specificity | Precision | Parameters |
|----------------------|------------|--------|----------|--------|-------------|-----------|------------|
| Vision Lab           | 0.9396     | 0.8864 | 0.9862   | 0.9270 | 0.9948      | 0.9530    | 29.7M      |
| VisionFuseNet        | 0.9409     | 0.8888 | 0.9865   | 0.9460 | 0.9927      | 0.9367    | 18.2M      |

Average Training FPS: 2.537794794362772
Average Validation FPS: 50.984521796444064
Average Test FPS: 52.019017981653846

<h3>2. Kvasir Seg</h3>

Backbone : EfficientNetV2B3

| Models               | Dice Score | mIoU   | Accuracy | Recall | Specificity | Precision | Parameters |
|----------------------|------------|--------|----------|--------|-------------|-----------|------------|
| Vision Lab           | 0.9059     | 0.8299 | 0.9626   | 0.8999 | 0.9845      | 0.9153    | 37M        |
| VisionFuseNet        | 0.9231     | 0.8582 | 0.9699   | 0.9065 | 0.9879      | 0.9411    | 16.7M      |

Number of Images: 100
Total Inference Time: 4.5868 seconds
Average Inference Time per Batch: 1.5289 seconds
Average FPS: 21.80

