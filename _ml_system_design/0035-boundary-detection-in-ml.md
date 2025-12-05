---
title: "Boundary Detection in ML"
day: 35
collection: ml_system_design
categories:
  - ml_system_design
tags:
  - computer-vision
  - segmentation
  - edge-detection
  - unet
  - autonomous-driving
subdomain: "Computer Vision"
tech_stack: [OpenCV, PyTorch, U-Net, Canny]
scale: "Real-time, 60 FPS"
companies: [Tesla, Waymo, Adobe, Medical AI]
---

**"Defining where one object ends and another begins."**

## 1. The Problem: Edges vs. Boundaries

- **Edge Detection:** Finding sharp changes in pixel intensity (low-level).
  - Example: A checkerboard pattern has many edges.
- **Boundary Detection:** Finding semantically meaningful contours of objects (high-level).
  - Example: The outline of a "Dog" or "Car".

**Applications:**
- **Autonomous Driving:** Lane detection, road boundaries.
- **Medical Imaging:** Tumor segmentation, organ boundaries.
- **Photo Editing:** "Select Subject" tool in Photoshop.

## 2. Classical Approaches

Before Deep Learning, we used math.

### 1. Canny Edge Detector (1986)
The gold standard for decades.
1.  **Gaussian Blur:** Remove noise.
2.  **Gradient Calculation:** Find intensity change ($\nabla I$) using Sobel filters.
3.  **Non-Maximum Suppression:** Thin out edges to 1-pixel width.
4.  **Hysteresis Thresholding:** Keep strong edges, and weak edges connected to strong ones.

**Pros:** Fast, precise localization.
**Cons:** Detects *all* edges (texture, shadows), not just object boundaries.

### 2. Structured Forests
- Uses Random Forests to classify patches as "edge" or "non-edge".
- Uses hand-crafted features (color, gradient histograms).

## 3. Deep Learning Approaches

Modern systems use CNNs to learn semantic boundaries.

### 1. Holistically-Nested Edge Detection (HED)
- **Architecture:** VGG-16 backbone.
- **Multi-Scale:** Predicts edges at multiple layers (conv3, conv4, conv5).
- **Fusion:** Combines side-outputs into a final edge map.
- **Loss:** Weighted Cross-Entropy (to handle class imbalance: 90% pixels are non-edge).

### 2. CASENet (Category-Aware Semantic Edge Detection)
- Not just "is this an edge?", but "is this a Dog edge or a Car edge?".
- **Architecture:** ResNet-101 with multi-label loss.
- **Output:** $K$ channels, one for each class boundary.

## 4. Deep Dive: U-Net for Boundary Detection

**U-Net** is the standard for biomedical segmentation, but it excels at boundaries too.

**Architecture:**
- **Encoder (Contracting Path):** Captures context (What is this?).
- **Decoder (Expanding Path):** Precise localization (Where is it?).
- **Skip Connections:** Concatenate high-res features from encoder to decoder to recover fine details.

**Loss Function for Thin Boundaries:**
Standard Cross-Entropy produces thick, blurry boundaries.
**Solution:** **Dice Loss** or **Tversky Loss**.
$$Dice = \frac{2 |P \cap G|}{|P| + |G|}$$
Where $P$ is prediction, $G$ is ground truth.

## 5. System Design: Lane Detection System

**Scenario:** Self-driving car needs to stay in lane.

**Pipeline:**
1.  **Input:** Camera feed (1080p, 60fps).
2.  **Preprocessing:** ROI cropping (focus on road), Perspective Transform (Bird's Eye View).
3.  **Model:** Lightweight CNN (e.g., ENet or LaneNet).
    - Output: Binary mask of lane lines.
4.  **Post-processing:**
    - Curve Fitting: Fit a 2nd or 3rd degree polynomial ($y = ax^2 + bx + c$) to the points.
    - Kalman Filter: Smooth predictions over time (lanes don't jump).

**Challenges:**
- **Occlusion:** Car in front blocks view.
- **Lighting:** Shadows, glare, night.
- **Worn Markings:** Faded lines.

## 6. Deep Dive: Active Contour Models (Snakes)

A hybrid approach: Deep Learning gives a rough mask, **Snakes** refine it.

**Concept:**
- Define a curve (snake) around the object.
- Define an **Energy Function**:
  - $E_{internal}$: Smoothness (don't bend too sharply).
  - $E_{external}$: Image forces (snap to high gradients).
- Minimize energy iteratively. The snake "shrinks-wraps" the object.

**Modern Twist:** **Deep Snake**. Use a GNN to predict vertex offsets for the polygon contour.

## 7. Evaluation Metrics

1.  **F-Measure (ODS/OIS):**
    - **ODS (Optimal Dataset Scale):** Best fixed threshold for the whole dataset.
    - **OIS (Optimal Image Scale):** Best threshold per image.
2.  **Boundary IoU:**
    - Standard IoU is dominated by the object interior.
    - Boundary IoU computes intersection only along the contour band.

## 8. Real-World Case Studies

### Case Study 1: Adobe Photoshop "Select Subject"
- **Problem:** User wants to cut out a person.
- **Solution:** Deep Learning model (Sensei) predicts a "trimap" (Foreground, Background, Unknown).
- **Refinement:** Matting Laplacian to solve the alpha value for hair/fur pixels.

### Case Study 2: Tesla Autopilot
- **Problem:** Map the drivable space.
- **Solution:** "HydraNet" multi-task learning.
- **Heads:** Lane lines, Road edges, Curbs.
- **Vector Space:** Projects image-space predictions into 3D vector space for planning.

## 9. Summary

| Component | Technology |
| :--- | :--- |
| **Low-Level** | Canny, Sobel |
| **Deep Learning** | HED, CASENet, U-Net |
| **Refinement** | Active Contours, CRF |
| **Metrics** | Boundary IoU, F-Score |

## 10. Deep Dive: U-Net Architecture Implementation

Let's implement a production-ready U-Net in PyTorch.

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512 (upconv) + 512 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out(d1))
```

## 11. Deep Dive: Loss Functions for Boundary Detection

Standard Binary Cross-Entropy (BCE) produces thick boundaries. We need specialized losses.

### 1. Weighted BCE (Class Imbalance)
Boundary pixels are rare (< 5% of image). Weight them higher.

```python
def weighted_bce_loss(pred, target, pos_weight=10.0):
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    return bce(pred, target)
```

### 2. Dice Loss (Overlap Metric)
Directly optimizes for IoU.

```python
def dice_loss(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice
```

### 3. Tversky Loss (Precision/Recall Trade-off)
Generalization of Dice. Control false positives vs. false negatives.

```python
def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    
    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    FN = (target * (1 - pred)).sum()
    
    tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    return 1 - tversky
```

### 4. Focal Loss (Hard Examples)
Down-weight easy examples, focus on hard ones.

```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()
```

## 12. Deep Dive: Post-Processing Techniques

Raw model output is noisy. Refine it.

### 1. Morphological Operations
```python
import cv2
import numpy as np

def post_process_boundary(mask):
    # Convert to uint8
    mask = (mask * 255).astype(np.uint8)
    
    # Morphological closing (fill small gaps)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Skeletonization (thin to 1-pixel width)
    mask = cv2.ximgproc.thinning(mask)
    
    return mask
```

### 2. Non-Maximum Suppression (NMS)
Keep only local maxima along the gradient direction.

```python
def non_max_suppression(edge_map, gradient_direction):
    M, N = edge_map.shape
    suppressed = np.zeros((M, N))
    
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255
            
            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = edge_map[i, j+1]
                r = edge_map[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = edge_map[i+1, j-1]
                r = edge_map[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = edge_map[i+1, j]
                r = edge_map[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = edge_map[i-1, j-1]
                r = edge_map[i+1, j+1]
            
            if (edge_map[i,j] >= q) and (edge_map[i,j] >= r):
                suppressed[i,j] = edge_map[i,j]
    
    return suppressed
```

## 13. Deep Dive: Real-Time Deployment Optimizations

For autonomous driving, we need 60 FPS (16ms per frame).

### 1. Model Quantization
Convert FP32 to INT8.

```python
import torch.quantization

model_fp32 = UNet()
model_fp32.eval()

# Post-training static quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Conv2d, torch.nn.Linear},
    dtype=torch.qint8
)

# Speedup: 3-4x on CPU
```

### 2. TensorRT Optimization
NVIDIA's inference optimizer.

```python
import tensorrt as trt

# Convert PyTorch model to ONNX
torch.onnx.export(model, dummy_input, "unet.onnx")

# Build TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("unet.onnx", 'rb') as model_file:
    parser.parse(model_file.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # Use FP16

engine = builder.build_engine(network, config)

# Speedup: 5-10x on GPU
```

### 3. Spatial Pyramid Pooling
Process multiple scales simultaneously.

```python
class SPPLayer(nn.Module):
    def __init__(self, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
    
    def forward(self, x):
        batch_size, channels, h, w = x.size()
        pooled = []
        
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (h // level, w // level)
            stride = kernel_size
            pooling = nn.AdaptiveMaxPool2d((level, level))
            tensor = pooling(x).view(batch_size, channels, -1)
            pooled.append(tensor)
        
        return torch.cat(pooled, dim=2)
```

## 14. Deep Dive: Data Augmentation for Boundary Detection

Boundaries are thin. Augmentation must preserve them.

```python
import albumentations as A

transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
    ], p=0.3),
    A.RandomBrightnessContrast(p=0.3),
])

# Apply to both image and mask
augmented = transform(image=image, mask=boundary_mask)
```

## 15. Deep Dive: Multi-Task Learning

Instead of just boundaries, predict boundaries + segmentation + depth.

**Architecture:**
```
        Shared Encoder
              |
    ┌─────────┼─────────┐
    |         |         |
Boundary   Segment   Depth
  Head      Head      Head
```

**Loss:**
$$L_{total} = \lambda_1 L_{boundary} + \lambda_2 L_{segment} + \lambda_3 L_{depth}$$

**Benefit:** Shared features improve all tasks. Segmentation provides context for boundaries.

## 16. System Design: Medical Image Boundary Detection

**Scenario:** Detect tumor boundaries in MRI scans.

**Pipeline:**
1.  **Preprocessing:**
    - Normalize intensity (Z-score).
    - Resize to 512x512.
    - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
2.  **Model:** 3D U-Net (process volumetric data).
3.  **Post-processing:**
    - 3D Connected Components (remove small noise).
    - Surface smoothing (Laplacian smoothing).
4.  **Validation:** Radiologist review (Human-in-the-loop).

**Metrics:**
- **Dice Score:** Overlap with ground truth.
- **Hausdorff Distance:** Maximum boundary error.

- **Hausdorff Distance:** Maximum boundary error.

## 17. Deep Dive: Conditional Random Fields (CRF) for Boundary Refinement

**Problem:** CNN outputs are often blurry at boundaries due to pooling and upsampling.

**Solution:** Post-process with a CRF to enforce spatial consistency.

**Dense CRF (Fully Connected CRF):**
- Every pixel is connected to every other pixel.
- **Unary Potential:** CNN prediction for pixel $i$.
- **Pairwise Potential:** Encourages similar pixels to have similar labels.

$$E(x) = \sum_i \psi_u(x_i) + \sum_{i<j} \psi_p(x_i, x_j)$$

Where:
- $\psi_u(x_i) = -\log P(x_i)$ (from CNN).
- $\psi_p(x_i, x_j) = \mu(x_i, x_j) \cdot k(f_i, f_j)$ (similarity kernel based on color and position).

**Implementation (PyDenseCRF):**
```python
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def crf_refine(image, prob_map):
    h, w = image.shape[:2]
    
    # Create CRF
    d = dcrf.DenseCRF2D(w, h, 2)  # 2 classes: boundary/non-boundary
    
    # Unary potential
    U = unary_from_softmax(prob_map)
    d.setUnaryEnergy(U)
    
    # Pairwise potentials
    # Appearance kernel (color similarity)
    d.addPairwiseGaussian(sxy=3, compat=3)
    
    # Smoothness kernel (spatial proximity)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    
    # Inference
    Q = d.inference(5)  # 5 iterations
    refined = np.argmax(Q, axis=0).reshape((h, w))
    
    return refined
```

**Result:** Sharp, clean boundaries aligned with object edges.

## 18. Deep Dive: Attention Mechanisms for Boundary Detection

**Observation:** Not all regions are equally important. Focus on boundary regions.

**Spatial Attention:**
```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Aggregate across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention
```

**Channel Attention (SE Block):**
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pooling
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

## 19. Case Study: Instance Segmentation (Mask R-CNN)

**Problem:** Detect boundaries of individual instances (e.g., 3 separate cars).

**Mask R-CNN Architecture:**
1. **Backbone:** ResNet-50 + FPN (Feature Pyramid Network).
2. **RPN (Region Proposal Network):** Proposes bounding boxes.
3. **RoI Align:** Extract features for each box (better than RoI Pooling, preserves spatial alignment).
4. **Heads:**
   - **Classification:** What class?
   - **Box Regression:** Refine box coordinates.
   - **Mask:** Binary mask for the instance (28x28, upsampled to box size).

**Boundary Extraction:**
- The mask head outputs a soft mask.
- Apply threshold (0.5) to get binary mask.
- Use `cv2.findContours()` to extract boundary polygon.

**Production Optimization:**
```python
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

predictor = DefaultPredictor(cfg)

# Inference
outputs = predictor(image)
instances = outputs["instances"]

# Extract boundaries
for i in range(len(instances)):
    mask = instances.pred_masks[i].cpu().numpy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours[0] is the boundary polygon
```

## 20. Advanced: Differentiable Rendering for Boundary Optimization

**Concept:** Treat boundary detection as an inverse rendering problem.

**Pipeline:**
1. **Predict:** 3D mesh of the object.
2. **Render:** Project mesh to 2D using differentiable renderer (PyTorch3D).
3. **Loss:** Compare rendered silhouette with target boundary.
4. **Backprop:** Gradients flow through the renderer to update the mesh.

**Code Sketch:**
```python
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftSilhouetteShader,
    RasterizationSettings, PerspectiveCameras
)

# Define mesh
verts, faces = load_mesh()

# Differentiable renderer
cameras = PerspectiveCameras()
raster_settings = RasterizationSettings(image_size=512, blur_radius=1e-5)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader()
)

# Render
silhouette = renderer(meshes)

# Loss
loss = F.mse_loss(silhouette, target_boundary)
loss.backward()

# Update mesh vertices
optimizer.step()
```

**Use Case:** 3D reconstruction from 2D images (e.g., NeRF, 3D Gaussian Splatting).

## 21. Ethical Considerations

**1. Bias in Medical Imaging:**
- If training data is mostly from one demographic (e.g., Caucasian patients), boundary detection might fail on others.
- **Fix:** Diverse, representative datasets.

**2. Surveillance:**
- Boundary detection enables person tracking and re-identification.
- **Mitigation:** Privacy-preserving techniques (on-device processing, federated learning).

**3. Deepfakes:**
- Precise boundary detection enables realistic face swaps.
- **Safeguard:** Watermarking, detection models.

- **Safeguard:** Watermarking, detection models.

## 22. Benchmark Datasets for Boundary Detection

**1. BSDS500 (Berkeley Segmentation Dataset):**
- 500 natural images with human-annotated boundaries.
- **Metric:** F-measure (ODS/OIS).
- **SOTA:** F-ODS = 0.82 (HED).

**2. Cityscapes:**
- 5,000 street scene images with fine annotations.
- **Task:** Instance-level boundary detection for cars, pedestrians, etc.
- **Metric:** Boundary IoU.

**3. NYU Depth V2:**
- 1,449 indoor RGB-D images.
- **Task:** Depth discontinuities (boundaries in 3D).
- **Use Case:** Robotics, AR/VR.

**4. Medical Datasets:**
- **ISIC (Skin Lesions):** Melanoma boundary detection.
- **BraTS (Brain Tumors):** 3D tumor boundaries in MRI.
- **DRIVE (Retinal Vessels):** Blood vessel segmentation.

## 23. Production Monitoring and Debugging

**Challenge:** Model works in lab, fails in production.

**Monitoring Metrics:**
1. **Boundary Precision/Recall:** Track over time.
2. **Inference Latency:** P50, P95, P99.
3. **GPU Utilization:** Should be > 80% for efficiency.
4. **Error Cases:** Log images where Dice < 0.5.

**Debugging Tools:**
```python
import wandb

# Log predictions
wandb.log({
    "prediction": wandb.Image(pred_mask),
    "ground_truth": wandb.Image(gt_mask),
    "dice_score": dice,
    "inference_time_ms": latency
})

# Alert if performance degrades
if dice < 0.7:
    wandb.alert(
        title="Low Dice Score",
        text=f"Dice = {dice} on image {image_id}"
    )
```

**A/B Testing:**
- Deploy new model to 5% of traffic.
- Compare boundary quality (human eval or automated metrics).
- Gradual rollout if metrics improve.

## 24. Common Pitfalls and How to Avoid Them

**Pitfall 1: Ignoring Class Imbalance**
- Boundary pixels are < 5% of the image.
- **Fix:** Use weighted loss or focal loss.

**Pitfall 2: Over-smoothing**
- Too much pooling/upsampling blurs boundaries.
- **Fix:** Use skip connections (U-Net) or dilated convolutions.

**Pitfall 3: Inconsistent Annotations**
- Different annotators draw boundaries differently.
- **Fix:** Multi-annotator consensus, use soft labels (average of multiple annotations).

**Pitfall 4: Domain Shift**
- Train on sunny day images, deploy on rainy nights.
- **Fix:** Domain adaptation (CycleGAN), diverse training data.

**Pitfall 5: Not Testing on Edge Cases**
- Occlusion, motion blur, low light.
- **Fix:** Curate a "hard examples" test set.

## 25. Advanced: Boundary-Aware Data Augmentation

Standard augmentation (rotation, flip) isn't enough for thin boundaries.

**Elastic Deformation:**
```python
import elasticdeform

# Deform image and mask together
[image_deformed, mask_deformed] = elasticdeform.deform_random_grid(
    [image, mask],
    sigma=25,  # Deformation strength
    points=3,  # Grid resolution
    order=[3, 0],  # Interpolation order (cubic for image, nearest for mask)
    axis=(0, 1)
)
```

**Boundary-Specific Augmentation:**
```python
def augment_boundary(mask, dilation_range=(1, 3)):
    # Randomly dilate or erode boundary
    kernel_size = np.random.randint(*dilation_range)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if np.random.rand() > 0.5:
        mask = cv2.dilate(mask, kernel)
    else:
        mask = cv2.erode(mask, kernel)
    
    return mask
```

## 26. Advanced: Multi-Scale Boundary Detection

Objects have boundaries at different scales (fine hair vs. body outline).

**Laplacian Pyramid:**
```python
def build_laplacian_pyramid(image, levels=4):
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    
    laplacian_pyramid = []
    for i in range(levels):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    
    return laplacian_pyramid

# Process each scale
for level in laplacian_pyramid:
    boundary_map = model(level)
    # Fuse multi-scale outputs
```

    # Fuse multi-scale outputs
```

## 27. Hardware Considerations for Real-Time Boundary Detection

**Challenge:** Autonomous vehicles need 60 FPS at 1080p.

**Hardware Options:**
1. **NVIDIA Jetson AGX Xavier:**
   - 32 TOPS (INT8).
   - Power: 30W.
   - **Use Case:** Embedded systems, drones.

2. **Tesla FSD Chip:**
   - Custom ASIC for neural networks.
   - 144 TOPS.
   - **Use Case:** Tesla Autopilot.

3. **Google Edge TPU:**
   - 4 TOPS.
   - Power: 2W.
   - **Use Case:** Mobile devices, IoT.

**Optimization for Edge:**
```python
# Model pruning
import torch.nn.utils.prune as prune

# Prune 30% of weights
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Knowledge distillation
teacher = UNet(channels=64)  # Large model
student = UNet(channels=16)  # Small model

# Train student to mimic teacher
loss = F.mse_loss(student(x), teacher(x).detach())
```

loss = F.mse_loss(student(x), teacher(x).detach())
```

**Performance Benchmarks (1080p Image):**

| Hardware | Model | FPS | Latency (ms) | Power (W) |
| :--- | :--- | :--- | :--- | :--- |
| **RTX 3090** | U-Net (FP32) | 120 | 8.3 | 350 |
| **RTX 3090** | U-Net (INT8) | 350 | 2.9 | 350 |
| **Jetson Xavier** | U-Net (INT8) | 45 | 22 | 30 |
| **Edge TPU** | MobileNet-UNet | 15 | 67 | 2 |
| **CPU (i9)** | U-Net (FP32) | 3 | 333 | 125 |

**Takeaway:** For real-time edge deployment, use INT8 quantization + lightweight architecture.

## 28. Interview Tips for Boundary Detection Problems

**Q1: How would you handle class imbalance in boundary detection?**
*Answer:* Use weighted loss (weight boundary pixels 10x higher), focal loss, or Dice loss which is robust to imbalance.

**Q2: Why use skip connections in U-Net?**
*Answer:* Pooling loses spatial information. Skip connections concatenate high-res features from the encoder to the decoder, recovering fine details needed for precise boundaries.

**Q3: How to deploy a boundary detection model at 60 FPS?**
*Answer:* Model quantization (FP32 → INT8), TensorRT optimization, use lightweight architectures (MobileNet backbone), process at lower resolution and upsample.

**Q4: How to evaluate boundary quality?**
*Answer:* Boundary IoU (intersection over union along the contour band), F-measure (precision/recall on boundary pixels), Hausdorff distance (maximum error).

**Q5: What's the difference between edge detection and boundary detection?**
*Answer:* Edge detection finds all intensity changes (low-level, includes texture). Boundary detection finds semantically meaningful object contours (high-level, requires understanding).

## 29. Further Reading

1. **"U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015):** The U-Net paper.
2. **"Holistically-Nested Edge Detection" (Xie & Tu, 2015):** HED architecture.
3. **"Mask R-CNN" (He et al., 2017):** Instance segmentation standard.
4. **"DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs" (Chen et al., 2018):** CRF refinement.
5. **"Attention U-Net: Learning Where to Look for the Pancreas" (Oktay et al., 2018):** Attention for medical imaging.

## 30. Conclusion

Boundary detection has evolved from simple gradient operators (Canny) to sophisticated deep learning models (U-Net, Mask R-CNN) that understand semantic context. The key challenges—thin structures, class imbalance, real-time performance—are being addressed through specialized loss functions (Dice, Tversky), attention mechanisms, and deployment optimizations (TensorRT, quantization). As we move toward 3D understanding and multi-modal fusion (LiDAR + Camera), boundary detection will remain a critical building block for autonomous systems, medical AI, and creative tools.

## 31. Summary

| Component | Technology |
| :--- | :--- |
| **Low-Level** | Canny, Sobel |
| **Deep Learning** | HED, CASENet, U-Net |
| **Refinement** | Active Contours, CRF |
| **Metrics** | Boundary IoU, F-Score |
| **Deployment** | TensorRT, Quantization |
| **Advanced** | Attention, Differentiable Rendering |

---

**Originally published at:** [arunbaby.com/ml-system-design/0035-boundary-detection-in-ml](https://www.arunbaby.com/ml-system-design/0035-boundary-detection-in-ml/)
