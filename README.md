# Open-Street-View Classification with Convolutional Neural Networks

<table>
<thead>Table of Contents</thead>
    <tbody>
        <tr>
            <th>Section 1</th>
            <th><a href="#disclaimer">Disclaimer</a></th>
        </tr>
        <tr>
            <th>Section 2</th>
            <th><a href="#overview">Overview</a></th>
        </tr>
        <tr>
            <th>Section 3</th>
            <th><a href="#technical-details">Technical Details</a></th>
        </tr>
        <tr>
            <th>Section 4</th>
            <th><a href="#challenges--solutions">Challenges & Solutions</a></th>
        </tr>
        <tr>
            <th>Section 5</th>
            <th><a href="#replicability">Replicability</a></th>
        </tr>
    </tbody>
</table>

---

<h2 id="disclaimer">Disclaimer</h2>

This project researches the capabilities and limits of Convolutional Neural Networks. It is not intended to be used in ranked GeoGuessr matches, or any other way that goes against the terms & conditions listed on GeoGuessr. I am not responsible for any complications that may arrive due to the misuse of this project.

---

<h2 id="overview">Overview</h2>

This project focuses on the effects of computer vision methods—specifically a custom deep Convolutional Neural Network (CNN) built with TensorFlow—on geographic classification from street-level imagery. 

By training a sequential model on the `sylshaw/streetview-by-country` dataset sourced from Kaggle, this project investigates how accurately a neural network can identify a country or region based purely on visual cues such as flora, infrastructure, and landscape features. 

The primary goal is to benchmark a custom, regularized CNN against raw, localized environmental data, tracking optimization metrics over time.

---

<h2 id="technical-details">Technical Details</h2>

### Data Collection & Preprocessing
* **Dataset:** Sourced dynamically via `kagglehub` using the `sylshaw/streetview-by-country` repository.
* **Data Split:** A $70/30$ train/validation split using a deterministic seed (`161`) to ensure consistency across runs.
* **Resolution:** Images are loaded at $250 \times 250$ pixels using nearest-neighbor interpolation and batched in groups of $32$. 
* **Data Pipeline:** Scaled to `float32` tensors and optimized using TensorFlow's `AUTOTUNE` prefetching to maximize GPU throughput.

### Model Architecture
The network is built sequentially using Keras layers, incorporating explicit data augmentation and regularization techniques to combat overfitting:

1. **In-Model Augmentation:** Integrates a `RandomZoom` ($30\%$ factor) and `RandomContrast` ($20\%$ factor) layer directly into the execution graph.
2. **Feature Extraction:** * Conv2D ($32$ filters, $3 \times 3$ kernel, $L_1$ bias regularization) $\rightarrow$ Batch Normalization $\rightarrow$ MaxPool
   * Conv2D (64 filters, $3 \times 3$ kernel, $L_1$ bias regularization) $\rightarrow$ `MaxPool2D()`
   * Conv2D (128 filters, $3 \times 3$ kernel) $\rightarrow$ Conv2D ($128$ filters) $\rightarrow$ `MaxPool2D()`
3. **Classification Head:** * `GlobalAveragePooling2D` to condense spatial feature maps.
   * Dense layer ($512$ units, ReLU activation).
   * Dropout layer ($50\%$ rate) to enforce structural redundancy.
   * Final Dense layer utilizing a Softmax activation to predict categorical probabilities for each country class.

### Training Configuration
* **Loss Function:** Categorical Cross-Entropy.
* **Optimizer:** Adam with a highly tuned learning rate ($LR = 10^{-5}$) and stability epsilon ($\varepsilon$ = $10^{-5}$).
* **Callbacks:** Automated model checkpointing saving the absolute best iteration based on validation accuracy (`val_accuracy`), alongside periodic fallback checkpoints every $25$ batches.

---

<h2 id="challenges--solutions">Challenges & Solutions</h2>

* **Challenge 1: High Training Variance.** Raw street-view images contain drastic shifts in lighting and framing that easily confuse shallow networks.
  * *Solution:* Injected real-time augmentations (`RandomZoom`, `RandomContrast`) at the input layer to artificially increase dataset variety and teach the network translation-invariant features.
* **Challenge 2: Severe Overfitting Risk.** Given the complexity of geographic features, a deep convolutional model can easily memorize background textures.
  * *Solution:* Applied $L_1$ bias regularization to the structural convolutional weights and placed a heavy 50% Dropout barrier right before the final classification head.
* **Challenge 3: Training Tracking.** Monitoring training convergence across epochs without manual intervention.
  * *Solution:* Programmed automated generation of metric logs using `pandas` and `matplotlib`, outputting `loss_graph.png` and `accuracy_graph.png` instantly upon script completion.

---

<h2 id="replicability">Replicability</h2>

Follow these steps to set up and run the script locally.

### Prerequisites
* Python 3.10+
* Active Kaggle API credentials (for `kagglehub` integration)
* TensorFlow 2.x

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/open-street-view-cnn.git
   cd open-street-view-cnn
   ```
2. Install the necessary machine learning libraries:
   ```bash
   pip install numpy pandas matplotlib tensorflow sklearn kagglehub
   ```
*Note: Ensure long-file paths are enabled and that the python interpreter is compatible with the version of tensorflow.*

### Running the Pipeline
Simply execute the main script. The pipeline will automatically download the dataset from Kaggle, handle preprocessing, build the model, and run training for 15 epochs:
```bash
python train.py
```
*Note: Ensure the local output file directory pathways inside the `ModelCheckpoint` callbacks exist, or update them to match your local operating system structure before running.*

---
<h2 id="future-roadmap">Future Roadmap</h2>

To improve the accuracy and robustness of the geolocation model, the following milestones are planned for future iterations:

- **Hyperparameter Optimization:** Implement `keras-tuner` to systematically test optimal combinations of learning rates, dropout ratios, and convolutional filter depths.
- **Advanced Spatial Resolution:** Shift the categorical classification model to a coordinate-regression setup, configuring the network to directly predict Latitude and Longitude variables instead of discrete country tags. Plan to use the Haversine formula.
- **Smart Cropping Modules:** Introduce an automated preprocessing step utilizing a light object-detection network to segment and isolate country-specific details.