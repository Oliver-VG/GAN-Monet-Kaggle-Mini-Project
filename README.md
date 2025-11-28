# GAN Kaggle Mini-Project

## GAN – Generative Adversarial Network

This repository contains the Kaggle notebook **“GAN Kaggle Mini-Project”**, which builds and trains a Generative Adversarial Network (GAN) to generate Monet-style images.

The project is based on the original GAN paper (Goodfellow et al.) and follows the Kaggle competition **“I’m Something of a Painter Myself”** from the `gan-getting-started` competition.

---

## Kaggle Competition: “I’m Something of a Painter Myself”

Notebook section: **Problem / Kaggle Competition: "I’m Something of a Painter Myself"**  
Competition: https://www.kaggle.com/competitions/gan-getting-started/overview

From the notebook’s competition description (Kaggle citation):

- A GAN consists of at least two neural networks: a **generator model** and a **discriminator model**.
- A generative AI model modifies and enriches input data.
- In this competition, the goal is to transform input photographs into Monet-style images.
- The generator is trained using the discriminator; the two models work against each other, with the generator trying to fool the discriminator and the discriminator trying to distinguish real from generated images.
- **Task:** Build a GAN that generates **7,000 to 10,000 Monet-style images**.

### Evaluation (FID / Leaderboard)

Notebook section: **Kaggle score**  
Citation: https://www.kaggle.com/competitions/gan-getting-started/leaderboard

From the notebook:

- **FID** (Fréchet Inception Distance) measures how close generated images are to real images.
- Lower FID = better quality.
- **0** would mean a perfect match (but does not occur in practice).
- On the Kaggle leaderboard: **lower score = better**.

**Submission result recorded in the notebook**  
Section: *Submission Results*

- Kaggle score for `notebook03b8491926 - Version3`: **84.30637**

---

## Data

Notebook section: **Data** and **Kaggle Data Citation**  
Citation: Kaggle – https://www.kaggle.com/competitions/gan-getting-started/data

From the notebook:

- The competition dataset contains four main directories:
  - `monet_tfrec`
  - `photo_tfrec`
  - `monet_jpg`
  - `photo_jpg`
- The **Monet directories** contain Monet paintings. These images are used to train the model.
- The **photo directories** contain real-world photos. The goal is to add Monet-style characteristics to these images.
- The notebook notes that the `photo_tfrec` and `photo_jpg` directories contain the same photos.
- Submissions are limited to **10,000 transformed images**.

### Paths and loading (from the code)

- Competition name: `COMPETITION_NAME = "gan-getting-started"`
- Data is accessed via:
  ```python
  GCS_PATH = KaggleDatasets().get_gcs_path(COMPETITION_NAME)
  MONET_TFRECS = tf.io.gfile.glob(os.path.join(GCS_PATH, "monet_tfrec", "*.tfrec"))
  PHOTO_TFRECS = tf.io.gfile.glob(os.path.join(GCS_PATH, "photo_tfrec", "*.tfrec"))
  ```
- JPEGs used for visual inspection and EDA:
  ```python
  monet_files = sorted(tf.io.gfile.glob("/kaggle/input/gan-getting-started/monet_jpg/*.jpg"))
  photo_files = sorted(tf.io.gfile.glob("/kaggle/input/gan-getting-started/photo_jpg/*.jpg"))
  ```
- The notebook prints basic dataset statistics, including the number of Monet and Photo TFRecord files and approximate image counts:
  ```python
  print("Monet TFRecord files:", len(MONET_TFRECS))
  print("Photo TFRecord files:", len(PHOTO_TFRECS))
  n_monet = count_data_items(MONET_TFRECS)
  n_photo = count_data_items(PHOTO_TFRECS)
  print("Approx Monet images:", n_monet)
  print("Approx Photo images:", n_photo)
  ```

---

## Environment & Notebook Setup

Notebook section: **Notebook**

- The notebook is designed to **run in the Kaggle environment**.
- In Kaggle, set the **Accelerator** to **“GPU T4x2”**.
- A similar notebook was used in Kaggle and submitted as:  
  `notebook03b8491926 - Version3`

The notebook uses the following key libraries (from the imports):

- `numpy`
- `tensorflow`, `keras`, `tensorflow.keras.layers`
- `matplotlib`
- `PIL` (Pillow)
- `tqdm`
- `kaggle_datasets`
- Standard Python libraries: `os`, `re`, `zipfile`, `pathlib.Path`, `random`

The training strategy uses TensorFlow’s default distribution strategy:

```python
strategy = tf.distribute.get_strategy()
print("Using default strategy")
print("Replicas in sync:", strategy.num_replicas_in_sync)
```

---

## Notebook Plan / Structure

Notebook section: **Plan**

The notebook is organized into the following steps:

1. Import Libraries  
2. Setup configuration  
3. Check the data  
4. Define Functions  
5. Load the data  
6. EDA (Exploratory Data Analysis)  
7. Model Setup  
8. Model Training  
9. Generate file for Kaggle submission  
10. Submission  
11. Conclusions  
12. Discussion  
13. Link GitHub Repository  
14. Citation  
15. AI Acknowledgements  

---

## Configuration & Hyperparameters

From the configuration code in the notebook:

```python
IMAGE_SIZE = (256, 256)
CHANNELS = 3

BATCH_SIZE = 1 * strategy.num_replicas_in_sync     # Up to 4
EPOCHS = 10
BUFFER_SIZE = 1024

LR = 2e-4
LAMBDA_CYCLE = 10.0
LAMBDA_ID = 0.5 * LAMBDA_CYCLE

N_GENERATED_IMAGES = 7000
```

Other key settings:

- Data loading uses `tf.data` pipelines with options for **augmentation**, **shuffle**, **repeat**, and **batching**.
- `STEPS_PER_EPOCH` is computed as:
  ```python
  STEPS_PER_EPOCH = min(n_monet, n_photo) // BATCH_SIZE
  ```
- The notebook raises an error if `STEPS_PER_EPOCH == 0` to guard against missing data or incompatible batch size.

---

## Data Loading & Preprocessing

The notebook uses TFRecords for training data and includes:

- Parsing TFRecords into images.
- Normalizing images to the `[-1, 1]` range.
- Optional **augmentation** steps:
  - Random horizontal flips.
  - Random resizing and cropping between 256 and 286 pixels back down to `(IMG_HEIGHT, IMG_WIDTH)`.

Example (from the `load_dataset` function):

```python
def load_dataset(filenames, augment=False, shuffle=False, repeat=False, batch_size=1):
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.map(read_tfrecord, num_parallel_calls=AUTO)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=AUTO)
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds
```

Datasets used:

- `monet_ds_train`: Monet images with augmentation, shuffling, repeat, and batching.
- `photo_ds_train`: Photo images with augmentation, shuffling, repeat, and batching.
- `photo_ds_inference`: Photo dataset without augmentation, shuffling, or repeat, used for generating Monet-style outputs.

Training pairs the datasets as:

```python
train_ds = tf.data.Dataset.zip((photo_ds_train, monet_ds_train))
```

---

## Exploratory Data Analysis (EDA)

Notebook sections: **EDA**, **Check the data**, **Insights**

The notebook performs:

- **Visual inspection of samples**:  
  - Random Monet images.  
  - Random Photo images.
- **Metadata statistics**:
  - Computes image widths, heights, pixel mean, and pixel standard deviation for a sample of images.
- **Distribution plots**:
  - Image width distribution (Monet vs Photo).
  - Pixel-intensity mean histograms (Monet vs Photo).

From the notebook’s **Insights** section:

- The width distribution indicates that all images are uniformly sized, so no resizing issues are present.
- Pixel-intensity histograms reveal that Monet images tend to have slightly higher and more widely spread brightness values.  
  The photos cluster more tightly at lower intensities, reflecting their more balanced, natural lighting.

---

## Model Setup

Notebook section: **Model Setup** and corresponding code

High-level summary from the notebook:

- **InstanceNorm**: Normalizes each sample independently across spatial dimensions.
- **Residual Block**: A pair of convolutional layers with a skip connection that lets the input flow directly to the output.
- **ResNet Generator**: A generator built from several residual blocks, allowing it to learn complex transformations.
- **Discriminator**: A CNN that judges whether an image is real or generated and guides the generator by providing feedback on how realistic its outputs appear.

### Instance Normalization

Implemented as a custom Keras layer:

```python
class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight(
            shape=(channels,), initializer="ones", trainable=True, name="gamma"
        )
        self.beta = self.add_weight(
            shape=(channels,), initializer="zeros", trainable=True, name="beta"
        )
        super().build(input_shape)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(var + self.epsilon)
        x_norm = (x - mean) * inv
        return self.gamma * x_norm + self.beta
```

### Residual Block

```python
def residual_block(x, filters, use_norm=True, name=None):
    init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    skip = x

    y = layers.Conv2D(
        filters, 3, strides=1, padding="same",
        kernel_initializer=init, use_bias=not use_norm,
        name=None if name is None else name + "_conv1",
    )(x)
    if use_norm:
        y = InstanceNormalization(name=None if name is None else name + "_in1")(y)
    y = layers.ReLU()(y)

    y = layers.Conv2D(
        filters, 3, strides=1, padding="same",
        kernel_initializer=init, use_bias=not use_norm,
        name=None if name is None else name + "_conv2",
    )(y)
    if use_norm:
        y = InstanceNormalization(name=None if name is None else name + "_in2")(y)

    out = layers.Add(name=None if name is None else name + "_add")([skip, y])
    return out
```

### ResNet Generators

Two generators are built using the same architecture:

- `G_photo_to_monet`
- `G_monet_to_photo`

Generator definition:

```python
def build_resnet_generator(name="generator", n_res_blocks=9):
    init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    x = layers.Conv2D(
        64, 7, strides=1, padding="same",
        kernel_initializer=init, use_bias=False,
    )(inputs)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    for filters in [128, 256]:
        x = layers.Conv2D(
            filters, 3, strides=2, padding="same",
            kernel_initializer=init, use_bias=False,
        )(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)

    for i in range(n_res_blocks):
        x = residual_block(x, 256, name=f"res{i+1}")

    for filters in [128, 64]:
        x = layers.Conv2DTranspose(
            filters, 3, strides=2, padding="same",
            kernel_initializer=init, use_bias=False,
        )(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.Conv2D(
        CHANNELS, 7, strides=1, padding="same",
        kernel_initializer=init,
    )(x)
    outputs = layers.Activation("tanh")(x)
    return keras.Model(inputs, outputs, name=name)
```

### Discriminators

Two discriminators are used:

- `D_monet`
- `D_photo`

Discriminator definition (Patch-based CNN):

```python
def build_discriminator(name="discriminator"):
    init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    def disc_block(x, filters, stride, use_norm=True):
        x = layers.Conv2D(
            filters, 4, strides=stride, padding="same",
            kernel_initializer=init, use_bias=not use_norm,
        )(x)
        if use_norm:
            x = InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    x = disc_block(inputs, 64, stride=2, use_norm=False)
    x = disc_block(x, 128, stride=2)
    x = disc_block(x, 256, stride=2)
    x = disc_block(x, 512, stride=1)
    x = layers.Conv2D(1, 4, strides=1, padding="same", kernel_initializer=init)(x)
    return keras.Model(inputs, x, name=name)
```

---

## Losses & Optimization

Within the `strategy.scope()` block, the notebook defines:

```python
mse_loss = keras.losses.MeanSquaredError()
mae_loss = keras.losses.MeanAbsoluteError()

def generator_adversarial_loss(fake_logits):
    return mse_loss(tf.ones_like(fake_logits), fake_logits)

def discriminator_loss(real_logits, fake_logits):
    real_loss = mse_loss(tf.ones_like(real_logits), real_logits)
    fake_loss = mse_loss(tf.zeros_like(fake_logits), fake_logits)
    return 0.5 * (real_loss + fake_loss)

def cycle_consistency_loss(real, cycled):
    return mae_loss(real, cycled)

def identity_loss(real, same):
    return mae_loss(real, same)
```

Optimizers:

```python
gen_G_optimizer = keras.optimizers.Adam(LR, beta_1=0.5, beta_2=0.999)
gen_F_optimizer = keras.optimizers.Adam(LR, beta_1=0.5, beta_2=0.999)
disc_monet_optimizer = keras.optimizers.Adam(LR, beta_1=0.5, beta_2=0.999)
disc_photo_optimizer = keras.optimizers.Adam(LR, beta_1=0.5, beta_2=0.999)
```

The training step computes:

- Adversarial loss for both generators.
- Cycle-consistency loss for both domains.
- Identity loss for both domains.

Total generator losses:

```python
total_gen_G_loss = gen_G_adv + LAMBDA_CYCLE * total_cycle_loss + LAMBDA_ID * id_loss_monet
total_gen_F_loss = gen_F_adv + LAMBDA_CYCLE * total_cycle_loss + LAMBDA_ID * id_loss_photo
```

Discriminator losses:

```python
disc_monet_loss = discriminator_loss(disc_real_monet, disc_fake_monet)
disc_photo_loss = discriminator_loss(disc_real_photo, disc_fake_photo)
```

Gradients are computed with `tf.GradientTape` and applied to each model’s trainable variables.

---

## Training Loop

Notebook section: **Model Training**

Key elements of the training loop:

```python
history = {
    "gen_G_loss": [],
    "gen_F_loss": [],
    "disc_monet_loss": [],
    "disc_photo_loss": [],
}

for epoch in range(1, EPOCHS + 1):
    start_time = time()
    epoch_metrics = {
        "gen_G_loss": 0.0,
        "gen_F_loss": 0.0,
        "disc_monet_loss": 0.0,
        "disc_photo_loss": 0.0,
    }
    for step, batch in enumerate(train_ds.take(STEPS_PER_EPOCH)):
        metrics = train_step(batch)
        for k in epoch_metrics:
            epoch_metrics[k] += metrics[k]

        if (step + 1) % 10 == 0 or (step + 1) == STEPS_PER_EPOCH:
            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Step [{step+1}/{STEPS_PER_EPOCH}] "
                f"G_G: {metrics['gen_G_loss']:.2f} "
                f"G_F: {metrics['gen_F_loss']:.2f} "
                f"D_M: {metrics['disc_monet_loss']:.2f} "
                f"D_P: {metrics['disc_photo_loss']:.2f}",
                end="\r",
            )

    epoch_time = time() - start_time
    for k in epoch_metrics:
        epoch_metrics[k] /= STEPS_PER_EPOCH
        history[k].append(epoch_metrics[k])

    print(
        f"\nEpoch {epoch}/{EPOCHS} time: {epoch_time:.1f}s | "
        f"G_G: {epoch_metrics['gen_G_loss']:.3f} "
        f"G_F: {epoch_metrics['gen_F_loss']:.3f} "
        f"D_M: {epoch_metrics['disc_monet_loss']:.3f} "
        f"D_P: {epoch_metrics['disc_photo_loss']:.3f}"
    )

print("Training finished.")
```

---

## Generating Monet-Style Images & Submission File

Notebook section: **Generate file for Kaggle competition**

After training, the notebook generates Monet-style images from the photo dataset and writes them into a zip archive for submission:

- Output directory and zip path:
  ```python
  OUTPUT_DIR = Path("/kaggle/working")
  ZIP_PATH = OUTPUT_DIR / "images.zip"
  ```

- Number of images to generate:
  ```python
  N_GENERATED_IMAGES = 7000
  ```

- Denormalization helper:

  ```python
  def denormalize_to_uint8(image):
      image = (image + 1.0) * 0.5
      image = tf.clip_by_value(image, 0.0, 1.0)
      image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
      return image
  ```

- Generation loop (excerpt):

  ```python
  print("Generating Monet-style images into images.zip ...")

  with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
      idx = 0
      for batch in photo_ds_inference:
          if idx >= N_GENERATED_IMAGES:
              break
          photo = batch
          fake_monet = G_photo_to_monet(photo, training=False)[0]
          img_uint8 = denormalize_to_uint8(fake_monet)
          jpg_bytes = tf.io.encode_jpeg(img_uint8).numpy()

          filename = f"image_{idx:05d}.jpg"
          zf.writestr(filename, jpg_bytes)
          idx += 1

          if idx % 100 == 0:
              print(f"Generated {idx}/{N_GENERATED_IMAGES} images...", end="\r")

  print(f"\nDone. Wrote {idx} images to {ZIP_PATH}.")
  ```

This `images.zip` is used as the Kaggle competition submission file.

---

## Conclusions & Discussion

Notebook sections: **Conclusions**, **Discussion**

From the notebook **Conclusions** section:

- The GAN pipeline successfully generated Monet-style images and achieved a competitive score on the Kaggle leaderboard.
- The training process demonstrated stable adversarial learning.
- The results suggest that further gains are possible through hypertuning with alternative GAN losses, or increasing model capacity.

From the **Discussion** section:

- The notebook is running in the Kaggle environment. It is not necessary to download any files.  
  You can download the files if you want to create a notebook which is running locally on your PC.
- Hyperparameter tuning that could be done:
  - More epochs
  - Different learning rates for generator and/or discriminator
  - Larger batches
  - Other loss functions
  - Different number of filters
  - Image augmentation

---

## Citation / References

Notebook section: **Citation / References**

- Kaggle competition: https://www.kaggle.com/competitions/gan-getting-started/overview
- GAN paper: https://arxiv.org/abs/1406.2661

---

## AI Acknowledgement

Notebook section: **AI Acknowledgement**

- ChatGPT-5.1 (OpenAI, 2025) was used to assist in proofreading an...hanges to the original ideas or analysis were made by the model.
