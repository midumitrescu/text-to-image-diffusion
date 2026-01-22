# Generation of Images starting from Text using diffusion latent models

Toy machine learning project made for capstone 2 of the [DataTalksClub Machine Learning Zoomcamp](https://courses.datatalks.club/ml-zoomcamp-2025)
organized by [Datatalks.club](https://datatalks.club/blog/machine-learning-zoomcamp.html)

## Table of Contents

---

### Introduction
- [Project Overview](#project-overview)
- [About This Project](#about-this-project)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)

---

## Project Overview
This project is meant to showcase modern GenAI techniques that have been recently developed. 
Most of the inspiration I have found on this [Youtube video](youtube.com/watch?v=FslFZx08beM&feature=youtu.be), which I recommend watching.

## Architecture diagram 

                                           +----------------------+
                 Text Prompt/Context       |   CLIP / Text Encoder |
                                     +---->+ (Language + Vision    |
                                     |     |  Embeddings)         |
                                     |     +----------------------+
                                     |                |
                                     |                v
                                     |     +----------------------+
                                     |     |   Conditional Latent |
                                     |     |    Conditioning      |
                                     |     +----------------------+
                                     |                |
                                     |                v
                                     |     +----------------------+
                                     |     |   Latent Diffusion   |
                                     |     |   (Denoising Model)  |
                                     |     | + Attention Layers*  |
                                     |     | + Cross-Attention    |
                                     |     +----------------------+
                                     |                |
          +--------------------------+                |
          |                                           v
    +-----------------+                  +--------------------------+
    |  Input Text     |                  |   Conditional VAE (CVAE)  |
    |  or Prompt      |                  |   Encoder/Decoder         |
    +-----------------+                  | + Latent Representations  |
              |                          | + CLIP-conditioned output |
              v                           +-----------+--------------+
      Tokenize/Embed                                  |
              |                                       v
              +--------------------------> Reconstruction / Sampled Output
                                                      |
                                                      v
                                             +------------------+
                                             |  Generated Image |
                                             | (or sequence)    |
                                             +------------------+


## Main ideas

1. Take a text, encode it, pass it though an attention and a conditional variational autoencoder module to produce a conditioned latent probability distribution.
2. This probability distribution is then used as initialization of our denoising model.
3. The denoising model learns to sequentially denoise an image to produce sharp, realistic images that resemble qualitative picture, available on the internet.
   (sequentially denoising is a fancy name to say that we take a starting image and we denoise it N steps of time. They have to be in a sequence because
the noise has to be removed from increasingly less noisy intermediary steps)

## Dataset
**Source**: [Kaggle CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/code)
The project is based on a large number of publicly available face pictures of celebrities.

### Dataset Characteristics
- **Size**: 202,599 number of face images of various celebrities. 10,177 unique identities, but names of identities are not given
- **Features**: 40 binary attribute annotations per image 
- **Quality**: Benchmark for estimating the quality of reconstruction is not provided yet. This will be an interesting addition

### Data Files
1. *img_align_celeba.zip*: All the face images, cropped and aligned
2. *list_eval_partition.csv*: Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
3. *list_bbox_celeba.csv*: Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
4. *list_landmarks_align_celeba.csv*: Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth
5. *list_attr_celeba.csv*: Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative

A more detailed analysis on means, distribution of values, first 3 statististical moments, examples, etc, can be found in the
[Exploratory Data Analysis Notebook](Exploratory_Data_Analysis.ipynb).

## Technology Stack

### Data Science & ML
- **Python**: 3.10
- **Data Processing**: pandas, numpym pytorch
- **Visualization**: matplotlib
- **ML Models**: transformers, diffusers, xgboost
- **Testing**: pytest (unit and integration tests)
- **Notebook Environment**: Jupyter

### Deployment
- **API Framework**: FastAPI
- **Containerization**: Docker (still coming :)
- **Package Management**: poetry
- **Environment Management**: miniconda

## Getting Started

### Prerequisites

1. **Python 3.10+**
   ```bash
   python --version 
   ```

2. **Conda (via Miniconda)**
   ### Linux / macOS
   ```bash
      curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname)-x86_64.sh
      bash Miniconda3-latest-$(uname)-x86_64.sh
   ```
   ### Apple Silicon
   ```bash
       curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
       bash Miniconda3-latest-MacOSX-arm64.sh
   ```



3**Poetry Package Manager**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

4**Kaggle API Credentials (Optional)**
  - Create an account at [kaggle.com](https://www.kaggle.com)
  - Go to Account settings → API → Create New Token
  - Place the downloaded `kaggle.json` in `~/.kaggle/`
  - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## How to install?

1. **Clone the repository**
   ```bash
   git clone git@github.com:midumitrescu/text-to-image-diffusion.git
   cd text-to-image-diffusion
   ```

2. **Create conda environment**
   
   ```bash
       conda create -n diffusion python=3.10 -y
       conda activate diffusion
   ```
   
3. **Install dependencies**
   
   ```bash
        poetry install
   ```
   
4. **Run tests**
   
   ```bash
        poetry run pytest
   ```
   

5. Download the dataset either from Kaggle directly or from the [EDA Notebook](Exploratory_Data_Analysis.ipynb), uncomment, and run the first cell.
   
   ```bash
        poetry run pytest
   ```

6. **Train your VAE starting from [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)**
   
   ```bash
        poetry run python -m src.vae.train  --batch_size 8   --validation_ratio 0.2   --learning-rate 0.01   --epochs 20   --save-dir ./models/
   ```


7. **Build docker image**
```bash
docker build --no-cache -t vae-image:1 .
```

8. **Start docker image locally**

```bash
docker run -p 8000:8000 vae-image:1
```


```bash
  poetry config virtualenvs.create false
  git clone git@github.com:midumitrescu/text-to-image-diffusion.git
  cd text-to-image-diffusion
  conda create -n diffusion python=3.10 -y
  conda activate diffusion
  poetry install --with dev
  poetry run pytest
```

## Docker
Is still in TODO and will be my next task.


```bash
cd ..
conda deactivate
conda remove --name diffusion --all -y
rm -rf text-to-image-diffusion/
```