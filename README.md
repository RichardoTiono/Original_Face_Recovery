# Original_Face_Recovery
Deep learning implementation for recovering the original skin color of face images that under color lighting


## Sample Result
Input Sample <br>
<img  width="100" height="100" src="https://github.com/RichardoTiono/Original_Face_Recovery/blob/main/sample_result/input/21_0_b_mask.png"/>
<img  width="100" height="100" src="https://github.com/RichardoTiono/Original_Face_Recovery/blob/main/sample_result/input/22_1_b_mask.png"/>
<img  width="100" height="100" src="https://github.com/RichardoTiono/Original_Face_Recovery/blob/main/sample_result/input/25_4_r_mask.png"/>
<img width="100" height="100" src="https://github.com/RichardoTiono/Original_Face_Recovery/blob/main/sample_result/input/26_2_g_mask.png"/>

Generated Sample <br />
<img  width="100" height="100" src="https://github.com/RichardoTiono/Original_Face_Recovery/blob/main/sample_result/result/21_0_b_mask.png">
<img width="100" height="100" src="https://github.com/RichardoTiono/Original_Face_Recovery/blob/main/sample_result/result/22_0_b_mask.png">
<img width="100" height="100" src="https://github.com/RichardoTiono/Original_Face_Recovery/blob/main/sample_result/result/25_4_r_mask.png">
<img width="100" height="100" src="https://github.com/RichardoTiono/Original_Face_Recovery/blob/main/sample_result/result/26_2_g_mask.png">

## Source Description
```
├─── data_loading.py: data loading and pre-processing methods <br>
├─── generate_dataset_csv.py: generate csv given dataset directory <br>
├─── interpolate.py: Create interpolation result from provided image directory <br>
├─── main_gen_pseudo-data.py: Train Skip-Connection based network on Synthetic dataset and generate Pseudo-Supervision data for CelebA dataset<br>
├─── main_gen_synthetic_and_full.py: Train on synthetic data, generate pseudo-supervision data, train on mix data ss<br>
├─── main_mix_training.py: Train SfSNet on mix data. Need to provide both CelebA and Synthetic dataset directory <br>
├─── main_light_training.py: Train or test Light Removal Network. Need to provide both Real and Synthetic dataset directory<br>
├─── models.py: Definition of all the models used. Skip-Net and SfSNet <br>
├─── train.py: Train and test rountines <br>
├─── utils.py: Help rountines <br>
├─── black2white.py: Changes black background of images into white background <br>
├─── create_mask.py: Create mask from original images <br>
├─── masking.py: Apply mask genearted from create_mask.py to original image <br>
├─── ContrastEnhancer.ipynb: Apply post processing to the generated image <br>
```

## Usage of main_light_training to train and test the model
```
usage: main_mix_training.py [-h] [--batch_size N] [--epochs N] [--lr LR]
                            [--wt_decay W] [--no_cuda] [--seed S]
                            [--read_first READ_FIRST] [--details DETAILS]
                            [--load_pretrained_model LOAD_PRETRAINED_MODEL]
                            [--syn_data SYN_DATA] [--celeba_data CELEBA_DATA]
                            [--log_dir LOG_DIR] [--load_model LOAD_MODEL]

SfSNet - Residual

optional arguments:
  -h, --help            show this help message and exit
  --batch_size N        input batch size for training (default: 8)
  --epochs N            number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.001)
  --wt_decay W          SGD momentum (default: 0.0005)
  --no_cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --read_first READ_FIRST
                        read first n rows (default: -1) from the dataset
                        This is helpful to load part of the data. Note that, internally
                        we change this to sample randomly with seed value 100
  --details DETAILS     Explaination of the run
                        String provided will be written into root log directory
                        We perform many experiments and then get lost on the results and what was this experiment for.
                        This txt file will help us understand what was the purpose of this experiment
  --sample_data         Sample Dataset path - Data for testing/inferring with your pretrained network
  --rgb_syn_data        Path to RGB synthetic Dataset csv file
  --rgb_real_data		Path to RGB real Dataset csv file
  --log_dir             Log Path
  --load_model          Path for Existing Pretrained Network
  --mode                Choose mode for doing Train-rgb (Training Network) or metrics(test using metrics) or Infer (Infer pretrained network) '
                        'or Visualization (visuale each convolution output in grayscale)
```

## Guidelines

Infer using pretrained network
1. Download the pretrained model using [link](https://drive.google.com/drive/folders/1FcthqkhLtdfMm7hYgNISGYntez1YNXKr?usp=sharing)
2. run the main_light_training code
```
python main_light_training --mode Infer --sample_data /path/to/your/data --load_model /path/to/pretrained/network/
```
3. run black2white.py if you want white background
4. run ContrastEnhancer to do post-processing

Train with new dataset
if the new dataset still has background, do pre-processing first. This is the steps for pre-processing :
1. run create_mask.py to create mask for the new dataset
2. run masking.py to apply those mask to the new dataset

if the new dataset is cleaned :
1. run generate_dataset_csv on the new dataset for both synthetic and real dataset seperately 
2. run the main_light_training code
```
python main_light_training --mode Train-rgb --rgb_syn_data /path/to/your/new/synthetic_data_csv --rgb_real_data /path/to/your/new/real_data_csv
```
