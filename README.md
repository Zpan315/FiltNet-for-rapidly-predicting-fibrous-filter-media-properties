# FiltNet for rapidly predicting fibrous filter media properties

## 1.Description

**Deep Learning-driven SEM Image Analysis for Rapidly Predicting Structural Characteristics and Filtration Performance of Fibrous Filter Media**

## 2.File Structure
```
FiltNet-for-rapidly-predicting-fibrous-filter-media-properties
├─ FiltNet_prediction
│  ├─ ImgDisplay.ipynb
│  ├─ Prediction.ipynb
│  ├─ RandomShowCase.ipynb
│  ├─ Imgs
│  │  ├─ CurvedFiber
│  │  │  └─ Artificial_SEM
│  │  │     ├─ Matched_x_y.png
│  │  │     ├─ Matched_x_z.png
│  │  │     └─ Matched_y_z.png
│  │  ├─ EllipticalFiber
│  │  │  └─ Artificial_SEM
│  │  │     ├─ Matched_x_y.png
│  │  │     ├─ Matched_x_z.png
│  │  │     └─ Matched_y_z.png
│  │  ├─ MF-1
│  │  │  ├─ CT
│  │  │  │  ├─ Matched_x_y.png
│  │  │  │  ├─ Matched_x_z.png
│  │  │  │  └─ Matched_y_z.png
│  │  │  └─ SEM
│  │  │     ├─ cross_section_cut.png
│  │  │     └─ surface_cut.png
│  │  ├─ NF-1
│  │  │  ├─ Artifical_SEM
│  │  │  │  ├─ Matched_x_y.png
│  │  │  │  ├─ Matched_x_z.png
│  │  │  │  └─ Matched_y_z.png
│  │  │  └─ SEM
│  │  │     ├─ cross_section.png
│  │  │     └─ surface.png
│  │  └─ OrientedFiber
│  │     └─ Artificial_SEM
│  │        ├─ Matched_x_y.png
│  │        ├─ Matched_x_z.png
│  │        └─ Matched_y_z.png
│  └─ Predictions
│     └─ showcase
│        └─ results_40_10066d.txt
├─ FiltNet_training
│  ├─ FiltNet.py
│  ├─ FiltNet_CNNs.py
│  ├─ Model_training.ipynb
│  ├─ GeneralEvaluation
│  ├─ Logs
│  └─ MinMax
│     ├─ minmax_model40_10066d.npy
│     └─ minmax_res_model40_10066d.npy
├─ GTD_generation
│  ├─ BubblePoint.py
│  ├─ Fiber_Orientations.py
│  ├─ FilterEfficiency.py
│  ├─ Generate_Inputs.py
│  ├─ Intergrate_values.py
│  ├─ Matdict_1DStatistics.py
│  ├─ Papergeo_Circle.py
│  ├─ PoreSizeDistribution.py
│  ├─ Porosimetry.py
│  ├─ ProcessGeo_EmbedFibers.py
│  └─ Tortuosity.py
├─ HDF
│  ├─ FiltNet_1000d.h5 (download use the link below)
│  ├─ List
│  │  ├─ list_40_1000d.h5
│  │  └─ list_40_10066d.h5
│  └─ Trained
│     └─ Model40FineTune_10066d.h5
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ Particle_size.txt
├─ VarNames.txt
└─ timesbd.ttf

```

## 3.File Function

### 3.1 FiltNet_prediction
1. **Prediction.ipynb**: Main notebook for predicting fibrous filter media properties using trained FiltNet models.
2. **ImgDisplay.ipynb**: Notebook for displaying and visualizing images from the training dataset (FiltNet_1000d.h5).
3. **RandomShowCase.ipynb**: Notebook for showcasing random prediction examples with detailed visualizations.
4. **Model40FineTune_10066d.h5**: Pre-trained FiltNet model for prediction. Use this link to download: [model](https://1drv.ms/u/c/4790a4376137575d/EQPth4OhBPJIrP8gm0E0w5IBBEhI2oJ0e7koLiEodgBa0A?e=ntUQbR)
5. **Imgs/**: Contains example images of different fiber types and filter media:
   - **CurvedFiber/Artificial_SEM/**: Artificial SEM images with matched projections in x-y, x-z, and y-z planes
   - **EllipticalFiber/Artificial_SEM/**: Artificial SEM images with matched projections in x-y, x-z, and y-z planes
   - **OrientedFiber/Artificial_SEM/**: Artificial SEM images with matched projections in x-y, x-z, and y-z planes
   - **MF-1/**: Real filter media with both CT and SEM images
   - **NF-1/**: Filter media with both Artificial SEM and real SEM images
6. **Predictions/showcase/**: Directory containing prediction results.

### 3.2 FiltNet_training
1. **Model_training.ipynb**: Notebook for training the FiltNet deep learning model.
2. **FiltNet.py**: Main FiltNet training script with data augmentation and preprocessing functions.
3. **FiltNet_CNNs.py**: CNN architecture definitions for FiltNet.
4. **MinMax/**: Contains min-max normalization parameters for model inputs and outputs.
5. **GeneralEvaluation/**: Directory for model evaluation results.
6. **Logs/**: Training logs directory.

### 3.3 GTD_generation
Configurations for generating the Ground Truth Data (GTD) using GeoDict macros. Each file, except "Generate_Inputs.py" and "Intergrate_values.py", refers to corresponding module of GeoDict. 
1. **Generate_Inputs.py**: Script for generating input data for the model.
2. **Papergeo_Circle.py**: Code for generating circular fiber geometry.
3. **ProcessGeo_EmbedFibers.py**: Code for embedding fibers in the filter media geometry.
4. **FilterEfficiency.py**: Code for calculating filter efficiency.
5. **PoreSizeDistribution.py**: Code for computing pore size distribution.
6. **Porosimetry.py**: Code for porosimetry analysis.
7. **BubblePoint.py**: Code for bubble point calculations.
8. **Tortuosity.py**: Code for tortuosity analysis.
9. **Fiber_Orientations.py**: Code for analyzing fiber orientations.
10. **Matdict_1DStatistics.py**: Code for extracting 1D statistics from material dictionaries.
11. **Intergrate_values.py**: Code for integrating various calculated values.

### 3.4 HDF
1. **FiltNet_1000d.h5**: HDF5 file containing training dataset (1000 data points) that can be downloaded from [dataset](https://1drv.ms/u/c/4790a4376137575d/ERUmG3tfaLtFl-Le5AfJIvoBIOWf4YKoQDJsb98WA1zVTg?e=6Xw7ta)
2. **List/**: Contains data list files for different dataset sizes (1000d and 10066d).
3. **Trained/**: Directory for storing trained models.


## 4.Software Requirements

### 4.1 Required packages

- tensorflow-gpu==2.6.2
- pandas
- Pillow
- matplotlib
- opencv-python
- numpy>=1.19.4,<2.0
- h5py==3.1.0
- scikit-learn

## 5.Usage

### 5.1 Environment configuration
For convenience, execute the following command to install all required packages:

```bash
pip install -r requirements.txt
```

### 5.2 Make predictions using pre-trained model
- Use the `./FiltNet_prediction/Prediction.ipynb` notebook to predict filter media properties from SEM images or Artificial SEM iamges from 3D geoemtries  .
- The pre-trained model `Model40FineTune_10066d.h5` is provided for immediate use.
- Example images are available in `./FiltNet_prediction/Imgs/` directory.

### 5.3 Train your own model
- Prepare your training dataset in HDF5 format (see `./HDF/` directory for examples).
- Use the `./FiltNet_training/Model_training.ipynb` notebook to train your own FiltNet model.
- Adjust hyperparameters and architecture in `FiltNet_CNNs.py` as needed.

### 5.4 Generate training data
- Use the macro scripts in `./GTD_generation/` with GeoDict software to generate synthetic training data from filter geometry parameters.


## 6.Citation

If you find this repo useful, please cite our paper.

```
[Citation to be added upon publication]
```


## 7.Contact
If you have any questions or suggestions, feel free to contact:
- zpan@umn.edu
