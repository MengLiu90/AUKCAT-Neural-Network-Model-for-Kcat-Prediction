# AUKCAT-Neural-Network-Model-for-Kcat-Prediction
AUKCAT NN model is a neural network–based framework designed to predict enzyme turnover numbers (kcat) from structured biochemical features. The model takes as input a triplet of features representing either the substrate (or product), enzyme EC number, and species, and outputs predicted kcat values.

This repository contains a unified architecture trained under two input feature modes and two training strategies, resulting in four variants of the model.

## Dependencies
1. pytorch 1.10.0
2. pandas 1.4.2
3. numpy 1.24.3
4. sklearn 1.0.2
5. scipy 1.5.3
6. CUDA 11.1

## Multi-Species Kcat Prediction (Substrate-EC-Species As Input)
### 5-Fold Cross-Validation Evaluation
This step is used to estimate the generalization performance of the model.
#### Model Training
The original data ```./Datasets/Substrate_ec_species data/Original_data.csv``` was partitioned into 5 subsets. For each fold, four subsets together with their corresponding synthetic instances were used for training, and the remaining subset of the original data was used for evaluation.
#### Trained Models
The 5-fold cross-validation checkpoints for the multi-species substrate model are available in ```./Trained models/General models/Substrate_model_5_fold_cv/```.

### Final Deployed Model
This model is trained on the full set of original data and all synthetic data, and is used as the final deployed model for multi-species kcat prediction.
#### Model Training

#### Trained Model

#### Kcat Prediction Using the Trained model


## Human-Specialist kcat Prediction (Substrate-EC-Species As Input)

## Multi-Species Kcat Prediction (Product-EC-Species As Input)

## Human-Specialist kcat Prediction (Product-EC-Species As Input)
