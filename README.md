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
This step is used to assess the generalization performance of the model.
#### Model Training
The original data ```./Datasets/Substrate_ec_species data/Original_data.csv``` was partitioned into 5 subsets. For each fold, four subsets together with their corresponding synthetic instances were used for training, and the remaining subset of the original data was used for evaluation.
#### Trained Models
The 5-fold cross-validation checkpoints for the multi-species substrate model are available in ```./Trained models/General models/Substrate_model_5_fold_cv/```.

### Final Deployed Model
This model is trained on the full set of original data and all synthetic data, and is used as the final deployed model for multi-species kcat prediction.

#### Trained Model
The trained multi-species kcat prediction model with substrate–EC–species inputs is available at ```./Trained_models/General_models/Substrate_model.pth```.
#### Kcat Prediction Using the Trained model
To predict the kcat from your data, simply run 

```python predict_kcat.py --ckpt model.pth --input data.csv --out pred.csv```


where ```--ckpt``` specifies the path to the trained model checkpoint, ```--input``` specifies the path to the CSV file for prediction (which must contain the required feature columns), and ```--out``` specifies the name of the output prediction file. The output results will be saved under ```./Results``` directory.

#### Simple Example
Here, we provide a simple example demonstrating how to use the model for kcat prediction.

Run ```python predict_kcat.py --ckpt ./Trained_models/General_models/Substrate_model.pth --input ./Datasets/Substrate_ec_species_data/Embedded_original_data_example.csv --out pred_substrate_example.csv```

After running this command, the program outputs the following evaluation metrics (if the input data contains kcat labels; otherwise, only the predicted kcat values are generated):

```AvgLoss=0.358112  MSE=0.358112  R2=0.7413  Pearson=0.8869```

```Saved predictions → Results/pred_substrate_example.csv```

The predicted kcat values are saved in the ```Results``` directory as ```pred_substrate_example.csv```.

## Human-Specialist kcat Prediction (Substrate-EC-Species As Input)
This model is specifically trained for human kcat prediction. The trained model is available at ```./Trained_models/Human-specialist_models/Substrate_model_for_human_kcat_prediction.pth```

To run this model for human kcat prediction, simply run 

```python predict_kcat.py --ckpt ./Trained_models/Human-specialist_models/Substrate_model_for_human_kcat_prediction.pth --input your_data_path/your_data.csv --out pred_human_kcat.csv```

This model takes substrate embedding, EC number embedding, and human species embedding as input features.
## Multi-Species Kcat Prediction (Product-EC-Species As Input)

## Human-Specialist kcat Prediction (Product-EC-Species As Input)


### Remark
The required feature columns for a successful prediction includes
1. Molecular feature embedding for the metabolite (either substrate or product, depending on the model you want to use)

   This feature embedding can be obtained through [Mol2Vec](https://github.com/samoturk/mol2vec)
3. EC number embedding

   This feature embedding can be obtained via [EC2Vec](https://github.com/MengLiu90/EC2Vec)
5. Species embedding

   This feature embedding can be obtained using Node2Vec.

The feature embeddings should be concatenated in the order of metabolite-EC number-species to construct the input to the model.
