# LA_MC2C
This is my repo for my MSc thesis, titled,
"Exploiting Label Attention and Clustering for Multi-label Codes Classification"
  

Step 1: Clone the official metric repos for CANTEMIST and CLEF eHealth 2020: https://github.com/TeMU-BSC/cantemist-evaluation-library & https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script
```
git clone https://github.com/TeMU-BSC/cantemist-evaluation-library.git
git clone https://github.com/TeMU-BSC/codiesp-evaluation-script.git
```
Step 2: Unzip data and HEMKit:
```
unzip data.zip
unzip HEMKit.zip
```

Step 3: Compile the HEMKit (hierarchical metric library):
```
cd HEMKit/software/
make
cd ../../
```

Step 4: Create a virtual environment from the .yml file. IMPORTANT: The transformer library must be ~2.4.1
```
conda env create --file ICDEnv.yml
```

To train the MC2C model:
```
python run_classifier.py 
        --data_dir processed_data/german/ 
        --encoder_name_or_path bert-base-multilingual-cased 
        --encoder_type bert 
        --model mc2c 
        --loss bbce 
        --num_train_epochs 12 
        --doc_max_seq_length 512 
        --lmbda 0.31103291023330726 
        --min_cluster_size 4 
        --max_cluster_size 23 
        --max_cluster_threshold 0.29389531264559815 
        --hierarchical_clustering
        --pass_mlcc_preds_to_mccs 
        --learning_rate 6.980178471054263e-05 
        --weight_decay 1.0437691001673701 
        --adam_epsilon 3.749239874139236e-06 
        --max_m 0.32289308625543495 
        --warmup_proportion 0.2951384375813258 
        --do_train 
        --do_test

```

To train the label description attention model:
```
python run_classifier.py 
        --data_dir processed_data/cantemist/ 
        --encoder_name_or_path bert-base-multilingual-cased 
        --encoder_type bert 
        --model label_attn 
        --loss bbce 
        --num_train_epochs 288 
        --doc_max_seq_length 256 
        --learning_rate 3.708441754237857e-05 
        --weight_decay 2.4197691267693067 
        --adam_epsilon 0.0001255758423684405 
        --warmup_proportion 0.2808958299478227  
        --do_train
        --do_test
```

To train the baseline model:
```
python run_classifier.py 
        --data_dir processed_data/german/ 
        --encoder_name_or_path bert-base-multilingual-cased 
        --encoder_type bert 
        --model baseline 
        --loss bbce 
        --num_train_epochs 25 
        --doc_max_seq_length 256 
        --do_train 
        --do_test
```

For the SKML models:
```
python run_SKMLearnExps.py 
        --data processed_data/german/ 
        --parameter_search
        --do_train 
        --load_best_parameters
        --eval_data test
```

For the hyperparameter search using Optuna:
```
 python hyperparam_search.py 
         --data_dir processed_data/german/ 
         --encoder_name_or_path bert-base-multilingual-cased 
         --encoder_type bert 
         --model mc2c 
         --output_dir parameter_search/german/
```
