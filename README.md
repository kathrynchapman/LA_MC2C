# LA_MC2C
This is my repo for my MSc thesis, titled,
"Exploiting Label Attention and Clustering for Multi-label Codes Classification"
  

To run the classifiers, one must first clone the official metric repo: https://github.com/TeMU-BSC/cantemist-evaluation-library
```
git clone https://github.com/TeMU-BSC/cantemist-evaluation-library.git
```
Then, from within the CANTEMIST2020 directory: <br>
-run the classifier with:
```
python run_dev_classifier.py 
--model_name_or_path bert-base-multilingual-cased 
--model_type bert 
--data_dir processed_data/cantemist/ 
--doc_max_seq_length 512 
--num_train_epochs 20 
--output_dir DEBUGGIN 
--loss_fct bbce 
--do_train 
--logging_steps 5 
--overwrite_output_dir

```
This is still a work in progress (obv)