# README

Thanks for your attention. The following instructions can help you reproduce the experiments.

## Platform

Our experiments are conducted on a platform with Intel(R) Xeon(R) Gold 6248R CPU @3.00GHz and single GPU NVIDIA TITAN RTX 24GB.

## Environment

```
conda env create -f environment.yaml
```

## Running

```
cd code
bash train.sh
```

The detailed configurations can be found in the ```train.sh```. As the Bert model is too large (cannot
upload to the ACL system), you can download the Bert model from [Hugging Face(```bert-base-uncased```)](https://huggingface.co/bert-base-uncased).

## Files Definition

- ```data``` : contains three public datasets: FewAsp, FewAsp(multi), FewAsp(single)

- ```code``` : contains python files of our framework

    - ```data_loader_multi.py``` : used to sample each episode's data
    - ```bert_atten.py``` : model file
    - ```mfb.py``` : part of label-enhanced prototypical network 
    - ```losses.py``` : contains loss function
    - ```parser_util.py``` : parse parameters
    - ```train.py``` : train the model
    - ```train.sh``` : parameters used to train models
   
