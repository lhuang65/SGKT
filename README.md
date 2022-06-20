# SGKT

The code is related to the paper SGKT: Session Graph-Based Knowledge Tracing for Student Performance Prediction


## Datasets
We have placed the processed ASSIST2012 dataset in the 'data' folder.

If you want to process the datasets by yourself, you can download them by the following links:

ASSIST09:[download](https://drive.google.com/file/d/1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE/view)

ASSIST2012: [download](https://drive.google.com/file/d/0BxCxNjHXlkkHczVDT2kyaTQyZUk/edit?usp=sharing)

EdNet: [download](https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view)

You should create a folder named 'checkpoint' and 'logs' for saving model and log in our experiments.

## Environment Requirement

python == 3.6.5

tensorflow == 1.15.0

numpy == 1.15.2

## Examples to run the model

### ASSIST12 dataset
* Command
```
python main.py --dataset assist12_3
```

If you have more questions about our experiments, you can contact us. 
email: lhuang@m.scnu.edu.cn
