#3D Shape Classifier 

##Getting started
To get the project running locally first install the requirements using
```
pip install -r requiremnets.txt
```

##Data formation
Once all the requirements are installed, run the `data_formation.py` to generate the data. If data is already available it will overwrite it with new random shapes.

```
.
├── data
│   ├── annulus
│   │   ├── test
│   │   ├── train
│   │   └── validation
│   ├── capsule
│   ├── cone
│   ├── cube
│   ├── cylinder
└── └── sphere

```
##Training
After data formation and run the `classifier.py` to train the on the labeled data. You can change the parameters in the `__main__` module of the file.

The model will be saved after the training is completed. The default name for saved model is `saved_model.pt`.

##Inference
To get inference on the a `.obj` file, run the `infer.py` by assigning the object file path and saved model path to the following variables `obj_file_path` and `model_path` respectively.

The `infer.py` will print out the predicted class label.