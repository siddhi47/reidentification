# Reidentification

## Training
For training script, Download the VeRi.zip file and export it. Make sure that you have all the requirements fulfilled for your python.

You can run the training script using the train.py script as follows:

```
python train.py
```

You can pass arguments to the train.py script. For detailed help type in 

```
python train.py --help
```


The arguments that you can pass to this scripts are :

- model: name of the model. Available models are "res18", "mobile", "google", "shuffle", "reid"
- epoch: Number of epoch to train.
- batch_size: The batch size to be used. Use lower batch size if memory issues are encountered.
- path: path to the VeRi Folder.

```
python train.py -model google --epoch 50 --batch_size 32 --path VeRi
```

## Testing
There are two scripts that are used for testing. One for accuracy and framerate and another for mAP.
Make sure that `checkpoints` folderr is inside the VeRi folder.
The command to run these are quite similar.
### Accuracy

`test_accuracy.py` is used for accuracy and framerate. The command to run the accuracy test is:

```
python test_accuracy.py
```
There are other arguments that can be used for this script. 

- model
- epoch 
- batch_size
- path

The  are stored in the file acc_fps.txt file in the current directory.
### mAP
Before running this script. Make sure that you have downloaded and extracted list.zip file to VeRi folder.
You can run `test_map.py` script to generate the mAP score for the model. The arguments used are:

- model
- epoch
- path


The results for both testing scripts are stored in the file acc_fps.txt and map.txt file in the current directory.
