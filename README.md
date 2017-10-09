# MNIST Digits Classification with MLP
## Get started
1. Put `data` folder in the root path of the project
2. Run `run_mlp.py` using `python2.7`
3. Get results from log file and png plots when the training is done

## `run_mlp.py` usage
- **--model \[model\] (optional)** '2-layer' or '3-layer' for network with 1 hidden layer or 2 hidden layer. Default is '2-layer'.
- **--name \[name\] (optional)** Your name for the model, used to name log file and png plots. Default is 'model'.
- **--lr \[lr\] (optional)** Initial learning rate, default is 0.1.
- **--momentum \[momentum\] (optional)** Momentum for SGD, default is 0.
- **--wd \[weight decay\] (optional)** Weight decay for SGD, default is 0.
- **--scheduler \[scheduler\] (optional)** Learning rate scheduler, 'empty' or 'step' for EmptyScheduler or StepScheduler. Default is 'empty'.
- **--batch \[batch\] (optional)** Training batch size, default is 128.
- **--epoch \[epoch\] (optional)** Training epoch, default is 200.
- **--activation \[activation function\] (optional)** Activation function after a linear layer. 'relu' or 'sigmoid' for Relu layer or Sigmoid Layer. Default is 'relu'.

## Examples
`python run_mlp.py --model 2-layer --name mymodel --lr 0.1 --scheduler step --epoch 200`

`python run_mlp.py --model 3-layer --name mymodel --lr 0.01 --scheduler step --activation sigmoid --epoch 200`