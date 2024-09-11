# 数据集  DATASET

本项目已经涵盖了由我们提供的 转化为mnist格式的数据集，存储位置为

./data/MNIST/raw

如果你想要使用自己的数据集，请参照https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format

项目内已有我使用过的格式转换代码，只需将28*28的单通道图片以 ./Convert png to mnist./test-imgaes内的格式存放好，即可运行

./Convert png to mnist./convert-images-to-mnist-format.py 

来转化您自己的数据集（注意，windows系统调用gzip可能会出错，需要手动压缩一下



This project has covered the data set that we provided which is converted to mnist format, stored in the location

./data/MNIST/raw

If you want to use your own data set, please see → https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format

I already use the format conversion code in the project, just put 28*28 single-channel image in the format of./Convert png to mnist./test-imgaes, you can run

./Convert png to mnist./convert-images-to-mnist-format.py

To convert your own data set (note that windows system calls to gzip may fail and need to be manually compressed

# 使用 Usage

如果您要使用vgg16模型，请修改models.vgg16对应的vgg.py

if you wanna use VGG16, plz set the in_channels as 1

路径通常为  python\Python311\site-packages\torchvision\models\vgg.py

normally path is：python\Python311\site-packages\torchvision\models\vgg.py

文件中的：

from

```python
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
```



改为：

to

```python
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1
```



# 配置说明 Config

./utils/config.json 控制参数

```json
{
  "model_name" : "resnet50",  //model name, plz choose in {resnet50, resnet18,vgg16,alexnet}
  "algorithm" : "fedavg",   //al name, plz choose in {fedavg,fedsgd,fedprox}
  "no_models" : 10,  //number of clients
  "type" : "mnist",
  "global_epochs" : 40, 
  "local_epochs" : 1,
  "batch_size" : 32, //dont be greater than 32 if ur PC is not that strong!
  "lr" : 0.001,
  "momentum" : 0.0001,
  "lambda" : 0.1,
  "poison_client": 3, //client num that will poison
  "poison_times": 10,//how many time that each poison client will do the harm!
  "krum": 0,  //weather krum or not
  "krum_client": 3,  //how many rats will the krum eliminate
  "DEVICE": "cuda",  //dont have a gpu? change it to cpu
  "download_dataset": 0, //download mnist/cifar or not?
  "num_classes": 11,  //the out_features
  "mu_for_fedprox" : 0.01 
}



```

# 运行 run

 run main and there will be xlsx file in ./result



