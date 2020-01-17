# ESIM-PyTorch
ESIM implementation by PyTorch, get **89.29%** accuracy on Quora Questions Pairs dataset.

# Requirements

+ Python 3.5+
+ PyTorch 1.0+

# Usage

+ train

```
python run.py --gpu gpudevice(e.g. 0) 
```
+ test

```
python test.py ./saved_models/best.pth.tar --gpu gpudevice(e.g. 0)
```

The dataset, glove.npy, best.pth.tar can be downloaded from the given url in the above files.

The accuracy on the test set is **89.29%**, I use the Quora Queations Pairs dataset from [https://github.com/zhiguowang/BiMPM](https://github.com/zhiguowang/BiMPM), which can be downloaded from [https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing)

# keras ESIM

A keras ESIM implementation can be found here: [https://github.com/Deep1994/ESIM-keras](https://github.com/Deep1994/ESIM-keras), which can get **88.12%** accuracy on the test set.

# Reference

+ [https://github.com/coetaur0/ESIM](https://github.com/coetaur0/ESIM)
