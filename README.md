# Deploy deep learning models on your iPhone/iPad

_using the mxnet c predict api_
  <br />
  <br />
![](https://github.com/ZwX1616/placeholder.jpg)


Initially I tried to run a 50MB ssd object detection network on the iPhone and failed due to insufficient RAM. It currently runs a 5MB mobilenet (from the model zoo) for classification among 1000 classes. In order to detect objects (or even do segmentation), small networks like tinyyolo or tinyssd are essential for mobile devices.


Mxnet is used here because it is the main deep learning framework I learn and use, and it has a rich model zoo (gluoncv.model_zoo) to offer. It is also a lightweight framework to be used anywhere and, for example, I can train a model with it on some workstation/AWS and easily deploy this model onto the end-use device (e.g. a phone) without the need/pain to convert it.
  <br />
  <br />
  <br />
(tested on iPhoneX)<br />
how to run on your iPhone: (files in step 2,3 are not included for being too large)<br />
1. clone and open the Xcode project and <br />
2. put opencv2.framework (downloadable from official website) in folder 'ios_mx_det/opencv' <br />
3. compile the mxnet prediction api (static library libmxnet_ios_lib.a file) for iOS and put it in 'ios_mx_det/mxnet' <br />
  (I used mxnet 1.2.0 since 1.3 doesn't work for some reason. Refer to https://github.com/apache/incubator-mxnet/pull/11184 or https://www.jianshu.com/p/8f703c10540e)
4. build and run on device
<br />
<br />
