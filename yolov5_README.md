To run the COTS detection using YoloV5 on PyTorch:
1. Clone the https://github.com/ultralytics/yolov5 into the yolov5 folder
2. cd into the folder 
3. Download the weights file yolov5n6.pt(or any other mdoel) from the same repo.
4. Run the notebook kaggle-yolov5.ipynb
5. Inside the notebook tweak the hyperparatmeters of the yolo_v5 train process , for example:
`!python train.py --single-cls --img 1280 --rect --batch 20 --epochs 64 --data cots.yml --weights weights/yolov5n6.pt --workers 0` We are hear choosing --signle-cls because we are dealing with a single class, we chose the image size to be 1280 because our images has that size. We chose for 20 batch size and 64 epochs. Don't forget the cots.yml which includes the needed information for yolo_V5 to run. With those parameters I could start the training on my personal PC that has 32GB RAM, and nVidia GeForce GTX 1060 6GB wih CUDA installed on Ubuntu 20.04
6. Within the same notebook you can see how to run the detection
`!python detect.py --img 1280 --source /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/train_images/video_1/9115.jpg --weights ./weights/best.pt --conf-thres 0.4` . Here, you can specify which image or images to choose for detection

7. You run the tensorbord to see the training metrics by executing:
 `tensorboard --logdir yolov5/runs/train/exp` Make sure you have already installed tensorbord `pip install tensorbord`