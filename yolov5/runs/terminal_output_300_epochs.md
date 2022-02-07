(py3711) peshmerge@akersloot:~/ml_2_project/yolov5$ python train.py --single-cls --img 1280 --rect --batch 20 --epochs 300 --data cots.yml --weights weights/yolov5n6.pt --workers 0
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: (30 second timeout) 3
wandb: You chose 'Don't visualize my results'
train: weights=weights/yolov5n6.pt, cfg=, data=cots.yml, hyp=data/hyps/hyp.scratch.yaml, epochs=300, batch_size=20, imgsz=1280, rect=True, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=True, optimizer=SGD, sync_bn=False, workers=0, project=runs/train, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: skipping check (not a git repository), for updates see https://github.com/ultralytics/yolov5
YOLOv5 🚀 4ce6737 torch 1.10.1 CUDA:0 (GeForce GTX 1060 6GB, 6075MiB)

hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=1

                 from  n    params  module                                  arguments                     
  0                -1  1      1760  models.common.Conv                      [3, 16, 6, 2, 2]              
  1                -1  1      4672  models.common.Conv                      [16, 32, 3, 2]                
  2                -1  1      4800  models.common.C3                        [32, 32, 1]                   
  3                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  4                -1  2     29184  models.common.C3                        [64, 64, 2]                   
  5                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  6                -1  3    156928  models.common.C3                        [128, 128, 3]                 
  7                -1  1    221568  models.common.Conv                      [128, 192, 3, 2]              
  8                -1  1    167040  models.common.C3                        [192, 192, 1]                 
  9                -1  1    442880  models.common.Conv                      [192, 256, 3, 2]              
 10                -1  1    296448  models.common.C3                        [256, 256, 1]                 
 11                -1  1    164608  models.common.SPPF                      [256, 256, 5]                 
 12                -1  1     49536  models.common.Conv                      [256, 192, 1, 1]              
 13                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 14           [-1, 8]  1         0  models.common.Concat                    [1]                           
 15                -1  1    203904  models.common.C3                        [384, 192, 1, False]          
 16                -1  1     24832  models.common.Conv                      [192, 128, 1, 1]              
 17                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 18           [-1, 6]  1         0  models.common.Concat                    [1]                           
 19                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 20                -1  1      8320  models.common.Conv                      [128, 64, 1, 1]               
 21                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 22           [-1, 4]  1         0  models.common.Concat                    [1]                           
 23                -1  1     22912  models.common.C3                        [128, 64, 1, False]           
 24                -1  1     36992  models.common.Conv                      [64, 64, 3, 2]                
 25          [-1, 20]  1         0  models.common.Concat                    [1]                           
 26                -1  1     74496  models.common.C3                        [128, 128, 1, False]          
 27                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 28          [-1, 16]  1         0  models.common.Concat                    [1]                           
 29                -1  1    179328  models.common.C3                        [256, 192, 1, False]          
 30                -1  1    332160  models.common.Conv                      [192, 192, 3, 2]              
 31          [-1, 12]  1         0  models.common.Concat                    [1]                           
 32                -1  1    329216  models.common.C3                        [384, 256, 1, False]          
 33  [23, 26, 29, 32]  1     11592  models.yolo.Detect                      [1, [[19, 27, 44, 40, 38, 94], [96, 68, 86, 152, 180, 137], [140, 301, 303, 264, 238, 542], [436, 615, 739, 380, 925, 792]], [64, 128, 192, 256]]
[W NNPACK.cpp:79] Could not initialize NNPACK! Reason: Unsupported hardware.
Model Summary: 355 layers, 3094312 parameters, 3094312 gradients, 4.3 GFLOPs

Transferred 451/459 items from weights/yolov5n6.pt
Scaled weight_decay = 0.00046875
optimizer: SGD with parameter groups 75 weight (no decay), 79 weight, 79 bias
WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False
train: Scanning '/home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/training' images and labels...4242 found, 0 missing, 0 empty, 1 corrupt: 100%|█| 4242/4242 [00:00<00:00, 73
train: WARNING: /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/images/training/video_0_9470.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [     1.0021]
train: New cache created: /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/training.cache
val: Scanning '/home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/validation' images and labels...677 found, 0 missing, 0 empty, 0 corrupt: 100%|█| 677/677 [00:00<00:00, 6686.
val: New cache created: /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/validation.cache
Plotting labels to runs/train/exp/labels.jpg... 

AutoAnchor: 4.95 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to runs/train/exp
Starting training for 300 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     0/299     5.05G   0.06359   0.03304         0         0      1280: 100%|██████████| 213/213 [06:21<00:00,  1.79s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.39s/it]                                                                       
                 all        677          0          0          0          0          0

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     1/299     4.89G    0.0479   0.03484         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.39s/it]                                                                       
                 all        677       2449     0.0682      0.126     0.0251    0.00763

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     2/299     4.89G   0.04229   0.03177         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.40s/it]                                                                       
                 all        677       2449      0.263      0.182      0.139     0.0384

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     3/299     4.89G   0.03914   0.02895         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:24<00:00,  1.43s/it]                                                                       
                 all        677       2449      0.299      0.192      0.138     0.0434

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     4/299     4.89G   0.03861    0.0288         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:24<00:00,  1.46s/it]                                                                       
                 all        677       2449      0.362      0.258      0.222     0.0791

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     5/299     4.89G   0.03721   0.02712         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.40s/it]                                                                       
                 all        677       2449      0.532       0.24      0.253     0.0833

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     6/299     4.89G   0.03563   0.02601         0         0      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.40s/it]                                                                       
                 all        677       2449      0.732      0.365      0.443      0.176

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     7/299     4.89G   0.03382   0.02487         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.38s/it]                                                                       
                 all        677       2449      0.684      0.396      0.456      0.178

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     8/299     4.89G   0.03296   0.02372         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:24<00:00,  1.42s/it]                                                                       
                 all        677       2449      0.719      0.334      0.384      0.167

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     9/299     4.89G   0.03123   0.02301         0         0      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.38s/it]                                                                       
                 all        677       2449      0.774      0.323      0.402      0.169

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    10/299     4.89G   0.03046   0.02205         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449        0.7       0.43      0.514      0.233

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    11/299     4.89G   0.03059   0.02117         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.787      0.518       0.61      0.266

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    12/299     4.89G   0.02962   0.02125         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.859      0.478      0.581      0.268

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    13/299     4.89G   0.02877   0.02033         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.868      0.466      0.579      0.252

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    14/299     4.89G   0.02879   0.02015         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.819      0.486      0.591      0.268

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    15/299     4.89G   0.02744   0.01933         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.749      0.508      0.577      0.256

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    16/299     4.89G    0.0262   0.01874         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.38s/it]                                                                       
                 all        677       2449      0.836      0.512      0.613      0.282

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    17/299     4.89G   0.02672   0.01819         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.843      0.558      0.658      0.311

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    18/299     4.89G   0.02597   0.01771         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.843      0.534      0.631      0.298

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    19/299     4.89G   0.02654   0.01758         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.849      0.529      0.617      0.295

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    20/299     4.89G   0.02679   0.01771         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.821      0.512      0.607      0.288

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    21/299     4.89G   0.02559   0.01704         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.778      0.513      0.591      0.255

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    22/299     4.89G   0.02574   0.01655         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.897      0.501      0.603      0.295

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    23/299     4.89G   0.02442   0.01681         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.847      0.557      0.654      0.314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    24/299     4.89G   0.02448   0.01607         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.815      0.531       0.62      0.274

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    25/299     4.89G   0.02387   0.01571         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.836      0.499       0.58      0.262

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    26/299     4.89G   0.02432   0.01565         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.884      0.523      0.619      0.297

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    27/299     4.89G   0.02329   0.01536         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.835      0.542      0.632      0.309

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    28/299     4.89G   0.02266   0.01492         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.738      0.485      0.543       0.22

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    29/299     4.89G   0.02385   0.01487         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.839      0.509      0.601      0.264

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    30/299     4.89G   0.02337   0.01462         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.757      0.399      0.455      0.186

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    31/299     4.89G   0.02282   0.01435         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.832      0.536       0.62      0.269

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    32/299     4.89G   0.02277   0.01409         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.825      0.515      0.598      0.268

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    33/299     4.89G   0.02207   0.01427         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449       0.88      0.527      0.613      0.282

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    34/299     4.89G   0.02226   0.01408         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.861       0.53      0.629      0.281

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    35/299     4.89G   0.02188   0.01361         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.892      0.531      0.633      0.294

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    36/299     4.89G    0.0228   0.01361         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.846      0.519      0.601      0.286

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    37/299     4.89G    0.0223   0.01334         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.789      0.452      0.531      0.238

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    38/299     4.89G   0.02223   0.01331         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.911       0.41      0.514      0.241

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    39/299     4.89G    0.0221   0.01347         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.917      0.509      0.617      0.296

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    40/299     4.89G   0.02165     0.013         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.849      0.547       0.64      0.308

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    41/299     4.89G   0.02256   0.01315         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.885      0.538      0.637      0.308

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    42/299     4.89G   0.02166   0.01306         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.842      0.527      0.615      0.301

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    43/299     4.89G    0.0217     0.013         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.881      0.516      0.625      0.305

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    44/299     4.89G   0.02131   0.01285         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.866      0.538      0.613      0.293

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    45/299     4.89G   0.02109    0.0126         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.868      0.555      0.633      0.309

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    46/299     4.89G    0.0212   0.01262         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449       0.83      0.507      0.583      0.281

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    47/299     4.89G   0.02155   0.01266         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.877      0.528      0.606      0.303

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    48/299     4.89G   0.02119   0.01243         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.854      0.512      0.603      0.299

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    49/299     4.89G   0.02076   0.01233         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.856      0.519      0.609      0.304

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    50/299     4.89G   0.02121   0.01224         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449       0.83      0.522      0.599      0.284

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    51/299     4.89G   0.02083   0.01213         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.786      0.542      0.599      0.278

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    52/299     4.89G    0.0203   0.01205         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.887      0.517      0.607      0.304

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    53/299     4.89G   0.01992   0.01192         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.793       0.52      0.579      0.282

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    54/299     4.89G   0.01993    0.0118         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.866      0.501      0.592      0.293

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    55/299     4.89G   0.02006    0.0117         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.849      0.548      0.629      0.304

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    56/299     4.89G   0.02042   0.01179         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.884      0.515      0.602      0.306

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    57/299     4.89G   0.01995   0.01166         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.844        0.5      0.605      0.307

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    58/299     4.89G   0.01986    0.0116         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.849      0.498      0.595      0.298

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    59/299     4.89G   0.02003   0.01155         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.866      0.527      0.612      0.303

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    60/299     4.89G   0.01998   0.01151         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.866      0.518      0.628        0.3

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    61/299     4.89G   0.01999    0.0114         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.838      0.546      0.622      0.312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    62/299     4.89G   0.01978    0.0116         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.879      0.504      0.583      0.295

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    63/299     4.89G   0.02025    0.0113         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.912      0.502       0.61       0.31

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    64/299     4.89G   0.01946   0.01133         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.885       0.53      0.621      0.312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    65/299     4.89G   0.01952   0.01097         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.878      0.516      0.608      0.309

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    66/299     4.89G   0.01931   0.01108         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.856      0.539      0.617       0.31

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    67/299     4.89G   0.01925   0.01109         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.904      0.527       0.63      0.315

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    68/299     4.89G   0.01976   0.01115         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.853      0.494      0.579        0.3

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    69/299     4.89G   0.01977   0.01124         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.865      0.532      0.622      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    70/299     4.89G   0.01896   0.01093         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.872      0.527      0.612      0.294

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    71/299     4.89G   0.01933   0.01106         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.874      0.519      0.601      0.305

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    72/299     4.89G   0.01903   0.01112         0         0      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.866      0.498      0.589      0.281

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    73/299     4.89G   0.01871   0.01082         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.866      0.522      0.604      0.301

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    74/299     4.89G    0.0193   0.01099         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.833      0.511      0.592      0.293

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    75/299     4.89G   0.01885   0.01088         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.879      0.506       0.59      0.305

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    76/299     4.89G   0.01941   0.01122         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.855      0.535      0.627      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    77/299     4.89G   0.01913   0.01087         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.861      0.566       0.66      0.325

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    78/299     4.89G   0.01882   0.01072         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.834      0.509      0.598      0.284

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    79/299     4.89G   0.01848   0.01069         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.878      0.535      0.633      0.309

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    80/299     4.89G   0.01861   0.01052         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.38s/it]                                                                       
                 all        677       2449      0.839      0.529      0.625      0.297

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    81/299     4.89G   0.01873    0.0107         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.858      0.519      0.605      0.302

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    82/299     4.89G   0.01962   0.01061         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.826      0.539      0.621      0.312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    83/299     4.89G   0.01835    0.0105         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.892      0.507      0.615      0.299

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    84/299     4.89G   0.01818   0.01035         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449       0.88      0.537      0.633      0.311

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    85/299     4.89G   0.01827   0.01051         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.851      0.541      0.631      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    86/299     4.89G   0.01829   0.01051         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.883      0.512      0.612      0.298

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    87/299     4.89G   0.01903   0.01065         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.842      0.534      0.618      0.307

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    88/299     4.89G   0.01797   0.01064         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.843      0.519      0.611       0.29

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    89/299     4.89G   0.01843   0.01043         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.865      0.525      0.628      0.303

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    90/299     4.89G   0.01825    0.0105         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.886      0.529      0.625      0.299

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    91/299     4.89G   0.01831   0.01035         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.853       0.53      0.632      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    92/299     4.89G   0.01866    0.0104         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449        0.9      0.514      0.625      0.307

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    93/299     4.89G   0.01846   0.01035         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.844      0.531      0.631      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    94/299     4.89G   0.01805   0.01023         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.842      0.533      0.629      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    95/299     4.89G   0.01774   0.01043         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.871      0.532      0.629      0.306

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    96/299     4.89G   0.01835   0.01022         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.862       0.55      0.646      0.311

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    97/299     4.89G   0.01776   0.01023         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.848      0.528      0.618      0.309

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    98/299     4.89G     0.018   0.01024         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.894      0.506      0.615      0.304

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    99/299     4.89G   0.01804   0.01016         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.824      0.532      0.615      0.297

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   100/299     4.89G   0.01762   0.01009         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.875      0.535      0.627      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   101/299     4.89G   0.01723  0.009912         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.857      0.542      0.632      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   102/299     4.89G   0.01748  0.009959         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.855      0.533      0.627      0.307

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   103/299     4.89G    0.0177  0.009939         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.885      0.542      0.648      0.323

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   104/299     4.89G   0.01785  0.009968         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.865      0.545      0.641      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   105/299     4.89G   0.01767  0.009963         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.86      0.548      0.646      0.307

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   106/299     4.89G   0.01769  0.009987         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.874      0.542      0.642      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   107/299     4.89G   0.01778  0.009963         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.873      0.548      0.638      0.321

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   108/299     4.89G    0.0179  0.009916         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.831      0.544      0.637      0.321

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   109/299     4.89G   0.01764   0.00994         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.835      0.535      0.624      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   110/299     4.89G   0.01703  0.009949         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449       0.84      0.538      0.623       0.31

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   111/299     4.89G    0.0174  0.009924         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.856       0.53      0.616      0.304

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   112/299     4.89G   0.01724  0.009717         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.88      0.532       0.62      0.305

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   113/299     4.89G   0.01798  0.009973         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.832       0.56      0.631       0.31

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   114/299     4.89G   0.01732    0.0098         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.86      0.543      0.633      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   115/299     4.89G   0.01765  0.009811         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.878      0.539      0.636      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   116/299     4.89G   0.01736  0.009794         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.868      0.539      0.632      0.315

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   117/299     4.89G   0.01713  0.009768         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.872      0.536      0.637      0.314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   118/299     4.89G    0.0176  0.009928         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.861      0.532      0.633      0.307

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   119/299     4.89G   0.01756  0.009756         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.879      0.545      0.651      0.324

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   120/299     4.89G   0.01744  0.009639         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.874      0.538      0.635      0.315

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   121/299     4.89G   0.01729  0.009634         0         0      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.897      0.531       0.64       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   122/299     4.89G   0.01653  0.009792         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.876      0.533      0.644      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   123/299     4.89G   0.01646  0.009582         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.863      0.543      0.648      0.324

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   124/299     4.89G   0.01697  0.009561         0         0      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.887       0.53      0.653      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   125/299     4.89G   0.01676  0.009511         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.852      0.543      0.651      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   126/299     4.89G   0.01672  0.009477         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.867      0.548      0.657      0.332

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   127/299     4.89G   0.01645  0.009527         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.887      0.538      0.658      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   128/299     4.89G   0.01647  0.009564         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.886      0.531      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   129/299     4.89G   0.01665  0.009576         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.817      0.558      0.647       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   130/299     4.89G   0.01692  0.009527         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.85      0.542      0.648      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   131/299     4.89G   0.01672  0.009443         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.84      0.551      0.654       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   132/299     4.89G   0.01698  0.009517         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.88      0.535      0.653       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   133/299     4.89G   0.01652  0.009441         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.885      0.534      0.651      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   134/299     4.89G   0.01643  0.009373         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.865      0.545      0.658      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   135/299     4.89G    0.0163  0.009302         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.848       0.55       0.66      0.333

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   136/299     4.89G   0.01606  0.009256         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.897      0.528      0.658      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   137/299     4.89G   0.01629  0.009177         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.883       0.53      0.658      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   138/299     4.89G   0.01639  0.009237         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.854      0.543      0.655      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   139/299     4.89G    0.0165   0.00934         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.881      0.527      0.658      0.338

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   140/299     4.89G   0.01645  0.009367         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.842      0.543      0.654      0.334

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   141/299     4.89G   0.01673  0.009351         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.826       0.55      0.648      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   142/299     4.89G   0.01623  0.009506         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.858      0.537      0.651      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   143/299     4.89G   0.01612  0.009174         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.848      0.542      0.652      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   144/299     4.89G   0.01575  0.009213         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.835      0.545      0.653      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   145/299     4.89G   0.01564  0.009129         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449       0.87      0.535      0.658      0.339

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   146/299     4.89G   0.01625  0.009188         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.841      0.542      0.654      0.334

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   147/299     4.89G   0.01572  0.009108         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.849      0.537      0.651      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   148/299     4.89G   0.01625  0.009235         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.847      0.539      0.655      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   149/299     4.89G   0.01609  0.009243         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.806      0.558      0.653      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   150/299     4.89G   0.01568   0.00911         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.819      0.547       0.64      0.332

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   151/299     4.89G   0.01554  0.008926         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.838      0.533      0.634      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   152/299     4.89G   0.01581  0.009135         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.878      0.517      0.631      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   153/299     4.89G   0.01567  0.009062         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.841      0.535      0.635      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   154/299     4.89G   0.01587  0.009168         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.865       0.53      0.637      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   155/299     4.89G    0.0153  0.008966         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.842      0.546      0.639      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   156/299     4.89G   0.01605  0.009085         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.852      0.538       0.64       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   157/299     4.89G   0.01538  0.008947         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.888      0.521       0.64       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   158/299     4.89G   0.01572  0.009089         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.894      0.519      0.639      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   159/299     4.89G   0.01557  0.009015         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.855      0.534      0.639       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   160/299     4.89G   0.01554  0.008988         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.867      0.529      0.637      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   161/299     4.89G   0.01518  0.008933         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.867      0.528      0.639      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   162/299     4.89G   0.01542  0.008844         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.877      0.521      0.638      0.325

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   163/299     4.89G   0.01501  0.008891         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.866      0.525      0.638      0.324

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   164/299     4.89G   0.01601  0.008814         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.851       0.53      0.636      0.324

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   165/299     4.89G   0.01532  0.008922         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.849      0.534      0.638      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   166/299     4.89G   0.01518  0.008865         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.864      0.529      0.643       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   167/299     4.89G   0.01491  0.008867         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.854      0.535      0.648      0.332

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   168/299     4.89G   0.01533  0.008779         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.855      0.535      0.647       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   169/299     4.89G   0.01492  0.008772         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.895       0.52      0.647       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   170/299     4.89G   0.01564  0.008845         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.894      0.522      0.648      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   171/299     4.89G   0.01568  0.008837         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.89      0.524      0.649      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   172/299     4.89G   0.01522  0.008783         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.898       0.52      0.645      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   173/299     4.89G   0.01487  0.008785         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.881      0.525      0.645      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   174/299     4.89G   0.01513  0.008818         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.891      0.521      0.645      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   175/299     4.89G   0.01531  0.008737         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.868      0.528      0.644      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   176/299     4.89G   0.01564  0.008854         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.857      0.531      0.644      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   177/299     4.89G   0.01498  0.008809         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.862       0.53      0.643      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   178/299     4.89G   0.01431  0.008544         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.856       0.53      0.644      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   179/299     4.89G   0.01452  0.008603         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449        0.9      0.514      0.645      0.331

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   180/299     4.89G   0.01487  0.008681         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.883       0.52      0.645      0.333

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   181/299     4.89G   0.01477  0.008636         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449       0.86      0.532      0.648      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   182/299     4.89G   0.01464  0.008732         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449       0.88      0.523      0.648      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   183/299     4.89G   0.01479   0.00854         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.881       0.52      0.646      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   184/299     4.89G   0.01475  0.008595         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.893      0.516      0.649      0.338

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   185/299     4.89G    0.0145  0.008521         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.878      0.521      0.649      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   186/299     4.89G   0.01436  0.008619         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.891      0.517      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   187/299     4.89G   0.01504  0.008654         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.886      0.518       0.65      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   188/299     4.89G   0.01414  0.008537         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.873       0.52      0.651      0.338

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   189/299     4.89G   0.01462  0.008607         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.869      0.524      0.652      0.338

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   190/299     4.89G   0.01437   0.00861         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.873      0.522      0.651      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   191/299     4.89G   0.01438  0.008472         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.878      0.519       0.65      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   192/299     4.89G   0.01386  0.008262         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.879      0.518       0.65      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   193/299     4.89G   0.01404  0.008414         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.884      0.515      0.651      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   194/299     4.89G   0.01441  0.008478         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.879      0.516      0.651      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   195/299     4.89G   0.01435  0.008404         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.875      0.518       0.65      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   196/299     4.89G   0.01402  0.008428         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.872      0.519       0.65      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   197/299     4.89G   0.01429  0.008383         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.892      0.512      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   198/299     4.89G   0.01426  0.008401         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.876      0.517      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   199/299     4.89G   0.01404  0.008374         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.884      0.514      0.648      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   200/299     4.89G   0.01458  0.008282         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.872      0.516      0.648      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   201/299     4.89G   0.01454  0.008269         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.859      0.522       0.65      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   202/299     4.89G   0.01433  0.008435         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.869      0.517      0.649      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   203/299     4.89G   0.01421  0.008337         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.871      0.518      0.651      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   204/299     4.89G   0.01446  0.008367         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.873      0.517       0.65      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   205/299     4.89G   0.01461  0.008334         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.867      0.518      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   206/299     4.89G   0.01434  0.008422         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.886       0.51      0.648      0.334

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   207/299     4.89G   0.01421  0.008362         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449       0.88      0.511      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   208/299     4.89G   0.01425  0.008466         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.895      0.505      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   209/299     4.89G   0.01378  0.008336         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.895      0.504      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   210/299     4.89G   0.01406  0.008283         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.872      0.511      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   211/299     4.89G   0.01411  0.008397         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.865      0.512      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   212/299     4.89G   0.01428  0.008445         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.876      0.508       0.65      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   213/299     4.89G   0.01425  0.008351         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.875      0.508      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   214/299     4.89G   0.01406  0.008178         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.858      0.513       0.65      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   215/299     4.89G   0.01375  0.008171         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.865      0.512       0.65      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   216/299     4.89G   0.01395  0.008216         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.871      0.509       0.65      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   217/299     4.89G   0.01394  0.008235         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.869       0.51       0.65      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   218/299     4.89G   0.01365  0.008219         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.868      0.509       0.65      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   219/299     4.89G   0.01426  0.008322         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.866      0.509      0.649      0.334

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   220/299     4.89G   0.01361  0.008105         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.865       0.51      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   221/299     4.89G   0.01379  0.008203         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.868      0.509      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   222/299     4.89G   0.01374   0.00814         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.859      0.512      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   223/299     4.89G   0.01378  0.008185         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.865      0.511      0.649      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   224/299     4.89G   0.01333  0.008086         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.866      0.511      0.648      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   225/299     4.89G   0.01331  0.008061         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.864      0.511      0.648      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   226/299     4.89G   0.01333  0.008029         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.869      0.508      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   227/299     4.89G   0.01305  0.008084         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.869      0.508      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   228/299     4.89G   0.01318  0.007961         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.874      0.506      0.648      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   229/299     4.89G   0.01305  0.008204         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.77s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.869      0.507      0.648      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   230/299     4.89G   0.01304  0.007919         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449       0.87      0.507      0.648      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   231/299     4.89G   0.01304   0.00794         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.871      0.506      0.648      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   232/299     4.89G   0.01278  0.007896         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.868      0.507      0.648      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   233/299     4.89G   0.01269  0.007807         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.867      0.508      0.648      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   234/299     4.89G   0.01314  0.007999         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.868      0.508      0.647      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   235/299     4.89G   0.01267  0.007917         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.869      0.508      0.649      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   236/299     4.89G   0.01313  0.007859         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.864      0.508      0.648      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   237/299     4.89G   0.01267  0.007866         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.862      0.508      0.648      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   238/299     4.89G    0.0124  0.007796         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.873      0.505      0.649      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   239/299     4.89G   0.01286  0.007799         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.871      0.505      0.648      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   240/299     4.89G   0.01249  0.007798         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.871      0.505      0.649      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   241/299     4.89G   0.01253   0.00771         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449       0.87      0.505      0.649      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   242/299     4.89G    0.0125  0.007678         0         0      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.868      0.507       0.65      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   243/299     4.89G   0.01254  0.007715         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.873      0.506       0.65      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   244/299     4.89G   0.01256  0.007755         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.874      0.506       0.65      0.337

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   245/299     4.89G   0.01268  0.007712         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        677       2449      0.874      0.506      0.651      0.337
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 145, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

246 epochs completed in 27.213 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 6.9MB
Optimizer stripped from runs/train/exp/weights/best.pt, 6.9MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model Summary: 280 layers, 3087256 parameters, 0 gradients, 4.2 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:25<00:00,  1.50s/it]                                                                       
                 all        677       2449       0.87      0.535      0.658      0.339
Results saved to runs/train/exp
