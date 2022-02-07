(py3711) peshmerge@akersloot:~/ml_2_project/yolov5$ python train.py --single-cls --img 1280 --rect --batch 20 --epochs 150 --data cots.yml --weights weights/yolov5n6.pt --workers 0
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: (30 second timeout) 3
wandb: You chose 'Don't visualize my results'
train: weights=weights/yolov5n6.pt, cfg=, data=cots.yml, hyp=data/hyps/hyp.scratch.yaml, epochs=150, batch_size=20, imgsz=1280, rect=True, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=True, optimizer=SGD, sync_bn=False, workers=0, project=runs/train, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
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
val: Scanning '/home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/validation' images and labels...657 found, 0 missing, 0 empty, 0 corrupt: 100%|█| 657/657 [00:00<00:00, 6600.
val: New cache created: /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/validation.cache
Plotting labels to runs/train/exp2/labels.jpg... 

AutoAnchor: 4.95 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to runs/train/exp2
Starting training for 150 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     0/149     5.05G   0.06383   0.03304         0         0      1280: 100%|██████████| 213/213 [06:21<00:00,  1.79s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.34s/it]                                                                       
                 all        657       2287   1.52e-05    0.00131   7.63e-06    2.8e-06

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     1/149     4.89G   0.04812   0.03428         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.34s/it]                                                                       
                 all        657       2287     0.0567     0.0647     0.0184    0.00419

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     2/149     4.89G   0.04255   0.03127         0         0      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.34s/it]                                                                       
                 all        657       2287      0.243      0.173      0.115     0.0381

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     3/149     4.89G   0.03878   0.02823         0         1      1280: 100%|██████████| 213/213 [06:22<00:00,  1.80s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:25<00:00,  1.47s/it]                                                                       
                 all        657       2287       0.33       0.21       0.15     0.0368

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     4/149     4.89G   0.03863   0.02765         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:24<00:00,  1.47s/it]                                                                       
                 all        657       2287      0.455      0.265       0.25     0.0884

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     5/149     4.89G    0.0378   0.02725         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.35s/it]                                                                       
                 all        657       2287      0.478      0.202      0.208     0.0731

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     6/149     4.89G   0.03586   0.02564         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:24<00:00,  1.42s/it]                                                                       
                 all        657       2287       0.44      0.233      0.221     0.0807

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     7/149     4.89G   0.03395   0.02479         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.659      0.399       0.46      0.187

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     8/149     4.89G   0.03284    0.0236         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.608       0.32      0.357      0.157

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     9/149     4.89G   0.03104   0.02302         0         0      1280: 100%|██████████| 213/213 [06:08<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.819      0.367      0.462      0.207

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    10/149     4.89G   0.03069   0.02234         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.34s/it]                                                                       
                 all        657       2287      0.671      0.408       0.47      0.192

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    11/149     4.89G   0.02995   0.02125         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.752      0.512      0.591      0.268

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    12/149     4.89G   0.02894    0.0209         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.757      0.474       0.54      0.243

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    13/149     4.89G   0.02886    0.0206         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287        0.7      0.488      0.547      0.223

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    14/149     4.89G    0.0284   0.01988         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.826      0.498      0.599      0.259

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    15/149     4.89G   0.02722   0.01935         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.888      0.524       0.64      0.294

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    16/149     4.89G   0.02633   0.01861         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287       0.81      0.486      0.578      0.264

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    17/149     4.89G   0.02695   0.01842         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.827      0.533      0.632      0.287

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    18/149     4.89G   0.02635   0.01843         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.793      0.532      0.628        0.3

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    19/149     4.89G   0.02744   0.01785         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.842      0.538      0.636      0.301

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    20/149     4.89G   0.02612   0.01749         0         0      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.832      0.551      0.633      0.296

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    21/149     4.89G   0.02606    0.0174         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.857      0.529      0.624      0.273

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    22/149     4.89G   0.02562   0.01692         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.799      0.435        0.5      0.234

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    23/149     4.89G   0.02458   0.01682         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.816       0.54      0.624      0.314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    24/149     4.89G   0.02432   0.01624         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.821       0.53      0.624      0.297

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    25/149     4.89G   0.02401   0.01622         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.33s/it]                                                                       
                 all        657       2287      0.794      0.442      0.519      0.226

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    26/149     4.89G   0.02384   0.01591         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.823      0.545      0.639      0.291

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    27/149     4.89G    0.0232   0.01531         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.849      0.554      0.645      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    28/149     4.89G   0.02304     0.015         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.851      0.524      0.635      0.304

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    29/149     4.89G   0.02366   0.01492         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.848      0.529      0.617      0.289

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    30/149     4.89G   0.02322   0.01472         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.781      0.509      0.592      0.277

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    31/149     4.89G   0.02283    0.0143         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.792      0.521      0.611      0.296

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    32/149     4.89G   0.02268   0.01404         0         0      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.849      0.509      0.606      0.284

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    33/149     4.89G   0.02223   0.01416         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.774      0.491      0.564      0.271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    34/149     4.89G   0.02263   0.01405         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.834      0.464       0.54      0.259

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    35/149     4.89G   0.02142   0.01369         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.889      0.516      0.614      0.288

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    36/149     4.89G   0.02221   0.01355         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.802       0.48      0.565      0.273

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    37/149     4.89G   0.02269   0.01357         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287        0.8      0.527      0.607      0.273

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    38/149     4.89G   0.02242   0.01345         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287        0.8       0.53      0.615      0.294

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    39/149     4.89G   0.02177   0.01323         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.832      0.505      0.599      0.271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    40/149     4.89G   0.02157   0.01305         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.817      0.538      0.614        0.3

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    41/149     4.89G   0.02205   0.01318         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.825       0.49      0.573      0.271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    42/149     4.89G   0.02097   0.01294         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.807      0.487      0.574      0.271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    43/149     4.89G   0.02101   0.01284         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.844      0.516      0.598      0.288

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    44/149     4.89G   0.02102   0.01296         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.806      0.511      0.588      0.288

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    45/149     4.89G   0.02031   0.01265         0         1      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.861      0.525      0.613      0.309

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    46/149     4.89G   0.02079   0.01229         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287       0.83      0.546      0.619      0.298

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    47/149     4.89G    0.0209   0.01236         0         0      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.866      0.542      0.629      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    48/149     4.89G   0.02045   0.01214         0         0      1280: 100%|██████████| 213/213 [06:09<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.886      0.512      0.606      0.301

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    49/149     4.89G      0.02   0.01214         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.818      0.531      0.608      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    50/149     4.89G    0.0209   0.01211         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.882      0.505      0.607      0.282

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    51/149     4.89G   0.01989   0.01182         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.835      0.505      0.595        0.3

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    52/149     4.89G   0.01989   0.01195         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.811      0.554      0.626      0.301

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    53/149     4.89G    0.0197   0.01183         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.859        0.5      0.584      0.294

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    54/149     4.89G   0.01992   0.01176         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.872      0.539      0.624      0.307

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    55/149     4.89G   0.01968   0.01147         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.829      0.527       0.61      0.301

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    56/149     4.89G   0.01993   0.01167         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.894      0.495      0.587      0.297

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    57/149     4.89G     0.019   0.01145         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.871      0.505      0.602      0.299

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    58/149     4.89G   0.01887   0.01129         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.889      0.539      0.644      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    59/149     4.89G   0.01939   0.01138         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.887       0.53      0.621      0.314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    60/149     4.89G   0.01937   0.01133         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.862      0.541       0.63      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    61/149     4.89G   0.01942   0.01128         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.871      0.529      0.618      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    62/149     4.89G   0.01944    0.0113         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.875      0.521      0.613      0.315

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    63/149     4.89G   0.01935   0.01114         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287       0.84      0.496      0.585      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    64/149     4.89G   0.01895   0.01103         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.801      0.505      0.588      0.316

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    65/149     4.89G   0.01851    0.0109         0         0      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.852      0.527       0.62      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    66/149     4.89G   0.01818   0.01081         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.897        0.5      0.606      0.306

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    67/149     4.89G   0.01834    0.0108         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.895       0.52      0.615      0.308

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    68/149     4.89G   0.01834   0.01076         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.856      0.512      0.608      0.306

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    69/149     4.89G   0.01861   0.01089         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.825        0.5      0.592      0.296

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    70/149     4.89G   0.01801   0.01057         0         0      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.861      0.522      0.623      0.321

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    71/149     4.89G   0.01837   0.01089         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.878      0.509      0.606      0.311

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    72/149     4.89G   0.01843   0.01075         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.905      0.512      0.624      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    73/149     4.89G    0.0178   0.01056         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287       0.89      0.495      0.604      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    74/149     4.89G   0.01806   0.01058         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.864      0.535      0.631      0.316

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    75/149     4.89G   0.01822   0.01045         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.894      0.519      0.617      0.308

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    76/149     4.89G   0.01869   0.01079         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.871      0.518      0.628      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    77/149     4.89G   0.01824   0.01036         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.835      0.554      0.636       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    78/149     4.89G   0.01813   0.01036         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.859       0.52      0.613      0.311

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    79/149     4.89G   0.01784   0.01027         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.815      0.515      0.598      0.312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    80/149     4.89G    0.0177   0.01027         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.874      0.516      0.612       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    81/149     4.89G   0.01766   0.01028         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.846      0.528       0.62      0.306

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    82/149     4.89G   0.01773    0.0102         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.879      0.516       0.62      0.311

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    83/149     4.89G   0.01725   0.01017         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.874      0.515      0.617       0.31

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    84/149     4.89G   0.01749   0.01008         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.866      0.537       0.63      0.325

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    85/149     4.89G   0.01747   0.01014         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.889       0.53      0.634      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    86/149     4.89G   0.01724   0.01012         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.865      0.526       0.63      0.315

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    87/149     4.89G   0.01673  0.009988         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287       0.86      0.543      0.647      0.323

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    88/149     4.89G   0.01698  0.009937         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.876      0.537      0.636      0.317

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    89/149     4.89G   0.01683  0.009854         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.855       0.53      0.635       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    90/149     4.89G   0.01662  0.009896         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.833      0.568      0.656      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    91/149     4.89G   0.01674  0.009821         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.866       0.54      0.646      0.316

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    92/149     4.89G     0.017   0.00987         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.872      0.523      0.637      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    93/149     4.89G    0.0172  0.009911         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.843       0.55      0.649      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    94/149     4.89G   0.01635  0.009717         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.856      0.547      0.648      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    95/149     4.89G    0.0164  0.009831         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.821      0.549      0.649      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    96/149     4.89G   0.01675  0.009625         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.841      0.549      0.649      0.315

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    97/149     4.89G   0.01656  0.009649         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.881       0.52      0.635      0.314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    98/149     4.89G   0.01671  0.009601         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.875      0.526      0.639      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    99/149     4.89G   0.01618  0.009566         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.848      0.525      0.633       0.31

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   100/149     4.89G   0.01647   0.00962         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.897       0.52      0.637       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   101/149     4.89G   0.01605   0.00944         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.891      0.535      0.651      0.323

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   102/149     4.89G   0.01584  0.009354         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.883      0.526      0.652      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   103/149     4.89G    0.0162  0.009367         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.861      0.536      0.652      0.333

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   104/149     4.89G   0.01593  0.009342         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.899      0.524      0.655      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   105/149     4.89G     0.016  0.009459         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.32s/it]                                                                       
                 all        657       2287      0.886      0.522      0.643      0.317

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   106/149     4.89G   0.01571  0.009363         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.865      0.532       0.65      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   107/149     4.89G   0.01581   0.00932         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287       0.91      0.517       0.65      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   108/149     4.89G    0.0157  0.009223         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.858      0.533      0.647      0.323

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   109/149     4.89G   0.01574  0.009387         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.872      0.525      0.644      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   110/149     4.89G   0.01506  0.009279         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.878      0.523       0.64       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   111/149     4.89G   0.01518  0.009257         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.874      0.518      0.633      0.317

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   112/149     4.89G    0.0149  0.008986         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.847      0.531       0.64      0.321

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   113/149     4.89G   0.01523  0.009123         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.894      0.512      0.639      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   114/149     4.89G   0.01509  0.009067         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.888      0.512      0.637      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   115/149     4.89G   0.01515  0.009078         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.861       0.52      0.636      0.312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   116/149     4.89G   0.01508  0.008986         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.854      0.521      0.633      0.312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   117/149     4.89G   0.01476  0.009113         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.865      0.525      0.637      0.314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   118/149     4.89G   0.01481  0.009012         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.852      0.527      0.643      0.317

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   119/149     4.89G   0.01504  0.008995         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.866      0.518      0.644       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   120/149     4.89G   0.01512  0.008938         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.849      0.523      0.644      0.321

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   121/149     4.89G   0.01484  0.008859         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.871      0.516      0.637       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   122/149     4.89G   0.01478  0.009017         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.902      0.508      0.642      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   123/149     4.89G   0.01431  0.008853         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.881      0.515      0.643      0.317

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   124/149     4.89G    0.0147  0.008783         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287       0.91      0.503      0.646      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   125/149     4.89G   0.01461  0.008862         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.904      0.506      0.648      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   126/149     4.89G    0.0143  0.008758         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.894      0.511      0.645       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   127/149     4.89G   0.01413  0.008861         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287       0.89      0.511      0.641      0.316

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   128/149     4.89G   0.01423  0.008707         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.861      0.517      0.644      0.323

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   129/149     4.89G   0.01426  0.008736         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.897      0.501      0.642      0.323

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   130/149     4.89G   0.01521  0.008729         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.867      0.513      0.647       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   131/149     4.89G   0.01443  0.008717         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287       0.86      0.514      0.646       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   132/149     4.89G   0.01443   0.00873         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.866      0.512      0.647       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   133/149     4.89G   0.01418  0.008613         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.887        0.5      0.646      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   134/149     4.89G   0.01416  0.008642         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.851      0.512      0.644      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   135/149     4.89G   0.01398  0.008567         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.893      0.501      0.647       0.33

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   136/149     4.89G   0.01402  0.008592         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.884      0.501      0.644      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   137/149     4.89G   0.01404  0.008532         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.878      0.504      0.642      0.325

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   138/149     4.89G   0.01402  0.008476         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.876      0.506      0.645      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   139/149     4.89G   0.01391   0.00856         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.849      0.513      0.645      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   140/149     4.89G   0.01381  0.008638         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.876      0.505      0.645      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   141/149     4.89G     0.014  0.008597         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287       0.87      0.506      0.644      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   142/149     4.89G   0.01386  0.008658         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.858      0.507      0.645      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   143/149     4.89G   0.01391  0.008458         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.883      0.498      0.646      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   144/149     4.89G   0.01368  0.008624         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.898      0.495      0.647      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   145/149     4.89G   0.01335  0.008488         0         0      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287       0.88      0.503      0.647      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   146/149     4.89G   0.01389  0.008546         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.889      0.498      0.642      0.321

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   147/149     4.89G   0.01345   0.00854         0         0      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.879      0.502      0.643      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   148/149     4.89G   0.01377  0.008532         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.869      0.506      0.642      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   149/149     4.89G   0.01357  0.008537         0         1      1280: 100%|██████████| 213/213 [06:13<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:22<00:00,  1.31s/it]                                                                       
                 all        657       2287      0.893      0.498      0.643      0.319

150 epochs completed in 16.469 hours.
Optimizer stripped from runs/train/exp2/weights/last.pt, 6.9MB
Optimizer stripped from runs/train/exp2/weights/best.pt, 6.9MB

Validating runs/train/exp2/weights/best.pt...
Fusing layers... 
Model Summary: 280 layers, 3087256 parameters, 0 gradients, 4.2 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:24<00:00,  1.47s/it]                                                                       
                 all        657       2287      0.878      0.543      0.644      0.336
Results saved to runs/train/exp2
