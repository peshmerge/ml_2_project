python train.py --single-cls --img 1280 --rect --batch 20 --epochs 64 --data cots.yml --weights weights/yolov5n6.pt --workers 0
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: (30 second timeout) 3
wandb: You chose 'Don't visualize my results'
train: weights=weights/yolov5n6.pt, cfg=, data=cots.yml, hyp=data/hyps/hyp.scratch.yaml, epochs=64, batch_size=20, imgsz=1280, rect=True, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=True, optimizer=SGD, sync_bn=False, workers=0, project=runs/train, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
remote: Enumerating objects: 17, done.
remote: Counting objects: 100% (17/17), done.
remote: Compressing objects: 100% (12/12), done.
remote: Total 17 (delta 6), reused 14 (delta 5), pack-reused 0
Unpacking objects: 100% (17/17), 5.15 KiB | 879.00 KiB/s, done.
From https://github.com/ultralytics/yolov5
   bd815d4..c43439a  master     -> origin/master
   bced04e..32fb86e  tests/aws  -> origin/tests/aws
github: ⚠️ YOLOv5 is out of date by 1 commit. Use `git pull` or `git clone https://github.com/ultralytics/yolov5` to update.
YOLOv5 🚀 v6.0-205-gbd815d4 torch 1.10.1 CUDA:0 (GeForce GTX 1060 6GB, 6075MiB)

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
train: Scanning '/home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/training' images and labels...3936 found, 0 missing, 0 empty, 1 corrupt: 100%|█| 3936/3936 [00:00<00:00, 71
train: WARNING: /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/images/training/video_0_9470.jpg: ignoring corrupt image/label: non-normalized or out of bounds coordinates [     1.0021]
train: New cache created: /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/training.cache
val: Scanning '/home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/validation' images and labels...983 found, 0 missing, 0 empty, 0 corrupt: 100%|█| 983/983 [00:00<00:00, 6918.
val: New cache created: /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/validation.cache
Plotting labels to runs/train/exp/labels.jpg... 

AutoAnchor: 4.93 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to runs/train/exp
Starting training for 64 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      0/63     5.04G   0.06408   0.03532         0        15      1280: 100%|██████████| 197/197 [05:51<00:00,  1.79s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:33<00:00,  1.34s/it]                                                                       
                 all        983       2382   1.02e-05    0.00126   5.09e-06    1.7e-06

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      1/63     4.84G   0.04811   0.03716         0        14      1280: 100%|██████████| 197/197 [05:43<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:35<00:00,  1.40s/it]                                                                       
                 all        983       2382     0.0361     0.0979     0.0136    0.00398

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      2/63     4.84G   0.04256   0.03519         0        15      1280: 100%|██████████| 197/197 [05:42<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.181      0.141      0.086      0.031

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      3/63     4.84G   0.03974   0.03286         0        14      1280: 100%|██████████| 197/197 [05:42<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.23      0.149      0.102     0.0357

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      4/63     4.84G   0.03775   0.03222         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:35<00:00,  1.42s/it]                                                                       
                 all        983       2382      0.289      0.217      0.146       0.04

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      5/63     4.84G   0.03766   0.03048         0        13      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:35<00:00,  1.41s/it]                                                                       
                 all        983       2382      0.443      0.278      0.257      0.106

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      6/63     4.84G   0.03548   0.02807         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.617      0.395      0.433      0.155

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      7/63     4.84G   0.03386    0.0264         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.43      0.259      0.216     0.0531

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      8/63     4.84G   0.03308   0.02523         0        13      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.688      0.456      0.489      0.178

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      9/63     4.84G   0.03101   0.02514         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.30s/it]                                                                       
                 all        983       2382      0.693      0.397      0.467      0.198

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     10/63     4.84G    0.0303   0.02403         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.832      0.516      0.612      0.235

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     11/63     4.84G   0.03055   0.02287         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.795      0.523      0.614      0.238

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     12/63     4.84G   0.02925   0.02238         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:33<00:00,  1.35s/it]                                                                       
                 all        983       2382      0.791      0.521      0.611      0.252

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     13/63     4.84G   0.02814   0.02185         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.82      0.593      0.674      0.271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     14/63     4.84G   0.02758   0.02079         0        13      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.834       0.66      0.748      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     15/63     4.84G   0.02718   0.02069         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.854      0.648       0.74       0.34

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     16/63     4.84G    0.0274   0.02064         0        15      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.813      0.546      0.649      0.291

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     17/63     4.84G    0.0267   0.01999         0        13      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.817      0.612      0.699      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     18/63     4.84G   0.02648   0.01934         0        13      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.32s/it]                                                                       
                 all        983       2382      0.828      0.625      0.731      0.359

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     19/63     4.84G   0.02606   0.01899         0        13      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.873      0.665      0.765      0.347

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     20/63     4.84G   0.02566   0.01852         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:33<00:00,  1.32s/it]                                                                       
                 all        983       2382      0.867      0.693      0.788      0.365

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     21/63     4.84G   0.02497    0.0185         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.854      0.712      0.802      0.396

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     22/63     4.84G   0.02452   0.01805         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.32s/it]                                                                       
                 all        983       2382      0.869      0.715      0.797      0.384

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     23/63     4.84G    0.0247   0.01751         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.888       0.72      0.819      0.375

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     24/63     4.84G   0.02401   0.01723         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.879      0.729      0.819      0.401

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     25/63     4.84G   0.02378   0.01665         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.32s/it]                                                                       
                 all        983       2382      0.844      0.744      0.821       0.41

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     26/63     4.84G   0.02384   0.01686         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.903       0.76       0.85      0.424

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     27/63     4.84G   0.02381   0.01631         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.919      0.743      0.842      0.427

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     28/63     4.84G   0.02393   0.01603         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.888       0.77      0.849       0.43

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     29/63     4.84G   0.02305   0.01564         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.922      0.786      0.883      0.455

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     30/63     4.84G   0.02252   0.01552         0        13      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.91      0.801       0.89      0.459

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     31/63     4.84G   0.02158   0.01516         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.922      0.809      0.896      0.461

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     32/63     4.84G   0.02145   0.01486         0        15      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.932      0.819      0.902       0.47

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     33/63     4.84G   0.02221   0.01467         0        13      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.916      0.825      0.902      0.468

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     34/63     4.84G   0.02129   0.01446         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.94      0.821      0.902       0.48

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     35/63     4.84G   0.02118   0.01438         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.907      0.833      0.904      0.471

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     36/63     4.84G   0.02139    0.0143         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.93      0.837      0.915      0.478

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     37/63     4.84G   0.02097    0.0141         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.933      0.854      0.923      0.496

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     38/63     4.84G   0.02055   0.01393         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.932      0.853      0.924      0.497

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     39/63     4.84G    0.0206    0.0135         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.906      0.848      0.904      0.488

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     40/63     4.84G   0.02075   0.01356         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.93      0.853      0.922      0.501

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     41/63     4.84G   0.02048   0.01355         0        15      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.30s/it]                                                                       
                 all        983       2382      0.953      0.848      0.928      0.507

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     42/63     4.84G   0.02015   0.01322         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.94      0.873      0.938      0.515

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     43/63     4.84G   0.02004   0.01312         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.927      0.875      0.928       0.51

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     44/63     4.84G   0.01994    0.0131         0        15      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.94      0.871      0.938      0.513

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     45/63     4.84G   0.01979   0.01303         0        15      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.933      0.883      0.946      0.527

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     46/63     4.84G   0.01942   0.01273         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.932      0.876      0.934      0.513

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     47/63     4.84G   0.01911   0.01271         0        13      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.94      0.889      0.945       0.53

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     48/63     4.84G   0.01902   0.01257         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382       0.94       0.88      0.946       0.53

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     49/63     4.84G   0.01825   0.01241         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.30s/it]                                                                       
                 all        983       2382      0.947      0.895      0.954      0.536

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     50/63     4.84G   0.01846    0.0121         0        15      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.954      0.883      0.954      0.537

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     51/63     4.84G   0.01849   0.01213         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.30s/it]                                                                       
                 all        983       2382      0.938       0.91      0.954      0.548

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     52/63     4.84G   0.01849     0.012         0        15      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.937      0.901      0.952      0.537

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     53/63     4.84G   0.01825   0.01209         0        15      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.938       0.91      0.956      0.539

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     54/63     4.84G   0.01781   0.01186         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.946      0.909      0.958      0.547

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     55/63     4.84G   0.01784   0.01186         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.949      0.902      0.958      0.545

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     56/63     4.84G   0.01812   0.01186         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.939      0.907      0.956      0.538

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     57/63     4.84G   0.01778   0.01175         0        13      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.941      0.915       0.96       0.54

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     58/63     4.84G   0.01792   0.01167         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.943      0.917       0.96      0.547

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     59/63     4.84G   0.01728   0.01168         0        13      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.948      0.913      0.961      0.544

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     60/63     4.84G   0.01747   0.01148         0        14      1280: 100%|██████████| 197/197 [05:41<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.949      0.911      0.963       0.54

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     61/63     4.84G   0.01738   0.01151         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.30s/it]                                                                       
                 all        983       2382      0.955      0.906      0.962      0.538

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     62/63     4.84G   0.01742   0.01144         0        13      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.949       0.91      0.961      0.547

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     63/63     4.84G   0.01745   0.01143         0        14      1280: 100%|██████████| 197/197 [05:40<00:00,  1.73s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:32<00:00,  1.31s/it]                                                                       
                 all        983       2382      0.942      0.913      0.961      0.531

64 epochs completed in 6.662 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 6.9MB
Optimizer stripped from runs/train/exp/weights/best.pt, 6.9MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model Summary: 280 layers, 3087256 parameters, 0 gradients, 4.2 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 25/25 [00:35<00:00,  1.43s/it]                                                                       
                 all        983       2382      0.947      0.901      0.954      0.548
Results saved to runs/train/exp
