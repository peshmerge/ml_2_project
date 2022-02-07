(py3711) peshmerge@akersloot:~/ml_2_project/yolov5$ python train.py --single-cls --img 1280 --rect --batch 20 --epochs 64 --data cots.yml --weights weights/yolov5n6.pt --workers 0
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: (30 second timeout) 3
wandb: You chose 'Don't visualize my results'
train: weights=weights/yolov5n6.pt, cfg=, data=cots.yml, hyp=data/hyps/hyp.scratch.yaml, epochs=64, batch_size=20, imgsz=1280, rect=True, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=True, optimizer=SGD, sync_bn=False, workers=0, project=runs/train, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
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
val: Scanning '/home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/validation' images and labels...677 found, 0 missing, 0 empty, 0 corrupt: 100%|█| 677/677 [00:00<00:00, 6433.
val: New cache created: /home/peshmerge/.kaggle/tensorflow-great-barrier-reef/coco_style/convertor/labels/validation.cache
Plotting labels to runs/train/exp/labels.jpg... 

AutoAnchor: 4.95 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 1280 train, 1280 val
Using 0 dataloader workers
Logging results to runs/train/exp
Starting training for 64 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      0/63     4.89G   0.06353     0.033         0         0      1280: 100%|██████████| 213/213 [06:28<00:00,  1.83s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.39s/it]                                                                       
                 all        677       2449   4.92e-06   0.000408   2.47e-06   2.47e-07

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      1/63     4.89G   0.04785   0.03452         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.38s/it]                                                                       
                 all        677       2449      0.132      0.179     0.0661     0.0178

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      2/63     4.89G   0.04258   0.03107         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.38s/it]                                                                       
                 all        677       2449      0.279      0.188      0.147     0.0474

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      3/63     4.89G   0.03872   0.03003         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:24<00:00,  1.45s/it]                                                                       
                 all        677       2449      0.223      0.159      0.114     0.0363

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      4/63     4.89G   0.03888    0.0283         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:25<00:00,  1.48s/it]                                                                       
                 all        677       2449      0.286        0.2      0.151     0.0522

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      5/63     4.89G   0.03788    0.0276         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.442      0.306      0.283      0.103

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      6/63     4.89G   0.03512   0.02575         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.609      0.298      0.331      0.136

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      7/63     4.89G   0.03325   0.02463         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.38s/it]                                                                       
                 all        677       2449      0.651      0.377      0.428       0.16

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      8/63     4.89G   0.03258   0.02351         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.805      0.384      0.479      0.201

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      9/63     4.89G   0.03125   0.02274         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.817      0.441      0.544      0.229

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     10/63     4.89G   0.03025   0.02233         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.721      0.399      0.456      0.209

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     11/63     4.89G   0.03005    0.0211         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.843      0.477      0.603      0.267

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     12/63     4.89G   0.02869   0.02043         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.775      0.519      0.599      0.271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     13/63     4.89G   0.02841   0.02021         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.824       0.51      0.605      0.272

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     14/63     4.89G   0.02846   0.01989         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.38s/it]                                                                       
                 all        677       2449       0.77      0.522      0.598       0.28

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     15/63     4.89G   0.02752   0.01909         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449       0.81      0.534      0.617      0.304

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     16/63     4.89G   0.02599   0.01836         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.856      0.547      0.641      0.305

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     17/63     4.89G   0.02701   0.01796         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449       0.85      0.537      0.648      0.302

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     18/63     4.89G   0.02582   0.01795         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.795       0.51      0.578      0.262

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     19/63     4.89G   0.02703   0.01775         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.787       0.51      0.593      0.288

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     20/63     4.89G   0.02525    0.0171         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.883      0.546      0.654      0.307

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     21/63     4.89G   0.02507   0.01676         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.825      0.505        0.6      0.271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     22/63     4.89G   0.02504   0.01624         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.847      0.524      0.629      0.315

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     23/63     4.89G   0.02412   0.01649         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.837       0.54      0.637      0.303

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     24/63     4.89G   0.02424   0.01564         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.901      0.537      0.643      0.323

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     25/63     4.89G   0.02314   0.01527         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.881      0.549      0.654      0.316

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     26/63     4.89G   0.02307   0.01505         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.885      0.542      0.643      0.313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     27/63     4.89G   0.02225   0.01474         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.858       0.55      0.651      0.325

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     28/63     4.89G   0.02222   0.01443         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.827      0.577      0.657      0.298

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     29/63     4.89G   0.02326   0.01428         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.37s/it]                                                                       
                 all        677       2449      0.858      0.536      0.632      0.309

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     30/63     4.89G   0.02278   0.01417         0         1      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.851      0.559      0.652       0.31

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     31/63     4.89G   0.02216   0.01367         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.898      0.516      0.629      0.311

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     32/63     4.89G   0.02163    0.0135         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.822      0.557      0.646      0.305

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     33/63     4.89G   0.02092   0.01345         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.838       0.51      0.612      0.298

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     34/63     4.89G   0.02101   0.01327         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.847      0.522      0.622      0.302

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     35/63     4.89G   0.02017   0.01296         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.859      0.546      0.637      0.315

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     36/63     4.89G   0.02098   0.01269         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.885      0.516      0.615      0.312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     37/63     4.89G   0.02062   0.01268         0         0      1280: 100%|██████████| 213/213 [06:12<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.826      0.524      0.615      0.303

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     38/63     4.89G   0.02126   0.01257         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.847      0.479      0.576      0.283

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     39/63     4.89G   0.02027   0.01254         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.853      0.522      0.617      0.306

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     40/63     4.89G   0.02001   0.01216         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.75s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.845      0.524      0.608      0.317

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     41/63     4.89G   0.02014   0.01233         0         1      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.883      0.532      0.617      0.318

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     42/63     4.89G   0.01945     0.012         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.838      0.539      0.617      0.317

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     43/63     4.89G   0.01932   0.01194         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.81      0.505      0.592      0.294

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     44/63     4.89G   0.01908   0.01186         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.873      0.525      0.611      0.311

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     45/63     4.89G    0.0188    0.0117         0         1      1280: 100%|██████████| 213/213 [06:10<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.846      0.548      0.632      0.317

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     46/63     4.89G    0.0191   0.01151         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.864      0.539      0.634      0.316

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     47/63     4.89G     0.019   0.01137         0         0      1280: 100%|██████████| 213/213 [06:11<00:00,  1.74s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.832      0.543      0.636      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     48/63     4.89G   0.01865    0.0112         0         0      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.854      0.531      0.627      0.321

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     49/63     4.89G   0.01828   0.01128         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.877       0.56      0.659      0.332

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     50/63     4.89G   0.01854   0.01115         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.887      0.553       0.65      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     51/63     4.89G   0.01806   0.01101         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.817      0.566      0.651      0.324

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     52/63     4.89G   0.01761   0.01105         0         0      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.874      0.554       0.65      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     53/63     4.89G   0.01761   0.01091         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.865       0.54      0.636      0.316

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     54/63     4.89G   0.01745   0.01079         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.895      0.544      0.657      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     55/63     4.89G   0.01764   0.01065         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.855       0.56       0.66      0.332

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     56/63     4.89G   0.01779   0.01084         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449       0.86      0.534      0.635       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     57/63     4.89G   0.01741   0.01066         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.828      0.571      0.663      0.334

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     58/63     4.89G   0.01701   0.01061         0         1      1280: 100%|██████████| 213/213 [06:15<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.898      0.541      0.663      0.334

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     59/63     4.89G   0.01705   0.01062         0         0      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.869      0.559      0.668      0.336

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     60/63     4.89G   0.01723   0.01052         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.886       0.55      0.662      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     61/63     4.89G   0.01714   0.01035         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.828      0.569      0.656      0.321

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     62/63     4.89G   0.01702   0.01056         0         1      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.854      0.566      0.657      0.319

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     63/63     4.89G   0.01668   0.01022         0         0      1280: 100%|██████████| 213/213 [06:14<00:00,  1.76s/it]                                                                                     
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:23<00:00,  1.36s/it]                                                                       
                 all        677       2449      0.866      0.569      0.672       0.33

64 epochs completed in 7.044 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 6.9MB
Optimizer stripped from runs/train/exp/weights/best.pt, 6.9MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model Summary: 280 layers, 3087256 parameters, 0 gradients, 4.2 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 17/17 [00:25<00:00,  1.49s/it]                                                                       
                 all        677       2449      0.878      0.556      0.668      0.336
Results saved to runs/train/exp
