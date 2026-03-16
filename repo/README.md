Wells Hull
	HW2

We are given two datasets for this assignment and two models as well. The Penn-Fudan dataset is a single class detection benchmark containing 170 images of pedestrians photographed in urban environments across Penn and Fudan University campuses. This dataset presents some challenges such as partial occlusion and variable pedestrian scale. All images were resized to 512x512 pixels. The full Oxford-IIIT Pet dataset contains 37 cat and dog breeds across 7,349 images. A 10-breed subset was made containing Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, and Siamese. All these are capped at 50 images per breed for a total of around 500 images. Images were resized to 512x512 and split 70/15/15. 

Faster R-CNN 
Faster R-CNN is a two-stage detector. A Region Proposal Network (RPN) first generates candidate object regions, which are then classified into and refined in a second stage. The original VGG16 backbone is replace here with MobilityNetV3-Large combined with a Feature Pyramid Network (FPN), reducing memory consumption while retaining multi-scale feature extraction. 

Yolov8n (Nano)
YOLOv8 is a single-stage detector from Ultralytics that predicts bounding boxes and class probabilities. It has about 3.2 M parameters making it very fast and great for low memory GPUS. It simplifies the training objective and removes the need to tune anchor hyperparameters.

Results
I wont be able to include all the pictures of the data because it would clutter the report so I will just talk about the findings. 

YOLOv8n on Penn-Fudan — Precision-Recall & Confusion Matrix
The PR curve shows AP=0.882 with precision staying near 1.0 until recall reaches about 0.4 then declining. This shows the model lights up confidently on pedestrians it detects, but misses a large proportion. 48 out of 70 to be exact. This makes it a 69% miss rate per the normalised confusion matrix. This high precision- low recall pattern is the opposite of Faster R-CNN, which hit a 96.4% recall with 10 false positives. 

YOLOv8n on Oxford Pet — F1 Confidence Curve
The F1 confidence shows that the optimal operating threshold for YOLOv8n on the pet dataset is 0.139, having a mean F1 of 0.46. The low optimal confidence threshold reflects the models uncertainty on difficult breeds. Well performing breeds like the Siamese, Bombay, and British Shorthair all had a high F1 at much higher thresholds. 




Faster R-CNN on Penn-Fudan

The model was trained for 10 epochs. Training loss decreased sharply from 0.461 to 0.170. This shows rapid adaptation of the classification to the pedestrian detection task. Training and validation loss curves for Faster R-CNN on Penn Fudan over 10 epochs. Validation loss drifts upward from epoch 2 onward, showing that the model peaked early on this dataset. 


YOLOv8n on Penn-Fudan — Training Curves

YOLOv8n was trained on Penn-Fudan for up to 15 epochs with patience=5 early stopping. The model triggered early stopping at epoch 7. Training losses decrease steadily. Val metrics are noisy due to the tiny validation set of about 26 images. 

 Faster R-CNN on Oxford Pet — Training Curves

Faster R-CNN was trained for 12 epochs on 10 breed Oxford Pet subset. Training loss declined steadily from 1.254 to 0.240, which shows ongoing learning throughout. Validation loss was noisiest in the first 4 epochs, then stabilised at approximately 0.91 from epoch 6 onward before slowly moving to 0.966 by epoch 11. This indicated overfitting. 


Comparison Table


Findings

Penn-Fudan: Faster R-CNN wins clearly with 90% vs 79.3%. Its two-stage RPN recovers nearly every pedestrian with a 96.4% accuracy, missing only 2 of 56. YOLOv8n is still very precise with 94.2%, but missing 6% of pedestrians. Its conservative single-stage head under fires on small targets. 

Oxford Pet: YOLOv8n wins 63.98% vs 46%. With only about 35 images per breed, YOLOv8n’s decoupled classification head generalizes better than Faster R-CNN’s two stage pipeline, which overfits badly.

Speed: YOLOv8n is about 3.4x faster across both datasets.

Convergence: Both models converge within 2-7 epochs on the Penn-Fudan set. On the Oxford Pet task, YOLOv8n showed no change at epoch 18, which could mean more training or data per class would improve results. 

Precision vs Recall trade-off: On Penn-Fudan, Faster R-CNN prioritizes recall while YOLOv8n prioritizes precision. For safety critical pedestrian detection, Faster R-CNN high recall behavior is better. 


