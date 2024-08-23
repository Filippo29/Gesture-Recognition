# RoboCup 2024 Referee Recognition
The repository contains the entire code developed to recognize the Ready pose made by the referee during the Standby phase of the game. The successful recognition of the pose is essential in order to get all the robots well positioned before the game starts.
The architecture has the following structure:
![Architecture](https://i.imgur.com/gdB0Bmb.png)

## Models
This is the part contained inside the *models/* path and is made of two architectures trained to classify the pose extracted by [Movenet](https://www.tensorflow.org/hub/tutorials/movenet?hl=en) lightning.

First of all, given the limited hardware capabilities of the Nao robots, I chose the Movenet Lightning model (quantized to uint8) for extracting key points. While there are various architectures available for this purpose, with Yolo being one of the most popular, I required a model optimized for real-time applications. This is why I selected Movenet Lightning.
Inference time on Nao robots:
| Model | Run time (ms) |
|:---:|:---:|
| Yolo v6 | > 1000 |
| Movenet Lightning | **180** |

Speaking of the classification, I trained two similar models using different features:
 - The simplest approach for this problem is to directly classify the key points coordinates. Even if movenet extracts 17 different kp not all of them are useful for our classification, in fact we can directly consider only the coordinates of hips, shoulders, elbows and wrists. Since we are using only 8 points the classifier has a total of 16 inputs.
 - Despite a good accuracy has been obtained with the first method, a better approach is to calculate angles of the shoulders and elbows joints. In this case, the same key points are used to extract a different and stronger feature. Given three points A, B and C (for example A: shoulder; B: elbow; C: wrist) the angle is calculated with 
$\alpha=atan2(BC_y,BC_x)-atan2(BA_y,BA_x)$
Eventually we have a smaller input space from which it is much easier to capture patterns.
Here are the results of the two architectures:

| Feature | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| Key Points | 0.96 | 0.96 | 0.96 | 0.96 |
| Angles | 0.99 | 0.99 | 0.99 | 0.99 |

As shown in the table, the results were very high on the test set for both approaches. However, the key points model shown a worse generalization in the real usage. The main problem with this approach is that this model is not scale invariant, especially when trained on a limited amount of data. On the other hand, the model that harnesses angles as a feature has performed much better during the competition games as well.
In conclusion, the angle based architecture demonstrates clear advantages across all the points of view. It not only performs more effectively in real world scenarios but also achieves high quality results with fewer samples compared to the other approaches.

It is possible to test the architecture by running the *pose_visualization.py* script which can also be used to collect a dataset without labels.
## Nao implementation
Up to now our Nao framework is based on the [B-Human](https://b-human.de/index.html) 2021 release. The developed neural networks are imported with the [OnnxRuntime](https://onnxruntime.ai) library.
The main part of the code is divided in:
 - Representations: used to store variables content that has to be accessed by multiple modules;
 - Modules: responsible of doing every kind of operation relying on the data stored inside some representations.

When the game is in progress, during the standby phase, the robots are positioned on the field side lines waiting for a packet from the game controller indicating that they can take their place inside the field. However, a new rule for the RoboCup 2024 is that the head referee has to communicate through a pose that the robots can enter the field. If the recognition isn't succesful the robots will still wait for the game controller packet which will arrive 30 seconds later causing a positional disadvantage.

In the standby phase of the game, the behavior of the robots is managed by *StandbyCard.cpp* and it simply consists in looking at the referee position waiting for the pose recognition.

The recognition pipeline is implemented inside the *RefereeEstimatorProvider.cpp* file with all the variables defined in the related *RefereeEstimatorProvider.h*.
The function
`void RefereeEstimatorProvider::update(RefereeEstimator& estimator)`
performs some steps in this order:

 1. Convert the YUV image to RGB;
 2. Crop a rectangular ROI before scaling the image down. Doing the opposite would result in a big quality loss. The ROI has $width=width_0/4$ and $height=height_0/2$ and is positioned at $x=3*width_0/8$ and $y=-30+height_0/2$;
 3. The cropped image is resized while preserving the aspect ratio, so that its largest dimension becomes 192 pixels. Then, padding is added to the other dimension to make the final image size 192x192 pixels, which matches the input size required by Movenet Lightning;
 4. Now the Movenet input tensor is filled with the flattened image and the angles are calculated with the output;
 5. At this point, if at least 3 angles have been calculated (setting to -1.0 all the unseen angles), the classification is performed on them and the number of consecutive positive results is stored in `RefereeEstimator.measures`. If there are 4 consecutive positive readings the variable `RefereeEstimator.isDetected` is set to True and the time of last detection is stored. To succesfully calculate an angle, all the key points needed for it have to be calculated with a confidence greater or equal then a threshold set in the `confidence_threshold` variable in the file `RefereeEstimatorProvider.h`.

Every step involving the manipulation of images is implemented using [OpenCV](https://opencv.org/) structures and related functions since these operations are quite expensive and the hardware is limited.

The module `GameStateHandler` is then responsible of updating the current game state. For the referee gesture I've implemented the function `bool checkForReadyGesture()` which returns true either if the referee ready pose has been detected by the current robot or by a teammate. If the result of this function is True, the current game state is set to Ready and the robots can enter the field.
For this purpose, I've added the `RefereeEstimator` representation to the packet sent by the robots which is managed in the `TeamMessageHandler` module.

Additionally, the *RefereeEstimatorProvider* module uses debug drawings to show in real time how good are the key points extracted. This part is implemented in the `void drawKeypoints(cv::Rect ROI)` function.

# License
The files in the *models/* directory are covered by the license specified in the LICENSE file. For the files in the *spqrnao2024* directory, they are generally under the [B-Human license](https://github.com/bhuman/BHumanCodeRelease/blob/master/License.md), except for the files *RefereeEstimator.h*, *RefereeEstimatorProvider.h*, *RefereeEstimatorProvider.cpp*, and *StandbyCard.cpp* which are governed by the license specified in the LICENSE file.