# Decision Fusion-based Ensemble Deep Network For Hybrid Scene Recognition



## Proposed Method ##

Multiple feature extractors significantly meliorate the accuracy compared with a single model, thus various FC features are extracted using various off-the-shelf pre-trained DCNNs in phase I of the proposed algorithm (See Fig. 1). In phase II, the extracted highly abstract features are further processed by sets of classifiers to obtain the corresponding probabilistic confidences. In phase III, kernel operation is applied to fuse the prediction, and classification results are obtained. In this section, we will illustrate the model phase by phase. 

![Overview Model2 (1)](https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/Overview%20Model2%20(1).jpg)

​													**Figure 1.  Overview of the Proposed FEDNet** 

### Phase I

In this method, we use ImageNet pretrained weights for ResNet[1] and DenseNet [2]. Besides, AlexNet [3], VGG16 [4] and GoogLeNet [5] pretrained by Places365 large-scale scene context dataset are also included.

As a classic sequential neural network, AlexNet has strong ability for image recognition. VGG16 is capable of improving performances by continuously deepening the network structure. Deep residual networks, ResNet and DenseNet, are also involved. They solve the degradation problem that appears when the number of network layers is increasing. Eventually, we use the inception model, GoogLeNet [6], it reduces the computational burden of deep neural networks.

We selects those pretained DCNNs as multiple feature extractors to obtain effective deep features, and they are divided into two groups. Group one is VGG16, AlexNet, and GoogLeNet which is pretrained by Places365. It concentrates on scene-related representation and considers the relevance of various spatial objects. Besides, Group two is ResNet and DenseNet pretrained by ImageNet. It focus on object-centered feature effectively. To conclude, the deep features which are retrieved from those two groups of pretained DCNNs are complementary to capture both global and local representation.

![Deep Feature Extraction](/Users/wangxiang/Code/Github_Repository/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/Deep Feature Extraction.jpg)

​															**Figure 2. Layer Replacement** 

As shown in Fig. 3, these pretrained models as feature extractors are slightly modified. One additional FC layer with the dimension of 1000 is attached after the penultimate layer of these DCNN models respectively. According to the number of classes of pretrained dataset, the pretrained models have their default output sizes at the final FC layer. Thus, the output FC layer is replaced by the new FC layer containing our desire number of neurons (i.e. 397 neurons in SUN397).

### Phase II

The size-standardized FC features can be fused directly, which is regarded as an early fusion approach using a single-classifier. However, multi-classifiers surpass a single as mentioned in Section II. Therefore, this proposed FEDNet further propagates the individual high-level features into multi-classifiers. FEDNet extracts two groups of feature, first group is holistic related obtained by using Place365 pretrained weights, while another set of features is the local descriptors representing object-level information from ImageNet pretrained models. 

![Prediction-Fusion2](https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/Prediction-Fusion2.jpg)

The pool of top-level features forward propagate into classifiers. Two main kinds of classifiers can be implemented, Back-Propagated (BP) based and non-back-propagated based classifier. These supervised algorithms learn latent patterns to best represent the input data, and construct the linearly or non-linearly hyperplanes to categorize the data. In our FEDNet, we introduce multi-identical classifiers in which all five classifiers are the same kind rather than multi-diverse classifier. In fact, multi-identical classifiers perform categorization on either spatial-level or object-level features. Due to the diverse classification among classifiers, it is expected that prediction results are probably cross complemented against each other, resulting in ameliorating the classification performance. Four classifiers are tested, we chosen Multilayer Perceptron (MLP), Support Vector Machine (SVM), K-Nearest Neighbours(KNN), and Decision Tree(DT).

### Phase III

Similar to DCNNs, the probabilistic confidence is calculated at the final FC layer. All five sets of scores represent the prediction result of the input data. Then we perform the decision-level-fusion, specifically probabilistic-wise decision fusion, where the decision vector of a single $i^{th}$ classifier can be denoted as $[d_{i1},d_{i2},...d_{iC}]$ if there are total $C$ classes. 

Contributed by five classifiers, the posterior fused prediction is computed. The $\hat{d_j}$ is ranged between 0 and 1 reflecting the probability of image $x$ belonging to $C$ total classes. The highest value of likelihood is naturally classified as a result. In Alogrithm 1, the number of DCNNs and classifiers are predefined. We also set the desired dimension of deep features as 1000. It starts with phase I, five sets of deep features are extracted in the for-loop with five iterations. In the phase II, five classifiers are used to perform the classification individually. In the final phase, the probabilies are fused using a kernel, we can finalize and evaluate the result. 

## Experiments

#### Datasets

We conducted experiments on public large-scale datasets, SUN397, Scene15, MIT67. 

1. SUN397: It contains 108,754 images, with 397 indoor and outdoor scene categories in total. Each category contains at least 100 images. In our experiment, we randomly select 50 images in each category for training, and another 50 images for testing.
2. Scene15: There are a total of 4485 grayscale images, including indoor and outdoor 15 scene categories. Each category contains 200-400 images. We randomly select 100 images as the training set, and all the remaining images are used for testing
3. MIT67: This dataset contains 15620 images. It is worth mentioning that this dataset is all indoor scene images. Each indoor scene category contains at least 100 images, 80 of which are used for training, and the remaining 20 images are used for testing.

![dataset2](https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/dataset2.jpg)

​										**Fig. 3. Sample Images of SUN397 Dataset**



#### Comparisons with state-of-the-Art Methods

In this experiment, we measure FEDNet mainly on hybrid scene datasets ( SUN397 \& Scene15 ), but we also include the additional indoor dataset (MIT67) to further analyzed our model. Our final obtained accuracies of each dataset are: 76.18$\%$, 95.01$\%$ and 84.78$\%$. Experimental results can prove that FEDNet has a highly exceptional capacity to deal with scene recognition task, moreover it indicate FEDNet has a strong generalization ability.

In Table II, the best and second best results are presented in boldface font and underlined correspondingly. Compared with other DCNN methods, FEDNet achieves excellent performances on the large-scale hybrid scene dataset, SUN397. The testing accuracy of our method is the highest (76.18$\%$) over both single and multi model groups, followed by the second method, PatchNets (73$\%$). For another hybrid scene benchmark, Scene15, FEDNet still reaches comparable performances (Second best among all methods). Results have proved that FEDNet undoubtedly achieves higher accuracy than human expert. With regard to additional MIT67, our model generally outperforms other single model methods, it however achieves slightly low accuracy by comparing with other multi models. 

![compar](https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/compar.png)

​					**Table 1. Performance comparison of SUN397, MIT67 And Scene15**

## Conclusion

To conclude, scene recognition is an important research in the field of computer vision. In order to sketch both global and local representation from scene datasets, we proposed new network, FEDNet. Multi-feature extractors along with multi-identical classifiers are completely able to encode and categorise images. Instead of fusing the deep features from DCNNs and feeding into single-classifier, we observe that multi-identical classifiers precede single-classifier. From experiments, FEDNet makes a success in hybrid scene datasets, and it has a stable performance, compared with existing scene recognition methods. Inspired by the success of FEDNet using multi-identical classifiers, future research work can focus on developing the ensemble network with multi-diverse classifiers, the performance is expected to be further improved. 
<<<<<<< HEAD



https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/Deep%20Feature%20Extraction.jpg



https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/Overview%20Model2%20(1).jpg



https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/Prediction-Fusion.jpg



https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/Prediction-Fusion2.jpg



https://github.com/KennethXiang/Decision-Fusion-based-Ensemble-Deep-Network-For-Hybird-Scene-Recognition/blob/master/dataset2.jpg

=======
>>>>>>> 59f21e0ef988c2ff2fe708f650a40fb9258a181b
