<div align="center">

# Visual perspective taking in humans and machines

_Drew Linsley, Peisen Zhou, Alekh Karkada, Akash Nagaraj, Gaurav Gaonkar, Francis E Lewis, Zygmunt Pizio, Thomas Serre_
<p>Carney Institute for Brain Science, Brown University</p>

### [Projectpage](https://serre-lab.github.io/VPT/) · [Paper]

</div>

## Paper Abstract
Visual perspective taking (VPT), the ability to accurately perceive and reason about the perspectives of others, is an essential feature of human intelligence. VPT is a byproduct of capabilities for analyzing 3-Dimensional (3D) scenes, which develop over the first decade of life. Deep neural networks (DNNs) may be a good candidate for modeling VPT and its computational demands in light of a growing number of reports indicating that DNNs gain the ability to analyze 3D scenes after training on large static-image datasets. Here, we investigated this possibility by developing the _3D perception challenge_ (<tt>3D-PC</tt>) for comparing 3D perceptual capabilities in humans and DNNs. We tested over 30 human participants and ''linearly probed'' or text-prompted over 300 DNNs on several 3D-analysis tasks posed within natural scene images: (_i._) a simple test of object _depth order_, (_ii._) a basic VPT task (_VPT-basic_), and (_iii._) a more challenging version of VPT designed to limit the effectiveness of ''shortcut'' visual strategies. Nearly all of the DNNs approached or exceeded human accuracy in analyzing object _depth order_, and surprisingly, their accuracy on this task correlated with their object recognition performance. In contrast, there was an extraordinary gap between DNNs and humans on _VPT-basic_: humans were nearly perfect, whereas most DNNs were near chance. Fine-tuning DNNs on _VPT-basic_ brought them close to human performance, but they, unlike humans, dropped back to chance when tested on _VPT-perturbation_. Our challenge demonstrates that the training routines and architectures of today's DNN are well-suited for learning basic 3D properties of scenes and objects, but not for reasoning about these properties in the way that humans rely on for their everyday lives. We release our <tt>3D-PC</tt> datasets and code to help bridge this gap in 3D perception between humans and machines.

<img src="docs/assets/examples.png" width=80%>

## Data Access
We release the complete 3D-PC dataset 
#### Download full dataset
https://connectomics.clps.brown.edu/tf_records/VPT/
<details>
<summary>Data Structure</summary>
```
.
|
|
|_train
|
|
|_test
|
|
|_perspective
|
|
|_depth
```
</details>
