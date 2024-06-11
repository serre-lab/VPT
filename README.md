<div align="center">

# Visual perspective taking in humans and machines

_Drew Linsley<sup>1</sup>, Peisen Zhou<sup>1</sup>, Alekh Karkada<sup>1</sup>, Akash Nagaraj<sup>1</sup>, Gaurav Gaonkar<sup>1</sup>, Francis E Lewis<sup>1</sup>, Zygmunt Pizio<sup>2</sup>, Thomas Serre<sup>1</sup>_
<p><sup>1</sup>Carney Institute for Brain Science, Brown University, Providence, RI.</p>
<p><sup>2</sup>Department of Cognitive Sciences, University of California-Irvine, Irvine, CA.</p>

### [Project Page](https://serre-lab.github.io/VPT/) · [Paper](https://arxiv.org/abs/2406.04138) · [Data](https://huggingface.co/datasets/pzhou10/3D-PC)

</div>

## Paper Abstract
Visual perspective taking (VPT), the ability to accurately perceive and reason about the perspectives of others, is an essential feature of human intelligence. VPT is a byproduct of capabilities for analyzing 3-Dimensional (3D) scenes, which develop over the first decade of life. Deep neural networks (DNNs) may be a good candidate for modeling VPT and its computational demands in light of a growing number of reports indicating that DNNs gain the ability to analyze 3D scenes after training on large static-image datasets. Here, we investigated this possibility by developing the _3D perception challenge_ (<tt>3D-PC</tt>) for comparing 3D perceptual capabilities in humans and DNNs. We tested over 30 human participants and ''linearly probed'' or text-prompted over 300 DNNs on several 3D-analysis tasks posed within natural scene images: (_i._) a simple test of object _depth order_, (_ii._) a basic VPT task (_VPT-basic_), and (_iii._) a more challenging version of VPT designed to limit the effectiveness of ''shortcut'' visual strategies. Nearly all of the DNNs approached or exceeded human accuracy in analyzing object _depth order_, and surprisingly, their accuracy on this task correlated with their object recognition performance. In contrast, there was an extraordinary gap between DNNs and humans on _VPT-basic_: humans were nearly perfect, whereas most DNNs were near chance. Fine-tuning DNNs on _VPT-basic_ brought them close to human performance, but they, unlike humans, dropped back to chance when tested on _VPT-perturbation_. Our challenge demonstrates that the training routines and architectures of today's DNN are well-suited for learning basic 3D properties of scenes and objects, but not for reasoning about these properties in the way that humans rely on for their everyday lives. We release our <tt>3D-PC</tt> datasets and code to help bridge this gap in 3D perception between humans and machines.

<img src="docs/assets/examples.png" width=80%>

## Data Access
#### Hugging Face
We release data for all three tasks: VPT-basic, VPT-strategy, and depth order on Hugging Face.

https://huggingface.co/datasets/pzhou10/3D-PC
```python
from datasets import load_dataset
# config_name: one of ["vpt-basic", "vpt-strategy", "depth"]
dataset = load_dataset("pzhou10/3D-PC", "vpt-basic")
```
#### Download the full dataset
We release the complete 3D-PC dataset along with data splits for training and testing.

https://connectomics.clps.brown.edu/tf_records/VPT/

#### Dataset Content
`train` contains all training images organized by categories. 
```
train
|
|_<category>
|  |_<object>
|    |_<setting>
|      |_<*.png>
```
The corresponding labels are `train_perspective.csv` and `depth_perspective.csv`. We also provide `train_perspective_balanced.csv` and `depth_perspective_balanced.csv`, where the numbers of positive and negative samples are equal.

`perspective` and `depth` contain all data splits for 'VPT' and 'depth order' tasks.
```
perspective/depth
|
|_<split>
|  |_<category> 0/1
|    |_<*.png>
```

## TIMM Evaluation
To linear probe a timm model
```
python run_linear_probe.py --task <task> --data_dir <data_folder>/<task>/ --model_name <model_name>
```

To fine-tune a timm model
```
python run_finetune.py --task <task> --data_dir <data_folder>/<task>/ --model_name <model_name>
```

`data_folder`: Root directory for the dataset

`task`: Either `perspective` or `depth`

`model_name`: TIMM model name

## Citation
```
@misc{linsley20243dpc,
      title={The 3D-PC: a benchmark for visual perspective taking in humans and machines}, 
      author={Drew Linsley and Peisen Zhou and Alekh Karkada Ashok and Akash Nagaraj and Gaurav Gaonkar and Francis E Lewis and Zygmunt Pizlo and Thomas Serre},
      year={2024},
      eprint={2406.04138},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

