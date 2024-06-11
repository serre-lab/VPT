
<div style="text-align:center" class="latex-font">
    <h1 style="text-align: center; font-weight: bold; color: inherit; margin-bottom: 0.2em"> Visual perspective taking in humans and machines </h1>

    <span class="author" style="font-weight: bold"> Drew Linsley*<sup>1</sup>, Peisen Zhou*<sup>1</sup>, Alekh Karkada*<sup>1</sup>, Akash Nagaraj<sup>1</sup>, <br> Gaurav Gaonkar<sup>1</sup>, Francis E Lewis<sup>1</sup>, Zygmunt Pizlo<sup>2</sup>, Thomas Serre<sup>1</sup></span> <br>
    <span class="affiliations"> <sup>1</sup>Carney Institute for Brain Science, Brown University, Providence, RI 02912 </span> <br>
    <span class="affiliations> <sup>2</sup>Department of Cognitive Sciences, University of California-Irvine, Irvine, CA. </span> <br>
    <span class="mono"> drew_linsley@brown.edu </span>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2406.04138"><strong>Read the official paper »</strong></a>
  <br>
  <br>
  <a href="https://huggingface.co/datasets/pzhou10/3D-PC/">Data</a>
  ·
  <a href="https://github.com/serre-lab/VPT">Github</a>
</p>

<div>
    <a href="#">
        <img src="https://img.shields.io/badge/License-CC--BY--4.0-efefef">
    </a>
</div>

<img src="./assets/examples.png" width="90%" align="center">

## Abstract


Visual perspective taking (VPT), the ability to accurately perceive and reason about the perspectives of others, is an essential feature of human intelligence. 
VPT is a byproduct of capabilities for analyzing 3-Dimensional (3D) scenes, which develop over the first decade of life. 
Deep neural networks (DNNs) may be a good candidate for modeling VPT and 
its computational demands in light of a growing number of reports indicating that DNNs gain the ability to analyze 3D scenes after training on large static-image datasets. 
Here, we investigated this possibility by developing the 3D perception challenge (3D-PC) for comparing 3D perceptual capabilities in humans and DNNs. 
We tested over 30 human participants and _'linearly probed'_ or text-prompted over 300 DNNs on several 3D-analysis tasks posed within natural scene images: 
(i.) a simple test of object depth order, (ii.) a basic VPT task (VPT-basic), and (iii.) a more challenging version of VPT designed to limit the effectiveness of _'shortcut'_ visual strategies. 
Nearly all of the DNNs approached or exceeded human accuracy in analyzing object depth order, and surprisingly, their accuracy on this task correlated with their object recognition performance. 
In contrast, there was an extraordinary gap between DNNs and humans on VPT-basic: humans were nearly perfect, whereas most DNNs were near chance. 
Fine-tuning DNNs on VPT-basic brought them close to human performance, but they, unlike humans, dropped back to chance when tested on VPT-perturbation. 
Our challenge demonstrates that the training routines and architectures of today's DNN are well-suited for learning basic 3D properties of scenes and objects, 
but not for reasoning about these properties in the way that humans rely on for their everyday lives.
We release our _3D-PC_ datasets and code to help bridge this gap in 3D perception between humans and machines.


## 🗞️ Citation

If you use or build on our work as part of your workflow in a scientific publication, please consider citing the [official paper](https://arxiv.org/abs/2406.04138):

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
## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/cc-by-4.0/"> CC-BY-4.0 license</a>.
