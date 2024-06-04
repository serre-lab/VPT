

<div style="text-align:center" class="latex-font">
    <h1 style="text-align: center; font-weight: bold; color: inherit; margin-bottom: 0.2em"> Visual perspective taking in humans and machines </h1>

    <span class="author" style="font-weight: bold"> Drew Linsley*, Peisen Zhou*, Alekh Karkada*, Akash Nagaraj, <br> Gaurav Gaonkar, Francis E Lewis, Zygmunt Pizlo, Thomas Serre</span> <br>
    <span class="affiliations"> Carney Institute for Brain Science, Brown University, Providence, RI 02912 </span> <br>
    <span class="mono"> drew_linsley@brown.edu </span>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2211.04533"><strong>Read the official paper »</strong></a>
  <br>
  <br>
  <a href="https://serre-lab.github.io/Harmonization/results">Explore results</a>
  ·
  <a href="https://github.com/serre-lab/VPT">Github</a>
</p>

<div>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>


## Paper summary

<img src="./assets/examples.png" width="100%" align="center">

Visual perspective taking (VPT), the ability to accurately perceive and reason about the perspectives of others, is an essential feature of human intelligence. VPT is a byproduct of capabilities for analyzing 3-Dimensional (3D) scenes, which develop over the first decade of life. Deep neural networks (DNNs) may be a good candidate for modeling VPT and its computational demands in light of a growing number of reports indicating that DNNs gain the ability to analyze 3D scenes after training on large static-image datasets. Here, we investigated this possibility by developing the 3D perception challenge (3D-PC) for comparing 3D perceptual capabilities in humans and DNNs. We tested over 30 human participants and _'linearly probed'_ or text-prompted over 300 DNNs on several 3D-analysis tasks posed within natural scene images: (i.) a simple test of object depth order, (ii.) a basic VPT task (VPT-basic), and (iii.) a more challenging version of VPT designed to limit the effectiveness of _'shortcut'_ visual strategies. Nearly all of the DNNs approached or exceeded human accuracy in analyzing object depth order, and surprisingly, their accuracy on this task correlated with their object recognition performance. In contrast, there was an extraordinary gap between DNNs and humans on VPT-basic: humans were nearly perfect, whereas most DNNs were near chance. Fine-tuning DNNs on VPT-basic brought them close to human performance, but they, unlike humans, dropped back to chance when tested on VPT-perturbation. Our challenge demonstrates that the training routines and architectures of today's DNN are well-suited for learning basic 3D properties of scenes and objects, but not for reasoning about these properties in the way that humans rely on for their everyday lives. We release our _3D-PC_ datasets and code to help bridge this gap in 3D perception between humans and machines.


<!-- ## Authors

<p align="center">

<div class="authors-container">

  <div class="author-block">
    <img src="./assets/thomas.png" width="2%" align="center">
    <a href="mailto:thomas_fel@brown.edu"> Drew Linsley* </a>
  </div>


  <div class="author-block">
    <img src="./assets/ivan.png" width="25%" align="center">
    <a href="mailto:ivan_felipe_rodriguez@brown.edu"> Peisen Zhou* </a>
  </div>


  <div class="author-block">
    <img src="./assets/drew.png" width="25%" align="center">
    <a href="mailto:drew_linsley@brown.edu"> Alekh Karakada* </a>
  </div>

  <div class="author-block">
    <img src="./assets/tserre.png" width="25%" align="center">
    <a href="mailto:thomas_serre@brown.edu"> Akash Nagaraj </a>
  </div>

  <div class="author-block">
    <img src="./assets/ivan.png" width="25%" align="center">
    <a href="mailto:ivan_felipe_rodriguez@brown.edu"> Gaurav Gaonkar </a>
  </div>


  <div class="author-block">
    <img src="./assets/drew.png" width="25%" align="center">
    <a href="mailto:drew_linsley@brown.edu"> Francis E Lewis </a>
  </div>

  <div class="author-block">
    <img src="./assets/tserre.png" width="25%" align="center">
    <a href="mailto:thomas_serre@brown.edu"> Zygmunt Pizlo</a>
  </div>

  <div class="author-block">
    <img src="./assets/tserre.png" width="25%" align="center">
    <a href="mailto:thomas_serre@brown.edu"> Thomas Serre</a>
  </div>

</div>

<br>
<p align="right"> 
<i> * : all authors have contributed equally. </i>
</p>

</p> -->


<!-- ## 🗞️ Citation

If you use or build on our work as part of your workflow in a scientific publication, please consider citing the [official paper](https://arxiv.org/abs/2211.04533):

```
@article{fel2022aligning,
  title={Harmonizing the object recognition strategies of deep neural networks with humans},
  author={Fel, Thomas and Felipe, Ivan and Linsley, Drew and Serre, Thomas},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

Moreover, this paper relies heavily on previous work from the Lab, notably [Learning What and Where to Attend](https://arxiv.org/abs/1805.08819) where the ambitious ClickMe dataset was collected.

```
@article{linsley2018learning,
  title={Learning what and where to attend},
  author={Linsley, Drew and Shiebler, Dan and Eberhardt, Sven and Serre, Thomas},
  journal={International Conference on Learning Representations (ICLR)},
  year={2019}
}
```

## Tutorials

**Evaluate your own model (pytorch and tensorflow)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mp0vxUcIsX1QY-_Byo1LU2IRVcqu7gUl) 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/230px-Tensorflow_logo.svg.png" width=35>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bttp-hVnV_agJGhwdRRW6yUBbf-eImRN) 
<img src="https://pytorch.org/assets/images/pytorch-logo.png" width=35> -->


## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
