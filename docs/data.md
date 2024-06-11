## Data Access
#### Hugging Face
We host data for all three tasks: VPT-basic, VPT-strategy, and depth order on Hugging Face.

https://huggingface.co/datasets/pzhou10/3D-PC
```python
from datasets import load_dataset
# config_name: one of ["vpt-basic", "vpt-strategy", "depth"]
dataset = load_dataset("pzhou10/3D-PC", "vpt-basic")
```
#### Download the full dataset
We also release the complete 3D-PC dataset along with data splits for training and testing.

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