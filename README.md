# Learning Precise Temporal Point Event Detection with Misaligned Labels
**AAAI 2021**

<div style="text-align: justify">
This work addresses the problem of robustly learning precise temporal point event detection despite only having access to poorly aligned labels for training. While standard (cross entropy-based) methods work well in noise-free setting, they often fail when labels are unreliable since they attempt to strictly fit the annotations. A common solution to this drawback is to transform the point prediction problem into a distribution prediction problem. We show however that this approach raises several issues that negatively affect the robust learning of temporal localization. Thus, in an attempt to overcome these shortcomings, we introduce a simple and versatile training paradigm combining soft localization learning with counting-based sparsity regularization. In fact, unlike its counterparts, our approach allows to directly infer clear-cut point predictions in an end-to-end fashion while relaxing the reliance of the training on the exact position of labels. We achieve state-of-the-art performance against standard benchmarks in a number of challenging experiments (e.g., detection of instantaneous events in videos and music transcription) by simply replacing the original loss function with our novel alternative---without any additional fine-tuning.
</div>

---
### (Section 5.1) Golf Swing Sequencing
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Benchmark Code and Dataset]](https://github.com/wmcnally/golfdb) by McNally et al.

To run the **original code (CE)** with noise level (n) on split (s), (default: n=0, s=1):
```
python original_train.py $n $s
python original_eval.py $n $s
```

To run either the **classical one-sided-smoothing** (classic=True) or our **SoftLoc approach** (classic=False) with noise level (n) on split (s), (default: n=0, s=1, SoftLoc loss):
```
python soft_train.py $n $s $classic
python soft_eval.py $n $s $classic
```

The results are then save in .txt file in /results.

For the **causal** experiments, manually change the bidirectional argument in the EventDetect defintion to False (soft_train.py (l.69), soft_eval.py (l.92), original_train.py (l.42), and original_eval.py (l.80)).  

---
### (Section 5.2) Time Series Detection
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Benchmark Code]](https://github.com/mlds-lab/weakly_supervised_timeseries_detection) by Adams and Marlin

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Dataset Request]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4631252/) The dataset has to be requested from the authors of _puffMarker: A Multi-Sensor Approach for Pinpointing the Timing of First Lapse in Smoking Cessation_.

To run the experiment on split (s) with noise level (n):
```
python PuffDetection.py $loss_id $s $n $distribution_id
```
- **loss_id**: SoftLoc (loss_id=0), cross-entropy (loss_id=1), one-sided-smoothing (loss_id=2)

- **distribution_id**: normal (distribution_id=0), binary (distribution_id=1), skew normal (distribution_id=2)

---
### (Section 5.3) Piano Onset Experiment
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Benchmark Code]](https://github.com/tensorflow/magenta/tree/9885adef56d134763a89de5584f7aa18ca7d53b6) by Hawthorne et al.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Dataset Request]](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for MAPS Database.

To run the **SoftLoc** pipeline (to modify the noise level, change value in bash):
```
bash piepline.sh
```
Modify the oneSided variable in pipeline.sh in order to run the **one-sided smoothing benchmark**.

The project is structured as follows:

- pipeline.sh (Full pipeline)
- SoftNetworkModel.py (Tensorflow model with SoftLoc loss)
- main.py (Main script that runs the **training**)
- createDataset.py and google_create_dataset.py (**dataset** creation)
- infer.py and final_score.py (**inference**)
- config.py (Configuration file)

In addition, *subfolders* contains all utility functions used throughout the project.

---
### (Section 5.4) Drum Detection Experiment
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Benchmark Code]](https://github.com/SchroeterJulien/ICML-2019-Weakly-Supervised-Temporal-Localization-via-Occurrence-Count-Learning) by Schroeter et al.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Dataset Request]](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/drums.html) for IDMT-SMT-Drums.

In order to compute and add one random point to the heatmap (Figure 6), run the following line:
```
python multi_main.py
```
This will run a single run of the experiment with random softness and noise level.
- The results are computed and added to a results file in addSweep.py.
- The heatmap is generated with sweepVisualization.py
