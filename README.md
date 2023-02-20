# COMPASS
Context and Motion Primitive Aggregate Surgical Set

### Context Labeling App 
Contains code and instructions for labeling videos.

<img src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/poap_app_2.png" width="50%">

### Context Labels
Contains labels for each trial organized by Labeler id.

### Datasets
Contains kinematic and video data organized by task and trial.
Includes the Suturing (S), Needle Passing (NP), and Knot Tying (KT) tasks from the JIGSAWS dataset, Peg Transfer (PT) from the DESK dataset, and Pea on a Peg (PoaP) and Post and Sleeve (PaS) from the ROSMA dataset.


<p align="middle" float="left">
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/suturing_frame.png" alt="Suturing" width="25%"/>
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/needle_passing_frame.png" alt="Needle Passing" width="25%"/>
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/knot_tying_frame.png" alt="Knot Tying" width="25%"/>
</p>

<p align="middle" float="left">
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/PT_frame.png" alt="Peg Transfer" width="25%"/>
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/PoaP_frame.png" alt="Pea on a Peg" width="25%"/>
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/PaS_frame.png" alt="Post and Sleeve" width="25%"/>
<\p>



### TCN
Contains code and instructions for running ML models; and eventually our trained models.

### Translation Scripts
Contains scripts for converting context labels to motion primitives, and motion primitives to gestures (for trials that came from datasets that had original gesture or surgeme labels).

