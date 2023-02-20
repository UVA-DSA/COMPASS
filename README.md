# COMPASS
Context and Motion Primitive Aggregate Surgical Set

### Context Labeling App 
Contains code and instructions for labeling videos.

<img align="middle" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/poap_app_2.png" width="50%">

### Context Labels
Contains labels for each trial organized by Labeler id.

### Datasets
Contains kinematic and video data organized by task and trial.
Includes the Suturing (S), Needle Passing (NP), and Knot Tying (KT) tasks from the JIGSAWS dataset [[2]](#2), Peg Transfer (PT) from the DESK dataset [[3]](#3), and Pea on a Peg (PoaP) and Post and Sleeve (PaS) from the ROSMA dataset [[4]](#4).


<p align="middle" float="left">
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/suturing_frame.png" alt="Suturing" title="Suturing" width="25%"/>
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/needle_passing_frame.png" alt="Needle Passing" title="Needle Passing" width="25%"/>
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/knot_tying_frame.png" alt="Knot Tying" title="Knot Tying" width="25%"/>
</p>

<p align="middle" float="left">
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/PT_frame.png" alt="Peg Transfer" title="Peg Transfer" width="25%"/>
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/PoaP_frame.png" alt="Pea on a Peg" title="Pea on a Peg" width="25%"/>
  <img align="top" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/PaS_frame.png" alt="Post and Sleeve" title="Post and Sleeve" width="25%"/>
</p>



### TCN
Contains code and instructions for running ML models; and eventually our trained models.

### Translation Scripts
Contains scripts for converting context labels to motion primitives, and motion primitives to gestures (for trials that came from datasets that had original gesture or surgeme labels).



### References
<a id="1">[1]</a>
Hutchinson, K., Reyes, I., Li, Z., & Alemzadeh, H. (2022). COMPASS: A Formal Framework and Aggregate Dataset for Generalized Surgical Procedure Modeling. arXiv preprint arXiv:2209.06424.

<a id="2">[2]</a> 
Gao, Y., Vedula, S. S., Reiley, C. E., Ahmidi, N., Varadarajan, B., Lin, H. C., ... & Hager, G. D. (2014, September). Jhu-isi gesture and skill assessment working set (jigsaws): A surgical activity dataset for human motion modeling. In MICCAI workshop: M2cai (Vol. 3, No. 3).

<a id="3">[3]</a>
Gonzalez, G. T., Kaur, U., Rahma, M., Venkatesh, V., Sanchez, N., Hager, G., ... & Wachs, J. (2020). From the DESK (Dexterous Surgical Skill) to the Battlefield--A Robotics Exploratory Study. arXiv preprint arXiv:2011.15100.

<a id="4">[4]</a>
Rivas-Blanco, I., Pérez-del-Pulgar, C. J., Mariani, A., Quaglia, C., Tortora, G., Menciassi, A., & Muñoz, V. F. (2021). A surgical dataset from the da Vinci Research Kit for task automation and recognition. arXiv preprint arXiv:2102.03643.

