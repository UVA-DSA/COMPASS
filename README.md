# COMPASS
Context and Motion Primitive Aggregate Surgical Set

### Context Labeling App 
Contains code and instructions for labeling videos.

<img align="middle" src="https://github.com/UVA-DSA/COMPASS/blob/main/Figures/poap_app_2.png" width="50%">

### Datasets
Contains kinematic and video data organized by task, subject, and trial.
Includes the Suturing (S), Needle Passing (NP), and Knot Tying (KT) tasks from the JIGSAWS dataset [[3]](#3), Peg Transfer (PT) from the DESK dataset [[4]](#4), and Pea on a Peg (PoaP) and Post and Sleeve (PaS) from the ROSMA dataset [[5]](#5).


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

| Motion Primitive  | Suturing          | Needle Passing    | Knot Tying        | Peg Transfer      | Pea on a Peg      | Post and Sleeve   | COMPASS           |
| ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Grasp             | 471               | 373               | 283               | 323               | 577               | 824               | 2851              |
| Release           | 441               | 365               | 247               | 313               | 556               | 776               | 2698              |
| Touch             | 518               | 330               | 135               | 539               | 1782              | 1598              | 4902              |
| Untouch           | 314               | 206               | 111               | 364               | 1261              | 1131              | 3387              |
| Pull              | 194               | 114               | 235               | 0                 | 525               | 0                 | 1068              |
| Push              | 179               | 119               | 0                 | 0                 | 2                 | 0                 | 300               |



### TCN
Contains code and instructions for running ML models; and eventually our trained models.

### Translation Scripts
Contains scripts for converting context labels to motion primitives.


### Papers
<a id="1">[1]</a>
Hutchinson, K., Reyes, I., Li, Z., & Alemzadeh, H. (2023). <a href="https://link.springer.com/article/10.1007/s11548-023-02922-1">COMPASS: a formal framework and aggregate dataset for generalized surgical procedure modeling.</a> _International Journal of Computer Assisted Radiology and Surgery_, 1-12. 
[<a href="https://arxiv.org/abs/2209.06424">arXiv</a>, <a href="https://kch4fk.github.io/papers/COMPASS_1_accepted.pdf">pdf</a>]

<a id="2">[2]</a>
Hutchinson, K., Reyes, I., Li, Z., & Alemzadeh, H. (2023). <a href="https://ieeexplore.ieee.org/abstract/document/10173628">Evaluating the Task Generalization of Temporal Convolutional Networks for Surgical Gesture and Motion Recognition using Kinematic Data.</a> _IEEE Robotics and Automation Letters_.
[<a href="https://arxiv.org/abs/2306.16577">arXiv</a>]

### Related References
<a id="3">[1]</a> 
Gao, Y., Vedula, S. S., Reiley, C. E., Ahmidi, N., Varadarajan, B., Lin, H. C., ... & Hager, G. D. (2014, September). Jhu-isi gesture and skill assessment working set (jigsaws): A surgical activity dataset for human motion modeling. In _MICCAI workshop: M2cai_ (Vol. 3, No. 3).

<a id="4">[2]</a>
Gonzalez, G. T., Kaur, U., Rahma, M., Venkatesh, V., Sanchez, N., Hager, G., ... & Wachs, J. (2020). From the DESK (Dexterous Surgical Skill) to the Battlefield--A Robotics Exploratory Study. _arXiv preprint arXiv:2011.15100_.

<a id="5">[3]</a>
Rivas-Blanco, I., Pérez-del-Pulgar, C. J., Mariani, A., Quaglia, C., Tortora, G., Menciassi, A., & Muñoz, V. F. (2021). A surgical dataset from the da Vinci Research Kit for task automation and recognition. _arXiv preprint arXiv:2102.03643_.

