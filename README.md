# CMCL6D
This repository is used to store some of the results from the paper "Multiple Modality Fusion for Object Pose Estimation: A Cross-layer Cross-modal Hybrid CNN Architecture". When the paper is published, the source code will be opened and a tutorial will be released.

## Results of CM&CL Module Based Segmentation Network

Fig 1: Visualization of Input and Output of the Segmentation Network.

![segResult](figs\segResult.png)



## Results of Improved Pose Predict Network

Table 1: Evaluation of 6D Pose (AUC) on the YCB-Video Dataset.Bold numbers are the best indicators.

| Objects               | PointFuion   AUC | PoseCNN+ICP   AUC | DenseFusion   AUC | MaskedFusion  AUC | FFB6D  AUC | Proposed Method AUC |
| --------------------- | ---------------- | ----------------- | ----------------- | ----------------- | ---------- | ------------------- |
| 002_master_chef_can   | 90.9             | 95.8              | 96.4              | 95.5              | 96.3       | **97.4**            |
| 003_checker_box       | 80.5             | 92.7              | 95.5              | 96.7              | 96.3       | **97.4**            |
| 004_sugar_box         | 90.4             | **98.2**          | 97.5              | **98.1**          | 97.6       | 98.0                |
| 005_tomato_soup_can   | 91.9             | 94.5              | 94.6              | **94.3**          | **95.6**   | 94.5                |
| 006_mustard_bottle    | 88.5             | 98.6              | 97.2              | **98.0**          | 97.8       | 97.4                |
| 007_tuna_fish_can     | 93.8             | 97.1              | 96.6              | 96.9              | 96.8       | **98.0**            |
| 008_pudding_box       | 87.5             | 97.9              | 96.5              | 97.3              | 97.1       | **98.3**            |
| 009_geltain_box       | 95.0             | **98.8**          | 98.1              | 98.3              | 98.1       | 98.6                |
| 010_potted_meat_can   | 86.4             | 92.7              | 91.3              | 89.6              | 94.7       | **95.7**            |
| 011_banana            | 84.7             | 97.1              | 96.6              | 97.6              | 97.2       | **98.0**            |
| 019_pitcher_base      | 85.5             | **97.8**          | 97.1              | 97.7              | 97.6       | 96.2                |
| 021_bleach_cleanser   | 81.0             | **96.9**          | 95.8              | 95.4              | 96.8       | 95.5                |
| 024_bowl              | 75.7             | 81.0              | 88.2              | 89.6              | **96.3**   | 88.5                |
| 025_mug               | 94.2             | 95.0              | 97.1              | 97.1              | 97.3       | **98.2**            |
| 035_power_drill       | 71.5             | **98.2**          | 96.0              | 96.7              | 97.2       | 97.0                |
| 036_wood_block        | 68.1             | 87.6              | 89.7              | 91.8              | 92.6       | **94.5**            |
| 037_scissors          | 76.7             | 91.7              | 95.2              | 92.7              | 97.7       | **98.5**            |
| 040_large_marker      | 87.9             | 97.2              | 97.5              | 97.5              | 96.6       | **98.6**            |
| 051_large_clamp       | 65.9             | 75.2              | 72.9              | 71.9              | **96.8**   | 75.0                |
| 052_extra_large_clamp | 60.4             | 64.4              | 69.8              | 71.4              | **96.0**   | 72.9                |
| 061_foam_brick        | 91.8             | 97.2              | 92.5              | 94.3              | 97.3       | **97.7**            |

Table 2: Evaluation of 6D Pose (percentage of ADD-S smaller than 2cm) on the YCB-Video Dataset.Bold numbers are the best indicators.

| Objects               | PointFuion   <2cm | PoseCNN+ICP   <2cm | DenseFusion   <2cm | MaskedFusion  <2cm | Proposed Method <2cm |
| --------------------- | ----------------- | ------------------ | ------------------ | ------------------ | -------------------- |
| 002_master_chef_can   | 99.8              | **100.0**          | **100.0**          | **100.0**          | **100.0**            |
| 003_checker_box       | 62.6              | 91.6               | 99.5               | **99.8**           | **99.8**             |
| 004_sugar_box         | 95.4              | **100.0**          | **100.0**          | **100.0**          | **100.0**            |
| 005_tomato_soup_can   | **96.9**          | 96.6               | **96.9**           | **96.9**           | **96.9**             |
| 006_mustard_bottle    | 84.0              | **100.0**          | **100.0**          | **100.0**          | **100.0**            |
| 007_tuna_fish_can     | 99.8              | **100.0**          | **100.0**          | 99.7               | **100.0**            |
| 008_pudding_box       | 96.7              | **100.0**          | **100.0**          | **100.0**          | **100.0**            |
| 009_geltain_box       | **100.0**         | **100.0**          | **100.0**          | **100.0**          | **100.0**            |
| 010_potted_meat_can   | 88.5              | 93.6               | 93.1               | 94.2               | **98.0**             |
| 011_banana            | 70.5              | 99.7               | **100.0**          | **100.0**          | **100.0**            |
| 019_pitcher_base      | 79.8              | **100.0**          | **100.0**          | **100.0**          | **100.0**            |
| 021_bleach_cleanser   | 65.0              | 99.4               | **100.0**          | 99.4               | 99.8                 |
| 024_bowl              | 24.1              | 54.9               | 98.8               | 95.4               | **100.0**            |
| 025_mug               | 99.8              | 99.8               | **100.0**          | **100.0**          | **100.0**            |
| 035_power_drill       | 22.8              | **99.6**           | 98.7               | 99.5               | **99.6**             |
| 036_wood_block        | 18.2              | 80.2               | 94.6               | **100.0**          | 98.8                 |
| 037_scissors          | 35.9              | 95.6               | **100.0**          | 99.9               | **100.0**            |
| 040_large_marker      | 80.4              | 99.7               | **100.0**          | 99.9               | **100.0**            |
| 051_large_clamp       | 50.0              | 74.9               | 79.2               | 78.7               | **80.9**             |
| 052_extra_large_clamp | 20.1              | 48.8               | 76.3               | 75.9               | **82.1**             |
| 061_foam_brick        | **100.0**         | **100.0**          | **100.0**          | **100.0**          | **100.0**            |

Fig 2: Visualization of the Overall Effectiveness of the Framework

![6dReult](figs\6dReult.png)
