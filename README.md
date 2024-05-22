# Vessel Probability Guided Attenuation Learning
This is the official repo of our paper "3D Vessel Reconstruction from Sparse-View Dynamic DSA Images via Vessel Probability Guided Attenuation Learning". We will release our code and some test cases once our paper is acceptted. For more details, you refer to our [paper](https://arxiv.org/abs/2405.10705).


# Introduction
## What is DSA
DSA (Digital Subtraction Angiography) is one of the gold standards in vascular disease diagnosing. The patient undergoes two rotational X-ray scans at identical positions. The first scan is performed before the injection of contrast agent (mask run), and the second scan is conducted after injection (fill run). Following this, the DSA sequence is generated by subtracting the X-ray images acquired during the fill run from those taken during the mask run. This process highlights the blood flow information marked by the contrast agent while removing other irrelevant tissues. Each DSA image captures a particular blood flow state as the contrast agent gradually fills the vessels. You may refer to the following video for more insights. 


## Our method



To achieve a holistic understanding of vessel anatomy, the DSA sequence is then utilized to reconstruct 3D vascular structures.




# Citation
Please cite our paper if you think it is interesting.

      @ARTICLE{VPAL,
      title={3D Vessel Reconstruction from Sparse-View Dynamic DSA Images via Vessel Probability Guided Attenuation Learning}, 
      author={Zhentao Liu and Huangxuan Zhao and Wenhui Qin and Zhenghong Zhou and Xinggang Wang and Wenping Wang and Xiaochun Lai and Chuansheng Zheng and Dinggang Shen and Zhiming Cui},
      year={2024},
      eprint={2405.10705},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
      }
