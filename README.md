# VPAL: Vessel Probability Guided Attenuation Learning

## [Arxiv](https://arxiv.org/abs/2405.10705)

This is the official repo of our paper "3D Vessel Reconstruction from Sparse-View Dynamic DSA Images via Vessel Probability Guided Attenuation Learning". We will release our code and some test cases once our paper is accepted. 

![dsaimaging](https://github.com/Zhentao-Liu/VPAL/assets/81148025/cac8df9b-f0b8-45e8-805b-3dd760c0ec3f)


# Introduction
## What is DSA?

https://github.com/Zhentao-Liu/VPAL/assets/81148025/78cf2bfc-a211-4a82-927f-4b5d7b2417c4

DSA (Digital Subtraction Angiography) is one of the gold standards in vascular disease diagnosing. The patient undergoes two rotational X-ray scans at identical positions. The first scan is performed before the injection of contrast agent (mask run), and the second scan is conducted after injection (fill run). Following this, the DSA sequence is generated by subtracting the X-ray images acquired during the fill run from those taken during the mask run. This process highlights the blood flow information marked by the contrast agent while removing other irrelevant tissues. Each DSA image captures a particular blood flow state as the contrast agent gradually fills the vessels. Time-resolved 2D DSA sequence delivers comprehensive insights into blood flow information and vessel anatomy, aiding in the diagnosis of vascular occlusions, abnormalities, and aneurysms. You may refer to the above video (case #1) for intuitive observation. Note that, in our study, each DSA sequence contains 133 frames.

To achieve a holistic understanding of vessel anatomy, the DSA sequence is then utilized to reconstruct 3D vascular structures. Traditional algorithm (FDK) requires hundreds of scanning views (133) to perform reconstruction, results in significant radiation exposure. Moreover, the dynamic imaging nature of DSA scanning also presents a significant challenge. We aim to (1) effectively model the dynamic nature of DSA imaging, and (2) reduce the number of required scanning views to decrease radiation dosage.

## Our Method
![flowchart](https://github.com/Zhentao-Liu/VPAL/assets/81148025/0eab7e4a-fc69-4f5f-8969-7c603f9e4671)

In this study, we propose to use a time-agnostic vessel probability field to solve this problem effectively. Our approach, termed as vessel probability guided attenuation learning, represents the DSA imaging as a complementary weighted combination of static and dynamic attenuation fields, with the weights derived from the vessel probability field. Functioning as a dynamic mask, vessel probability provides proper gradients for both static and dynamic fields adaptive to different scene types. This mechanism facilitates a self-supervised decomposition between static backgrounds and dynamic contrast agent flow, and significantly improves the reconstruction quality. Our model is trained by minimizing the disparity between synthesized projections and real captured DSA images. We further employ two training strategies to improve our reconstruction quality: (1) coarse-to-fine progressive training to achieve better geometry and (2) temporal perturbed rendering loss to enforce temporal consistency.

# Interesting Results
## Self-Supervised Static-Dynamic Decomposition

https://github.com/Zhentao-Liu/VPAL/assets/81148025/4320cfc7-f4aa-41c6-979b-e76a01c2a2e7

Use 40 training views (uniformly spaced) to recover complete 133 views. Our methods achieves self-supervised static-dynamic decomposition and high-quality novel view synthesis.

https://github.com/Zhentao-Liu/VPAL/assets/81148025/fdff2da1-ba80-49ac-a264-02b8d668f499

But if we do not use vessel probability to guide our model training, the decomposition will become blurry and the view synthesis will deteriorate a lot especially for the dynamic one.


![reconstruction_comparision](https://github.com/Zhentao-Liu/VPAL/assets/81148025/60092920-6833-4c46-98b6-1462f2c49a65)

Our vessel probability captures meaningful vascular patterns, assisting in providing high-quality vessel reconstruction. The reconstruction will deteriorate a lot without vessel probability (naive solution). All results here come from case #1.

## High-Quality Vessel Reconstructions

![vesselreconstruction](https://github.com/Zhentao-Liu/VPAL/assets/81148025/237cb88e-0d39-45b6-8028-0c0c36f31518)

Vessel reconstruction results from 40 training views. Our method significantly outperforms all the other methods, which looks quite close to the reference one provided by DSA scanner with full 133 views. We produce reconstructions with less noise, more complete vascular topology, and smoother surfaces. For more visualizations, please refer to our paper. 

## High-Quality Renderings

https://github.com/Zhentao-Liu/VPAL/assets/81148025/c360bb5c-6e25-4eaf-a841-4308e3487917


https://github.com/Zhentao-Liu/VPAL/assets/81148025/f88d8621-32b3-4484-a8b0-cb34bc04b0fc


Use 40 training views to recover complete 133 views. Our methods achieves high-quality novel view synthesis compared to other methods.

## Ablations

<img src="https://github.com/Zhentao-Liu/VPAL/assets/81148025/2eef8366-be2a-4f06-a01f-3834d50b8eea" alt="ablation_reconstruction" width="700"/>

Ablation results on vessel reconstruction with 40 training views from case #1. The values labeled above are CD(mm)/HD(mm).

https://github.com/Zhentao-Liu/VPAL/assets/81148025/11e812d9-d716-4679-b9b0-399aaec33a53

Ablation results on view synthesis with 40 training views from case #15. Especially look at discontinuous initial frames of (c), resulting from training frames overfitting issue.

# Releasing
We will release our code and some test cases once our paper is accepted. We will continue updating this repo. To be continue. If you have any question, just reach out to the author: liuzht2022@shanghaitech.edu.cn

# Related Links
- Traditional FDK algorithm is implemented based on powerful [Astra-toolbox](https://github.com/astra-toolbox/astra-toolbox)
- Pioneer NeRF-based framework for CBCT reconstruction: [NAF](https://github.com/Ruyi-Zha/naf_cbct), [SNAF](https://arxiv.org/abs/2211.17048)
- Pioneer NeRF-based framework for DSA reconstruction: [TiAVox](https://arxiv.org/abs/2309.02318)
- Pioneer 3DGS-based framework for DSA reconstruction: [TOGS](https://github.com/hustvl/TOGS)
  
Thanks for all these great works.

# Citation
If you think our work and repo are interesting, you may cite our paper.

    @article{VPAL,
      title={3D Vessel Reconstruction from Sparse-View Dynamic DSA Images via Vessel Probability Guided Attenuation Learning},
      author={Liu, Zhentao and Zhao, Huangxuan and Qin, Wenhui and Zhou, Zhenghong and Wang, Xinggang and Wang, Wenping and Lai, Xiaochun and Zheng, Chuansheng and Shen, Dinggang and Cui, Zhiming},
      journal={arXiv preprint arXiv:2405.10705},
      year={2024}
    }
