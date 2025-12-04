# **Deep Learning for SAR ATR in Complex Environmental Interference - System Review**

***Jinrui Liao***, ***Qingsong Wang***, ***Tao Lai***, ***Chengran Yang***, ***Haifeng Huang***

**This paper explores the cross-application between artificial intelligence and SAR ATR, focusing on detection and recognition tasks under two environmental interference: intricate natural scenes and electromagnetic interference.**

****

For more details, kindly refer to our [paper].

## :books: Citation 

If you find this work helpful for your research, please kindly consider citing our paper:
```bib
@article{liao2025deepL,
      title={Deep Learning for SAR ATR in Complex Environmental Interference: A Systematic Review},
      author={Liao, Jinrui and Wang, Qingsong and Lai, Tao and Yang, Chengran and Huang, Haifeng},
      journal={XXX (Under Review/Published)}, 
      year={2025},
      url={}, 
      doi={}
}
````

# Table of Contents

  - [**Summary of Surveys in SAR ATR**](#summary-of-surveys-in-sar-atr)
  - [**SAR Target Datasets**](#sar-target-datasets)
      - [SAR Target Classification Datasets](#sar-target-classification-datasets)
      - [SAR Target Detection Datasets](sar-target-detection-datasets)
  - [**SAR Target Detection**](sar-target-detection)
      - [Traditional Methods](#traditional-methods)
      - [Deep Learning Methods](#deep-learning-methods)
  - [**SAR Target Classification**](#sar-target-classification)
      - [Based on Real-Valued Amplitude Visual Representation](#based-on-real-valued-amplitude-visual-representation)
      - [Based on Part-Topology Driving](#based-on-part-topology-driving)
      - [Based on Electromagnetic Scattering Features](#based-on-electromagnetic-scattering-features)
  - [**Complex Background Interference**](#complex-background-interference)
      - [Electromagnetic Scattering Interference](#electromagnetic-scattering-interference)
          - [Suppression Jamming](#suppression-jamming)
          - [Deception Jamming](#deception-jamming)
  - [**Anti-Jamming & Robustness Methods**](#anti-jamming-robustness-methods)
      - [Adversarial Attacks in SAR](#adversarial-attacks-in-sar)
      - [Domain Adaptation for SAR ATR](#domain-adaptation-for-sar-atr)
      - [Robust Detection/Recognition](#robust-detectionrecognition)
  - [**Promising Research Directions**](#promising-research-directions)
      - [Camouflaged Object Detection](#camouflaged-object-detection)
      - [Vision-Language Models (VLMs)](#vision-language-models)
      - [Fusion of Physical Mechanism and Deep Learning](#fusion-of-physical-mechanism-and-deep-learning)

-----

# **Summary of Surveys in SAR ATR**

| Year | Venue | Topic | Scope | Survey Title | Link |
|:---:|:---:|:---|:---:|:---|:---|
| 2020 | J. Radars | Aircraft detection/recognition | Tradition+DL | Research progress on aircraft detection and recognition in SAR imagery | [Link](https://radars.ac.cn/en/article/doi/10.12000/JR20020) |
| 2020 | Signal Proc. | Robustness against attacks | DL | Robustness of SAR Ship Recognition CNN based on Adversarial Attacks | [Link](https://signal.ejournal.org.cn/cn/article/doi/10.16798/j.issn.1003-0530.2020.12.002) |
| 2024 | J. Radars | Adversarial attacks (Digital to Physical) | DL | SAR Target Recognition Adversarial Attack: From Digital Domain to Physical Domain | [Link](https://radars.ac.cn/cn/article/doi/10.12000/JR24142) |
| 2021 | Acta Auto. Sin. | Deep domain adaptation | DL | Deep Domain Adaptation: General and Complex Cases | [Link]((http://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c200238?viewType=HTML)) |
| 2024 | IEEE TNNLS | Transfer adaptation learning | DL | Transfer Adaptation Learning: A Decade Survey | [Link](https://doi.org/10.1109/TNNLS.2022.3183326) |

-----

# **SAR Target Datasets**

## SAR Target Classification Datasets

| Year | Dataset | Targets | Scenes | Size | Band | Note | Link |
|:---:|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1995 | **MSTAR** | 10 | 1 | 128\~193 | X | Fine-grained vehicle benchmark | [Link](https://www.sdms.afrl.af.mil/index.php?collection=mstar) |
| 2004 | **QinetiQ** | 9 | - | - | - | Non-centered, natural clutter | [Link](https://publica.fraunhofer.de/entities/publication/495cf534-215a-4913-962b-a5df7f9fdf39/details) |
| 2010 | **CV Domes** | - | - | - | X | Civil vehicle simulation (wide angle) | [Link](https://doi.org/10.1117/12.850151) |
| 2012 | **Gotcha** | - | - | - | - | Civil vehicle measured (360°) | [Link](https://doi.org/10.1117/12.925077) |
| 2017 | **SARSim** | 14 | 3 | 139 | X | Simulation vehicle dataset | [Link](https://doi.org/10.1109/LGRS.2017.2717486) |
| 2017 | **OpenSARShip** | 14 | 10 | 9\~445 | C | Fine-grained ship slices | [Link](https://doi.org/10.1109/JSTARS.2017.2755672) |
| 2017 | **MGTD** | 4 | 5 | - | - | Weak environment correlation | [Link](https://doi.org/10.1117/12.2277914) |
| 2019 | **SAMPLE** | 10 | 2 | 128 | X | Simulation and measured vehicle | [Link](https://doi.org/10.1117/12.2523460) |
| 2019 | **FUSAR-Ship 1.0** | 15 | 3 | 512 | C | High-resolution ship classification | [Link](https://doi.org/10.1007/s11432-019-2772-5) |
| 2022 | **SAR-ACD** | 6 | 14 | - | - | Spaceborne fine-grained aircraft | [Link](https://doi.org/10.1109/TGRS.2022.3166174) |
| 2025 | **NUDT4MSTAR** | 40 | 5 | 128 | X, Ku | Large-scale multi-class vehicle | [Link](https://doi.org/10.16356/j.1005-2615.2022.05.022) |
| 2025 | **FAIR-CSAR** | 5 | 32 | 800 | C | Single-Look Complex (SLC) dataset | [Link](https://data.nasa.gov/dataset/sentinel-1-single-look-complex-slc-bursts) |

## SAR Target Detection Datasets

| Year | Dataset | Targets | Size | Res. (m) | Band | Note | Link |
|:---:|:---|:---:|:---:|:---:|:---:|:---|:---|
| 2006 | **Sandia MiniSAR** | 1 | 224 | 0.1 | Ku | Urban/Desert vehicles | [Link](https://www.sandia.gov/radar/pathfinder-radar-isr-and-synthetic-aperture-radar-sar-systems/complex-data/) |
| 2018 | **SSDD** | 1 | 214\~668 | 1\~15 | C/X | Benchmark ship detection | [Link](https://doi.org/10.11999/JEIT180050) |
| 2019 | **AIR-SARShip** | 1 | 512\~1K | 1\~3 | C | High-resolution ship | [Link](https://doi.org/10.12000/JR19097) |
| 2019 | **SAR-Ship** | 1 | 256 | 3\~25 | C | Complex scenes | [Link](https://www.mdpi.com/2072-4292/11/7/765) |
| 2020 | **HRSID** | 1 | 800 | 0.5\~3 | C/X | Instance segmentation ships | [Link](https://doi.org/10.1109/ACCESS.2020.3005861) |
| 2020 | **LS-SSDD** | 1 | 800 | 5x20 | L | Large scene small ship | [Link](https://doi.org/10.3390/rs12182997) |
| 2021 | **DSSDD** | 1 | - | - | - | Dual-Polarimetric Ship Detection | [Link](https://doi.org/10.3390/s21248478) |
| 2021 | **SRSDD** | 6 | 1024 | 1 | C | Rotated ship detection | [Link](https://doi.org/10.3390/rs13245104) |
| 2022 | **SADD** | 1 | 224 | 0.5\~3 | X | Aircraft small sample | [Link](https://ieeexplore.ieee.org/document/9791234/) |
| 2022 | **LGSVOD** | 3 | - | - | - | Large scene military vehicle | [Link](https://doi.org/10.1109/ICIVC55077.2022.9887095) |
| 2022 | **RSDD-SAR** | 1 | 512 | 1\~3 | C | Ship dataset | [Link](https://ieeexplore.ieee.org/servlet/Login?logout=/document/9127939/) |
| 2022 | **MSAR** | 4 | 256\~2K | 1 | C | Terrestrial and maritime | [Link](https://radars.ac.cn/web/data/getData?dataType=MSAR) |
| 2023 | **SAR-AIRcraft** | 7 | 512 | 1 | C | Fine-grained Aircraft | [Link](https://radars.ac.cn/web/data/getData%3FdataType%3DSAR-AIRcraft) |
| 2023 | **SIVED** | 1 | 512 | 0.1\~0.3 | X/Ku/Ka | Rotatable vehicle | [Link](https://doi.org/10.3390/rs15112825) |
| 2023 | **OGSOD** | 3 | 256 | 3 | C | Oil tanks, Bridges, Harbors | [link](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13539/3057720/OGSOD-20--a-challenging-multimodal-benchmark-for-optical-SAR/10.1117/12.3057720.full) |
| 2024 | **SARDet-100K** | 6 | 512 | 0.1\~25 | Multi | Large-scale COCO level | [Link](https://arxiv.org/abs/2403.10755) |
| 2025 | **RSAR** | 6 | 512 | 0.1\~25 | Multi | Large-scale Rotated detection | [Link](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_RSAR_Restricted_State_Angle_Resolver_and_Rotated_SAR_Benchmark_CVPR_2025_paper.html) |

-----

# **SAR Target Detection**

## Traditional Methods for SAR Target Detection

| Year | Venue | Topic | Scope | Reference Title | Link |
|:---:|:---:|:---|:---:|:---|:---|
| 2004 | IEEE TGRS | Noise radar jamming | Tradition | Noise radar using random phase and frequency modulation | [Link](https://doi.org/10.1109/TGRS.2004.834589) |
| 2017 | IGARSS | Single-stage ship det. | Tradition/CNN | A fully convolutional neural network for low-complexity... | [Link](https://doi.org/10.1109/IGARSS.2017.8127094) |
| 2019 | J. Radars | ISRJ Suppression | Tradition | Identification and Suppression of ISRJ in Time-Frequency Domain | [Link](https://doi.org/10.12000/JR18080) |

## Deep Learning Methods for SAR Target Detection

| Year | Venue | Method Type | Method Name | Reference Title |
|:---:|:---:|:---|:---|:---|
| 2017 | BIGSARDATA | Two-stage | **Improved Faster R-CNN** | [Ship detection in SAR images based on an improved faster R-CNN](https://ieeexplore.ieee.org/abstract/document/8124934/) |
| 2019 | IEEE TGRS | Single-stage (OBB) | **DRBox-v2** | [DRBox-v2: An Improved Detector With Rotatable Boxes...](https://ieeexplore.ieee.org/abstract/document/8746781/) |
| 2022 | IEEE TGRS | Single-stage | **MGCAN** | [Geospatial Transformer Is What You Need for Aircraft Detection...](https://ieeexplore.ieee.org/abstract/document/9741706/) |
| 2020 | CVPR | Single-stage | **EfficientDet** | [EfficientDet: Scalable and Efficient Object Detection](http://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html) |
| 2020 | IEEE TGRS | Anchor-free | **SSE-CenterNet** | [Ship Detection in Large-Scale SAR Images Via Spatial Shuffle-Group...](https://ieeexplore.ieee.org/abstract/document/9106758/) |
| 2021 | IEEE TGRS | Anchor-free | **FBR-Net** | [An Anchor-Free Method Based on Feature Balancing...](https://ieeexplore.ieee.org/abstract/document/9134934/) |
| 2021 | IEEE JSTARS | Anchor-free | **CP-FCOS** | [An Anchor-Free Detection Method for Ship Targets...](https://ieeexplore.ieee.org/abstract/document/9496207/) |
| 2022 | IEEE JSTARS | One-stage (Pyramid) | **SEFEPNet** | [SEFEPNet: Scale Expansion and Feature Enhancement Pyramid Network...](https://ieeexplore.ieee.org/abstract/document/9761751/) |
| 2023 | IEEE TGRS | Lightweight | **HRLE-SARDet** | [HRLE-SARDet: A Lightweight SAR Target Detection Algorithm...](https://ieeexplore.ieee.org/abstract/document/10057265/) |
| 2024 | IEEE TGRS | Diffusion | **DiffDet4SAR** | [DiffDet4SAR: Diffusion-Based Aircraft Target Detection Network...](https://ieeexplore.ieee.org/abstract/document/10494361/) |
| 2024 | IEEE TGRS | Transformer | **MD-DETR** | [Multilevel Denoising for High-Quality SAR Object Detection...](https://ieeexplore.ieee.org/abstract/document/10741545/) |
| 2025 | IEEE TCSVT | Mamba+Diffusion | **MaDiNet** | [MaDiNet: Mamba Diffusion Network for SAR Target Detection](https://ieeexplore.ieee.org/abstract/document/11016924/) |
| 2025 | IEEE TGRS | Anchor-free (Dynamic) | **DAFDet** | [DAFDet: A Unified Dynamic SAR Target Detection Architecture...](https://ieeexplore.ieee.org/abstract/document/10813601/) |
| 2025 | IEEE TGRS | Transformer | **RDB-DINO** | [RDB-DINO: An Improved End-to-End Transformer...](https://ieeexplore.ieee.org/abstract/document/10798480/) |
| 2024 | IEEE TGRS | Topology | **SRT-Net** | [SRT-Net: Scattering Region Topology Network for Oriented Ship...](https://ieeexplore.ieee.org/abstract/document/10384614/) |

-----

# **SAR Target Classification**

## Based on Real-Valued Amplitude Visual Representation

| Year | Venue | Topic | Method | Link |
|:---:|:---:|:---|:---|:---|
| 2022 | IEEE TGRS | HOG Feature Fusion | **HOG-ShipCLSNet** | [Link](https://doi.org/10.1109/TGRS.2021.3082759) |
| 2023 | IEEE TGRS | Topological Fusion | **VSFA** | [Link](https://doi.org/10.1109/TGRS.2023.3317828) |
| 2024 | ISPRS | Self-Supervised | **SAR-JEPA** | [Link](https://doi.org/10.1016/j.isprsjprs.2024.09.013) |
| 2024 | IEEE TCSVT | Graph Attention | **MIGA-Net** | [Link](https://doi.org/10.1109/TCSVT.2024.3418979) |
| 2024 | ISPRS | Multi-view Trans. | **MT-Net** | [Link](https://doi.org/10.1016/j.isprsjprs.2024.03.009) |

## Based on Part-Topology Driving

| Year | Venue | Topic | Method | Link |
|:---:|:---:|:---|:---|:---|
| 2024 | IEEE TGRS | Scattering Topology | **MoFFL** | [Link](https://doi.org/10.1109/TGRS.2023.3340651) |
| 2025 | IEEE JSTARS | Graph Network | **LDSF** (Dual-Stream) | [Link](https://doi.org/10.1109/JSTARS.2024.3498327) |
| 2025 | IEEE TAES | Structure Guided | **MTSGL** | [Link](https://doi.org/10.1109/TAES.2025.3574673) |

## Based on Electromagnetic Scattering Features

| Year | Venue | Sub-Type | Method | Link |
|:---:|:---:|:---|:---|:---|
| 2021 | IEEE TGRS | Physics Embedded | **FEC** | [Link](https://doi.org/10.1109/TGRS.2020.3003264) |
| 2022 | IEEE TGRS | Physics Embedded | **CA-MCNN** | [Link](https://doi.org/10.1109/TGRS.2021.3100137) |
| 2024 | IEEE TGRS | Physics Embedded | **EMI-Net** | [Link](https://doi.org/10.1109/TGRS.2024.3362334) |
| 2024 | ISPRS | Physics Embedded | **PIHA** | [Link](https://doi.org/10.1016/j.isprsjprs.2023.12.004) |
| 2025 | IEEE TGRS | Physics Embedded | **EMWaveNet** | [Link](https://doi.org/10.1109/TGRS.2025.3557705) |
| 2017 | IEEE TGRS | Complex-Valued | **CV-CNN** | [Link](https://doi.org/10.1109/TGRS.2017.2743222) |
| 2020 | IEEE LGRS | Complex-Valued | **CV-FCNN** | [Link](https://doi.org/10.1109/LGRS.2019.2953892) |
| 2022 | IEEE TGRS | Complex-Valued | **MSCVNets** | [Link](https://doi.org/10.1109/TGRS.2022.3177323) |
| 2024 | IEEE TAES | Complex-Valued | **CV-SAR-Det** | [Link](https://doi.org/10.1109/TAES.2024.3425392) |
| 2025 | IEEE JSTARS | Complex-Valued | **FDC-TA-DSN** | [Link](https://doi.org/10.1109/JSTARS.2025.3542436) |
| 2025 | IEEE JSTARS | Complex-Valued | **CRMC-Net** | [Link](https://doi.org/10.1109/JSTARS.2025.3559656) |

-----

# **Complex Background Interference**

## Electromagnetic Scattering Interference

### Suppression Jamming

| Year | Venue | Topic | Method | Link |
|:---:|:---:|:---|:---|:---|
| 2004 | IEEE TGRS | Noise Radar | Random Phase/Freq Mod. | [Link](https://doi.org/10.1109/TGRS.2004.834589) |
| 2018 | IEEE TGRS | Target Recon. | DSA (Dynamic SA) | [Link](https://doi.org/10.1109/TGRS.2017.2744178) |
| 2022 | IEEE TGRS | LFM Interference | **2-D SPECAN** | [Link](https://doi.org/10.1109/TGRS.2021.3132495) |
| 2022 | IEEE TGRS | RFI Suppression | **BSF** (Block Subspace) | [Link](https://doi.org/10.1109/TGRS.2021.3096538) |
| 2019 | J. Radars | ISRJ | Time-Frequency Filter | [Link](https://doi.org/10.12000/JR18080) |

### Deception Jamming

| Year | Venue | Topic | Key Mechanism | Link |
|:---:|:---:|:---|:---|:---|
| 2002 | IEE Proc. | ISAR Counter | Digital false-target synthesiser | [Link](https://www.google.com/search?q=https://doi.org/10.1049/ip-rsn:20020522) |
| 2013 | IEEE TGRS | Large-Scene | Deceptive Jamming | [Link](https://doi.org/10.1109/TGRS.2013.2259178) |
| 2023 | IEEE LGRS | Scatter-Wave | **LFMCW** Product Mod. | [Link](https://doi.org/10.1109/LGRS.2023.3294556) |
| 2024 | IEEE TGRS | SAR-GMTI | Cooperative False Images | [Link](https://doi.org/10.1109/TGRS.2024.3359317) |

-----

# **Anti-Jamming & Robustness Methods**

## Adversarial Attacks in SAR

| Year | Venue | Type | Title/Topic | Link |
|:---:|:---:|:---|:---|:---|
| 2021 | IEEE TGRS | Empirical | An Empirical Study of Adversarial Examples... | [Link](https://doi.org/10.1109/TGRS.2021.3051641) |
| 2023 | IEEE LGRS | Pseudo-Physical | **ASC-STA** (Attributed Scattering Center) | [Link](https://doi.org/10.1109/LGRS.2023.3235051) |
| 2024 | J. Radars | Review | SAR Target Recognition Adversarial Attack: From Digital to Physical | [Link](https://doi.org/10.12000/JR24142) |
| 2025 | J. Radars | Physical (Active) | Intelligent Recognition Adversarial Method via Active Jammer | [Link](https://radars.ac.cn/web/journal/paper_detail%3Fid%3D4474) |

## Domain Adaptation for SAR ATR

| Year | Venue | Topic | Reference Title | Link |
|:---:|:---:|:---|:---|:---|
| 2016 | ECCV | Deep CORAL | Deep CORAL: Correlation Alignment... | [Link](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35) |
| 2023 | IEEE TGRS | Feature Align | **VSFA** (Visual Scattering Topological) | [Link](https://doi.org/10.1109/TGRS.2023.3317828) |
| 2024 | IEEE TNNLS | Survey | Transfer Adaptation Learning: A Decade Survey | [Link](https://doi.org/10.1109/TNNLS.2022.3183326) |

## Robust Detection/Recognition

| Year | Venue | Method/Data | Reference Title | Link |
|:---:|:---:|:---|:---|:---|
| 2017 | EUSIPCO | Shadow Feature | SAR deception jamming target recognition... | [Link](https://doi.org/10.23919/EUSIPCO.2017.8081659) |
| 2023 | Remote Sens. | GAN Augmentation | A GAN-Based Augmentation Scheme for SAR Deceptive Jamming... | [Link](https://doi.org/10.3390/rs15194756) |
| 2023 | Sys. Eng. Elec. | ISRJ Detection | SAR Image ISRJ Detection Based on Deep Learning | [Link](https://doi.org/10.12305/j.issn.1001-506X.2023.11.12) |

-----

# **Promising Research Directions**

## Camouflaged Object Detection

  - **Focus:** Separating targets from high-clutter/similar-texture backgrounds.
  - **Reference:** [Fan et al., CVPR 2020] (General Vision COD foundation).

## Vision-Language Models

| Year | Venue | Model Name | Scope | Link |
|:---:|:---:|:---|:---|:---|
| 2024 | IEEE TGRS | **RemoteCLIP** | Foundation Model for RS | [Link](https://doi.org/10.1109/TGRS.2024.3390838) |
| 2024 | IEEE TGRS | **GeoRSCLIP** | Large-scale VLM for RS | [Link](https://doi.org/10.1109/TGRS.2024.3449154) |
| 2024 | AAAI | **Sky-CLIP** (SkyScript) | Semantically Diverse VLM | [Link](https://doi.org/10.1609/aaai.v38i6.28393) |
| 2024 | ISPRS | **ChangeCLIP** | Change Detection | [Link](https://doi.org/10.1016/j.isprsjprs.2024.01.004) |
| 2025 | IEEE WACV | **SenCLIP** | Zero-Shot Land-Use (Sentinel-2) | [Link](https://doi.org/10.1109/WACV61041.2025.00552) |
| 2025 | ISPRS | **SkyEyeGPT** | Unified Task / Instruction Tuning | [Link](https://doi.org/10.1016/j.isprsjprs.2025.01.020) |
| 2025 | ArXiv | **Falcon** | Foundation Model | [Link](https://arxiv.org/abs/2503.11070) |

## Fusion of Physical Mechanism and Deep Learning

  - **Concept:** Embedding electromagnetic scattering models into DNNs.
  - **Key Works:** [EMI-Net](https://doi.org/10.1109/TGRS.2024.3362334), [EMWaveNet](https://www.google.com/search?q=https://doi.org/10.1109/TGRS.2025.3557705), [Physics Inspired Hybrid Attention](https://doi.org/10.1016/j.isprsjprs.2023.12.004).

