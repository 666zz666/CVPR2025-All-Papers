# **CVPR 2025 论文分类整理**

## 📚 **目录**

### **一、基础模型与架构 (Foundation Models & Architectures)**
- [**Backbone**](#backbone)
- [**MoE (Mixture of Experts)**](#moe-mixture-of-experts)
- [**MLP**](#mlp)
- [**CNN**](#cnn)
- [**Transformer**](#transformer)
- [**ViT (Vision Transformer)**](#vit-vision-transformer)
- [**Mamba / 状态空间模型 (State Space Models)**](#mamba--状态空间模型-state-space-models)

### **二、生成式模型与AIGC (Generative Models & AIGC)**
- [**扩散模型 (Diffusion Model)**](#扩散模型-diffusion-model)
- [**世界模型 (World Model)**](#世界模型-world-model)
- [**大语言模型 (LLM for Vision)**](#大语言模型-llm-for-vision)
- [**多模态大模型 (Large Multimodal Models)**](#多模态大模型-large-multimodal-models)
- [**视觉语言模型 (Vision-Language / VLM)**](#视觉语言模型-vision-language--vlm)
- [**图像生成 (Image Generation)**](#图像生成-image-generation)
- [**视频生成 (Video Generation)**](#视频生成-video-generation)
- [**文生图 (Text-to-Image)**](#文生图-text-to-image)
- [**图像编辑 (Image Editing)**](#图像编辑-image-editing)
- [**图生图 (Image-to-Image Translation)**](#图生图-image-to-image-translation)
- [**风格迁移 (Style Transfer)**](#风格迁移-style-transfer)
- [**图像抠图 (Image Matting)**](#图像抠图-image-matting)
- [**图像处理 (Image Manipulation)**](#图像处理-image-manipulation)

### **三、3D视觉 (3D Vision)**
- [**3D视觉 (3D Vision)**](#3d视觉-3d-vision)
- [**3D生成 (3D Generation)**](#3d生成-3d-generation)
- [**3D高斯溅射 (3D Gaussian Splatting)**](#3d高斯溅射-3d-gaussian-splatting)
- [**神经辐射场 (NeRF)**](#神经辐射场-nerf)
- [**3D重建 (3D Reconstruction)**](#3d重建-3d-reconstruction)

### **四、感知与理解 (Perception & Understanding)**
- [**目标检测 (Object Detection)**](#目标检测-object-detection)
- [**目标跟踪 (Object Tracking)**](#目标跟踪-object-tracking)
- [**通用分割 (Segmentation)**](#通用分割-segmentation)
- [**图像分割 (Image Segmentation)**](#图像分割-image-segmentation)
- [**视频目标分割 (Video Object Segmentation)**](#视频目标分割-video-object-segmentation)
- [**医学图像分割 (Medical Image Segmentation)**](#医学图像分割-medical-image-segmentation)
- [**交互式分割 (Interactive Segmentation)**](#交互式分割-interactive-segmentation)
- [**视频理解 (Video Understanding)**](#视频理解-video-understanding)
- [**动作识别 (Action Recognition)**](#动作识别-action-recognition)
- [**视频-语言 (Video-Language)**](#视频-语言-video-language)

### **五、人体与人脸技术 (Human & Face Technology)**
- [**人体分析 (Human-related)**](#人体分析-human-related)
- [**姿态估计 (Pose Estimation)**](#姿态估计-pose-estimation)
- [**人-物交互 (Human-Object Interaction)**](#人-物交互-human-object-interaction)
- [**行人重识别 (Person Re-identification)**](#行人重识别-person-re-identification)
- [**人体网格恢复 (Human Mesh Recovery)**](#人体网格恢复-human-mesh-recovery)
- [**人脸技术 (Face)**](#人脸技术-face)
- [**人脸识别 (Face Recognition)**](#人脸识别-face-recognition)
- [**人脸生成 (Face Generation)**](#人脸生成-face-generation)
- [**活体检测/反欺诈 (Anti-Spoofing)**](#活体检测反欺诈-anti-spoofing)

### **六、底层视觉与图像处理 (Low-Level Vision & Image Processing)**
- [**图像复原 (Image Restoration)**](#图像复原-image-restoration)
- [**底层视觉 (Low-level Vision)**](#底层视觉-low-level-vision)
- [**超分辨率 (Super-Resolution)**](#超分辨率-super-resolution)
- [**图像去噪 (Denoising)**](#图像去噪-denoising)
- [**图像去模糊 (Deblurring)**](#图像去模糊-deblurring)
- [**图像去雾 (Dehazing)**](#图像去雾-dehazing)
- [**图像压缩 (Image Compression)**](#图像压缩-image-compression)
- [**图像质量评价 (Image Quality Assessment)**](#图像质量评价-image-quality-assessment)
- [**视频去雨 (Video Deraining)**](#视频去雨-video-deraining)

### **七、特定应用领域 (Specific Application Domains)**
- [**具身智能 (Embodied AI)**](#具身智能-embodied-ai)
- [**自动驾驶 (Autonomous Driving)**](#自动驾驶-autonomous-driving)
- [**医学影像 (Medical Image)**](#医学影像-medical-image)
- [**遥感影像 (Remote Sensing)**](#遥感影像-remote-sensing)
- [**文字识别 (OCR)**](#文字识别-ocr)

### **八、学习范式与模型优化 (Learning Paradigms & Model Optimization)**
- [**自监督学习 (Self-Supervised Learning)**](#自监督学习-self-supervised-learning)
- [**对比学习 (Contrastive Learning)**](#对比学习-contrastive-learning)
- [**持续学习 (Continual Learning)**](#持续学习-continual-learning)
- [**联邦学习 (Federated Learning)**](#联邦学习-federated-learning)
- [**主动学习 (Active Learning)**](#主动学习-active-learning)
- [**小样本学习 (Few-Shot Learning)**](#小样本学习-few-shot-learning)
- [**知识蒸馏 (Knowledge Distillation)**](#知识蒸馏-knowledge-distillation)
- [**模型量化 (Quantization)**](#模型量化-quantization)
- [**网络剪枝 (Network Pruning)**](#网络剪枝-network-pruning)
- [**网络架构搜索 (NAS)**](#网络架构搜索-nas)
- [**轻量化模型 (Lightweight Models)**](#轻量化模型-lightweight-models)

### **九、其他 (Others)**
- [**对抗性攻击与防御 (Adversarial Attack & Defense)**](#对抗性攻击与防御-adversarial-attack--defense)
- [**其他 (Others)**](#其他-others)

---

# **一、基础模型与架构 (Foundation Models & Architectures)**
## **Backbone**

* **\[Oral\] OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels**, Meng Lou, Yizhou Yu
* *\[Highlight\] Your ViT is Secretly an Image Segmentation Model*, Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, Daan de Geus
* Argus: A Compact and Versatile Foundation Model for Vision, Weiming Zhuang, Chen Chen, Zhizhong Li, Sina Sajadmanesh, Jingtao Li, Jiabo Huang, Vikash Sehwag, Vivek Sharma, Hirotaka Shinozaki, Felan Carlo Garcia, Yihao Zhan, Naohiro Adachi, Ryoji Eki, Michael Spranger, Peter Stone, Lingjuan Lyu
* Building Vision Models upon Heat Conduction, Zhaozhi Wang, Yue Liu, Yunjie Tian, Yunfan Liu, Yaowei Wang, Qixiang Ye
* LSNet: See Large, Focus Small, Ao Wang, Hui Chen, Zijia Lin, Jungong Han, Guiguang Ding
* MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining, Yunze Liu, Li Yi
* ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect, Dachong Li, Li Li, Zhuangzhuang Chen, Jianqiang Li
* Star with Bilinear Mapping, Zelin Peng, Yu Huang, Zhengqin Xu, Feilong Tang, Ming Hu, Xiaokang Yang
* Frequency Dynamic Convolution for Dense Image Prediction, Linwei Chen, Lin Gu, Liang Li, Chenggang Yan, Ying Fu

## **MoE (Mixture of Experts)**


* *\[Highlight\] SPMTrack: Spatio-Temporal Parameter-Efficient Fine-Tuning with Mixture of Experts for Scalable Visual Tracking*, Wenrui Cai, Qingjie Liu, Yunhong Wang
* DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models, Yongqi Huang, Peng Ye, Chenyu Huang, Jianjian Cao, Lin Zhang, Baopu Li, Gang Yu, Tao Chen
* dFLMOE: Decentralized Federated Learning via Mixture of Experts for Medical Data Analysis, Luyuan Xie, Tianyu Luan, Wenyuan Cai, Guochen Yan, Zhaoyu Chen, Nan Xi, Yuejian Fang, Qingni Shen, Zhonghai Wu, Junsong Yuan
* MOEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation, Huaize Liu, Wenzhang Sun, Donglin Di, Shibo Sun, Jiahui Yang, Changqing Zou, Hujun Bao
* LiMoE: Mixture of LiDAR Representation Learners from Automotive Scenes, Xiang Xu, Lingdong Kong, Hui Shuai, Liang Pan, Ziwei Liu, Qingshan Liu
* Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis, Jiapeng Zhu, Ceyuan Yang, Kecheng Zheng, Yinghao Xu, Zifan Shi, Yifei Zhang, Qifeng Chen, Yujun Shen
* CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering, Tianyu Huai, Jie Zhou, Xingjiao Wu, Qin Chen, Qingchun Bai, Ze Zhou, Liang He

## **MLP**

* RivuletMLP: An MLP-based Architecture for Efficient Compressed Video Quality Enhancement, Gang He, Weiran Wang, Guancheng Quan, Shihao Wang, Dajiang Zhou, Yunsong Li

## **CNN**

* SLVR: Super-Light Visual Reconstruction via Blueprint Controllable Convolutions and Exploring Feature Diversity Representation, Ning Ni, Libao Zhang
* DiC: Rethinking Conv3x3 Designs in Diffusion Models, Yuchuan Tian, Jing Han, Chengcheng Wang, Yuchen Liang, Chao Xu, Hanting Chen

## **Transformer**


* **\[Oral\] LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions**, Faridoun Mehri, Mahdieh Soleymani Baghshah, Mohammad Taher Pilehvar
* **\[Oral\] Rethinking Spiking Self-Attention Mechanism: Implementing α-XNOR Similarity Calculation in Spiking Transformers**, Yichen Xiao, Shuai Wang, Dehao Zhang, Wenjie Wei, Yimeng Shan, Xiaoli Liu, Yulin Jiang, Malu Zhang
* *\[Highlight\] BWFormer: Building Wireframe Reconstruction from Airborne LIDAR Point Cloud with Transformer*, Yuzhou Liu, Lingjie Zhu, Hanqiao Ye, Shangfeng Huang, Xiang Gao, Xianwei Zheng, Shuhan Shen
* *\[Highlight\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation*, Zhuguanyu Wu, Shihe Wang, Jiayi Zhang, Jiaxin Chen, Yunhong Wang
* *\[Highlight\] HIPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation*, Hongwei Zheng, Han Li, Wenrui Dai, Ziyang Zheng, Chenglin Li, Junni Zou, Hongkai Xiong
* *\[Highlight\] Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models*, Zichen Miao, Wei Chen, Qiang Qiu
* SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception, Yaniv Benny, Lior Wolf
* AMR-Transformer: Enabling Efficient Long-range Interaction for Complex Neural Fluid Simulation, Zeyi Xu, Jinfan Liu, Kuangxu Chen, Ye Chen, Zhangli Hu, Bingbing Ni
* Associative Transformer, Yuwei Sun, Hideya Ochiai, Zhirong Wu, Stephen Lin, Ryota Kanai
* SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers, Nick Nikzad, Yi Liao, Yongsheng Gao, Jun Zhou
* ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention for White Balance, Yu-Cheng Chiu, Guan-Rong Chen, Zihao Chen, Yan-Tsung Peng
* Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer, Yufei Guo, Xiaode Liu, Yuanpei Chen, Weihang Peng, Yuhan Zhang, Zhe Ma
* Text Augmented Correlation Transformer For Few-shot Classification & Segmentation, Srinivasa Rao Nandam, Sara Atito, Zhenhua Feng, Josef Kittler, Muhammad Awais
* A Polarization-Aided Transformer for Image Deblurring via Motion Vector Decomposition, Duosheng Chen, Shihao Zhou, Jinshan Pan, Jinglei Shi, Lishen Qu, Jufeng Yang
* Dual Focus-Attention Transformer for Robust Point Cloud Registration, Kexue Fu, Mingzhi Yuan, Changwei Wang, Weiguang Pang, Jing Chi, Manning Wang, Longxiang Gao
* GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding, Haoyi Jiang, Liu Liu, Tianheng Cheng, Xinjie Wang, Tianwei Lin, Zhizhong Su, Wenyu Liu, Xinggang Wang
* A Universal Scale-Adaptive Deformable Transformer for Image Restoration across Diverse Artifacts, Xuyi He, Yuhui Quan, Ruotao Xu, Hui Ji
* Dual Prompting Image Restoration with Diffusion Transformers, Dehong Kong, Fan Li, Zhixin Wang, Jiaqi Xu, Renjing Pei, Wenbo Li, WenQi Ren
* D^2iT: Dynamic Diffusion Transformer for Accurate Image Generation, Weinan Jia, Mengqi Huang, Nan Chen, Lei Zhang, Zhendong Mao
* BlockDance: Reuse Structurally Similar Spatio-Temporal Features to Accelerate Diffusion Transformers, Hui Zhang, Tingwei Gao, Jie Shao, Zuxuan Wu
* Spiking Transformer with Spatial-Temporal Attention, Donghyun Lee, Yuhang Li, Youngeun Kim, Shiting Xiao, Priyadarshini Panda
* TADFormer: Task-Adaptive Dynamic TransFormer for Efficient Multi-Task Learning, Seungmin Baek, Soyul Lee, Hayeon Jo, Hyesong Choi, Dongbo Min
* Transformers without Normalization, Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu
* LaVin-DiT: Large Vision Diffusion Transformer, Zhaoqing Wang, Xiaobo Xia, Runnan Chen, Dongdong Yu, Changhu Wang, Mingming Gong, Tongliang Liu
* Your Scale Factors are My Weapon: Targeted Bit-Flip Attacks on Vision Transformers via Scale Factor Manipulation, Jialai Wang, Yuxiao Wu, Weiye Xu, Yating Huang, Chao Zhang, Zongpeng Li, Mingwei Xu, Zhenkai Liang
* CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction, Yuan Zhou, Qingshan Xu, Jiequan Cui, Junbao Zhou, Jing Zhang, Richang Hong, Hanwang Zhang
* DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation, Bo-Wen Yin, Jiao-Long Cao, Ming-Ming Cheng, Qibin Hou
* Mask^2DIT: Dual Mask-based Diffusion Transformer for Multi-Scene Long Video Generation, Tianhao Qi, Jianlong Yuan, Wanquan Feng, Shancheng Fang, Jiawei Liu, SiYu Zhou, Qian He, Hongtao Xie, Yongdong Zhang
* TinyFusion: Diffusion Transformers Learned Shallow, Gongfan Fang, Kunjun Li, Xinyin Ma, Xinchao Wang
* Towards Transformer-Based Aligned Generation with Self-Coherence Guidance, Shulei Wang, Wang Lin, Hai Huang, Hanting Wang, Sihang Cai, WenKang Han, Tao Jin, Jingyuan Chen, Jiacheng Sun, Jieming Zhu, Zhou Zhao

## **ViT (Vision Transformer)**


* BHVIT: Binarized Hybrid Vision Transformer, Tian Gao, Yu Zhang, Zhiyuan Zhang, Huajun Liu, Kaijie Yin, Chengzhong Xu, Hui Kong
* DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers, Li Ren, Chen Chen, Liqiang Wang, Kien Hua
* APHQ-VIT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers, Zhuguanyu Wu, Jiayi Zhang, Jiaxin Chen, Jinyang Guo, Di Huang, Yunhong Wang
* Token Cropr: Faster ViTs for Quite a Few Tasks, Benjamin Bergner, Christoph Lippert, Aravindh Mahendran
* Hypergraph Vision Transformers: Images are More than Nodes, More than Edges, Joshua Fixelle
* MambaVision: A Hybrid Mamba-Transformer Vision Backbone, Ali Hatamizadeh, Jan Kautz
* DeepCompress-ViT: Rethinking Model Compression to Enhance Efficiency of Vision Transformers at the Edge, Sabbir Ahmed, Abdullah Al Arafat, Deniz Najafi, Akhlak Mahmood, Mamshad Nayeem Rizve, Mohaiminul Al Nahian, Ranyang Zhou, Shaahin Angizi, Adnan Siraj Rakin
* Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency, Feng Wang, Timing Yang, Yaodong Yu, Sucheng Ren, Guoyizhe Wei, Angtian Wang, Wei Shao, Yuyin Zhou, Alan Yuille, Cihang Xie
* Rethinking Token Reduction with Parameter-Efficient Fine-Tuning in ViT for Pixel-Level Tasks, Cheng Lei, Ao Li, Hu Yao, Ce Zhu, Le Zhang
* Split Adaptation for Pre-trained Vision Transformers, Lixu Wang, Bingqi Shang, Yi Li, Payal Mohapatra, Wei Dong, Xiao Wang, Qi Zhu

## **Mamba / 状态空间模型 (State Space Models)**


* *\[Highlight\] Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation*, Xin Zhang, Robby T. Tan
* MambaVO: Deep Visual Odometry Based on Sequential Matching Refinement and Training Smoothing, Shuo Wang, Wanting Li, Yongcai Wang, Zhaoxin Fan, Zhe Huang, Xudong Cai, Jian Zhao, Deying Li
* UniMamba: Unified Spatial-Channel Representation Learning with Group-Efficient Mamba for LiDAR-based 3D Object Detection, Xin Jin, Haisheng Su, Kai Liu, Cong Ma, Wei Wu, Fei HUI, Junchi Yan
* Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation, Xingguang Zhang, Nicholas Chimitt, Xijun Wang, Yu Yuan, Stanley H. Chan
* 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification, Jingwei Zhang, Anh Tien Nguyen, Xi Han, Vincent Quoc-Huy Trinh, Hong Qin, Dimitris Samaras, Mahdi S. Hosseini
* DefMamba: Deformable Visual State Space Model, Leiye Liu, Miao Zhang, Jihao Yin, Tingwei Liu, Wei Ji, Yongri Piao, Huchuan Lu
* MambaOut: Do We Really Need Mamba for Vision?, Weihao Yu, Xinchao Wang
* MobileMamba: Lightweight Multi-Receptive Visual Mamba Network, Haoyang He, Jiangning Zhang, Yuxuan Cai, Hongxu Chen, Xiaobin Hu, Zhenye Gan, Yabiao Wang, Chengjie Wang, Yunsheng Wu, Lei Xie
* LMO: Linear Mamba Operator for MRI Reconstruction, Wei Li, Jiawei Jiang, Jie Wu, Kaihao Yu, Jianwei Zheng
* OSMamba: Omnidirectional Spectral Mamba with Dual-Domain Prior Generator for Exposure Correction, Gehui Li, Bin Chen, Chen Zhao, Lei Zhang, Jian Zhang
* MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration, Boyun Li, Haiyu Zhao, Wenxin Wang, Peng Hu, Yuanbiao Gou, Xi Peng
* MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking, Xingi Liu, Li Zhou, Zikun Zhou, Jianqiu Chen, Zhenyu He
* QMambaBSR: Burst Image Super-Resolution with Query State Space Model, Xin Di, Long Peng, Peizhe Xia, Wenbo Li, Renjing Pei, Yang Cao, Yang Wang, Zheng-Jun Zha
* AlignMamba: Enhancing Multimodal Mamba with Local and Global Cross-modal Alignment, Yan Li, Yifei Xing, Xiangyuan Lan, Xin Li, Haifeng Chen, Dongmei Jiang
* Samba: A Unified Mamba-based Framework for General Salient Object Detection, Jiahao He, Keren Fu, Xiaohong Liu, Qijun Zhao
* Incomplete Multi-modal Brain Tumor Segmentation via Learnable Sorting State Space Model, Zheyu Zhang, Yayuan Lu, Feipeng Ma, Yueyi Zhang, Huanjing Yue, Xiaoyan Sun
* Spectral State Space Model for Rotation-Invariant Visual Representation Learning, Sahar Dastani, Ali Bahri, Moslem Yazdanpanah, Mehrdad Noori, David Osowiechi, Gustavo Adolfo Vargas Hakim, Farzad Beizaee, Milad Cheraghalikhani, Arnab Kumar Mondal, Herve Lombaert, Christian Desrosiers
* MambalRv2: Attentive State Space Restoration, Hang Guo, Yong Guo, Yaohua Zha, Yulun Zhang, Wenbo Li, Tao Dai, Shu-Tao Xia, Yawei Li
* TSP-Mamba: The Travelling Salesman Problem Meets Mamba for Image Super-resolution and Beyond, Kun Zhou, Xinyu Lin, Jiangbo Lu
* SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer, Hongda Liu, Longguang Wang, Ye Zhang, Ziru Yu, Yulan Guo
* GG-SSMs: Graph-Generating State Space Models, Nikola Zubic, Davide Scaramuzza
* OccMamba: Semantic Occupancy Prediction with State Space Models, Heng Li, Yuenan Hou, Xiaohan Xing, Yuexin Ma, Xiao Sun, Yanyong Zhang
* Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM, Yizhou Huang, Yihua Cheng, Kezhi Wang
* FlowRAM: Grounding Flow Matching Policy with Region-Aware Mamba Framework for Robotic Manipulation, Sen Wang, Le Wang, Sanping Zhou, Jingyi Tian, Jiayi Li, Haowen Sun, Wei Tang
* Event-based Video Super-Resolution via State Space Models, Zeyu Xiao, Xinchao Wang
* GroupMamba: Efficient Group-Based Visual State Space Model, Abdelrahman Shaker, Syed Talal Wasim, Salman Khan, Juergen Gall, Fahad Shahbaz Khan
* EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality, Sanghyeok Lee, Joonmyung Choi, Hyunwoo J. Kim
* JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba, Xiaoyong Lu, Songlin Du
* Mamba-Reg: Vision Mamba Also Needs Registers, Feng Wang, Jiahao Wang, Sucheng Ren, Guoyizhe Wei, Jieru Mei, Wei Shao, Yuyin Zhou, Alan Yuille, Cihang Xie
* Mamba-Adaptor: State Space Model Adaptor for Visual Recognition, Fei Xie, Jiahao Nie, Yujin Tang, Wenkang Zhang, Hongshen Zhao
* Spectral Informed Mamba for Robust Point Cloud Processing, Ali Bahri, Moslem Yazdanpanah, Mehrdad Noori, Sahar Dastani, Milad Cheraghalikhani, Gustavo Adolfo Vargas Hakim, David Osowiechi, Farzad Beizaee, Ismail Ben Ayed, Christian Desrosiers
* WeatherGen: A Unified Diverse Weather Generator for LIDAR Point Clouds via Spider Mamba Diffusion, Yang Wu, Yun Zhu, Kaihua Zhang, Jianjun Qian, Jin Xie, Jian Yang
* Mamba4D: Efficient 4D Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models, Jiuming Liu, Jinru Han, Lihao Liu, Angelica I. Aviles-Rivero, Chaokang Jiang, Zhe Liu, Hesheng Wang
* LC-Mamba: Local and Continuous Mamba with Shifted Windows for Frame Interpolation, Min Wu Jeong, Chae Eun Rhee
* Cross-Modal Interactive Perception Network with Mamba for Lung Tumor Segmentation in PET-CT Images, Jie Mei, Chenyu Lin, Yu Qiu, Yaonan Wang, Hui Zhang, Ziyang Wang, Dong Dai
* M3amba: Memory Mamba is All You Need for Whole Slide Image Classification, Tingting Zheng, Kui Jiang, Yi Xiao, Sicheng Zhao, Hongxun Yao
* SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation, Yunxiang Fu, Meng Lou, Yizhou Yu

# **二、生成式模型与AIGC (Generative Models & AIGC)**
## **扩散模型 (Diffusion Model)**


* **\[Oral\] Alias-Free Latent Diffusion Models: Improving Fractional Shift Equivariance of Diffusion Latent Space**, Yifan Zhou, Zeqi Xiao, Shuai Yang, Xingang Pan
* **\[Oral\] CleanDIFT: Diffusion Features without Noise**, Nick Stracke, Stefan Andreas Baumann, Kolja Bauer, Frank Fundel, Björn Ommer
* **\[Oral\] DiffFNO: Diffusion Fourier Neural Operator**, Xiaoyi Liu, Hao Tang
* **\[Oral\] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing**, Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, Yang Song
* **\[Oral\] Towards Universal Dataset Distillation via Task-Driven Diffusion**, Ding Qi, Jian Li, Junyao Gao, Shuguang Dou, Ying Tai, Jianlong Hu, Bo Zhao, Yabiao Wang, Chengjie Wang, Cairong Zhao
* **\[Oral\] IceDiff: High Resolution and High-Quality Arctic Sea Ice Forecasting with Generative Diffusion Prior**, Jingyi Xu, Siwei Tu, Weidong Yang, Ben Fei, Shuhao Li, Keyi Liu, Yeqi Luo, Lipeng Ma, Lei Bai
* **\[Oral\] Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models**, Jingfeng Yao, Bin Yang, Xinggang Wang
* *\[Highlight\] SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model*, Yucheng Mao, Boyang Wang, Nilesh Kulkarni, Jeong Joon Park
* *\[Highlight\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation*, Yiftach Edelstein, Or Patashnik, Dana Cohen-Bar, Lihi Zelnik-Manor
* *\[Highlight\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models*, Fernando Julio Cendra, Kai Han
* *\[Highlight\] Diffusion Drive: Truncated Diffusion Model for End-to-End Autonomous Driving*, Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, Xinggang Wang
* *\[Highlight\] Light Transport-aware Diffusion Posterior Sampling for Single-View Reconstruction of 3D Volumes*, Ludwic Leonard, Nils Thurey, Rüdiger Westermann
* LEDiff: Latent Exposure Diffusion for HDR Generation, Chao Wang, Zhihao Xia, Thomas Leimkuhler, Karol Myszkowski, Xuaner Zhang
* Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model, Yingmao Miao, Zhanpeng Huang, Rui Han, Zibin Wang, Chenhao Lin, Chao Shen
* Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction, Cecilia Curreli, Dominik Muhle, Abhishek Saroha, Zhenzhang Ye, Riccardo Marin, Daniel Cremers
* Nested Diffusion Models Using Hierarchical Latent Priors, Xiao Zhang, Ruoxi Jiang, Rebecca Willett, Michael Maire
* Adaptive Non-Uniform Timestep Sampling for Accelerating Diffusion Model Training, Myunsoo Kim, Donghyeon Ki, Seong-Woong Shim, Byung-Jun Lee
* Scaling Inference Time Compute for Diffusion Models, Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, Saining Xie
* TCFG: Tangential Damping Classifier-free Guidance, Mingi Kwon, Shin seong Kim, Jaeseok Jeong, Yi Ting Hsiao, Youngjung Uh
* FADA: Fast Diffusion Avatar Synthesis with Mixed-Supervised Multi-CFG Distillation, Tianyun Zhong, Chao Liang, Jianwen Jiang, Gaojie Lin, Jiaqi Yang, Zhou Zhao
* Early-Bird Diffusion: Investigating and Leveraging Timestep-Aware Early-Bird Tickets in Diffusion Models for Efficient Training, Lexington Whalen, Zhenbang Du, Haoran You, Chaojian Li, Sixu Li, Yingyan Lin
* Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability, Yingdong Shi, Changming Li, Yifan Wang, Yongxiang Zhao, Anqi Pang, Sibei Yang, Jingyi Yu, Kan Ren
* Efficient Diffusion as Low Light Enhancer, Guanzhou Lan, Qianli Ma, Yuqi Yang, Zhigang Wang, Dong Wang, Xuelong Li, Bin Zhao
* Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion, Hao Wen, Zehuan Huang, Yaohui Wang, Xinyuan Chen, Lu Sheng
* Arbitrary-steps Image Super-resolution via Diffusion Inversion, Zongsheng Yue, Kang Liao, Chen Change Loy
* Reconciling Stochastic and Deterministic Strategies for Zero-shot Image Restoration using Diffusion Model in Dual, Chong Wang, Lanqing Guo, Zixuan Fu, Siyuan Yang, Hao Cheng, Alex C. Kot, Bihan Wen
* Finding Local Diffusion Schrödinger Bridge using Kolmogorov-Arnold Network, Xingyu Qiu, Mengying Yang, Xinghua Ma, Fanding Li, Dong Liang, Gongning Luo, Wei Wang, Kuanquan Wang, Shuo Li
* CacheQuant: Comprehensively Accelerated Diffusion Models, Xuewen Liu, Zhikai Li, Qingyi Gu
* Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning, Qianli Ma, Xuefei Ning, Dongrui Liu, Li Niu, Linfeng Zhang
* Decoupling Training-Free Guided Diffusion by ADMM, Youyuan Zhang, Zehua Liu, Zenan Li, Zhaoyu Li, James J. Clark, Xujie Si
* Decentralized Diffusion Models, David McAllister, Matthew Tancik, Jiaming Song, Angjoo Kanazawa
* Schedule On the Fly: Diffusion Time Prediction for Faster and Better Image Generation, Zilyu Ye, Zhiyang Chen, Tiancheng Li, Zemin Huang, Weijian Luo, Guo-Jun Qi
* Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models, Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, Di Huang
* A Unified Latent Schrödinger Bridge Diffusion Model for Unsupervised Anomaly Detection and Localization, Shilhora Akshay, Niveditha Lakshmi Narasimhan, Jacob George, Vineeth N Balasubramanian
* Inversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification, Yanghao Wang, Long Chen
* Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment, Johannes Schusterbauer, Ming Gui, Frank Fundel, Björn Ommer
* GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration, Sudarshan Rajagopalan, Nithin Gopalakrishnan Nair, Jay N. Paranjape, Vishal M. Patel
* Adversarial Diffusion Compression for Real-World Image Super-Resolution, Bin Chen, Gehui Li, Rongyuan Wu, Xindong Zhang, Jie Chen, Jian Zhang, Lei Zhang
* Q-DIT: Accurate Post-Training Quantization for Diffusion Transformers, Lei Chen, Yuan Meng, Chen Tang, Xinzhu Ma, Jingyan Jiang, Xin Wang, Zhi Wang, Wenwu Zhu
* FlexiDiT: Your Diffusion Transformer can Easily Generate High-Quality Samples with Less Compute, Sotiris Anagnostidis, Gregor Bachmann, Yeongmin Kim, Jonas Kohler, Markos Georgopoulos, Artsiom Sanakoyeu, Yuming Du, Albert Pumarola, Ali Thabet, Edgar Schönfeld
* Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling, Junha Hyung, Kinam Kim, Susung Hong, Min-Jung Kim, Jaegul Choo
* Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation, Zhuoman Liu, Weicai Ye, Yan Luximon, Pengfei Wan, Di Zhang
* Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models, Hao Ren, Yiming Zeng, Zetong Bi, Zhaoliang Wan, Junlong Huang, Hui Cheng
* Repurposing Pre-trained Video Diffusion Models for Event-based Video Interpolation, Jingxi Chen, Brandon Y. Feng, Haoming Cai, Tianfu Wang, Levi Burner, Dehao Yuan, Cornelia Fermuller, Christopher A. Metzler, Yiannis Aloimonos
* InterDyn: Controllable Interactive Dynamics with Video Diffusion Models, Rick Akkerman, Haiwen Feng, Michael J. Black, Dimitrios Tzionas, Victoria Fernández Abrevaya
* STDD: Spatio-Temporal Dual Diffusion for Video Generation, Shuaizhen Yao, Xiaoya Zhang, Xin Liu, Mengyi Liu, Zhen Cui
* Visual-Instructed Degradation Diffusion for All-in-One Image Restoration, Wenyang Luo, Haina Qin, Zewen Chen, Libin Wang, Dandan Zheng, Yuming Li, Yufan Liu, Bing Li, Weiming Hu
* PassionSR: Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution, Libo Zhu, Jianze Li, Haotong Qin, Wenbo Li, Yulun Zhang, Yong Guo, Xiaokang Yang
* Edge-SD-SR: Low Latency and Parameter Efficient On-device Super-Resolution with Stable Diffusion via Bidirectional Conditioning, Isma Hadji, Mehdi Noroozi, Victor Escorcia, Anestis Zaganidis, Brais Martinez, Georgios Tzimiropoulos
* FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error, Beilin Chu, Xuan Xu, Xin Wang, Yufei Zhang, Weike You, Linna Zhou
* Classifier-Free Guidance Inside the Attraction Basin May Cause Memorization, Anubhav Jain, Yuya Kobayashi, Takashi Shibuya, Yuhta Takida, Nasir Memon, Julian Togelius, Yuki Mitsufuji
* Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability, Lei Wang, Senmao Li, Fei Yang, Jianye Wang, Ziheng Zhang, Yuhan Liu, Yaxing Wang, Jian Yang
* Diffusion Model is Effectively Its Own Teacher, Xinyin Ma, Runpeng Yu, Songhua Liu, Gongfan Fang, Xinchao Wang
* Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward, Zhiwei Jia, Yuesong Nan, Huixi Zhao, Gengdai Liu
* RaSS: Improving Denoising Diffusion Samplers with Reinforced Active Sampling Scheduler, Xin Ding, Lei Yu, Xin Li, Zhijun Tu, Hanting Chen, Jie Hu, Zhibo Chen
* A Closer Look at Time Steps is Worthy of Triple Speed-Up for Diffusion Model Training, Kai Wang, Mingjia Shi, Yukun Zhou, Zekai Li, Zhihang Yuan, Yuzhang Shang, Xiaojiang Peng, Hanwang Zhang, Yang You
* Scaling Properties of Diffusion Models For Perceptual Tasks, Rahul Ravishankar, Zeeshan Patel, Jathushan Rajasegaran, Jitendra Malik
* Rectified Diffusion Guidance for Conditional Generation, Mengfei Xia, Nan Xue, Yujun Shen, Ran Yi, Tieliang Gong, Yong-Jin Liu
* The Illusion of Unlearning: The Unstable Nature of Machine Unlearning in Text-to-Image Diffusion Models, Naveen George, Karthik Nandan Dasaraju, Rutheesh Reddy Chittepu, Konda Reddy Mopuri
* SyncSDE: A Probabilistic Framework for Diffusion Synchronization, Hyunjun Lee, Hyunsoo Lee, Sookwan Han
* Lifting Motion to the 3D World via 2D Diffusion, Jiaman Li, C. Karen Liu, Jiajun Wu
* Energy MoGen: Compositional Human Motion Generation with Energy-Based Diffusion Model in Latent Space, Jianrong Zhang, Hehe Fan, Yi Yang
* Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model, Leheng Zhang, Weiyi You, Kexuan Shi, Shuhang Gu
* Accelerating Diffusion Transformer via Increment-Calibrated Caching with Channel-Aware Singular Value Decomposition, Zhiyuan Chen, Keyi Li, Yifan Jia, Le Ye, Yufei Ma
* Optimizing for the Shortest Path in Denoising Diffusion Model, Ping Chen, Xingpeng Zhang, Zhaoxiang Liu, Huan Hu, Xiang Liu, Kai Wang, Min Wang, Yanlin Qian, Shiguo Lian
* MambalC: State Space Models for High-Performance Learned Image Compression, Fanhu Zeng, Hao Tang, Yihua Shao, Siyu Chen, Ling Shao, Yan Wang
* Simpler Diffusion: 1.5 FID on ImageNet512 with Pixel-space Diffusion, Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, Tim Salimans
* Layer-and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers, Haoran You, Connelly Barnes, Yuqian Zhou, Yan Kang, Zhenbang Du, Wei Zhou, Lingzhi Zhang, Yotam Nitzan, Xiaoyang Liu, Zhe Lin, Eli Shechtman, Sohrab Amirghodsi, Yingyan Celine Lin
* Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DIT Acceleration, Haipeng Fang, Sheng Tang, Juan Cao, Enshuo Zhang, Fan Tang, Tong-Yee Lee
* NoiseCtrl: A Sampling-Algorithm-Agnostic Conditional Generation Method for Diffusion Models, Longquan Dai, He Wang, Jinhui Tang
* See Further When Clear: Curriculum Consistency Model, Yunpeng Liu, Boxiao Liu, Yi Zhang, Xingzhong Hou, Guanglu Song, Yu Liu, Haihang You
* RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories, Huiyang Shao, Xin Xia, Yuhong Yang, Yuxi Ren, Xing Wang, Xuefeng Xiao
* Pioneering 4-Bit FP Quantization for Diffusion Models: Mixup-Sign Quantization and Timestep-Aware Fine-Tuning, Maosen Zhao, Pengtao Chen, Chong Yu, Yan Wen, Xudong Tan, Tao Chen
* EIDT-V: Exploiting Intersections in Diffusion Trajectories for Model-Agnostic, Zero-Shot, Training-Free Text-to-Video Generation, Diljeet Jagpal, Xi Chen, Vinay P. Namboodiri
* Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation, Byung Hyun Lee, Sungjin Lim, Se Young Chun
* Random Conditioning with Distillation for Data-Efficient Diffusion Model Compression, Dohyun Kim, Sehwan Park, Geonhee Han, Seung Wook Kim, Paul Hongsuck Seo
* Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models, Reza Shirkavand, Peiran Yu, Shangqian Gao, Gowthami Somepalli, Tom Goldstein, Heng Huang
* The Art of Deception: Color Visual Illusions and Diffusion Models, Alexandra Gomez-Villa, Kai Wang, C.Alejandro Parraga, Bartłomiej Twardowski, Jesus Malo, Javier Vazquez-Corral, Joost van den Weijer
* Harnessing Frequency Spectrum Insights for Image Copyright Protection Against Diffusion Models, Zhenguang Liu, Chao Shuai, Shaojing Fan, Ziping Dong, Jinwu Hu, Zhongjie Ba, Kui Ren
* Hiding Images in Diffusion Models by Editing Learned Score Functions, Haoyu Chen, Yunqiao Yang, Nan Zhong, Kede Ma
* CDI: Copyrighted Data Identification in Diffusion Models, Jan Dubiński, Antoni Kowalczuk, Franziska Boenisch, Adam Dziedzic
* UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models, Yuning Han, Bingyin Zhao, Rui Chu, Feng Luo, Biplab Sikdar, Yingjie Lao
* Explaining in Diffusion: Explaining a Classifier with Diffusion Semantics, Tahira Kazimi, Ritika Allada, Pinar Yanardag
* Correcting Deviations from Normality: A Reformulated Diffusion Model for Multi-Class Unsupervised Anomaly Detection, Farzad Beizaee, Gregory A. Lodygensky, Christian Desrosiers, Jose Dolz
* Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation, Kunpeng Qiu, Zhiqiang Gao, Zhiying Zhou, Mingjie Sun, Yongxin Guo
* VasTSD: Learning 3D Vascular Tree-state Space Diffusion Model for Angiography Synthesis, Zhifeng Wang, Renjiao Yi, Xin Wen, Chenyang Zhu, Kai Xu
* Diffusion-based Realistic Listening Head Generation via Hybrid Motion Modeling, Yinuo Wang, Yanbo Fan, Xuan Wang, Guo Yu, Fei Wang

## **世界模型 (World Model)**


* **\[Oral\] GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion**, Object Dynamics, and Scene Composition Control, Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Pedro M B Rezende, Yasaman Haghighi, David Brüggemann, Isinsu Katircioglu, Lin Zhang, Xiaoran Chen, Suman Saha, Marco Cannici, Elie Aljalbout, Botao Ye, Xi Wang, Aram Davtyan, Mathieu Salzmann, Davide Scaramuzza, Marc Pollefeys, Paolo Favaro, Alexandre Alahi
* **\[Oral\] Navigation World Models**, Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun
* ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration, Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Guan Huang, Chen Liu, Yuyin Chen, Yida Wang, Xueyang Zhang, Yifei Zhan, Kun Zhan, Peng Jia, Xianpeng Lang, Xingang Wang, Wenjun Mei
* Scene Diffuser++: City-Scale Traffic Simulation via a Generative World Model, Shuhan Tan, John Lambert, Hong Jeon, Sakshum Kulshrestha, Yijing Bai, Jing Luo, Dragomir Anguelov, Mingxing Tan, Chiyu Max Jiang
* GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction, Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, Jiwen Lu
* PhysGen3D: Crafting a Miniature Interactive World from a Single Image, Boyuan Chen, Hanxiao Jiang, Shaowei Liu, Saurabh Gupta, Yunzhu Li, Hao Zhao, Shenlong Wang
* MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction, Jingcheng Ni, Yuxin Guo, Yichen Liu, Rui Chen, Lewei Lu, Zehuan Wu
* DIO: Decomposable Implicit 4D Occupancy-Flow World Model, Christopher Diehl, Quinlan Sykora, Ben Agro, Thomas Gilles, Sergio Casas, Raquel Urtasun
* Neural Motion Simulator Pushing the Limit of World Models in Reinforcement Learning, Chenjie Hao, Weyl Lu, Yifan Xu, Yubei Chen
* EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance, Yang Yue, Yulin Wang, Haojun Jiang, Pan Liu, Shiji Song, Gao Huang
* DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation, Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, Wenjun Mei, Xingang Wang
* Is Your World Simulator a Good Story Presenter? A Consecutive Events-Based Benchmark for Future Long Video Generation, Yiping Wang, Xuehai He, Kuan Wang, Luyao Ma, Jianwei Yang, Shuohang Wang, Simon Shaolei Du, Yelong Shen
* CheXWorld: Exploring Image World Modeling for Radiograph Representation Learning, Yang Yue, Yulin Wang, Chenxin Tao, Pan Liu, Shiji Song, Gao Huang
* Towards real-time multimodal world models for interactive experiences, Tabish Rashid, Dave Bignell, Raluca Georgescu, Mariya Hendriksen, Abdelhak Lemkhenter, Shanzheng Tan, Linda Wen, Katja Hofmann, Sarah Parisot

## **大语言模型 (LLM for Vision)**


* **\[Oral\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding**, Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, Bo Zhao
* **\[Oral\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons**, Andrew Szot, Bogdan Mazoure, Omar Attia, Aleksei Timofeev, Harsh Agrawal, Devon Hjelm, Zhe Gan, Zsolt Kira, Alexander Toshev
* *\[Highlight\] FirePlace: Geometric Refinements of LLM Common Sense Reasoning for 3D Object Placement*, lan Huang, Yanan Bao, Karen Truong, Howard Zhou, Cordelia Schmid, Leonidas Guibas, Alireza Fathi
* ChatGarment: Garment Estimation, Generation and Editing via Large Language Models, Siyuan Bian, Chenghao Xu, Yuliang Xiu, Artur Grigorev, Zhen Liu, Cewu Lu, Michael J. Black, Yao Feng
* Bridging Gait Recognition and Large Language Models Sequence Modeling, Shaopeng Yang, Jilong Wang, Saihui Hou, Xu Liu, Chunshui Cao, Liang Wang, Yongzhen Huang
* COLLM: A Large Language Model for Composed Image Retrieval, Chuong Huynh, Jinyu Yang, Ashish Tawari, Mubarak Shah, Son Tran, Raffay Hamid, Trishul Chilimbi, Abhinav Shrivastava
* Patient-Level Anatomy Meets Scanning-Level Physics: Personalized Federated Low-Dose CT Denoising Empowered by Large Language Model, Ziyuan Yang, Yingyu Chen, Zhiwen Wang, Hongming Shan, Yang Chen, Yi Zhang
* 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models, Wanhua Li, Renping Zhou, Jiawei Zhou, Yingwei Song, Johannes Herter, Minghan Qin, Gao Huang, Hanspeter Pfister
* Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction, Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, Jiaqi Wang
* Protecting Your Video Content: Disrupting Automated Video-based LLM Annotations, Haitong Liu, Kuofeng Gao, Yang Bai, Jinmin Li, Jinxiao Shan, Tao Dai, Shu-Tao Xia
* Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models, Ronghuan Wu, Wanchao Su, Jing Liao
* A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation, Andrew Z. Wang, Songwei Ge, Tero Karras, Ming-Yu Liu, Yogesh Balaji
* Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models, Jinhui Yi, Syed Talal Wasim, Yanan Luo, Muzammal Naseer, Juergen Gall
* Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMS, Zeyi Huang, Yuyang Ji, Xiaofang Wang, Nikhil Mehta, Tong Xiao, Donghyun Lee, Sigmund Vanvalkenburgh, Shengxin Zha, Bolin Lai, Licheng Yu, Ning Zhang, Yong Jae Lee, Miao Liu
* SLADE: Shielding against Dual Exploits in Large Vision-Language Models, Md Zarif Hossain, Ahmed Imteaj, Shaden Alshammari
* LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models, Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, Jiajun Wu
* The Photographer's Eye: Teaching Multimodal Large Language Models to See, and Critique Like Photographers, Daiqing Qi, Handong Zhao, Jing Shi, Simon Jenni, Yifei Fan, Franck Dernoncourt, Scott Cohen, Sheng Li
* HiRes-LLaVA: Restoring Fragmentation Input in High-Resolution Large Vision-Language Models, Runhui Huang, Xinpeng Ding, Chunwei Wang, Jianhua Han, Yulong Liu, Hengshuang Zhao, Hang Xu, Lu Hou, Wei Zhang, Xiaodan Liang
* VoCo-LLaMA: Towards Vision Compression with Large Language Models, Xubing Ye, Yukang Gan, Xiaoke Huang, Yixiao Ge, Yansong Tang
* Accelerating Multimodal Large Language Models by Searching Optimal Vision Token Reduction, Shiyu Zhao, Zhenting Wang, Felix Juefei-Xu, Xide Xia, Miao Liu, Xiaofang Wang, Mingfu Liang, Ning Zhang, Dimitris N. Metaxas, Licheng Yu
* HalLoc: Token-level Localization of Hallucinations for Vision Language Models, Eunkyu Park, Minyeong Kim, Gunhee Kim
* Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy, Joonhyun Jeong, Seyun Bae, Yeonsung Jung, Jaeryong Hwang, Eunho Yang
* LLM-driven Multimodal and Multi-Identity Listening Head Generation, Peiwen Lai, Weizhi Zhong, Yipeng Qin, Xiaohang Ren, Baoyuan Wang, Guanbin Li
* GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation, Ning Gao, Yilun Chen, Shuai Yang, Xinyi Chen, Yang Tian, Hao Li, Haifeng Huang, Hanqing Wang, Tai Wang, Jiangmiao Pang
* FilmComposer: LLM-Driven Music Production for Silent Film Clips, Zhifeng Xie, Qile He, Youjia Zhu, Qiwei He, Mengtian Li
* LLMDet: Learning Strong Open-Vocabulary Object Detectors under the Supervision of Large Language Models, Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei, Jingke Meng, Xiaohua Xie, Wei-Shi Zheng
* PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation, Qiyao Xue, Xiangyu Yin, Boyuan Yang, Wei Gao
* CountLLM: Towards Generalizable Repetitive Action Counting via Large Language Model, Ziyu Yao, Xuxin Cheng, Zhiqi Huang, Lei Li
* SKE-Layout: Spatial Knowledge Enhanced Layout Generation with LLMs, Junsheng Wang, Nieqing Cao, Yan Ding, Mengying Xie, Fuqiang Gu, Chao Chen
* R2C: Mapping Room to Chessboard to Unlock LLM As Low-Level Action Planner, Ziyi Bai, Hanxuan Li, Bin Fu, Chuyan Xiong, Ruiping Wang, Xilin Chen
* Empowering LLMs to Understand and Generate Complex Vector Graphics, Ximing Xing, Juncheng Hu, Guotao Liang, Jing Zhang, Dong Xu, Qian Yu
* A Simple yet Effective Layout Token in Large Language Models for Document Understanding, Zhaoqing Zhu, Chuwei Luo, Zirui Shao, Feiyu Gao, Hangdi Xing, Qi Zheng, Ji Zhang
* CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation, Jiahao Li, Weijian Ma, Xueyang Li, Yunzhong Lou, Guichun Zhou, Xiangdong Zhou
* Video Summarization with Large Language Models, Min Jung Lee, Dayoung Gong, Minsu Cho

## **多模态大模型 (Large Multimodal Models)**


* **\[Oral\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models**, Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, Aniruddha Kembhavi
* **\[Oral\] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding**, Feilong Tang, Chengzhi Liu, Zhongxing Xu, Ming Hu, Zile Huang, Haochen Xue, Ziyang Chen, Zelin Peng, Zhiwei Yang, Sijin Zhou, Wenxue Li, Yulong Li, Wenxuan Song, Shiyan Su, Wei Feng, Jionglong Su, Mingquan Lin, Yifan Peng, Xuelian Cheng, Imran Razzak, Zongyuan Ge
* **\[Oral\] LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models**, Jian Liang, Wenke Huang, Guancheng Wan, Qu Yang, Mang Ye
* **\[Oral\] Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens**, Kaihang Pan, Wang Lin, Zhongqi Yue, Tenglong Ao, Liyu Jia, Wei Zhao, Juncheng Li, Siliang Tang, Hanwang Zhang
* **\[Oral\] Thinking in Space: How Multimodal Large Language Models See**, Remember, and Recall Spaces, Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, Saining Xie
* **\[Oral\] Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key**, Zhihe Yang, Xufang Luo, Dongqi Han, Yunjian Xu, Dongsheng Li
* **\[Oral\] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models**, Xinyu Tian, Shu Zou, Zhaoyuan Yang, Jing Zhang
* *\[Highlight\] Hyperbolic Safety-Aware Vision-Language Models*, Tobia Poppi, Tejaswi Kasarla, Pascal Mettes, Lorenzo Baraldi, Rita Cucchiara
* *\[Highlight\] DyFo: A Training-Free Dynamic Focus Visual Search for Enhancing LMMs in Fine-Grained Visual Understanding*, Geng Li, Jinglin Xu, Yunzhen Zhao, Yuxin Peng
* *\[Highlight\] Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis*, Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, Xing Sun
* *\[Highlight\] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding*, Wei Suo, Lijun Zhang, Mengyang Sun, Lin Yuanbo Wu, Peng Wang, Yanning Zhang
* *\[Highlight\] XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?*, Fengxiang Wang, Hongzhen Wang, Zonghao Guo, Di Wang, Yulin Wang, Mingshuo Chen, Qiang Ma, Long Lan, Wenjing Yang, Jing Zhang, Zhiyuan Liu, Maosong Sun
* S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation, Yichen Xie, Runsheng Xu, Tong He, Jyh-Jing Hwang, Katie Luo, Jingwei Ji, Hubert Lin, Letian Chen, Yiren Lu, Zhaoqi Leng, Dragomir Anguelov, Mingxing Tan
* JTD-UAV: MLLM-Enhanced Joint Tracking and Description Framework for Anti-UAV Systems, Yifan Wang, Jian Zhao, Zhaoxin Fan, Xin Zhang, Xuecheng Wu, Yudian Zhang, Lei Jin, Xinyue Li, Gang Wang, Mengxi Jia, Ping Hu, Zheng Zhu, Xuelong Li
* SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model, Chunlin Yu, Hanqing Wang, Ye Shi, Haoyang Luo, Sibei Yang, Jingyi Yu, Jingya Wang
* EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions, Kai Chen, Yunhao Gou, Runhui Huang, Zhili Liu, Daxin Tan, Jing Xu, Chunwei Wang, Yi Zhu, Yihan Zeng, Kuo Yang, Dingdong Wang, Kun Xiang, Haoyuan Li, Haoli Bai, Jianhua Han, Xiaohui Li, Weike Jin, Nian Xie, Yu Zhang, James T. Kwok, Hengshuang Zhao, Xiaodan Liang, Dit-Yan Yeung, Xiao Chen, Zhenguo Li, Wei Zhang, Qun Liu, Lanqing Hong, Lu Hou, Hang Xu
* V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents, Zhengrong Yue, Shaobin Zhuang, Kunchang Li, Yanbo Ding, Yali Wang
* LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences, Hongyan Zhi, Peihao Chen, Junyan Li, Shuailei Ma, Xinyu Sun, Tianhang Xiang, Yinjie Lei, Mingkui Tan, Chuang Gan
* SegAgent: Exploring Pixel Understanding Capabilities in MLLMS by Imitating Human Annotator Trajectories, Muzhi Zhu, Yuzhuo Tian, Hao Chen, Chunluan Zhou, Qingpei Guo, Yang Liu, Ming Yang, Chunhua Shen
* HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator, Fan Yang, Ru Zhen, Jianing Wang, Yanhao Zhang, Haoxiang Chen, Haonan Lu, Sicheng Zhao, Guiguang Ding
* PEACE: Empowering Geologic Map Holistic Understanding with MLLMs, Yangyu Huang, Tianyi Gao, Haoran Xu, Qihao Zhao, Yang Song, Zhipeng Gui, Tengchao Lv, Hao Chen, Lei Cui, Scarlett Li, Furu Wei
* MMTL-UniAD: A Unified Framework for Multimodal and Multi-Task Learning in Assistive Driving Perception, Wenzhuo Liu, Wenshuo Wang, Yicheng Qiao, Qiannan Guo, Jiayin Zhu, Pengfei Li, Zilong Chen, Huiming Yang, Zhiwei Li, Lening Wang, Tiao Tan, Huaping Liu
* BlueLM-V-3B: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices, Xudong Lu, Yinghao Chen, Cheng Chen, Hui Tan, Boheng Chen, Yina Xie, Rui Hu, Guanxin Tan, Renshou Wu, Yan Hu, Yi Zeng, Lei Wu, Liuyang Bian, Zhaoxiong Wang, Long Liu, Yanzhou Yang, Han Xiao, Aojun Zhou, Yafei Wen, Xiaoxin Chen, Shuai Ren, Hongsheng Li
* Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices, Junyan Lin, Haoran Chen, Yue Fan, Yingqi Fan, Xin Jin, Hui Su, Jinlan Fu, Xiaoyu Shen
* MBQ: Modality-Balanced Quantization for Large Vision-Language Models, Shiyao Li, Yingchun Hu, Xuefei Ning, Xihui Liu, Ke Hong, Xiaotao Jia, Xiuhong Li, Yaqi Yan, Pei Ran, Guohao Dai, Shengen Yan, Huazhong Yang, Yu Wang
* ICT: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models, Junzhe Chen, Tianshu Zhang, Shiyu Huang, Yuwei Niu, Linfeng Zhang, Lijie Wen, Xuming Hu
* Forensics-Bench: A Comprehensive Forgery Detection Benchmark Suite for Large Vision Language Models, Jin Wang, Chenghui Lv, Xian Li, Shichao Dong, Huadong Li, Kelu Yao, Chao Li, Wengi Shao, Ping Luo
* SnowMaster: Comprehensive Real-world Image Desnowing via MLLM with Multi-Model Feedback Optimization, Jianyu Lai, Sixiang Chen, Yunlong Lin, Tian Ye, Yun Liu, Song Fei, Zhaohu Xing, Hongtao Wu, Weiming Wang, Lei Zhu
* Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning, Di Zhang, Jingdi Lei, Junxian Li, Xunzhi Wang, Yujie Liu, Zonglin Yang, Jiatong Li, Weida Wang, Suorong Yang, Jianbo Wu, Peng Ye, Wanli Ouyang, Dongzhan Zhou
* Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly, Yexin Liu, Zhengyang Liang, Yueze Wang, Xianfeng Wu, Feilong Tang, Muyang He, Jian Li, Zheng Liu, Harry Yang, Sernam Lim, Bo Zhao
* UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation, Qihui Zhang, Munan Ning, Zheyuan Liu, Yue Huang, Shuo Yang, Yanbo Wang, Jiayi Ye, Xiao Chen, Yibing Song, Li Yuan
* Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models, Qirui Jiao, Daoyuan Chen, Yilun Huang, Bolin Ding, Yaliang Li, Ying Shen
* CASP: Compression of Large Multimodal Models Based on Attention Sparsity, Mohsen Gholami, Mohammad Akbari, Kevin Cannons, Yong Zhang
* Lifting the Veil on Visual Information Flow in MLLMs: Unlocking Pathways to Faster Inference, Hao Yin, Guangzong Si, Zilei Wang
* DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models, Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, Yong Zhang
* Libra-Merging: Importance-redundancy and Pruning-merging Trade-off for Acceleration Plug-in in Large Vision-Language Model, Longrong Yang, Dong Shen, Chaoxiang Cai, Kaibing Chen, Fan Yang, Tingting Gao, Di Zhang, Xi Li
* AdaMMS: Model Merging for Heterogeneous Multimodal Large Language Models with Unsupervised Coefficient Optimization, Yiyang Du, Xiaochen Wang, Chi Chen, Jiabo Ye, Yiru Wang, Peng Li, Ming Yan, Ji Zhang, Fei Huang, Zhifang Sui, Maosong Sun, Yang Liu
* Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization, Zefeng Zhang, Hengzhu Tang, Jiawei Sheng, Zhenyu Zhang, Yiming Ren, Zhenyang Li, Dawei Yin, Duohe Ma, Tingwen Liu
* From Head to Tail: Towards Balanced Representation in Large Vision-Language Models through Adaptive Data Calibration, Mingyang Song, Xiaoye Qu, Jiawei Zhou, Yu Cheng
* EfficientLLaVA: Generalizable Auto-Pruning for Large Vision-language Models, Yinan Liang, Ziwei Wang, Xiuwei Xu, Jie Zhou, Jiwen Lu
* Distraction is All You Need for Multimodal Large Language Model Jailbreaking, Zuopeng Yang, Jiluan Fan, Anli Yan, Erdun Gao, Xin Lin, Tao Li, Kanghua Mo, Changyu Dong
* On the Out-Of-Distribution Generalization of Large Multimodal Models, Xingxuan Zhang, Jiansheng Li, Wenjing Chu, junjia hai, Renzhe Xu, Yuqing Yang, Shikai Guan, Jiazheng Xu, Liping Jing, Peng Cui
* Weakly Supervised Temporal Action Localization via Dual-Prior Collaborative Learning Guided by Multimodal Large Language Models, Quan Zhang, Jinwei Fang, Rui Yuan, Xi Tang, Yuxin Qi, Ke Zhang, Chun Yuan
* Efficient Motion-Aware Video MLLM, Zijia Zhao, Yuqi Huo, Tongtian Yue, Longteng Guo, Haoyu Lu, Bingning Wang, Weipeng Chen, Jing Liu
* Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model, Yuting Zhang, Hao Lu, Qingyong Hu, Yin Wang, Kaishen Yuan, Xin Liu, Kaishun Wu
* ECBench: Can Multi-modal Foundation Models Understand the Egocentric World? A Holistic Embodied Cognition Benchmark, Ronghao Dang, Yuqian Yuan, Wenqi Zhang, Yifei Xin, Boqiang Zhang, Long Li, Liuyi Wang, Qinyang Zeng, Xin Li, Lidong Bing
* AesthetiQ: Enhancing Graphic Layout Design via Aesthetic-Aware Preference Alignment of Multi-modal Large Language Models, Sohan Patnaik, Rishabh Jain, Balaji Krishnamurthy, Mausoom Sarkar
* MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output, Yanyuan Chen, Dexuan Xu, Yu Huang, Songkun Zhan, Hanpin Wang, Dongxue Chen, Xueping Wang, Meikang Qiu, Hang Li
* Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Mutimodal Models, Xingrui Wang, Wufei Ma, Tiezheng Zhang, Celso M de Melo, Jieneng Chen, Alan Yuille
* Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training, Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jiawen Liu, Jifeng Dai, Yu Qiao, Xizhou Zhu
* Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment, Ziang Yan, Zhilin Li, Yinan He, Chenting Wang, Kunchang Li, Xinhao Li, Xiangyu Zeng, Zilei Wang, Yali Wang, Yu Qiao, Limin Wang, Yi Wang
* Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention, Wenbin An, Feng Tian, Sicong Leng, Jiahao Nie, Haonan Lin, Qianying Wang, Ping Chen, Xiaoqin Zhang, Shijian Lu
* BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models, Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, Lichao Sun
* MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities, Bizhu Wu, Jinheng Xie, Keming Shen, Zhe Kong, Jianfeng Ren, Ruibin Bai, Rong Qu, Linlin Shen
* Distilling Multi-modal Large Language Models for Autonomous Driving, Deepti Hegde, Rajeev Yasarla, Hong Cai, Shizhong Han, Apratim Bhattacharyya, Shweta Mahajan, Litian Liu, Rísheek Garrepalli, Vishal M. Patel, Fatih Porikli
* LLAVIDAL: A Large LAnguage Vision Model for Daily Activities of Living, Dominick Reilly, Rajatsubhra Chakraborty, Arkaprava Sinha, Manish Kumar Govind, Pu Wang, Francois Bremond, Le Xue, Srijan Das
* SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model, Zhenglin Huang, Jinwei Hu, Xiangtai Li, Yiwei He, Xingyu Zhao, Bei Peng, Baoyuan Wu, Xiaowei Huang, Guangliang Cheng
* Conformal Prediction and MLLM aided Uncertainty Quantification in Scene Graph Generation, Sayak Nag, Udita Ghosh, Calvin-Khang Ta, Sarosij Bose, Jiachen Li, Amit K. Roy-Chowdhury
* DTOS: Dynamic Time Object Sensing with Large Multimodal Model, Jirui Tian, Jinrong Zhang, Shenglan Liu, Luhao Xu, Zhixiong Huang, Gao Huang
* Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation, Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, Ping Luo
* LLAVA-Critic: Learning to Evaluate Multimodal Models, Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, Chunyuan Li
* M-LLM Based Video Frame Selection for Efficient Video Understanding, Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi
* VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding, Chaoyu Li, Eun Woo Im, Pooyan Fazli
* Inst3D-LMM: Instance-Aware 3D Scene Understanding with Multi-modal Instruction Tuning, Hanxun Yu, Wentong Li, Song Wang, Junbo Chen, Jianke Zhu
* Magma: A Foundation Model for Multimodal Al Agents, Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Jianfeng Gao
* Is Right' Right? Enhancing Object Orientation Understanding in Multimodal Large Language Models through Egocentric Instruction Tuning, Ji Hyeok Jung, Eun Tae Kim, Seoyeon Kim, Joo Ho Lee, Bumsoo Kim, Buru Chang
* ROD-MLLM: Towards More Reliable Object Detection in Multimodal Large Language Models, Heng Yin, Yuqiang Ren, Ke Yan, Shouhong Ding, Yongtao Hao
* Human-centered Interactive Learning via MLLMs for Text-to-Image Person Re-identification, Yang Qin, Chao Chen, Zhihang Fu, Dezhong Peng, Xi Peng, Peng Hu
* RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models, Haoran Hao, Jiaming Han, Changsheng Li, Yu-Feng Li, Xiangyu Yue
* Continual SFT Matches Multimodal RLHF with Negative Supervision, Ke Zhu, Yu Wang, Yanpeng Sun, Qiang Chen, Jiangjiang Liu, Gang Zhang, Jingdong Wang
* ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation in Multimodal Large Language Models, Hao Yin, Guangzong Si, Zilei Wang
* Nullu: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection, Le Yang, Ziwei Zheng, Boxu Chen, Zhengyu Zhao, Chenhao Lin, Chao Shen
* Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception, Yuanchen Wu, Lu Zhang, Hang Yao, Junlong Du, Ke Yan, Shouhong Ding, Yunsheng Wu, Xiaoqiang Li
* MLLM-as-a-Judge for Image Safety without Human Labeling, Zhenting Wang, Shuming Hu, Shiyu Zhao, Xiaowen Lin, Felix Juefei-Xu, Zhuowei Li, Ligong Han, Harihar Subramanyam, Li Chen, Jianfa Chen, Nan Jiang, Lingjuan Lyu, Shiqing Ma, Dimitris N. Metaxas, Ankit Jain
* SpatialLLM: A Compound 3D-Informed Design towards Spatially-Intelligent Large Multimodal Models, Wufei Ma, Luoxin Ye, Celso M de Melo, Alan Yuille, Jieneng Chen
* Apollo: An Exploration of Video Understanding in Large Multimodal Models, Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena Yeung-Levy, Xide Xia
* Notes-guided MLLM Reasoning: Enhancing MLLM with Knowledge and Visual Notes for Visual Question Answering, Wenlong Fang, Qiaofeng Wu, Jing Chen, Yun Xue
* GENIUS: A Generative Framework for Universal Multimodal Search, Sungyeon Kim, Xinliang Lin, Xiaofan Lin, Muhammet Bastan, Douglas Gray, Suha Kwak
* AdaDARE-gamma: Balancing Stability and Plasticity in Multi-modal LLMs through Efficient Adaptation, Jingyi Xie, Jintao Yang, Zhunchen Luo, Yunbo Cao, Qiang Gao, Mengyuan Zhang, Wenpeng Hu
* Cross-modal Information Flow in Multimodal Large Language Models, Zhi Zhang, Srishti Yadav, Fengze Han, Ekaterina Shutova
* ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models, Yahan Tu, Rui Hu, Jitao Sang
* PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset, Jiazhen Liu, Yuhan Fu, Ruobing Xie, Runquan Xie, Xingwu Sun, Fengzong Lian, Zhanhui Kang, Xirong Li
* SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models, Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin, Yu Qiao, Xuanjing Huang, Feng Zhao, Tao Gui, Jing Shao
* Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models?, Yanbo Wang, Jiyang Guan, Jian Liang, Ran He
* Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models, Jiacong Xu, Shao-Yuan Lo, \+Bardia Safaei, Vishal M. Patel, Isht Dwivedi
* RLAIF-V: Open-Source Al Feedback Leads to Super GPT-4V Trustworthiness, Tianyu Yu, Haoye Zhang, Qiming Li, Qixin Xu, Yuan Yao, Da Chen, Xiaoman Lu, Ganqu Cui, Yunkai Dang, Taiwen He, Xiaocheng Feng, Jun Song, Bo Zheng, Zhiyuan Liu, T at-Seng Chua, Maosong Sun

## **视觉语言模型 (Vision-Language / VLM)**


* **\[Oral\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector**, Xiao Guo, Xiufeng Song, Yue Zhang, Xiaohong Liu, Xiaoming Liu
* **\[Oral\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics**, Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, Stan Birchfield
* *\[Highlight\] Grounding Face: Fine-grained Face Understanding via Pixel Grounding Multimodal Large Language Model*, Yue Han, Jiangning Zhang, Junwei Zhu, Runze Hou, Xiaozhong Ji, Chuming Lin, Xiaobin Hu, Zhucun Xue, Yong Liu
* *\[Highlight\] Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation*, Xin Zhang, Robby T. Tan
* *\[Highlight\] Assessing and Learning Alignment of Unimodal Vision and Language Models*, Le Zhang, Qian Yang, Aishwarya Agrawal
* COT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models, Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Tsung-Yi Lin, Gordon Wetzstein, Ming-Yu Liu, Donglai Xiang
* MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation, Zhenyu Wu, Yuheng Zhou, Xiuwei Xu, Ziwei Wang, Haibin Yan
* VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary, Kevin Qinghong Lin, Mike Zheng Shou
* PAVE: Patching and Adapting Video Large Language Models, Zhuoming Liu, Yiquan Li, Khoi Duc Nguyen, Yiwu Zhong, Yin Li
* BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding, Shuming Liu, Chen Zhao, Tianqi Xu, Bernard Ghanem
* Synthetic Data is an Elegant GIFT for Continual Vision-Language Models, Bin Wu, Wuxuan Shi, Jinqiao Wang, Mang Ye
* PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability, Weijie Zhou, Manli Tao, Chaoyang Zhao, Haiyun Guo, Honghui Dong, Ming Tang, Jinqiao Wang
* GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks, Haoqiang Kang, Enna Sachdeva, Piyush Gupta, Sangjae Bae, Kwonjoon Lee
* Perception Tokens Enhance Visual Reasoning in Multimodal Language Models, Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G. Shapiro, Ranjay Krishna
* Do Visual Imaginations Improve Vision-and-Language Navigation Agents?, Akhil Perincherry, Jacob Krantz, Stefan Lee
* Words or Vision: Do Vision-Language Models Have Blind Faith in Text?, Ailin Deng, Tri Cao, Zhirui Chen, Bryan Hooi
* VisionArena: 230k Real World User-VLM Conversations with Preference Labels, Christopher Chou, Lisa Dunlap, Koki Mashita, Krishna Mandal, Trevor Darrell, lon Stoica, Joseph E. Gonzalez, Wei-Lin Chiang
* VladVA: Discriminative Fine-tuning of LVLMs, Yassine Ouali, Adrian Bulat, Alexandros Xenos, Anestis Zaganidis, loannis Maniadis Metaxas, Brais Martinez, Georgios Tzimiropoulos
* Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding, Tianyu Chen, Xingcheng Fu, Yisen Gao, Haodong Qian, Yuecen Wei, Kun Yan, Haoyi Zhou, Jianxin Li
* NVILA: Efficient Frontier Visual Language Models, Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, Xiuyu Li, Haotian Tang, Yunhao Fang, Yukang Chen, Cheng-Yu Hsieh, De-An Huang, An-Chieh Cheng, Jinyi Hu, Sifei Liu, Ranjay Krishna, Pavio Molchanov, Jan Kautz, Hongxu Yin, Song Han, Yao Lu
* Unveiling Visual Perception in Language Models: An Attention Head Analysis Approach, Jing Bi, Junjia Guo, Yunlong Tang, Lianggong Bruce Wen, Zhang Liu, Chenliang Xu, Bingjie Wang
* Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Large Model Enhancement, Qianhan Feng, Wenshuo Li, Tong Lin, Xinghao Chen
* VASparse: Towards Efficient Visual Hallucination Mitigation via Visual-Aware Token Sparsification, Xianwei Zhuang, Zhihong Zhu, Yuxin Xie, Liming Liang, Yuexian Zou
* Stop Learning it all to Mitigate Visual Hallucination, Focus on the Hallucination Target. Dokyoon Yoon, Youngsook Song, Woomyoung Park
* Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts, Qizhou Chen, Chengyu Wang, Dakan Wang, Taolin Zhang, Wangyue Li, Xiaofeng He
* Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift, Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Mingli Zhu, Xiaochun Cao, Dacheng Tao
* Vision-Language Model IP Protection via Prompt-based Learning, Lianyu Wang, Meng Wang, Huazhu Fu, Daoqiang Zhang
* VISCO: Benchmarking Fine-Grained Critique and Correction Towards Self-Improvement in Visual Reasoning, Xueqing Wu, Yuheng Ding, Bingxuan Li, Pan Lu, Da Yin, Kai-Wei Chang, Nanyun Peng
* CALICO: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models, Kiet A. Nguyen, Adheesh Juvekar, Tianjiao Yu, Muntasir Wahed, Ismini Lourentzou
* InteractVLM: 3D Interaction Reasoning from 2D Foundational Models, Sai Kumar Dwivedi, Dimitrije Antić, Shashank Tripathi, Omid Taheri, Cordelia Schmid, Michael J. Black, Dimitrios Tzionas
* Embodied Scene Understanding for Vision Language Models via MetaVQA, Weizhen Wang, Chenda Duan, Zhenghao Peng, Yuxin Liu, Bolei Zhou
* OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning, Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, Jose M. Alvarez
* RoboGround: Robotic Manipulation with Grounded Vision-Language Priors, Haifeng Huang, Xinyi Chen, Yilun Chen, Hao Li, Xiaoshen Han, Zehan Wang, Tai Wang, Jiangmiao Pang, Zhou Zhao
* Can Text-to-Video Generation help Video-Language Alignment?, Luca Zanella, Massimiliano Mancini, Willi Menapace, Sergey Tulyakov, Yiming Wang, Elisa Ricci
* Unveiling the Mist over 3D Vision-Language Understanding: Object-centric Evaluation with Chain-of-Analysis, Jiangyong Huang, Baoxiong Jia, Yan Wang, Ziyu Zhu, Xiongkun Linghu, Qing Li, Song-Chun Zhu, Siyuan Huang
* Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels, Yongshuo Zong, Qin Zhang, Dongsheng An, Zhihua Li, Xiang Xu, Linghan Xu, Zhuowen Tu, Yifan Xing, Onkar Dabeer
* VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models, Lei Li, Yuancheng Wei, Zhihui Xie, Xuqing Yang, Yifan Song, Peiyi Wang, Chenxin An, Tianyu Liu, Sujian Li, Bill Yuchen Lin, Lingpeng Kong, Qi Liu
* F-LMM: Grounding Frozen Large Multimodal Models, Size Wu, Sheng Jin, Wenwei Zhang, Lumin Xu, Wentao Liu, Wei Li, Chen Change Loy
* Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models, Davide Berasi, Matteo Farina, Massimiliano Mancini, Elisa Ricci, Nicola Strisciuglio
* Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion, Jiuhai Chen, Jianwei Yang, Haiping Wu, Dianqi Li, Jianfeng Gao, Tianyi Zhou, Bin Xiao
* PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models, Chenyu Yang, Xuan Dong, Xizhou Zhu, Weijie Su, Jiahao Wang, Hao Tian, Zhe Chen, Wenhai Wang, Lewei Lu, Jifeng Dai
* ATP-LLAVA: Adaptive Token Pruning for Large Vision Language Models, Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, Yansong Tang
* It's a (Blind) Match\! Towards Vision-Language Correspondence without Parallel Data, Dominik Schnaus, Nikita Araslanov, Daniel Cremers
* Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens, Zhangqi Jiang, Junkai Chen, Beier Zhu, Tingjin Luo, Yankun Shen, Xu Yang
* MMRL: Multi-Modal Representation Learning for Vision-Language Models, Yuncheng Guo, Xiaodong Gu
* Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment, Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi
* SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments, Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo
* Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models, Zhaoyi Liu, Huan Zhang
* PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models, Jenny Schmalfuss, Nadine Chang, Vibashan VS, Maying Shen, Andres Bruhn, Jose M. Alvarez
* ProkeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models, Yassir Bendou, Amine Ouasfi, Vincent Gripon, Adnane Boukhayma
* Realistic Test-Time Adaptation of Vision-Language Models, Maxime Zanella, Clément Fuchs, Christophe De Vleeschouwer, Ismail Ben Ayed
* VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks, Jinseong Jang, Chunfei Ma, Byeongwon Lee
* R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning, Lijun Sheng, Jian Liang, Zilei Wang, Ran He
* Bayesian Test-Time Adaptation for Vision-Language Models, Lihua Zhou, Mao Ye, Shuaifeng Li, Nianxin Li, Xiatian Zhu, Lei Deng, Hongbin Liu, Zhen Lei
* Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation, Xiaoqi Li, Jingyun Xu, Mingxu Zhang, Jiaming Liu, Yan Shen, laroslav Ponomarenko, Jiahui Xu, Liang Heng, Siyuan Huang, Shanghang Zhang, Hao Dong
* Vision-Language Embodiment for Monocular Depth Estimation, Jinchang Zhang, Guoyu Lu
* Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method, Xinshuai Song, Weixing Chen, Yang Liu, Weikai Chen, Guanbin Li, Liang Lin
* FireEdit: Fine-grained Instruction-based Image Editing via Region-aware Vision Language Model, Jun Zhou, Jiahao Li, Zunnan Xu, Hanhui Li, Yiji Cheng, Fa-Ting Hong, Qin Lin, Qinglin Lu, Xiaodan Liang
* Self-Evolving Visual Concept Library using Vision-Language Critics, Atharva Sehgal, Patrick Yuan, Ziniu Hu, Yisong Yue, Jennifer J. Sun, Swarat Chaudhuri
* HOVLE: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding, Chenxin Tao, Shiqian Su, Xizhou Zhu, Chenyu Zhang, Zhe Chen, Jiawen Liu, Wenhai Wang, Lewei Lu, Gao Huang, Yu Qiao, Jifeng Dai
* FlashSloth: Lightning Multimodal Large Language Models via Embedded Visual Compression, Bo Tong, Bokai Lai, Yiyi Zhou, Gen Luo, Yunhang Shen, Ke Li, Xiaoshuai Sun, Rongrong Ji
* PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models, Mohamed Dhouib, Davide Buscaldi, Sonia Vanier, Aymen Shabou
* Conical Visual Concentration for Efficient Large Vision-Language Models, Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, Dahua Lin
* Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?, Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, David Acuna
* Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks, Peng Xie, Yequan Bie, Jianda Mao, Yangqiu Song, Yang Wang, Hao Chen, Kani Chen
* COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training, Sanghwan Kim, Rui Xiao, Mariana-luliana Georgescu, Stephan Alaniz, Zeynep Akata
* Reproducible Vision-Language Models Meet Concepts Out of Pre-Training, Ziliang Chen, Xin Huang, Xiaoxuan Fan, Keze Wang, Yuyu Zhou, Quanlong Guan, Liang Lin
* Once-Tuning-Multiple-Variants: Tuning Once and Expanded as Multiple Vision-Language Model Variants, Chong Yu, Tao Chen, Zhongxue Gan
* Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves, Shihan Wu, Ji Zhang, Pengpeng Zeng, Lianli Gao, Jingkuan Song, Heng Tao Shen
* SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling, Qi Zhu, Jiangwei Lao, Deyi Ji, Junwei Luo, Kang Wu, Yingying Zhang, Lixiang Ru, Jian Wang, Jingdong Chen, Ming Yang, Dong Liu, Feng Zhao
* Task-Aware Clustering for Prompting Vision-Language Models, Fusheng Hao, Fengxiang He, Fuxiang Wu, Tichao Wang, Chengqun Song, Jun Cheng
* BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models, Taha Koleilat, Hojat Asgariandehkordi, Hassan Rivaz, Yiming Xiao
* VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge, Vishwesh Nath, Wenqi Li, Dong Yang, Andriy Myronenko, Mingxin Zheng, Yao Lu, Zhijian Liu, Hongxu Yin, Yee Man Law, Yucheng Tang, Pengfei Guo, Can Zhao, Ziyue Xu, Yufan He, Stephanie Harmon, Benjamin Simon, Greg Heinrich, Stephen Aylward, Marc Edgar, Michael Zephyr, Pavlo Molchanov, Baris Turkbey, Holger Roth, Daguang Xu
* Overcoming Shortcut Problem in VLM for Robust Out-of-Distribution Detection, Zhuo Xu, Xiang Xiang, Yifan Liang
* FastVLM: Efficient Vision Encoding for Vision Language Models, Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari
* VisionZip: Longer is Better but Not Necessary in Vision Language Models, Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, Jiaya Jia
* TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model, Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, Bo Yuan
* A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for Accelerating Large VLMs, Wangbo Zhao, Yizeng Han, Jiasheng Tang, Zhikai Li, Yibing Song, Kai Wang, Zhangyang Wang, Yang You
* MOVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders, Jiajun Cao, Yuan Zhang, Tao Huang, Ming Lu, Qizhe Zhang, Ruichuan An, Ningning Ma, Shanghang Zhang
* Exploring Visual Vulnerabilities via Multi-Loss Adversarial Search for Jailbreaking Vision-Language Models, Shuyang Hao, Bryan Hooi, Jun Liu, Kai-Wei Chang, Zi Huang, Yujun Cai
* Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models, Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, Dit-Yan Yeung
* TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models, Xin Wang, Kai Chen, Jiaming Zhang, Jingjing Chen, Xingjun Ma
* On the Zero-shot Adversarial Robustness of Vision-Language Models: A Truly Zero-shot and Training-free Approach, Baoshun Tong, Hanjiang Lai, Yan Pan, Jian Yin
* O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models, Ashshak Sharifdeen, \+Muhammad Akhtar Munir, Sanoojan Baliah, Salman Khan, Muhammad Haris Khan
* NLPrompt: Noise-Label Prompt Learning for Vision-Language Models, Bikang Pan, Qun Li, Xiaoying Tang, Wei Huang, Zhen Fang, Feng Liu, Jingya Wang, Jingyi Yu, Ye Shi
* F^3OCUS- Federated Finetuning of Vision-Language Foundation Models with Optimal Client Layer Updating Strategy via Multi-objective Meta-Heuristics, Pramit Saha, Felix Wagner, Divyanshu Mishra, Can Peng, Anshul Thakur, David A. Clifton, Konstantinos Kamnitsas, J. Alison Noble
* MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs through Disentangled Spatial-Temporal Representations, Kyungho Bae, Jinhyung Kim, Sihaeng Lee, Soonyoung Lee, Gunhee Lee, Jinwoo Choi
* GenEx: Generating an explorable world, TaiMing Lu, Jieneng Chen

## **图像生成 (Image Generation)**


* **\[Oral\] RandAR: Decoder-only Autoregressive Visual Generation in Random Orders**, Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T. Freeman, Yu-Xiong Wang
* **\[Oral\] Reanimating Images using Neural Representations of Dynamic Stimuli**, Jacob Yeung, Andrew F. Luo, Gabriel Sarch, Margaret M. Henderson, Deva Ramanan, Michael J. Tarr
* **\[Oral\] CustAny: Customizing Anything from A Single Example**, Lingjie Kong, Kai Wu, Chengming Xu, Xiaobin Hu, Wenhui Han, Jinlong Peng, Donghao Luo, Mengtian Li, Jiangning Zhang, Chengjie Wang, Yanwei Fu
* **\[Oral\] Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks**, Junying Wang, Hongyuan Zhang, Yuan Yuan
* **\[Oral\] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model**, Meilong Xu, Saumya Gupta, Xiaoling Hu, Chen Li, Shahira Abousamra, Dimitris Samaras, Prateek Prasanna, Chao Chen
* **\[Oral\] Language-Guided Image Tokenization for Generation**, Kaiwen Zha, Lijun Yu, Alireza Fathi, David A. Ross, Cordelia Schmid, Dina Katabi, Xiuye Gu
* **\[Oral\] Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis**, Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu
* **\[Oral\] DreamRelation: Bridging Customization and Relation Generation**, Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, Xiangtai Li
* *\[Highlight\] PGC: Physics-Based Gaussian Cloth from a Single Pose*, Michelle Guo, Matt Jen-Yuan Chiang, Igor Santesteban, Nikolaos Sarafianos, Hsiao-yu Chen, Oshri Halimi, Aljaž Božič, Shunsuke Saito, Jiajun Wu, C. Karen Liu, Tuur Stuyck, Egor Larionov
* *\[Highlight\] Parallelized Autoregressive Visual Generation*, Yuqing Wang, Shuhuai Ren, Zhijie Lin, Yujin Han, Haoyuan Guo, Zhenheng Yang, Difan Zou, Jiashi Feng, Xihui Liu
* *\[Highlight\] StarVector: Generating Scalable Vector Graphics Code from Images and Text*, Juan A. Rodriguez, Abhay Puri, Shubham Agarwal, Issam H. Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, Marco Pedersoli
* FAM Diffusion: Frequency and Attention Modulation for High-Resolution Image Generation with Stable Diffusion, Haosen Yang, Adrian Bulat, Isma Hadji, Hai X. Pham, Xiatian Zhu, Georgios Tzimiropoulos, Brais Martinez
* HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation, Hermann Kumbong, Xian Liu, Tsung-Yi Lin, Ming-Yu Liu, Xihui Liu, Ziwei Liu, Daniel Y. Fu, Christopher Re, David W. Romero
* TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation, Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, Xinglong Wu
* Learning Flow Fields in Attention for Controllable Person Image Generation, Zijian Zhou, Shikun Liu, Xiao Han, Haozhe Liu, Kam Woh Ng, Tian Xie, Yuren Cong, Hang Li, Mengmeng Xu, Juan-Manuel Perez-Rua, Aditya Patel, Tao Xiang, Miaojing Shi, Sen He
* Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation, Nadav Z. Cohen, Oron Nir, Ariel Shamir
* One Diffusion to Generate Them All, Duong H. Le, Tuan Pham, Sangho Lee, Christopher Clark, Aniruddha Kembhavi, Stephan Mandt, Ranjay Krishna, Jiasen Lu
* Dual Diffusion for Unified Image Generation and Understanding, Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, Peng Wang
* SerialGen: Personalized Image Generation by First Standardization Then Personalization, Cong Xie, Han Zou, Ruiqi Yu, Yan Zhang, Zhenpeng Zhan
* ID-Patch: Robust ID Association for Group Photo Personalization, Yimeng Zhang, Tiancheng Zhi, Jing Liu, Shen Sang, Liming Jiang, Qing Yan, Sijia Liu, Linjie Luo
* IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation, Yiren Song, Pei Yang, Hai Ci, Mike Zheng Shou
* Image Generation Diversity Issues and How to Tame Them, Mischa Dombrowski, Weitong Zhang, Sarah Cechnicka, Hadrien Reynaud, Bernhard Kainz
* Disentangled Pose and Appearance Guidance for Multi-Pose Generation, Tengfei Xiao, Yue Wu, Yuelong Li, Can Qin, Maoguo Gong, Qiguang Miao, Wenping Ma
* Nitro Fusion: High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training, Dar-Yen Chen, Hmrishav Bandyopadhyay, Kai Zou, Yi-Zhe Song
* DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention, Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, Xinggang Wang
* Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction, Teng Hu, Jiangning Zhang, Ran Yi, Jieyu Weng, Yabiao Wang, Xianfang Zeng, Zhucun Xue, Lizhuang Ma
* Generative Hard Example Augmentation for Semantic Point Cloud Segmentation, Qi Zhang, Jibin Peng, Zhao Huang, Wei Feng, Di Lin
* Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation, Nicolas Dufour, Vicky Kalogeiton, David Picard, Loic Landrieu
* ZoomLDM: Latent Diffusion Model for Multi-scale Image Generation, Srikar Yellapragada, Alexandros Graikos, Kostas Triaridis, Prateek Prasanna, Rajarsi Gupta, Joel Saltz, Dimitris Samaras
* Boost Your Human Image Generation Model via Direct Preference Optimization, Sanghyeon Na, Yonggyu Kim, Hyunjoon Lee
* BizGen: Advancing Article-level Visual Text Rendering for Infographics Generation, Yuyang Peng, Shishi Xiao, Keming Wu, Qisheng Liao, Bohan Chen, Kevin Lin, Danqing Huang, Ji Li, Yuhui Yuan
* WeGen: A Unified Model for Interactive Multimodal Generation as We Chat, Zhipeng Huang, Shaobin Zhuang, Canmiao Fu, Binxin Yang, Ying Zhang, Chong Sun, Zhizheng Zhang, Yali Wang, Chen Li, Zheng-Jun Zha
* Unseen Visual Anomaly Generation, Han Sun, Yunkang Cao, Hao Dong, Olga Fink
* Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation, Ying Jin, Jinlong Peng, Qingdong He, Teng Hu, Jiafu Wu, Hao Chen, Haoxuan Wang, Wenbing Zhu, Mingmin Chi, Jun Liu, Yabiao Wang
* Improving the Training of Data-Efficient GANs via Quality Aware Dynamic Discriminator Rejection Sampling, Zhaoyu Zhang, Yang Hua, Guanxiong Sun, Hui Wang, Seán McLoone
* Consistency Posterior Sampling for Diverse Image Synthesis, Vishal Purohit, Matthew Repasky, Jianfeng Lu, Qiang Qiu, Yao Xie, Xiuyuan Cheng
* DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching, Emanuele Aiello, Umberto Michieli, Diego Valsesia, Mete Ozay, Enrico Magli
* UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics, Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, Hui Ding, Zhe Lin, Hengshuang Zhao
* Training-free Dense-Aligned Diffusion Guidance for Modular Conditional Image Synthesis, Zixuan Wang, Duo Peng, Feng Chen, Yuwei Yang, Yinjie Lei
* OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows, Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Zichun Liao, Yusuke Kato, Kazuki Kozuka, Aditya Grover
* LoRACLR: Contrastive Adaptation for Customization of Diffusion Models, Enis Simsar, Thomas Hofmann, Federico Tombari, Pinar Yanardag
* Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization, Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Mingxi Cheng, Ji Li, Liang Zheng
* Composing Parts for Expressive Object Generation, Harsh Rangwani, Aishwarya Agarwal, Kuldeep Kulkarni, R. Venkatesh Babu, Srikrishna Karanam
* DYMO: Training-Free Diffusion Model Alignment with Dynamic Multi-Objective Scheduling, Xin Xie, Dong Gong
* OmniGen: Unified Image Generation, Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, Zheng Liu
* Rethinking Training for De-biasing Text-to-Image Generation: Unlocking the Potential of Stable Diffusion, Eunji Kim, Siwon Kim, Minjun Park, Rahim Entezari, Sungroh Yoon
* Multi-focal Conditioned Latent Diffusion for Person Image Synthesis, Jiaqi Liu, Jichao Zhang, Paolo Rota, Nicu Sebe
* Image is All You Need to Empower Large-scale Diffusion Models for In-Domain Generation, Pu Cao, Feng Zhou, Lu Yang, Tianrui Huang, Qing Song
* PatchDPO: Patch-level DPO for Finetuning-free Personalized Image Generation, Qihan Huang, Long Chan, Jinlong Liu, Wanggui He, Hao Jiang, Mingli Song, Jie Song
* Diffusion Self-Distillation for Zero-Shot Customized Image Generation, Shengqu Cai, Eric Ryan Chan, Yunzhi Zhang, Leonidas Guibas, Jiajun Wu, Gordon Wetzstein
* MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization, Siyuan Li, Luyuan Zhang, Zedong Wang, Juanxi Tian, Cheng Tan, Zicheng Liu, Chang Yu, Qingsong Xie, Haonan Lu, Haoqian Wang, Zhen Lei
* Auto Cherry-Picker: Learning from High-quality Generative Data Driven by Language, Yicheng Chen, Xiangtai Li, Yining Li, Yanhong Zeng, Jianzong Wu, Xiangyu Zhao, Kai Chen

## **视频生成 (Video Generation)**


* **\[Oral\] Motion Prompting: Controlling Video Generation with Motion Trajectories**, Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Yusuf Aytar, Michael Rubinstein, Chen Sun, Oliver Wang, Andrew Owens, Deqing Sun
* **\[Oral\] Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise**, Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, Michael Ryoo, Paul Debevec, Ning Yu
* **\[Oral\] GS-DIT: Advancing Video Generation with Dynamic 3D Gaussian Fields through Efficient Dense 3D Point Tracking**, Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yijin Li, Fu-Yun Wang, Hongsheng Li
* **\[Oral\] CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models**, Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T. Barron, Aleksander Holynski
* **\[Oral\] AudCast: Audio-Driven Human Video Generation by Cascaded Diffusion Transformers**, Jiazhi Guan, Kaisiyuan Wang, Zhiliang Xu, Quanwei Yang, Yasheng Sun, Shengyi He, Borong Liang, Yukang Cao, Yingying Li, Haocheng Feng, Errui Ding, Jingdong Wang, Youjian Zhao, Hang Zhou, Ziwei Liu
* *\[Highlight\] 4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion*, Chaoyang Wang, Peiye Zhuang, Tuan Duc Ngo, Willi Menapace, Aliaksandr Siarohin, Michael Vasilkovsky, Ivan Skorokhodov, Sergey Tulyakov, Peter Wonka, Hsin-Ying Lee
* StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models, Yunzhi Yan, Zhen Xu, Haotong Lin, Haian Jin, Haoyu Guo, Yida Wang, Kun Zhan, Xianpeng Lang, Hujun Bao, Xiaowei Zhou, Sida Peng
* SnapGen-V: Generating a Five-Second Video within Five Seconds on a Mobile Device, Yushu Wu, Zhixing Zhang, Yanyu Li, Yanwu Xu, Anil Kag, Yang Sui, Huseyin Coskun, Ke Ma, Aleksei Lebedev, Ju Hu, Dimitris N. Metaxas, Yanzhi Wang, Sergey Tulyakov, Jian Ren
* StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text, Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, Humphrey Shi
* AKiRa: Augmentation Kit on Rays for Optical Video Generation, Xi Wang, Robin Courant, Marc Christie, Vicky Kalogeiton
* Token Motion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation, Ruineng Li, Daitao Xing, Huiming Sun, Yuanzhou Ha, Jinglin Shen, Chiuman Ho
* Tora: Trajectory-oriented Diffusion Transformer for Video Generation, Zhenghao Zhang, Junchao Liao, Menghao Li, ZuoZhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, Weizhi Wang
* FloVD: Optical Flow Meets Video Diffusion Model for Enhanced Camera-Controlled Video Synthesis, Wonjoon Jin, Qi Dai, Chong Luo, Seung-Hwan Baek, Sunghyun Cho
* Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation, Xin Yan, Yuxuan Cai, Qiuyue Wang, Yuan Zhou, Wenhao Huang, Huan Yang
* IM-Zero: Instance-level Motion Controllable Video Generation in a Zero-shot Manner, Yuyang Huang, Yabo Chen, Li Ding, Xiaopeng Zhang, Wenrui Dai, Junni Zou, Hongkai Xiong, Qi Tian
* AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion, Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, Siyu Zhou, Qian He, Jing Liu
* Taming Teacher Forcing for Masked Autoregressive Video Generation, Deyu Zhou, Quan Sun, Yuang Peng, Kun Yan, Runpei Dong, Duomin Wang, Zheng Ge, Nan Duan, Xiangyu Zhang
* OpenHumanVid: A Large-Scale High-Quality Dataset for Enhancing Human-Centric Video Generation, Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, Siyu Zhu
* DiTCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Tuning-Free Multi-Prompt Longer Video Generation, Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, Xiangyu Yue
* Multi-subject Open-set Personalization in Video Generation, Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, Sergey Tulyakov
* DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes, Jinxiu Liu, Shaoheng Lin, Yinxiao Li, Ming-Hsuan Yang
* VideoDPO: Omni-Preference Alignment for Video Diffusion Generation, Runtao Liu, Haoyu Wu, Ziqiang Zheng, Chen Wei, Yingqing He, Renjie Pi, Qifeng Chen
* Optical-Flow Guided Prompt Optimization for Coherent Video Generation, Hyelin Nam, Jaemin Kim, Dohun Lee, Jong Chul Ye
* World-consistent Video Diffusion with Explicit 3D Modeling, Qihang Zhang, Shuangfei Zhai, Miguel Angel Bautista Martin, Kevin Miao, Alexander Toshev, Joshua Susskind, Jiatao Gu
* MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation, Shuwei Shi, Biao Gong, Xi Chen, Dandan Zheng, Shuai Tan, Zizheng Yang, Yuyuan Li, Jingwen He, Kecheng Zheng, Jingdong Chen, Ming Yang, Yinqiang Zheng
* AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers, Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B. Lindell, Sergey Tulyakov
* Video Motion Transfer with Diffusion Transformers, Alexander Pondaven, Aliaksandr Siarohin, Sergey Tulyakov, Philip Torr, Fabio Pizzati
* VidTwin: Video VAE with Decoupled Structure and Dynamics, Yuchi Wang, Junliang Guo, Xinyi Xie, Tianyu He, Xu Sun, Jiang Bian
* From Slow Bidirectional to Fast Autoregressive Video Diffusion Models, Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, Xun Huang
* Goku: Flow Based Video Generative Foundation Models, Shoufa Chen, Chongjian Ge, Yuqi Zhang, Yida Zhang, Fengda Zhu, Hao Yang, Hongxiang Hao, Hui Wu, Zhichao Lai, Yifei Hu, Ting-Che Lin, Shilong Zhang, Fu Li, Chuan Li, Xing Wang, Yanghua Peng, Peize Sun, Ping Luo, Yi Jiang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu
* Mimir: Improving Video Diffusion Models for Precise Text Understanding, Shuai Tan, Biao Gong, Yutong Feng, Kecheng Zheng, Dandan Zheng, Shuwei Shi, Yujun Shen, Jingdong Chen, Ming Yang
* Mind the Time: Temporally-Controlled Multi-Event Video Generation, Ziyi Wu, Aliaksandr Siarohin, Willi Menapace, Ivan Skorokhodov, Yuwei Fang, Varnith Chordia, Igor Gilitschenski, Sergey Tulyakov
* HOIGen-1M: A Large-scale Dataset for Human-Object Interaction Video Generation, Kun Liu, Qi Liu, Xinchen Liu, Jie Li, Yongdong Zhang, Jiebo Luo, Xiaodong He, Wu Liu
* Generating 3D-Consistent Videos from Unposed Internet Photos, Gene Chou, Kai Zhang, Sai Bi, Hao Tan, Zexiang Xu, Fujun Luan, Bharath Hariharan, Noah Snavely
* AnimateAnything: Consistent and Controllable Animation for Video Generation, Guojun Lei, Chi Wang, Rong Zhang, Yikai Wang, Hong Li, Weiwei Xu
* MotionPro: A Precise Motion Controller for Image-to-Video Generation, Zhongwei Zhang, Fuchen Long, Zhaofan Qiu, Yingwei Pan, Wu Liu, Ting Yao, Tao Mei
* Generative Inbetweening through Frame-wise Conditions-Driven Video Generation, Tianyi Zhu, Dongwei Ren, Qilong Wang, Xiaohe Wu, Wangmeng Zuo
* FreePCA: Integrating Consistency Information across Long-short Frames in Training-free Long Video Generation via Principal Component Analysis, Jiangtong Tan, Hu Yu, Jie Huang, Jie Xiao, Feng Zhao
* ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models, Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M. Rehg, Tobias Hinz
* MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation, Weijia Wu, Mingyu Liu, Zeyu Zhu, Xi Xia, Haoen Feng, Wen Wang, Kevin Qinghong Lin, Chunhua Shen, Mike Zheng Shou
* OSV: One Step is Enough for High-Quality Image to Video Generation, Xiaofeng Mao, Zhengkai Jiang, Fu-yun Wang, Jiangning Zhang, Hao Chen, Mingmin Chi, Yabiao Wang, Wenhan Luo
* ByTheWay: Boost Your Text-to-Video Generation Model to Higher Quality in a Training-free Way, Jiazi Bu, Pengyang Ling, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Dahua Lin, Jiaqi Wang
* Fiction: 4D Future Interaction Prediction from Video, Kumar Ashutosh, Georgios Pavlakos, Kristen Grauman
* One-Minute Video Generation with Test-Time Training, Jiarui Xu, Shihao Han, Karan Dalal, Daniel Koceja, Yue Zhao, Ka Chun Cheung, Yejin Choi, Jan Kautz, Yu Sun, Xiaolong Wang
* Generative Video Propagation, Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, Jiaya Jia
* LongDiff: Training-Free Long Video Generation in One Go, Zhuoling Li, Hossein Rahmani, Qiuhong Ke, Jun Liu
* Improved Video VAE for Latent Video Diffusion Model, Pingyu Wu, Kai Zhu, Yu Liu, Liming Zhao, Wei Zhai, Yang Cao, Zheng-Jun Zha
* Towards Precise Scaling Laws for Video Diffusion Transformers, Yuanyang Yin, Yaqi Zhao, Mingwu Zheng, Ke Lin, Jiarong Ou, Rui Chen, Victor Shea-Jay Huang, Jiahao Wang, Xin Tao, Pengfei Wan, Di Zhang, Baoqun Yin, Wentao Zhang, Kun Gai
* Encapsulated Composition of Text-to-Image and Text-to-Video Models for High-Quality Video Synthesis, Tongtong Su, Chengyu Wang, Bingyan Liu, Jun Huang, Dongming Lu
* DriveScape: High-Resolution Driving Video Generation by Multi-View Feature Fusion, Wei Wu, Xi Guo, Weixuan Tang, Tingxuan Huang, Chiyu Wang, Chenjing Ding
* Grounding Pixels in Facts: Distilled Knowledge Retrieval for Factual Text-to-Video Generation, Daniel Lee, Arjun Chandra, Yang Zhou, Yunyao Li, Simone Conia

## **文生图 (Text-to-Image)**


* **\[Oral\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation**, Pengfei Zhou, Xiaopeng Peng, Jiajun Song, Chuanhao Li, Zhaopan Xu, Yue Yang, Ziyao Guo, Hao Zhang, Yuqi Lin, Yefei He, Lirui Zhao, Shuo Liu, Tianhua Li, Yuxuan Xie, Xiaojun Chang, Yu Qiao, Wenqi Shao, Kaipeng Zhang
* **\[Oral\] DesignDiffusion: High-Quality Text-to-Design Image Generation with Diffusion Models**, Zhendong Wang, Jianmin Bao, Shuyang Gu, Dong Chen, Wengang Zhou, Houqiang Li
* **\[Oral\] Minority-Focused Text-to-Image Generation via Prompt Optimization**, Soobin Um, Jong Chul Ye
* **\[Oral\] Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models**, Andreas Müller, Denis Lukovnikov, Jonas Thietke, Asja Fischer, Erwin Quiring
* **\[Oral\] Q-Eval-100K: Evaluating Visual Quality and Alignment Level for Text-to-Vision Content**, Zicheng Zhang, Tengchuan Kou, Shushi Wang, Chunyi Li, Wei Sun, Wei Wang, Xiaoyu Li, Zongyu Wang, Xuezhi Cao, Xiongkuo Min, Xiaohong Liu, Guangtao Zhai
* *\[Highlight\] Type-R: Automatically Retouching Typos for Text-to-Image Generation*, Wataru Shimoda, Naoto Inoue, Daichi Haraguchi, Hayato Mitani, Seiichi Uchida, Kota Yamaguchi
* *\[Highlight\] Flowing from Words to Pixels: A Noise-Free Framework for Cross-Modality Evolution*, Qihao Liu, Xi Yin, Alan Yuille, Andrew Brown, Mannat Singh
* *\[Highlight\] Generative Photography: Scene-Consistent Camera Control for Realistic Text-to-Image Synthesis*, Yu Yuan, Xijun Wang, Yichen Sheng, Prateek Chennuri, Xingguang Zhang, Stanley Chan
* PreciseCam: Precise Camera Control for Text-to-Image Generation, Edurne Bernal-Berdun, Ana Serrano, Belen Masia, Matheus Gadelha, Yannick Hold-Geoffroy, Xin Sun, Diego Gutierrez
* Science-T21: Addressing Scientific Illusions in Image Synthesis, Jialuo Li, Wenhao Chai, Xingyu Fu, Haiyang Xu, Saining Xie
* GPS as a Control Signal for Image Generation, Chao Feng, Ziyang Chen, Aleksander Holynski, Alexei A. Efros, Andrew Owens
* Compass Control: Multi Object Orientation Control for Text-to-Image Generation, Rishubh Parihar, Vaibhav Agrawal, Sachidanand VS, Venkatesh Babu Radhakrishnan
* MC^2: Multi-concept Guidance for Customized Multi-concept Generation, Jiaxiu Jiang, Yabo Zhang, Kailai Feng, Xiaohe Wu, Wenbo Li, Renjing Pei, Fan Li, Wangmeng Zuo
* TFCustom: Customized Image Generation with Time-Aware Frequency Feature Guidance, Mushui Liu, Dong She, Jingxuan Pang, Qihan Huang, Jiacheng Ying, Wanggui He, Yuanlei Hou, Siming Fu
* ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation, Zirun Guo, Tao Jin
* Plug-and-Play Interpretable Responsible Text-to-Image Generation via Dual-Space Multi-facet Concept Control, Basim Azam, Naveed Akhtar
* Not Just Text: Uncovering Vision Modality Typographic Threats in Image Generation Models, Hao Cheng, Erjia Xiao, Jiayan Yang, Jiahang Cao, Qiang Zhang, Jize Zhang, Kaidi Xu, Jindong Gu, Renjing Xu
* Zero-Shot Styled Text Image Generation, but Make It Autoregressive, Vittorio Pippi, Fabio Quattrini, Silvia Cascianelli, Alessio Tonioni, Rita Cucchiara
* Multi-party Collaborative Attention Control for Image Customization, Han Yang, Chuanguang Yang, Qiuli Wang, Zhulin An, Weilun Feng, Libo Huang, Yongjun Xu
* UNIC-Adapter: Unified Image-instruction Adapter with Multi-modal Transformer for Image Generation, Lunhao Duan, Shanshan Zhao, Wenjun Yan, Yinglun Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, Mingming Gong, Gui-Song Xia
* Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator, Chaehun Shin, Jooyoung Choi, Heeseung Kim, Sungroh Yoon
* SnapGen: Taming High-Resolution Text-to-Image Models for Mobile Devices with Efficient Architectures and Training, Jierun Chen, Dongting Hu, Xijie Huang, Huseyin Coskun, Arpit Sahni, Aarush Gupta, Anujraaj Goyal, Dishani Lahiri, Rajesh Singh, Yerlan Idelbayev, Junli Cao, Yanyu Li, Kwang-Ting Cheng, S.-H. Gary Chan, Mingming Gong, Sergey Tulyakov, Anil Kag, Yanwu Xu, Jian Ren
* Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps, Jeeyung Kim, Erfan Esmaeili, Qiang Qiu
* VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness, SeungJu Cha, Kwanyoung Lee, Ye-Chan Kim, Hyunwoo Oh, Dong-Jin Kim
* Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation, Gianni Franchi, Nacim Belkhir, Dat Nguyen Trong, Guoxuan Xia, Andrea Pilzer
* PosterMaker: Towards High-Quality Product Poster Generation with Accurate Text Rendering, Yifan Gao, Zihuan Lin, Chuanbin Liu, Min Zhou, Tiezheng Ge, Bo Zheng, Hongtao Xie
* SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models, Zilan Wang, Junfeng Guo, Jiacheng Zhu, Yiming Li, Heng Huang, Muhao Chen, Zhengzhong Tu
* Acquire and then Adapt: Squeezing out Text-to-Image Model for Image Restoration, Junyuan Deng, Xinyi Wu, Yongxing Yang, Congchao Zhu, Song Wang, Zhenyao Wu
* ACE: Anti-Editing Concept Erasure in Text-to-Image Models, Zihao Wang, Yuxiang Wei, Fan Li, Renjing Pei, Hang Xu, Wangmeng Zuo
* Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects, Weimin Qiu, Jieke Wang, Meng Tang
* Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis, Boming Miao, Chunxiao Li, Xiaoxiao Wang, Andi Zhang, Rui Sun, Zizhe Wang, Yao Zhu
* LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending, Jian Jin, Zhenbo Yu, Yang Shen, Zhenyong Fu, Jian Yang
* Devil is in the Detail: Towards Injecting Fine Details of Image Prompt in Image Generation via Conflict-free Guidance and Stratified Attention, Kyungmin Jo, Jooyeol Yun, Jaegul Choo
* Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards, Zijing Hu, Fengda Zhang, Long Chen, Kun Kuang, Jiahui Li, Kaifeng Gao, Jun Xiao, Xin Wang, Wenwu Zhu
* Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation, Taeyoung Yun, Dinghuai Zhang, Jinkyoo Park, Ling Pan
* ReNeg: Learning Negative Embedding with Reward Guidance, Xiaomin Li, Yixuan Liu, Takashi Isobe, Xu Jia, Qinpeng Cui, Dong Zhou, Dong Li, You He, Huchuan Lu, Zhongdao Wang, Emad Barsoum
* Multi-Group Proportional Representations for Text-to-Image Models, Sangwon Jung, Alex Oesterling, Claudio Mayrink Verdun, Sajani Vithana, Taesup Moon, Flavio P. Calmon
* STEREO: A Two-Stage Framework for Adversarially Robust Concept Erasing from Text-to-Image Diffusion Models, Koushik Srivatsan, Fahad Shamshad, Muzammal Naseer, Vishal M. Patel, Karthik Nandakumar
* Exploring the Deep Fusion of Large Language Models and Diffusion Transformers for Text-to-Image Synthesis, Bingda Tang, Boyang Zheng, Sayak Paul, Saining Xie
* Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budget, Vikash Sehwag, Xianghao Kong, Jingtao Li, Michael Spranger, Lingjuan Lyu
* Enhancing Creative Generation on Stable Diffusion-based Models, Jiyeon Han, Dahee Kwon, Gayoung Lee, Junho Kim, Jaesik Choi
* STEPS: Sequential Probability Tensor Estimation for Text-to-Image Hard Prompt Search, Yuning Qiu, Andong Wang, Chao Li, Haonan Huang, Guoxu Zhou, Qibin Zhao
* PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction, Eduard Poesina, Adriana Valentina Costache, Adrian-Gabriel Chifu, Josiane Mothe, Radu Tudor lonescu
* Let's Verify and Reinforce Image Generation Step by Step, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Ziyu Guo, Haoquan Zhang, Manyuan Zhang, Jiaming Liu, Peng Gao, Hongsheng Li
* DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation, Jianzong Wu, Chao Tang, Jingbo Wang, Yanhong Zeng, Xiangtai Li, Yunhai Tong
* POSTA: A Go-to Framework for Customized Artistic Poster Generation, Haoyu Chen, Xiaojie Xu, Wenbo Li, Jingjing Ren, Tian Ye, Songhua Liu, Ying-Cong Chen, Lei Zhu, Xinchao Wang
* StageDesigner: Artistic Stage Generation for Scenography via Theater Scripts, Zhaoxing Gan, Mengtian Li, Ruhua Chen, Zhongxia Ji, Sichen Guo, Huanling Hu, Guangnan Ye, Zuo Hu
* Implicit Bias Injection Attacks against Text-to-Image Diffusion Models, Huayang Huang, Xiangye Jin, Jiaxu Miao, Yu Wu
* T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting, Yifei Qian, Zhongliang Guo, Bowen Deng, Chun Tong Lei, Shuai Zhao, Chun Pong Lau, Xiaopeng Hong, Michael P. Pound
* Continuous, Subject-Specific Attribute Control in T21 Models by Identifying Semantic Directions, Stefan Andreas Baumann, Felix Krause, Michael Neumayr, Nick Stracke, Melvin Sevi, Vincent Tao Hu, Björn Ommer
* Make It Count: Text-to-Image Generation with an Accurate Number of Objects, Lital Binyamin, Yoad Tewel, Hilit Segev, Eran Hirsch, Royi Rassin, Gal Chechik
* Detect-and-Guide: Self-regulation of Diffusion Models for Safe Text-to-Image Generation via Guideline Token Optimization, Feifei Li, Mi Zhang, Yiming Sun, Min Yang
* MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation, Mingcheng Li, Xiaolu Hou, Ziyang Liu, Dingkang Yang, Ziyun Qian, Jiawei Chen, Jinjie Wei, Yue Jiang, Qingyao Xu, Lihua Zhang
* StoryGPT-V: Large Language Models as Consistent Story Visualizers, Xiaoqian Shen, Mohamed Elhoseiny
* ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting, Chengyou Jia, Changliang Xia, Zhuohang Dang, Weijia Wu, Hangwei Qian, Minnan Luo
* ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts, Dmitry Petrov, Pradyumn Goyal, Divyansh Shivashok, Yuanming Tao, Melinos Averkiou, Evangelos Kalogerakis
* From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing, Jingxuan Wei, Cheng Tan, Qi Chen, Gaowei Wu, Siyuan Li, Zhangyang Gao, Linzhuang Sun, Bihui Yu, Ruifeng Guo
* T2ISafety: Benchmark for Assessing Fairness, Toxicity, and Privacy in Image Generation, Lijun Li, Zhelun Shi, Xuhao Hu, Bowen Dong, Yiran Qin, Xihui Liu, Lu Sheng, Jing Shao
* VODiff: Controlling Object Visibility Order in Text-to-Image Generation, Dong Liang, Jinyuan Jia, Yuhao Liu, Zhanghan Ke, Hongbo Fu, Rynson W. H. Lau
* Z-Magic: Zero-shot Multiple Attributes Guided Image Creator, Yingying Deng, Xiangyu He, Fan Tang, Weiming Dong
* Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis, Woojung Han, Yeonkyung Lee, Chanyoung Kim, Kwanghyun Park, Seong Jae Hwang
* Scaling Down Text Encoders of Text-to-Image Diffusion Models, Lifu Wang, Daqing Liu, Xinchen Liu, Xiaodong He
* Redefining \<Creative\> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation, Fu Feng, Yucheng Xie, Xu Yang, Jing Wang, Xin Geng
* Calibrated Multi-Preference Optimization for Aligning Diffusion Models, Kyungmin Lee, Xiahong Li, Qifei Wang, Junfeng He, Junjie Ke, Ming-Hsuan Yang, Irfan Essa, Jinwoo Shin, Feng Yang, Yinxiao Li
* SILMM: Self-Improving Large Multimodal Models for Compositional Text-to-Image Generation, Leigang Qu, Haochuan Li, Wenjie Wang, Xiang Liu, Juncheng Li, Liqiang Nie, Tat-Seng Chua
* IDEA-Bench: How Far are Generative Models from Professional Designing?, Chen Liang, Lianghua Huang, Jingwu Fang, Huanzhang Dou, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Junge Zhang, Xin Zhao, Yu Liu

## **图像编辑 (Image Editing)**


* **\[Oral\] AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea**, Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, Yueting Zhuang
* **\[Oral\] Pattern Analogies: Learning to Perform Programmatic Image Edits by Analogy**, Aditya Ganeshan, Thibault Groueix, Paul Guerrero, Radomir Mech, Matthew Fisher, Daniel Ritchie
* *\[Highlight\] Reference-Based 3D-Aware Image Editing with Triplanes*, Bahri Batuhan Bilecen, Yigit Yalin, Ning Yu, Aysegul Dundar
* *\[Highlight\] Style-Editor: Text-driven Object-centric Style Editing*, Jihun Park, Jongmin Gim, Kyoungmin Lee, Seunghun Lee, Sunghoon Im
* Align-A-Video: Deterministic Reward Tuning of Image Diffusion Models for Consistent Video Editing, Shengzhi Wang, Yingkang Zhong, Jiangchuan Mu, Kai Wu, Mingliang Xiong, Wen Fang, Mingqing Liu, Hao Deng, Bin He, Gang Li, Qingwen Liu
* FDS: Frequency-Aware Denoising Score for Text-Guided Latent Diffusion Image Editing, Yufan Ren, Zicong Jiang, Tong Zhang, Søren Forchhammer, Sabine Süsstrunk
* FeedEdit: Text-Based Image Editing with Dynamic Feedback Regulation, Fengyi Fu, Lei Zhang, Mengqi Huang, Zhendong Mao
* InsightEdit: Towards Better Instruction Following for Image Editing, Yingjing Xu, Jie Kong, Jiazhi Wang, Xiao Pan, Bo Lin, Qiang Liu
* MoEdit: On Learning Quantity Perception for Multi-object Image Editing, Yanfeng Li, Kahou Chan, Yue Sun, Chantong Lam, Tong Tong, Zitong Yu, Keren Fu, Xiaohong Liu, Tao Tan
* Pathways on the Image Manifold: Image Editing via Video Generation, Noam Rotstein, Gal Yona, Daniel Silver, Roy Velich, David Bensaid, Ron Kimmel
* PhyS-EdiT: Physics-aware Semantic Image Editing with Text Description, Ziqi Cai, Shuchen Weng, Yifei Xia, Boxin Shi
* Stable Flow: Vital Layers for Training-Free Image Editing, Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, Daniel Cohen-Or
* Improving Editability in Image Generation with Layer-wise Memory, Daneul Kim, Jaeah Lee, Jaesik Park
* EditAR: Unified Conditional Generation with Autoregressive Models, Jiteng Mu, Nuno Vasconcelos, Xiaolong Wang
* Visual Prompting for One-shot Controllable Video Editing without Inversion, Zhengbo Zhang, Yuxi Zhou, Duo Peng, Joo-Hwee Lim, Zhigang Tu, De Wen Soh, Lin Geng Foo
* SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion, Trong-Tung Nguyen, Quang Nguyen, Khoi Nguyen, Anh Tran, Cuong Pham
* Dragin3D: Image Editing by Dragging in 3D Space, Weiran Guang, Xiaoguang Gu, Mengqi Huang, Zhendong Mao
* Visual Representation Learning through Causal Intervention for Controllable Image Editing, Shanshan Huang, Haoxuan Li, Chunyuan Zheng, Lei Wang, Guorui Liao, Zhili Gong, Huayi Yang, Li Liu
* Preserve or Modify? Context-Aware Evaluation for Balancing Preservation and Modification in Text-Guided Image Editing, Yoonjeon Kim, Soohyun Ryu, Yeonsung Jung, Hyunkoo Lee, Joowon Kim, June Yong Yang, Jaeryong Hwang, Eunho Yang
* Edit Away and My Face Will not Stay: Personal Biometric Defense against Malicious Generative Editing, Hanhui Wang, Yihua Zhang, Ruizheng Bai, Yue Zhao, Sijia Liu, Zhengzhong Tu
* EmoEdit: Evoking Emotions through Image Manipulation, Jingyuan Yang, Jiawei Feng, Weibin Luo, Dani Lischinski, Daniel Cohen-Or, Hui Huang
* Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing, Pengcheng Xu, Boyuan Jiang, Xiaobin Hu, Donghao Luo, Qingdong He, Jiangning Zhang, Chengjie Wang, Yunsheng Wu, Charles Ling, Boyu Wang
* h-Edit: Effective and Flexible Diffusion-Based Editing via Doob's h-Transform, Toan Nguyen, Kien Do, Duc Kieu, Thin Nguyen
* Concept Lancet: Image Editing with Compositional Representation Transplant, Jinqi Luo, Tianjiao Ding, Kwan Ho Ryan Chan, Hancheng Min, Chris Callison-Burch, Rene Vidal
* Instruct-CLIP: Improving Instruction-Guided Image Editing with Automated Data Refinement Using Contrastive Learning, Sherry X. Chen, Misha Sra, Pradeep Sen
* GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing, Tong Wang, Ting Liu, Xiaochao Qu, Chengjing Wu, Luoqi Liu, Xiaolin Hu
* DreamOmni: Unified Image Generation and Editing, Bin Xia, Yuechen Zhang, Jingyao Li, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, Jiaya Jia
* Text-Driven Fashion Image Editing with Compositional Concept Learning and Counterfactual Abduction, Shanshan Huang, Haoxuan Li, Chunyuan Zheng, Mingyuan Ge, Wei Gao, Lei Wang, Li Liu
* EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting, Dong In Lee, Hyeongcheol Park, Jiyoung Seo, Eunbyung Park, Hyunje Park, Ha Dam Baek, Sangheon Shin, Sangmin Kim, Sangpil Kim
* MagicQuill: An Intelligent Interactive Image Editing System, Zichen Liu, Yue Yu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Wen Wang, Zhiheng Liu, Qifeng Chen, Yujun Shen
* FluxSpace: Disentangled Semantic Editing in Rectified Flow Models, Yusuf Dalva, Kavana Venkatesh, Pinar Yanardag
* Recognition-Synergistic Scene Text Editing, Zhengyao Fang, Pengyuan Lyu, Jingjing Wu, Chengquan Zhang, Jun Yu, Guangming Lu, Wenjie Pei
* RealEdit: Reddit Edits As a Large-scale Empirical Dataset for Image Transformations, Peter Sushko, Ayana Bharadwaj, Zhi Yang Lim, Vasily Ilin, Ben Caffee, Dongping Chen, Mohammadreza Salehi, Cheng-Yu Hsieh, Ranjay Krishna
* VEU-Bench: Towards Comprehensive Understanding of Video Editing, Bozheng Li, Yongliang Wu, Yi Lu, Jiashuo Yu, Licheng Tang, Jiawang Cao, Wenqing Zhu, Yuyang Sun, Jay Wu, Wenbo Zhu
* Towards Scalable Human-aligned Benchmark for Text-guided Image Editing, Suho Ryu, Kihyun Kim, Eugene Baek, Dongsoo Shin, Joonseok Lee
* PS-Diffusion: Photorealistic Subject-Driven Image Editing with Disentangled Control and Attention, Weicheng Wang, Guoli Jia, Zhongqi Zhang, Liang Lin, Jufeng Yang
* Mobile Diffusion for Video Editing, Amirhossein Habibian

## **图生图 (Image-to-Image Translation)**


* *\[Highlight\] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation*, Xiaoying Xing, Avinab Saha, Junfeng He, Susan Hao, Paul Vicol, Moonkyung Ryu, Gang Li, Sahil Singla, Sarah Young, Yinxiao Li, Feng Yang, Deepak Ramachandran
* MangaNinja: Line Art Colorization with Precise Reference Following, Zhiheng Liu, Ka Leong Cheng, Xi Chen, Jie Xiao, Hao Ouyang, Kai Zhu, Yu Liu, Yujun Shen, Qifeng Chen, Ping Luo
* Deterministic Image-to-Image Translation via Denoising Brownian Bridge Models with Dual Approximators, Bohan Xiao, Peiyong Wang, Qisheng He, Ming Dong
* Identity-preserving Distillation Sampling by Fixed-Point Iterator, SeonHwa Kim, Jiwon Kim, Soobin Park, Donghoon Ahn, Jiwon Kang, Seungryong Kim, Kyong Hwan Jin, Eunju Cha
* Extrapolating and Decoupling Image-to-Video Generation Models: Motion Modeling is Easier Than You Think, Jie Tian, Xiaoye Qu, Zhenyi Lu, Wei Wei, Sichen Liu, Yu Cheng
* LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis, Hanlin Wang, Hao Ouyang, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Qifeng Chen, Yujun Shen, Limin Wang
* Identity-Preserving Text-to-Video Generation by Frequency Decomposition, Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, Li Yuan
* Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation, Guy Yariv, Yuval Kirstain, Amit Zohar, Shelly Sheynin, Yaniv Taigman, Yossi Adi, Sagie Benaim, Adam Polyak
* Difference Inversion: Interpolate and Isolate the Difference with Token Consistency for Image Analogy Generation, Hyunsoo Kim, Donghyun Kim, Suhyun Kim
* MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation, Zhuangzhuang Chen, Hualiang Wang, Chubin Ou, Xiaomeng Li

## **风格迁移 (Style Transfer)**


* StyleMaster: Stylize Your Video with Artistic Generation and Translation, Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, Wenhan Luo
* LineArt: A Knowledge-guided Training-free High-quality Appearance Transfer for Design Drawing with Diffusion Model, Xi Wang, Hongzhen Li, Heng Fang, Yichen Peng, Haoran Xie, Xi Yang, Chuntao Li
* OmniStyle: Filtering High Quality Style Transfer Data at Scale, Ye Wang, Ruiqi Liu, Jiang Lin, Fei Liu, Zili Yi, Yilin Wang, Rui Ma
* Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization, Jamie Wynn, Zawar Qureshi, Jakub Powierza, Jamie Watson, Mohamed Sayed
* HSI: A Holistic Style Injector for Arbitrary Style Transfer, Shuhao Zhang, Hui Kang, Yang Liu, Fang Mei, Hongjuan Li
* StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements, Mingkun Lei, Xue Song, Beier Zhu, Hao Wang, Chi Zhang
* SGSST: Scaling Gaussian Splatting Style Transfer, Bruno Galerne, Jianling Wang, Lara Raad, Jean-Michel Morel
* Geometry in Style: 3D Stylization via Surface Normal Deformation, Nam Anh Dinh, Itai Lang, Hyunwoo Kim, Oded Stein, Rana Hanocka
* SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer, Chunnan Shang, Zhizhong Wang, Hongwei Wang, Xiangming Meng
* K-LORA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs, Ziheng Ouyang, Zhen Li, Qibin Hou
* StyleSSP: Sampling StartPoint Enhancement for Training-free Diffusion-based Method for Style Transfer, Ruojun Xu, Weijie Xi, XiaoDi Wang, Yongbo Mao, Zach Cheng
* Attention Distillation: A Unified Approach to Visual Characteristics Transfer, Yang Zhou, Xu Gao, Zichong Chen, Hui Huang

## **图像抠图 (Image Matting)**


* Polarized Color Screen Matting, Kenji Enomoto, Scott Cohen, Brian Price, TJ Rhodes
* MatAnyone: Stable Video Matting with Consistent Memory Propagation, Peiqing Yang, Shangchen Zhou, Jixin Zhao, Qingyi Tao, Chen Change Loy
* Generative Omnimatte: Learning to Decompose Video into Layers, Yao-Chih Lee, Erika Lu, Sarah Rumbley, Michal Geyer, Jia-Bin Huang, Tali Dekel, Forrester Cole
* MaSS13K: A Matting-level Semantic Segmentation Benchmark, Chenxi Xie, Minghan Li, Hui Zeng, Jun Luo, Lei Zhang

## **图像处理 (Image Manipulation)**


* **\[Oral\] Looking Glass: Generative Anamorphoses via Laplacian Pyramid Warping**, Pascal Chang, Sergio Sancho, Jingwei Tang, Markus Gross, Vinicius Azevedo
* **\[Oral\] Birth and Death of a Rose**, Chen Geng, Yunzhi Zhang, Shangzhe Wu, Jiajun Wu
* *\[Highlight\] Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories*, Susung Hong, Johanna Karras, Ricardo Martin-Brualla, Ira Kemelmacher-Shlizerman
* Instruction-based Image Manipulation by Watching How Things Move, Mingdeng Cao, Xuaner Zhang, Yinqiang Zheng, Zhihao Xia
* Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways, Yi Liu, Hao Zhou, Benlei Cui, Wenxiang Shang, Ran Lin
* RAD: Region-Aware Diffusion Models for Image Inpainting, Sora Kim, Sungho Suh, Minsik Lee
* TurboFill: Adapting Few-step Text-to-image Model for Fast Image Inpainting, Liangbin Xie, Daniil Pakhomov, Zhonghao Wang, Zongze Wu, Ziyan Chen, Yuqian Zhou, Haitian Zheng, Zhifei Zhang, Zhe Lin, Jiantao Zhou, Chao Dong
* Generative Photomontage, Sean J. Liu, Nupur Kumari, Ariel Shamir, Jun-Yan Zhu
* ORIDa: Object-centric Real-world Image Composition Dataset, Jinwoo Kim, Sangmin Han, Jinho Jeong, Jiwoo Choi, Dongyeoung Kim, Seon Joo Kim
* SmartEraser: Remove Anything from Images using Masked-Region Guidance, Longtao Jiang, Zhendong Wang, Jianmin Bao, Wengang Zhou, Dongdong Chen, Lei Shi, Dong Chen, Houqiang Li
* Towards Enhanced Image Inpainting: Mitigating Unwanted Object Insertion and Preserving Color Consistency, Yikai Wang, Chenjie Cao, Junqiu Yu, Ke Fan, Xiangyang Xue, Yanwei Fu
* HomoGen: Enhanced Video Inpainting via Homography Propagation and Diffusion, Ding Ding, Yueming Pan, Ruoyu Feng, Qi Dai, Kai Qiu, Jianmin Bao, Chong Luo, Zhenzhong Chen
* IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement, Zhihao Shi, Dong Huo, Yuhongze Zhou, Yan Min, Juwei Lu, Xinxin Zuo
* 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency, Sheng-Yu Huang, Zi-Ting Chou, Yu-Chiang Frank Wang
* VIRES: Video Instance Repainting via Sketch and Text Guided Generation, Shuchen Weng, Haojie Zheng, Peixuan Zhang, Yuchen Hong, Han Jiang, Si Li, Boxin Shi
* FADE: Frequency-Aware Diffusion Model Factorization for Video Editing, Yixuan Zhu, Haolin Wang, Shilin Ma, Wenliang Zhao, Yansong Tang, Lei Chen, Jie Zhou
* Keyframe-Guided Creative Video Inpainting, Yuwei Guo, Ceyuan Yang, Anyi Rao, Chenlin Meng, Omer Bar-Tal, Shuangrui Ding, Maneesh Agrawala, Dahua Lin, Bo Dai
* SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion Models, Jaerin Lee, Daniel Sungho Jung, Kanggeon Lee, Kyoung Mu Lee
* Movie Weaver: Tuning-Free Multi-Concept Video Personalization with Anchored Prompts, Feng Liang, Haoyu Ma, Zecheng He, Tingbo Hou, Ji Hou, Kunpeng Li, Xiaoliang Dai, Felix Juefei-Xu, Samaneh Azadi, Animesh Sinha, Peizhao Zhang, Peter Vajda, Diana Marculescu
* ArtiFade: Learning to Generate High-quality Subject from Blemished Images, Shuya Yang, Shaozhe Hao, Yukang Cao, Kwan-Yee K. Wong
* Paint by Inpaint: Learning to Add Image Objects by Removing Them First, Navve Wasserman, Noam Rotstein, Roy Ganz, Ron Kimmel
* MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting, Jun Huang, Ting Liu, Yihang Wu, Xiaochao Qu, Luoqi Liu, Xiaolin Hu
* ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Inpainting, Yizhe Tang, Zhimin Sun, Yuzhen Du, Ran Yi, Guangben Lu, Teng Hu, Luying Li, Lizhuang Ma, Fangyuan Zou
* Unleashing In-context Learning of Autoregressive Models for Few-shot Image Manipulation, Bolin Lai, Felix Juefei-Xu, Miao Liu, Xiaoliang Dai, Nikhil Mehta, Chenguang Zhu, Zeyi Huang, James M. Rehg, Sangmin Lee, Ning Zhang, Tong Xiao
* ObjectMover: Generative Object Movement with Video Prior, Xin Yu, Tianyu Wang, Soo Ye Kim, Paul Guerrero, Xi Chen, Qing Liu, Zhe Lin, Xiaojuan Qi
* DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection, Jaewoo Song, Daemin Park, Kanghyun Baek, Sangyub Lee, Jooyoung Choi, Eunji Kim, Sungroh Yoon

# **三、3D视觉 (3D Vision)**
## **3D视觉 (3D Vision)**


* ***\[Best Paper\]\[Oral\] VGGT: Visual Geometry Grounded Transformer***, Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny
* **\[Oral\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation**, Haoyu Guo, He Zhu, Sida Peng, Haotong Lin, Yunzhi Yan, Tao Xie, Wenguan Wang, Xiaowei Zhou, Hujun Bao
* **\[Oral\] FoundationStereo: Zero-Shot Stereo Matching**, Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield
* **\[Oral\] MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision**, Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, Jiaolong Yang
* **\[Oral\] Time of the Flight of the Gaussians: Optimizing Depth Indirectly in Dynamic Radiance Fields**, Runfeng Li, Mikhail Okunev, Zixuan Guo, Anh Ha Duong, Christian Richardt, Matthew O'Toole, James Tompkin
* **\[Oral\] Continuous 3D Perception Model with Persistent State**, Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa
* **\[Oral\] Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos**, Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, Aleksander Holynski
* **\[Oral\] DORNet: A Degradation Oriented and Regularized Network for Blind Depth Super-Resolution**, Zhengxue Wang, Zhiqiang Yan, Jinshan Pan, Guangwei Gao, Kai Zhang, Jian Yang
* **\[Oral\] Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World**, Bangyan Liao, Zhenjun Zhao, Haoang Li, Yi Zhou, Yingping Zeng, Hao Li, Peidong Liu
* *\[Highlight\] MUSt3R: Multi-view Network for Stereo 3D Reconstruction*, Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, Vincent Leroy
* *\[Highlight\] CH3Depth: Efficient and Flexible Depth Foundation Model with Flow Matching*, Jiaqi Li, Yiran Wang, Jinghong Zheng, Junrui Zhang, Liao Shen, Tianqi Liu, Zhiguo Cao
* *\[Highlight\] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes*, Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi, Bernard Ghanem, Marc Van Droogenbroeck
* *\[Highlight\] MetricGrids: Arbitrary Nonlinear Approximation with Elementary Metric Grids based Implicit Neural Representation*, Shu Wang, Yanbo Gao, Shuai Li, Chong Lv, Xun Cai, Chuankun Li, Hui Yuan, Jinglin Zhang
* *\[Highlight\] Event Ellipsometer: Event-based Mueller-Matrix Video Imaging*, Ryota Maeda, Yunseong Moon, Seung-Hwan Baek
* *\[Highlight\] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation*, Hadi Alzayer, Philipp Henzler, Jonathan T. Barron, Jia-Bin Huang, Pratul P. Srinivasan, Dor Verbin
* *\[Highlight\] Structure from Collision*, Takuhiro Kaneko
* *\[Highlight\] Shading Meets Motion: Self-supervised Indoor 3D Reconstruction Via Simultaneous Shape-from-Shading and Structure-from-Motion*, Guoyu Lu
* UniK3D: Universal Camera Monocular 3D Estimation, Luigi Piccinelli, Christos Sakaridis, Mattia Segu, Yung-Hsu Yang, Siyuan Li, Wim Abbeloos, Luc Van Gool
* Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video, David Yifan Yao, Albert J. Zhai, Shenlong Wang
* You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale, Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo, Tiejun Huang, Lulu Tang, Xinlong Wang
* Radio Frequency Ray Tracing with Neural Object Representation for Enhanced RF Modeling, Xingyu Chen, Zihao Feng, Kun Qian, Xinyu Zhang
* Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes, Stefano Esposito, Anpei Chen, Christian Reiser, Samuel Rota Bulò, Lorenzo Porzi, Katja Schwarz, Christian Richardt, Michael Zollhöfer, Peter Kontschieder, Andreas Geiger
* RelationField: Relate Anything in Radiance Fields, Sebastian Koch, Johanna Wald, Mirco Colosi, Narunas Vaskevicius, Pedro Hermosilla, Federico Tombari, Timo Ropinski
* Chain of Semantics Programming in 3D Gaussian Splatting Representation for 3D Vision Grounding, Jiaxin Shi, Mingyue Xiang, Hao Sun, Yixuan Huang, Zhi Weng
* Proxy Transformation: Preshaping Point Cloud Manifold With Proxy Attention For 3D Visual Grounding, Qihang Peng, Henry Zheng, Gao Huang
* 3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination, Jianing Yang, Xuweiyi Chen, Nikhil Madaan, Madhavan lyengar, Shengyi Qian, David F. Fouhey, Joyce Chai
* Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range, Zorah Lähner, Vladislav Golyanik, Ziyuan Qu, Zihao Zou, Vivek Boominathan, Praneeth Chakravarthula, Adithya Pediredla
* Blurred LiDAR for Sharper 3D: Robust Handheld 3D Scanning with Diffuse LiDAR and RGB, Nikhil Behari, Aaron Young, Siddharth Somasundaram, Tzofi Klinghoffer, Akshat Dave, Ramesh Raskar
* GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting, Shujuan Li, Yu-Shen Liu, Zhizhong Han
* Mitigating Ambiguities in 3D Classification with Gaussian Splatting, Ruiqi Zhang, Hao Zhu, Jingyi Zhao, Qi Zhang, Xun Cao, Zhan Ma
* 3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping, Chenhui Shi, Fulin Tang, Ning An, Yihong Wu
* Continuous 3D Perception Model with Persistent State, Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa
* MVSAnywhere: Zero-Shot Multi-View Stereo, Sergio Izquierdo, Mohamed Sayed, Michael Firman, Guillermo Garcia-Hernando, Daniyar Turmukhambetov, Javier Civera, Oisin Mac Aodha, Gabriel Brostow, Jamie Watson
* Three-view Focal Length Recovery From Homographies, Yaqing Ding, Viktor Kocur, Zuzana Berger Haladova, Qianliang Wu, Shen Cai, Jian Yang, Zuzana Kukelova
* GeoDepth: From Point-to-Depth to Plane-to-Depth Modeling for Self-Supervised Monocular Depth Estimation, Haifeng Wu, Shuhang Gu, Lixin Duan, Wen Li
* HUSH: Holistic Panoramic 3D Scene Understanding using Spherical Harmonics, Jongsung Lee, Harin Park, Byeong-Uk Lee, Kyungdon Joo
* DORNet: A Degradation Oriented and Regularized Network for Blind Depth Super-Resolution, Zhengxue Wang, Zhiqiang Yan, Jinshan Pan, Guangwei Gao, Kai Zhang, Jian Yang
* MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data, Hanwen Jiang, Zexiang Xu, Desai Xie, Ziwen Chen, Haian Jin, Fujun Luan, Zhixin Shu, Kai Zhang, Sai Bi, Xin Sun, Jiuxiang Gu, Qixing Huang, Georgios Pavlakos, Hao Tan
* SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation, Duc-Hai Pham, Tung Do, Phong Nguyen, Binh-Son Hua, Khoi Nguyen, Rang Nguyen
* Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation, Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, Bingyi Kang
* Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting, Shu-Wei Lu, Yi-Hsuan Tsai, Yi-Ting Chen
* Buffer Anytime: Zero-Shot Video Depth and Normal from Image Priors, Zhengfei Kuang, Tianyuan Zhang, Kai Zhang, Hao Tan, Sai Bi, Yiwei Hu, Zexiang Xu, Milos Hasan, Gordon Wetzstein, Fujun Luan
* DepthCues: Evaluating Monocular Depth Perception in Large Vision Models, Duolikun Danier, Mehmet Aygün, Changjian Li, Hakan Bilen, Oisin Mac Aodha

## **3D生成 (3D Generation)**


* **\[Oral\] CraftsMan3D: High-fidelity Mesh Generation with 3D Native Diffusion and Interactive Geometry Refiner**, Weiyu Li, Jiarui Liu, Hongyu Yan, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, Xiaoxiao Long
* **\[Oral\] DNF: Unconditional 4D Generation with Dictionary-based Neural Fields**, Xinyi Zhang, Naiqi Li, Angela Dai
* *\[Highlight\] Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation*, Xiang Li, Zixuan Huang, Anh Thai, James M. Rehg
* *\[Highlight\] MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder and Generative Data Augmentation*, Zilong Chen, Yikai Wang, Wenqiang Sun, Feng Wang, Yiwen Chen, Huaping Liu
* *\[Highlight\] GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control*, Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, Jun Gao
* *\[Highlight\] ARM: Appearance Reconstruction Model for Relightable 3D Generation*, Xiang Feng, Chang Yu, Zoubin Bi, Yintong Shang, Feng Gao, Hongzhi Wu, Kun Zhou, Chenfanfu Jiang, Yin Yang
* *\[Highlight\] Structured 3D Latents for Scalable and Versatile 3D Generation*, Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, Jiaolong Yang
* *\[Highlight\] Few-shot Implicit Function Generation via Equivariance*, Suizhi Huang, Xingyi Yang, Hongtao Lu, Xinchao Wang
* SKDream: Controllable Multi-view and 3D Generation with Arbitrary Skeletons, Yuanyou Xu, Zongxin Yang, Yi Yang
* Fancy123: One Image to High-Quality 3D Mesh Generation via Plug-and-Play Deformation, Qiao Yu, Xianzhi Li, Yuan Tang, Xu Han, Long Hu, Yixue Hao, Min Chen
* ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion, Nissim Maruani, Wang Yifan, Matthew Fisher, Pierre Alliez, Mathieu Desbrun
* MeshArt: Generating Articulated Meshes with Structure-Guided Transformers, Daoyi Gao, Yawar Siddiqui, Lei Li, Angela Dai
* SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation, Aleksey Bokhovkin, Quan Meng, Shubham Tulsiani, Angela Dai
* LT3SD: Latent Trees for 3D Scene Diffusion, Quan Meng, Lei Li, Matthias Nießner, Angela Dai
* Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation, Yuanbo Yang, Jiahao Shao, Xinyang Li, Yujun Shen, Andreas Geiger, Yiyi Liao
* COSER: Towards Consistent Dense Multiview Text-to-Image Generator for 3D Creation, Bonan Li, Zicheng Zhang, Xingyi Yang, Xinchao Wang
* ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary, Zeqi Gu, Yin Cui, Zhaoshuo Li, Fangyin Wei, Yunhao Ge, Jinwei Gu, Ming-Yu Liu, Abe Davis, Yifan Ding
* ArtFormer: Controllable Generation of Diverse 3D Articulated Objects, Jiayi Su, Youhe Feng, Zheng Li, Jinhua Song, Yangfan He, Botao Ren, Botian Xu
* Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation, Jiantao Lin, Xin Yang, Meixi Chen, Yingjie Xu, Dongyu Yan, Leyi Wu, Xinli Xu, Lie Xu, Shunsi Zhang, Ying-Cong Chen
* PartGen: Part-level 3D Generation and Reconstruction with Multi-view Diffusion Models, Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, Andrea Vedaldi
* FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts, Tongyuan Bai, Wangyuanfan Bai, Dong Chen, Tieru Wu, Manyi Li, Rui Ma
* WonderWorld: Interactive 3D Scene Generation from a Single Image, Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, Jiajun Wu
* MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model, Chenjie Cao, Chaohui Yu, Shang Liu, Fan Wang, Xiangyang Xue, Yanwei Fu
* Scene Splatter: Momentum 3D Scene Generation from Single Image with Video Diffusion Model, Shengjun Zhang, Jinzhao Li, Xin Fei, Hao Liu, Yueqi Duan
* Generative Gaussian Splatting for Unbounded 3D City Generation, Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, Ziwei Liu
* MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation, Sankalp Sinha, Mohammad Sadil Khan, Muhammad Usama, Shino Sam, Didier Stricker, Sk Aziz Ali, Muhammad Zeshan Afzal
* Global-Local Tree Search in VLMs for 3D Indoor Scene Generation, Wei Deng, Mengshi Qi, Huadong Ma
* Hash3D: Training-free Acceleration for 3D Generation, Xingyi Yang, Songhua Liu, Xinchao Wang
* Robust Multi-Object 4D Generation for In-the-wild Videos, Wen-Hsuan Chu, Lei Ke, Jianmeng Liu, Mingxiao Huo, Pavel Tokmakov, Katerina Fragkiadaki
* GenAssets: Generating in-the-wild 3D Assets in Latent Space, Ze Yang, Jingkang Wang, Haowei Zhang, Sivabalan Manivasagam, Yun Chen, Raquel Urtasun
* MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation, Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, Lu Sheng
* Turbo3D: Ultra-fast Text-to-3D Generation, Hanzhe Hu, Tianwei Yin, Fujun Luan, Yiwei Hu, Hao Tan, Zexiang Xu, Sai Bi, Shubham Tulsiani, Kai Zhang
* Material Anything: Generating Materials for Any 3D Object via Diffusion, Xin Huang, Tengfei Wang, Ziwei Liu, Qing Wang
* 3DTopia-XL: Scaling High-quality 3D Asset Generation via Primitive Diffusion, Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, Liang Pan, Dahua Lin, Ziwei Liu
* BrepGiff: Lightweight Generation of Complex B-rep with 3D GAT Diffusion, Hao Guo, Xiaoshui Huang, Hao jiacheng, Yunpeng Bai, Hongping Gan, Yilei Shi
* TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing, Stefan Lionar, Jiabin Liang, Gim Hee Lee
* GenVDM: Generating Vector Displacement Maps From a Single Image, Yuezhi Yang, Qimin Chen, Vladimir G. Kim, Siddhartha Chaudhuri, Qixing Huang, Zhiqin Chen
* SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE, Yongwei Chen, Yushi Lan, Shangchen Zhou, Tengfei Wang, Xingang Pan
* Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data, Zhiyuan Ma, Xinyue Liang, Rongyuan Wu, Xiangyu Zhu, Zhen Lei, Lei Zhang
* CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images, Cheng Chen, Jiacheng Wei, Tianrun Chen, Chi Zhang, Xiaofeng Yang, Shangzhan Zhang, Bingchen Yang, Chuan-Sheng Foo, Guosheng Lin, Qixing Huang, Fayao Liu
* MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation, Jinnan Chen, Lingting Zhu, Zeyu Hu, Shengju Qian, Yugang Chen, Xin Wang, Gim Hee Lee
* Scaling Mesh Generation via Compressive Tokenization, Haohan Weng, Zibo Zhao, Biwen Lei, Xianghui Yang, Jian Liu, Zeqiang Lai, Zhuo Chen, Yuhong Liu, Jie Jiang, Chunchao Guo, Tong Zhang, Shenghua Gao, C.L. Philip Chen
* Hierarchical Gaussian Mixture Model Splatting for Efficient and Part Controllable 3D Generation, Qitong Yang, Mingtao Feng, Zijie Wu, Weisheng Dong, Fangfang Wu, Yaonan Wang, Ajmal Mian
* Transfer Your Perspective: Controllable 3D Generation from Any Viewpoint in a Driving Scene, Tai-Yu Pan, Sooyoung Jeon, Mengdi Fan, Jinsu Yoo, Zhenyang Feng, Mark Campbell, Kilian Q. Weinberger, Bharath Hariharan, Wei-Lun Chao
* SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation, Dekai Zhu, Yan Di, Stefan Gavranovic, Slobodan Ilic
* Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation, Shivam Duggal, Yushi Hu, Oscar Michel, Aniruddha Kembhavi, William T. Freeman, Noah A. Smith, Ranjay Krishna, Antonio Torralba, Ali Farhadi, Wei-Chiu Ma
* DirectTriGS: Triplane-based Gaussian Splatting Field Representation for 3D Generation, Xiaoliang Ju, Hongsheng Li
* Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders, Rui Chen, Jianfeng Zhang, Yixun Liang, Guan Luo, Weiyu Li, Jiarui Liu, Xiu Li, Xiaoxiao Long, Jiashi Feng, Ping Tan
* Acc3D: Accelerating Single Image to 3D Diffusion Models via Edge Consistency Guided Score Distillation, Kendong Liu, Zhiyu Zhu, Hui Liu, Junhui Hou
* CompGS: Unleashing 2D Compositionality for Compositional Text-to-3D via Dynamically Optimizing 3D Gaussians, Chongjian Ge, Chenfeng Xu, Yuanfeng Ji, Chensheng Peng, Masayoshi Tomizuka, Ping Luo, Mingyu Ding, Varun Jampani, Wei Zhan
* Apply Hierarchical-Chain-of-Generation to Complex Attributes Text-to-3D Generation, Yiming Qin, Zhu Xu, Yang Liu
* Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects, Shalini Maiti, Lourdes Agapito, Filippos Kokkinos

## **3D高斯溅射 (3D Gaussian Splatting)**


* **\[Oral\] EgoLM: Multi-Modal Language Model of Egocentric Motions**, Fangzhou Hong, Vladimir Guzov, Hyo Jin Kim, Yuting Ye, Richard Newcombe, Ziwei Liu, Lingni Ma
* **\[Oral\] 3D Student Splatting and Scooping**, Jialin Zhu, Jiangbei Yue, Feixiang He, He Wang
* **\[Oral\] 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting**, Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moënne-Loccoz, Zan Gojcic
* *\[Highlight\] Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs*, Yingji Zhong, Zhihao Li, Dave Zhenyu Chen, Lanqing Hong, Dan Xu
* *\[Highlight\] NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting*, Yulong Zheng, Zicheng Jiang, Shengfeng He, Yandu Sun, Junyu Dong, Huaidong Zhang, Yong Du
* *\[Highlight\] Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects*, Amir Barda, Matheus Gadelha, Vladimir G. Kim, Noam Aigerman, Amit H. Bermano, Thibault Groueix
* *\[Highlight\] OmniSplat: Taming Feed-Forward 3D Gaussian Splatting for Omnidirectional Images with Editable Capabilities*, Suyoung Lee, Jaeyoung Chung, Kihoon Kim, Jaeyoo Huh, Gunhee Lee, Minsoo Lee, Kyoung Mu Lee
* *\[Highlight\] ActiveGAMER: Active GAussian Mapping through Efficient Rendering*, Liyan Chen, Huangying Zhan, Kevin Chen, Xiangyu Xu, Qingan Yan, Changjiang Cai, Yi Xu
* Hardware-Rasterized Ray-Based Gaussian Splatting, Samuel Rota Bulò, Nemanja Bartolovic, Lorenzo Porzi, Peter Kontschieder
* Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering, Yutao Feng, Xiang Feng, Yintong Shang, Ying Jiang, Chang Yu, Zeshun Zong, Tianjia Shao, Hongzhi Wu, Kun Zhou, Chenfanfu Jiang, Yin Yang
* TexGaussian: Generating High-quality PBR Material via Octree-based 3D Gaussian Splatting, Bojun Xiong, Jialun Liu, Jiakui Hu, Chenming Wu, Jinbo Wu, Xing Liu, Chen Zhao, Errui Ding, Zhouhui Lian
* iSegMan: Interactive Segment-and-Manipulate 3D Gaussians, Yian Zhao, Wanshi Xu, Ruochong Zheng, Pengchong Qiao, Chang Liu, Jie Chen
* LOD-GS: Achieving Levels of Detail using Scalable Gaussian Soup, Jianxiong Shen, Yue Qian, Xiaohang Zhan
* MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks, Yifei Liu, Zhihang Zhong, Yifan Zhan, Sheng Xu, Xiao Sun
* NTR-Gaussian: Nighttime Dynamic Thermal Reconstruction with 4D Gaussian Splatting Based on Thermodynamics, Kun Yang, Yuxiang Liu, Zeyu Cui, Yu Liu, Maojun Zhang, Shen Yan, Qing Wang
* DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering, Yexing Xu, Longguang Wang, Minglin Chen, Sheng Ao, Li Li, Yulan Guo
* S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting, Yecong Wan, Mingwen Shao, Yuanshuo Cheng, Wangmeng Zuo
* DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering, Yihao Wang, Marcus Klasson, Matias Turkulainen, Shuzhe Wang, Juho Kannala, Arno Solin
* DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery, Jiadong Tang, Yu Gao, Dianyi Yang, Liqi Yan, Yufeng Yue, Yi Yang
* IndoorGS: Geometric Cues Guided Gaussian Splatting for Indoor Scene Reconstruction, Cong Ruan, Yuesong Wang, Tao Guan, Bin Zhang, Lili Ju
* 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video, Qiang Hu, Zihan Zheng, Houqiang Zhong, Sihua Fu, Li Song, Xiaoyun Zhang, Guangtao Zhai, Yanfeng Wang
* EnliveningGS: Active Locomotion of 3DGS, Siyuan Shen, Tianjia Shao, Kun Zhou, Chenfanfu Jiang, Yin Yang
* Gaussian Splatting Feature Fields for (Privacy-Preserving) Visual Localization, Maxime Pietrantoni, Gabriela Csurka, Torsten Sattler
* FlexDrive: Toward Trajectory Flexibility in Driving Scene Gaussian Splatting Reconstruction and Rendering, Jingqiu Zhou, Lue Fan, Linjiang Huang, Xiaoyu Shi, Si Liu, Zhaoxiang Zhang, Hongsheng Li
* POP-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality, Joey Wilson, Marcelino Almeida, Sachit Mahajan, Martin Labrie, Maani Ghaffari, Omid Ghasemalizadeh, Min Sun, Cheng-Hao Kuo, Arnab Sen
* Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting, Runsong Zhu, Shi Qiu, Zhengzhe Liu, Ka-Hei Hui, Qianyi Wu, Pheng-Ann Heng, Chi-Wing Fu
* EnvGS: Modeling View-Dependent Appearance with Environment Gaussian, Tao Xie, Xi Chen, Zhen Xu, Yiman Xie, Yudong Jin, Yujun Shen, Sida Peng, Hujun Bao, Xiaowei Zhou
* UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping, Aashish Rai, Dilin Wang, Mihir Jain, Nikolaos Sarafianos, Kefan Chen, Srinath Sridhar, Aayush Prakash
* 3D-GSW: 3D Gaussian Splatting for Robust Watermarking, Youngdong Jang, Hyunje Park, Feng Yang, Heeju Ko, Euijin Choo, Sangpil Kim
* PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting, Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana, Matthias Zwicker, Tom Goldstein
* Gaussian Splatting for Efficient Satellite Image Photogrammetry, Luca Savant Aira, Gabriele Facciolo, Thibaud Ehret
* HyperGS: Hyperspectral 3D Gaussian Splatting, Christopher Thirgood, Oscar Mendez, Erin Ling, Jon Storey, Simon Hadfield
* RoGSplat: Learning Robust Generalizable Human Gaussian Splatting from Sparse Multi-View Images, Junjin Xiao, Qing Zhang, Yonewei Nie, Lei Zhu, Wei-Shi Zheng
* GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping, Jinfeng Liu, Lingtong Kong, Bo Li, Dan Xu
* MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views, Antoine Guedon, Tomoki Ichikawa, Kohei Yamashita, Ko Nishino
* Feat2GS: Probing Visual Foundation Models with Gaussian Splatting, Yue Chen, Xingyu Chen, Anpei Chen, Gerard Pons-Moll, Yuliang Xiu
* Textured Gaussians for Enhanced 3D Scene Appearance Modeling, Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao, Tuotuo Li, Qinbo Li, Ayush Saraf, Jia-Bin Huang, Johannes Kopf, Gordon Wetzstein, Changil Kim
* DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Post-Capture Refocusing, Defocus Rendering and Blur Removal, Yujie Wang, Praneeth Chakravarthula, Baoquan Chen
* Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh, Xiangjun Gao, Xiaoyu Li, Yiyu Zhuang, Qi Zhang, Wenbo Hu, Chaopeng Zhang, Yao Yao, Ying Shan, Long Quan
* Deformable Radial Kernel Splatting, Yi-Hua Huang, Ming-Xian Lin, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, Xiaojuan Qi
* SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis, Hyojun Go, Byeongjun Park, Jiho Jang, Jin-Young Kim, Soonwoo Kwon, Changick Kim
* Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives, Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, Tom Goldstein
* GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction, Jinguang Tong, Xuesong Li, Fahira Afzal Maken, Sundaram Muthu, Lars Petersson, Chuong Nguyen, Hongdong Li
* MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models, Yifan Liu, Keyu Fan, Weihao Yu, Chenxin Li, Hao Lu, Yixuan Yuan
* LITA-GS: Illumination-Agnostic Novel View Synthesis via Reference-Free 3D Gaussian Splatting and Physical Priors, Han Zhou, Wei Dong, Jun Chen
* Splatter-360: Generalizable 360 Gaussian Splatting for Wide-baseline Panoramic Images, Zheng Chen, Chenming Wu, Zhelun Shen, Chen Zhao, Weicai Ye, Haocheng Feng, Errui Ding, Song-Hai Zhang
* DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting, Hyunwoo Park, Gun Ryu, Wonjun Kim
* SfM-Free 3D Gaussian Splatting via Hierarchical Training, Bo Ji, Angela Yao
* Improving Gaussian Splatting with Localized Points Management, Haosen Yang, Chenhao Zhang, Wenqing Wang, Marco Volino, Adrian Hilton, Li Zhang, Xiatian Zhu
* DIET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting, Seungjun Lee, Gim Hee Lee
* SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting, Gyeongjin Kang, Jisang Yoo, Jihyeon Park, Seungtae Nam, Hyeonsoo Im, Sangheon Shin, Sangpil Kim, Eunbyung Park
* DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting, Liao Shen, Tianqi Liu, Huiqiang Sun, Jiaqi Li, Zhiguo Cao, Wei Li, Chen Change Loy
* Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment, Ziteng Cui, Xuangeng Chu, Tatsuya Harada
* Ref-GS: Directional Factorization for 2D Gaussian Splatting, Youjia Zhang, Anpei Chen, Yumin Wan, Zikai Song, Junqing Yu, Yawei Luo, Wei Yang
* LeanGaussian: Breaking Pixel or Point Cloud Correspondence in Modeling 3D Gaussians, Jiamin Wu, Kenkun Liu, Han Gao, Xiaoke Jiang, Yuan Yao, Lei Zhang
* FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering, Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, Linning Xu, Zhilin Pei, Hengjie Li, Xiuhong Li, Ninghui Sun, Xingcheng Zhang, Bo Dai
* Steepest Descent Density Control for Compact 3D Gaussian Splatting, Peihao Wang, Yuehao Wang, Dilin Wang, Sreyas Mohan, Zhiwen Fan, Lemeng Wu, Ruisi Cai, Yu-Ying Yeh, Zhangyang Wang, Qiang Liu, Rakesh Ranjan
* GaussianSpa: An "Optimizing-Sparsifying" Simplification Framework for Compact and High-Quality 3D Gaussian Splatting, Yangming Zhang, Wenqi Jia, Wei Niu, Miao Yin
* HoGS: Unified Near and Far Object Reconstruction via Homogeneous Gaussian Splatting, Xinpeng Liu, Zeyi Huang, Fumio Okura, Yasuyuki Matsushita
* Generative Sparse-View Gaussian Splatting, Hanyang Kong, Xingyi Yang, Xinchao Wang
* 4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians, Hidenobu Matsuki, Gwangbin Bae, Andrew J. Davison
* EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering, Toshiya Yura, Ashkan Mirzaei, Igor Gilitschenski
* IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera, Jian Huang, Chengrui Dong, Xuanhua Chen, Peidong Liu
* ArticulatedGS: Self-supervised Digital Twin Modeling of Articulated Objects using 3D Gaussian Splatting, Junfu Guo, Yu Xin, Gaoyi Liu, Kai Xu, Ligang Liu, Ruizhen Hu
* Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians, Changfeng Ma, Ran Bi, Jie Guo, Chongjun Wang, Yanwen Guo
* TaoAvatar: Real-Time Lifelike Full-Body Talking Avatars for Augmented Reality via 3D Gaussian Splatting, Jianchuan Chen, Jingchuan Hu, Gaige Wang, Zhonghua Jiang, Tiansong Zhou, Zhiwen Chen, Chengfei Lv
* IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing, Chun Gu, Xiaofei Wei, Zixuan Zeng, Yuxuan Yao, Li Zhang
* Volumetrically Consistent 3D Gaussian Rasterization, Chinmay Talegaonkar, Yash Belhe, Ravi Ramamoorthi, Nicholas Antipa
* 3D-HGS: 3D Half-Gaussian Splatting, Haolin Li, Jinyang Liu, Mario Sznaier, Octavia Camps
* FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting, Fangyu Wu, Yuhao Chen
* DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds, Youyu Chen, Junjun Jiang, Kui Jiang, Xiao Tang, Zhihao Li, Xianming Liu, Yinyu Nie
* Efficient Decoupled Feature 3D Gaussian Splatting via Hierarchical Compression, Zhenqi Dai, Ting Liu, Yanning Zhang
* SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting, Jiahui Zhang, Fangneng Zhan, Ling Shao, Shijian Lu
* RestorGS: Depth-aware Gaussian Splatting for Efficient 3D Scene Restoration, Yuanjian Qiao, Mingwen Shao, Lingzhuang Meng, Kai Xu
* SPC-GS: Gaussian Splatting with Semantic-Prompt Consistency for Indoor Open-World Free-view Synthesis from Sparse Inputs, Guibiao Liao, Qing Li, Zhenyu Bao, Guoping Qiu, Kanglin Liu
* EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis, Sheng Miao, Jiaxin Huang, Dongfeng Bai, Xu Yan, Hongyu Zhou, Yue Wang, Bingbing Liu, Andreas Geiger, Yiyi Liao
* Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views, Jiang Wu, Rui Li, Yu Zhu, Rong Guo, Jinqiu Sun, Yanning Zhang
* MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting, Sangwoon Kwak, Joonsoo Kim, Jun Young Jeong, Won-Sik Cheong, Jihyong Oh, Munchurl Kim
* PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting, Cheng Zhang, Haofei Xu, Qianyi Wu, Camilo Cruz Gambardella, Dinh Phung, Jianfei Cai
* WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments, Jianhao Zheng, Zihan Zhu, Valentin Bieri, Marc Pollefeys, Songyou Peng, Iro Armeni
* SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving, Georg Hess, Carl Lindström, Maryam Fatemi, Christoffer Petersson, Lennart Svensson
* EigenGS Representation: From Eigenspace to Gaussian Image Space, Lo-Wei Tai, Ching-En Li, Cheng-Lin Chen, Chih-Jung Tsai, Hwann-Tzong Chen, Tyng-Luh Liu
* Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration, Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, Tae-Hyun Oh
* Creating Your Editable 3D Photorealistic Avatar with Tetrahedron-constrained Gaussian Splatting, Hanxi Liu, Yifang Men, Zhouhui Lian
* CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images, Jungho Lee, Suhwan Cho, Taeoh Kim, Ho-Deok Jang, Minhyeok Lee, Geonho Cha, Dongyoon Wee, Dogyoon Lee, Sangyoun Lee
* SpecTRe-GS: Modeling Highly Specular Surfaces with Reflected Nearby Objects by Tracing Rays in 3D Gaussian Splatting, Jiajun Tang, Fan Fei, Zhihao Li, Xiao Tang, Shiyong Liu, Youyu Chen, Binxiao Huang, Zhenyu Chen, Xiaofei Wu, Boxin Shi
* SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering, Hanxiao Sun, Yupeng Gao, Jin Xie, Jian Yang, Beibei Wang
* RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting, Qiyu Dai, Xingyu Ni, Qianfan Shen, Wenzheng Chen, Baoquan Chen, Mengyu Chu
* GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting, Zixuan Chen, Guangcong Wang, Jiahao Zhu, Jianhuang Lai, Xiaohua Xie
* FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting, Hengyu Liu, Yuehao Wang, Chenxin Li, Ruisi Cai, Kevin Wang, Wuyang Li, Pavlo Molchanov, Peihao Wang, Zhangyang Wang
* EVPGS: Enhanced View Prior Guidance for Splatting-based Extrapolated View Synthesis, Jiahe Li, Feiyu Wang, Xiaochao Qu, Chengjing Wu, Luoqi Liu, Ting Liu
* DepthSplat: Connecting Gaussian Splatting and Depth, Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, Marc Pollefeys
* EAP-GS: Efficient Augmentation of Pointcloud for 3D Gaussian Splatting in Few-shot Scene Reconstruction, Dongrui Dai, Yuxiang Xing
* Instant Gaussian Stream: Fast and Generalizable Streaming of Dynamic Scene Reconstruction via Gaussian Splatting, Jinbo Yan, Rui Peng, Zhiyan Wang, Luyang Tang, Jiayu Yang, Jie Liang, Jiahao Wu, Ronggang Wang
* BARD-GS: Blur-Aware Reconstruction of Dynamic Scenes via Gaussian Splatting, Yiren Lu, Yunlai Zhou, Disheng Liu, Tuo Liang, Yu Yin
* GauSTAR: Gaussian Surface Tracking and Reconstruction, Chengwei Zheng, Lixin Xue, Juan Zarate, Jie Song
* USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting, Kang Chen, Jiyuan Zhang, Zecheng Hao, Yajing Zheng, Tiejun Huang, Zhaofei Yu
* BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting, Jeongwan On, Kyeonghwan Gwak, Gunyoung Kang, Junuk Cha, Soohyun Hwang, Hyein Hwang, Seungryul Baek
* InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception, Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao Zhang, Ronggang Wang, Jian Zhang
* COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting, Jiaxin Zhang, Junjun Jiang, Youyu Chen, Kui Jiang, Xianming Liu
* PromptVFX: Text-Driven Fields for Open-World 3D Gaussian Animation, Mert Kiray, Paul Uhlenbruck, Benjamin Busam

## **神经辐射场 (NeRF)**


* **\[Oral\] Neural Inverse Rendering from Propagating Light**, Anagh Malik, Benjamin Attal, Andrew Xie, Matthew O'Toole, David B. Lindell
* Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations, Xunzhi Zheng, Dan Xu
* Exploiting Deblurring Networks for Radiance Fields, Haeyun Choi, Heemin Yang, Janghyeok Han, Sunghyun Cho
* DynaMoDe-NeRF: Motion-aware Deblurring Neural Radiance Field for Dynamic Scenes, Ashish Kumar, Rajagopalan A. N.
* GoLF-NRT: Integrating Global Context and Local Geometry for Few-Shot View Synthesis, You Wang, Li Fang, Hao Zhu, Fei Hu, Long Ye, Zhan Ma
* Factored-NeuS: Reconstructing Surfaces, Illumination, and Materials of Possibly Glossy Objects, Yue Fan, Ningjing Fan, Ivan Skorokhodov, Oleg Voynov, Savva Ignatyev, Evgeny Burnaev, Peter Wonka, Yiqun Wang
* GO-N3RDet: Geometry Optimized NeRF-enhanced 3D Object Detector, Zechuan Li, Hongshan Yu, Yihao Ding, Jinhao Qiao, Basim Azam, Naveed Akhtar
* Preconditioners for the Stochastic Training of Neural Fields, Shin-Fang Chng, Hemanth Saratchandran, Simon Lucey
* RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings, Aayush Dhakal, Srikumar Sastry, Subash Khanal, Adeel Ahmad, Eric Xing, Nathan Jacobs
* FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields, Kwan Yun, Chaelin Kim, Hangyeul Shin, Junyong Noh
* Neural Inverse Rendering from Propagating Light, Anagh Malik, O Benjamin Attal, Andrew Xie, Matthew O'Toole, David B. Lindell
* PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields, Sean Wu, Shamik Basu, Tim Broedermann, Luc Van Gool, Christos Sakaridis
* Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction, Li Fang, Hao Zhu, Longlong Chen, Fei Hu, Long Ye, Zhan Ma
* FrugalNeRF: Fast Convergence for Extreme Few-shot Novel View Synthesis without Learned Priors, Chin-Yang Lin, Chung-Ho Wu, Chang-Han Yeh, Shih-Han Yen, Cheng Sun, Yu-Lun Liu
* NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction, Wenyuan Zhang, Emily Yue-ting Jia, Junsheng Zhou, Baorui Ma, Kanle Shi, Yu-Shen Liu, Zhizhong Han
* Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video, Hoang Chuong Nguyen, Wei Mao, Jose M. Alvarez, Miaomiao Liu
* VI^3NR: Variance Informed Initialization for Implicit Neural Representations, Chamin Hewa Koneputugodage, Yizhak Ben-Shabat, Sameera Ramasinghe, Stephen Gould
* Advancing Adversarial Robustness in GNeRFs: The IL2-NeRF Attack, Nicole Meng, Caleb Manicke, Ronak Sahu, Caiwen Ding, Yingjie Lao

## **3D重建 (3D Reconstruction)**


* **\[Oral\] MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds**, Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, Zhicheng Yan
* **\[Oral\] DIFIX3D+: Improving 3D Reconstructions with Single-Step Diffusion Models**, Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, Huan Ling
* **\[Oral\] Fluid Nexus: 3D Fluid Reconstruction and Prediction from a Single Video**, Yue Gao, Hong-Xing Yu, Bo Zhu, Jiajun Wu
* **\[Oral\] Diffusion Renderer: Neural Inverse and Forward Rendering with Video Diffusion Models**, Ruofan Liang, Zan Gojcic, Huan Ling, Jacob Munkberg, Jon Hasselgren, Chih-Hao Lin, Jun Gao, Alexander Keller, Nandita Vijaykumar, Sanja Fidler, Zian Wang
* **\[Oral\] MegaSaM: Accurate**, Fast and Robust Structure and Motion from Casual Dynamic Videos, Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, Noah Snavely
* **\[Oral\] Camera Resection from Known Line Pencils and a Radially Distorted Scanline**, Juan C. Dibene, Enrique Dunn
* *\[Highlight\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging*, Ping Wang, Lishun Wang, Gang Qu, Xiaodong Wang, Yulun Zhang, Xin Yuan
* *\[Highlight\] Structure-from-Motion with a Non-Parametric Camera Model*, Yihan Wang, Linfei Pan, Marc Pollefeys, Viktor Larsson
* *\[Highlight\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model*, Yiyang Shen, Kun Zhou, He Wang, Yin Yang, Tianjia Shao
* *\[Highlight\] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization*, You Shen, Zhipeng Zhang, Xinyang Li, Yansong Qu, Yu Lin, Shengchuan Zhang, Liujuan Cao
* MultiGO: Towards Multi-level Geometry Learning for Monocular 3D Textured Human Reconstruction, Gangjian Zhang, Nanjie Yao, Shunsi Zhang, Hanfeng Zhao, Guoliang Pang, Jian Shu, Hao Wang
* Glossy Object Reconstruction with Cost-effective Polarized Acquisition, Bojian Wu, Yifan Peng, Ruizhen Hu, Xiaowei Zhou
* LIRM: Large Inverse Rendering Model for Progressive Reconstruction of Shape, Materials and View-dependent Radiance Fields, Zhengqin Li, Dilin Wang, Ka Chen, Zhaoyang Lv, Thu Nguyen-Phuoc, Milim Lee, Jia-Bin Huang, Lei Xiao, Yufeng Zhu, Carl S. Marshall, Yuheng Ren, Richard Newcombe, Zhao Dong
* Real-time Free-view Human Rendering from Sparse-view RGB Videos using Double Unprojected Textures, Guoxing Sun, Rishabh Dabral, Heming Zhu, Pascal Fua, Christian Theobalt, Marc Habermann
* MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction, Xiaohao Xu, Feng Xue, Shibo Zhao, Yike Pan, Sebastian Scherer, Xiaonan Huang
* ShowMak3r: Compositional TV Show Reconstruction, Sangmin Kim, Seunguk Do, Jaesik Park
* HiMoR: Monocular Deformable Gaussian Reconstruction with Hierarchical Motion Representation, Yiming Liang, Tianhan Xu, Yuta Kikuchi
* Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors, Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, Jerome Revaud
* A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions, Jiangbei Hu, Yanggeng Li, Fei Hou, Junhui Hou, Zhebin Zhang, Shengfa Wang, Na Lei, Ying He
* DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image, Hyeongjin Nam, Donghwan Kim, Jeongtaek Oh, Kyoung Mu Lee
* SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction, Xinran Yang, Donghao Ji, Yuanqi Li, Jie Guo, Yanwen Guo, Junyuan Xie
* Decompositional Neural Scene Reconstruction with Generative Diffusion Prior, Junfeng Ni, Yu Liu, Ruijie Lu, Zirui Zhou, Song-Chun Zhu, Yixin Chen, Siyuan Huang
* GenFusion: Closing the Loop between Reconstruction and Generation via Videos, Sibo Wu, Congrong Xu, Binbin Huang, Andreas Geiger, Anpei Chen
* Link to the Past: Temporal Propagation for Fast 3D Human Reconstruction from Monocular Video, Matthew Marchellus, Nadhira Noor, In Kyu Park
* Dense-SfM: Structure from Motion with Dense Consistent Matching, JongMin Lee, Sungjoo Yoo
* ArcPro: Architectural Programs for Structured 3D Abstraction of Sparse Points, Qirui Huang, Runze Zhang, Kangjun Liu, Minglun Gong, Hao Zhang, Hui Huang
* ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration, Johan Edstedt, André Mateus, Alberto Jaenal
* SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes, Cheng-De Fan, Chen-Wei Chang, Yi-Ruei Liu, Jie-Ying Lee, Jiun-Long Huang, Yu-Chee Tseng, Yu-Lun Liu
* MVBoost: Boost 3D Reconstruction with Multi-View Refinement, Xiangyu Liu, Xiaomei Zhang, Zhiyuan Ma, Xiangyu Zhu, Zhen Lei
* AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis, Khiem Vuong, Anurag Ghosh, Deva Ramanan, Srinivasa Narasimhan, Shubham Tulsiani
* AniGrad: Anisotropic Gradient-Adaptive Sampling for 3D Reconstruction From Monocular Video, Noah Stier, Alex Rich, Pradeep Sen, Tobias Höllerer
* ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos, Zetong Zhang, Manuel Kaufmann, Lixin Xue, Jie Song, Martin R. Oswald
* Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass, Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, Matt Feiszli
* Reconstructing People, Places, and Cameras, Lea Müller, Hongsuk Choi, Anthony Zhang, Brent Yi, Jitendra Malik, Angjoo Kanazawa
* Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild, Junhyeong Cho, Kim Youwang, Hunmin Yang, Tae-Hyun Oh
* Towards In-the-wild 3D Plane Reconstruction from a Single Image, Jiachen Liu, Rui Yu, Sili Chen, Sharon X. Huang, Hengkai Guo
* V2V3D: View-to-View Denoised 3D Reconstruction for Light Field Microscopy, Jiayin Zhao, Zhenqi Fu, Tao Yu, Hui Qiao
* Learning Partonomic 3D Reconstruction from Image Collections, Xiaoqian Ruan, Pei Yu, Dian Jia, Hyeonjeong Park, Peixi Xiong, Wei Tang
* SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction, Yutao Tang, Yuxiang Guo, Deming Li, Cheng Peng
* PMNI: Pose-free Multi-view Normal Integration for Reflective and Textureless Surface Reconstruction, Mingzhi Pei, Xu Cao, Xiangyi Wang, Heng Guo, Zhanyu Ma
* CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections, Thomas Walker, Salvatore Esposito, Daniel Rebain, Amir Vaxman, Arno Onken, Changjian Li, Oisin Mac Aodha
* Coherent 3D Portrait Video Reconstruction via Triplane Fusion, Shengze Wang, Xueting Li, Chao Liu, Matthew Chan, Michael Stengel, Henry Fuchs, Shalini De Mello, Koki Nagano
* ProbeSDF: Light Field Probes For Neural Surface Reconstruction, Briac Toussaint, Diego Thomas, Jean-Sébastien Franco
* Matrix3D: Large Photogrammetry Model All-in-One, Yuanxun Lu, Jingyang Zhang, Tian Fang, Jean-Daniel Nahmias, Yanghai Tsin, Long Quan, Xun Cao, Yao Yao, Shiwei Li
* MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos, Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, Noah Snavely
* OffsetOPT: Explicit Surface Reconstruction without Normals, Huan Lei
* High-Fidelity Lightweight Mesh Reconstruction from Point Clouds, Chen Zhang, Wentao Wang, Ximeng Li, Xinyao Liao, Wanjuan Su, Wenbing Tao
* Parametric Point Cloud Completion for Polygonal Surface Reconstruction, Zhaiyu Chen, Yuqing Wang, Liangliang Nan, Xiao Xiang Zhu
* ViiNeuS: Volumetric Initialization for Implicit Neural Surface Reconstruction of Urban Scenes with Limited Image Overlap, Hala Djeghim, Nathan Piasco, Moussab Bennehar, Luis Roldao, Dzmitry Tsishkou, Désiré Sidibé
* Recovering Dynamic 3D Sketches from Videos, Jaeah Lee, Changwoon Choi, Young Min Kim, Jaesik Park
* PSHuman: Photorealistic Single-image 3D Human Reconstruction using Cross-Scale Multiview Diffusion and Explicit Remeshing, Peng Li, Wangguandong Zheng, Yuan Liu, Tao Yu, Yangguang Li, Xingqun Qi, Xiaowei Chi, Siyu Xia, Yan-Pei Cao, Wei Xue, Wenhan Luo, Yike Guo
* SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement, Mark Boss, Zixuan Huang, Aaryaman Vasishta, Varun Jampani
* PyTorchGeoNodes: Enabling Differentiable Shape Programs for 3D Shape Reconstruction, Sinisa Stekovic, Arslan Artykov, Stefan Ainetter, Mattia D'Urso, Friedrich Fraundorfer
* SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos, Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, Baoquan Chen
* 4D-Fly: Fast 4D Reconstruction from a Single Monocular Video, Diankun Wu, Fangfu Liu, Yi-Hsin Hung, Yue Qian, Xiaohang Zhan, Yueqi Duan
* Light3R-SfM: Towards Feed-forward Structure-from-Motion, Sven Elflein, Qunjie Zhou, Laura Leal-Taixé
* BADGR: Bundle Adjustment Diffusion Conditioned by Gradients for Wide-Baseline Floor Plan Reconstruction, Yuguang Li, Ivaylo Boyadzhiev, Zixuan Liu, Linda Shapiro, Alex Colburn
* MESC-3D:Mining Effective Semantic Cues for 3D Reconstruction from a Single Image, Shaoming Li, Qing Cai, Songqi Kong, Runqing Tan, Heng Tong, Shiji Qiu, Yongguo Jiang, Zhi Liu
* EdgeDiff: Edge-aware Diffusion Network for Building Reconstruction from Point Clouds, Yujun Liu, Ruisheng Wang, Shangfeng Huang, Guorong Cai
* ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping, Shun Iwase, Muhammad Zubair Irshad, Katherine Liu, Vitor Guizilini, Robert Lee, Takuya Ikeda, Ayako Amma, Koichi Nishiwaki, Kris Kitani, Rares Ambrus, Sergey Zakharov

# **四、感知与理解 (Perception & Understanding)**
## **目标检测 (Object Detection)**


* **\[Oral\] Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning**, Kunyu Wang, Xueyang Fu, Xin Lu, Chengjie Ge, Chengzhi Cao, Wei Zhai, Zheng-Jun Zha
* **\[Oral\] Believing is Seeing: Unobserved Object Detection using Generative Models**, Subhransu S. Bhattacharjee, Dylan Campbell, Rahul Shome
* *\[Highlight\] Towards RAW Object Detection in Diverse Conditions*, Zhong-Yu Li, Xin Jin, Bo-Yuan Sun, Chun-Le Guo, Ming-Ming Cheng
* *\[Highlight\] COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts*, Jiansheng Li, Xingxuan Zhang, Hao Zou, Yige Guo, Renzhe Xu, Yilong Liu, Chuzhao Zhu, Yue He, Peng Cui
* *\[Highlight\] Cubify Anything: Scaling Indoor 3D Object Detection*, Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, Afshin Dehghan
* SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection, Xin Lin, Chong Shi, Zuopeng Yang, Haojin Tang, Zhili Zhou
* SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection, Phi Vu Tran
* Percept, Memory, and Imagine: World Feature Simulating for Open-Domain Unknown Object Detection, Aming Wu, Cheng Deng
* Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection, Marc-Antoine Lavoie, Anas Mahmoud, Steven L. Waslander
* MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism, Zhixiong Nan, Xianghong Li, Jifeng Dai, Tao Xiang
* SET: Spectral Enhancement for Tiny Object Detection, Huixin Sun, Rungi Wang, Yanjing Li, Linlin Yang, Shaohui Lin, Xianbin Cao, Baochang Zhang
* Brain-Inspired Spiking Neural Networks for Energy-Efficient Object Detection, Ziqi Li, Tao Gao, Yisheng An, Ting Chen, Jing Zhang, Yuanbo Wen, Mengkun Liu, Qianxi Zhang
* GauCho: Gaussian Distributions with Cholesky Decomposition for Oriented Object Detection, José Henrique Lima Marques, Jeffri Murrugarra-Llerena, Claudio R. Jung
* AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation, Datao Tang, Xiangyong Cao, Xuan Wu, Jialin Li, Jing Yao, Xueru Bai, Dongsheng Jiang, Yin Li, Deyu Meng
* Solving Instance Detection from an Open-World Perspective, Qianqian Shen, Yunhan Zhao, Nahyun Kwon, Jeeeun Kim, Yanan Li, Shu Kong
* Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection, Yun Zhu, Le Hui, Hang Yang, Jianjun Qian, Jin Xie, Jian Yang
* Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection, Boyong He, Yuxiang Ji, Qianwen Ye, Zhuoyue Tan, Liaoni Wu
* Mr. DETR: Instructive Multi-Route Training for Detection Transformers, Chang-Bin Zhang, Yujie Zhong, Kai Han
* MonoDGP: Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors, Fangi Pu, Yifan Wang, Jiru Deng, Wenming Yang
* MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection, Rishubh Parihar, Srinjay Sarkar, Sarthak Vora, Jogendra Nath Kundu, R. Venkatesh Babu
* ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction for Detecting Aerial Image Objects, Woojin Lee, Hyugjae Chang, Jaeho Moon, Jaehyup Lee, Munchurl Kim
* FSHNet: Fully Sparse Hybrid Network for 3D Object Detection, Shuai Liu, Mingyue Cui, Boyang Li, Quanmin Liang, Tinghe Hong, Kai Huang, Yunxiao Shan, Kai Huang
* Dual-view X-ray Detection: Can Al Detect Prohibited Items from Dual-view X-ray Images like Humans?, Renshuai Tao, Haoyu Wang, Yuzhe Guo, Hairong Chen, Li Zhang, Xianglong Liu, Yunchao Wei, Yao Zhao
* MonoTAKD: Teaching Assistant Knowledge Distillation for Monocular 3D Object Detection, Hou-l Liu, Christine Wu, Jen-Hao Cheng, Wenhao Chai, Shian-Yun Wang, Gaowen Liu, Hugo Latapie, Jhih-Ciang Wu, Jenq-Neng Hwang, Hong-Han Shuai, Wen-Huang Cheng
* RICCARDO: Radar Hit Prediction and Convolution for Camera-Radar 3D Object Detection, Yunfei Long, Abhinav Kumar, Xiaoming Liu, Daniel Morris
* SparseAlign: a Fully Sparse Framework for Cooperative Object Detection, Yunshuang Yuan, Yan Xia, Daniel Cremers, Monika Sester
* GBlobs: Explicit Local Structure via Gaussian Blobs for Improved Cross-Domain LiDAR-based 3D Object Detection, Dušan Malić, Christian Fruhwirth-Reisinger, Samuel Schulter, Horst Possegger
* V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection, Xun Huang, Jinlong Wang, Qiming Xia, Siheng Chen, Bisheng Yang, Xin Li, Cheng Wang, Chenglu Wen
* Leveraging Temporal Cues for Semi-Supervised Multi-View 3D Object Detection, Jinhyung Park, Navyata Sanghvi, Hiroki Adachi, Yoshihisa Shibata, Shawn Hunt, Shinya Tanaka, Hironobu Fujiyoshi, Kris Kitani
* CorrBEV: Multi-View 3D Object Detection by Correlation Learning with Multi-modal Prototypes, Ziteng Xue, Mingzhe Guo, Heng Fan, Shihui Zhang, Zhipeng Zhang
* Test-Time Backdoor Detection for Object Detection Models, Hangtao Zhang, Yichen Wang, Shihui Yan, Chenyu Zhu, Ziqi Zhou, Linshan Hou, Shengshan Hu, Minghui Li, Yanjun Zhang, Leo Yu Zhang
* ReDiffDet: Rotation-equivariant Diffusion Model for Oriented Object Detection, Jiaqi Zhao, Zeyu Ding, Yong Zhou, Hancheng Zhu, Wen-Liang Du, Rui Yao
* Detect Any Mirrors: Boosting Learning Reliability on Large-Scale Unlabeled Data with an Iterative Data Engine, Zhaohu Xing, Lihao Liu, Yijun Yang, Hongqiu Wang, Tian Ye, Sixiang Chen, Wenxue Li, Guang Liu, Lei Zhu
* SEEN-DA: SEmantic ENtropy guided Domain-aware Attention for Domain Adaptive Object Detection, Haochen Li, Rui Zhang, Hantao Yao, Xin Zhang, Yifan Hao, Xinkai Song, Shaohui Peng, Yongwei Zhao, Chen Zhao, Yanjun Wu, Ling Li
* OW-OVD: Unified Open World and Open Vocabulary Object Detection, Xing Xi, Yangyang Huang, Ronghua Luo, Yu Qiu
* Incremental Object Keypoint Learning, Mingfu Liang, Jiahuan Zhou, Xu Zou, Ying Wu
* Open-World Objectness Modeling Unifies Novel Object Detection, Shan Zhang, Yao Ni, Jinhao Du, Yuan Xue, Philip Torr, Piotr Koniusz, Anton van den Hengel
* Feature Information Driven Position Gaussian Distribution Estimation for Tiny Object Detection, Jinghao Bian, Mingtao Feng, Weisheng Dong, Fangfang Wu, Jianqiao Luo, Yaonan Wang, Guangming Shi
* UCOD-DPL: Unsupervised Camouflaged Object Detection via Dynamic Pseudo-label Learning, Weiqi Yan, Lvhai Chen, Huaijia Kou, Shengchuan Zhang, Yan Zhang, Liujuan Cao
* Ev-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras, Hoonhee Cho, Jae-Young Kang, Youngho Kim, Kuk-Jin Yoon
* Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset, Xiao Wang, Yu Jin, Wentao Wu, Wei Zhang, Lin Zhu, Bo Jiang, Yonghong Tian
* SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts, Shijia Zhao, Qiming Xia, Xusheng Guo, Pufan Zou, Maoji Zheng, Hai Wu, Chenglu Wen, Cheng Wang
* PointSR: Self-Regularized Point Supervision for Drone-View Detection, Weizhuo Li, Yue Xi, Wenjing Jia, Zehao Zhang, Fei Li, Xiangzeng Liu, Qiguang Miao
* ReRAW: RGB-to-RAW Image Reconstruction via Stratified Sampling for Efficient Object Detection on the Edge, Radu Berdan, Beril Besbinar, Christoph Reinders, Junji Otsuka, Daisuke Iso
* VIKIENet: Towards Efficient 3D Object Detection with Virtual Key Instance Enhanced Network, Zhuochen Yu, Bijie Qiu, Andy W. H. Khong
* Detection-Friendly Nonuniformity Correction: A Union Framework for Infrared UAV Target Detection, Houzhang Fang, Xiaolin Wang, Zengyang Li, Lu Wang, Qingshan Li, Yi Chang, Luxin Yan
* Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection, Zihao Zhang, Aming Wu, Yahong Han
* Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval, Mankeerat Sidhu, Hetarth Chopra, Ansel Blume, Jeonghwan Kim, Revanth Gangi Reddy, Heng Ji
* Fractal Calibration for Long-tailed Object Detection, Konstantinos Panagiotis Alexandridis, Ismail Elezi, Jiankang Deng, Anh Nguyen, Shan Luo
* DEIM: DETR with Improved Matching for Fast Convergence, Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, Xi Shen
* FlexUOD: The Answer to Real-world Unsupervised Image Outlier Detection, Zhonghang Liu, Kun Zhou, Changshuo Wang, Wen-Yan Lin, Jiangbo Lu
* FASTer: Focal token Acquiring-and-Scaling Transformer for Long-term 3D Objection Detection, Chenxu Dang, ZaiPeng Duan, Pei An, Xinmin Zhang, Xuzhong Hu, Jie Ma
* RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion, Xiaomeng Chu, Jiajun Deng, Guoliang You, Yifan Duan, Houqiang Li, Yanyong Zhang
* Can't Slow Me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices, Tianyi Wang, Zichen Wang, Cong Wang, Yuanchao Shu, Ruilong Deng, Peng Cheng, Jiming Chen
* Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances, Yi Yu, Botao Ren, Peiyuan Zhang, Mingxin Liu, Junwei Luo, Shaofeng Zhang, Feipeng Da, Junchi Yan, Xue Yang
* BOOTPLACE: Bootstrapped Object Placement with Detection Transformers, Hang Zhou, Xinxin Zuo, Rui Ma, Li Cheng
* Uncertainty Meets Diversity: A Comprehensive Active Learning Framework for Indoor 3D Object Detection, Jiangyi Wang, Na Zhao
* Revisiting Generative Replay for Class Incremental Object Detection, Shizhou Zhang, Xueqiang Lv, Yinghui Xing, Qirui Wu, Di Xu, Yanning Zhang

## **目标跟踪 (Object Tracking)**


* **\[Oral\] Descriptor-In-Pixel: Point-Feature Tracking For Pixel Processor Arrays**, Laurie Bose, Jianing Chen, Piotr Dudek
* *\[Highlight\] EBS-EKF: Accurate and High Frequency Event-based Star Tracking*, Albert W. Reed, Connor Hashemi, Dennis Melamed, Nitesh Menon, Keigo Hirakawa, Scott McCloskey
* Prior-free 3D Object Tracking, Xiuqiang Song, Li Jin, Zhengxian Zhang, Jiachen Li, Fan Zhong, Guofeng Zhang, Xueying Qin
* Exploring Historical Information for RGBE Visual Tracking with Mamba, Chuanyu Sun, Jiqing Zhang, Yang Wang, Huilin Ge, Qianchen Xia, Baocai Yin, Xin Yang
* Autoregressive Sequential Pretraining for Visual Tracking, Shiyi Liang, Yifan Bai, Yihong Gong, Xing Wei
* DreamTrack: Dreaming the Future for Multimodal Visual Object Tracking, Mingzhe Guo, Weiping Tan, Wenyu Ran, Liping Jing, Zhipeng Zhang
* Omnidirectional Multi-Object Tracking, Kai Luo, Hao Shi, Sheng Wu, Fei Teng, Mengfei Duan, Chang Huang, Yuhang Wang, Kaiwei Wang, Kailun Yang
* PURA: Parameter Update-Recovery Test-Time Adaption for RGB-T Tracking, Zekai Shao, Yufan Hu, Bin Fan, Hongmin Liu
* ACAttack: Adaptive Cross Attacking RGB-T Tracker via Multi-Modal Response Decoupling, Xinyu Xiang, Qinglong Yan, Hao Zhang, Jiayi Ma
* Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better, Zihang Lai, Andrea Vedaldi
* A Distractor-Aware Memory for Visual Object Tracking with SAM2, Jovana Videnovic, Alan Lukezic, Matej Kristan
* Multiple Object Tracking as ID Prediction, Ruopeng Gao, Ji Qi, Limin Wang
* MITracker: Multi-View Integration for Visual Object Tracking, Mengjie Xu, Yitao Zhu, Haotian Jiang, Jiaming Li, Zhenrong Shen, Sheng Wang, Haolin Huang, Xinyu Wang, Han Zhang, Qing Yang, Qian Wang
* ETAP: Event-based Tracking of Any Point, Friedhelm Hamann, Daniel Gehrig, Filbert Febryanto, Kostas Daniilidis, Guillermo Gallego
* Focusing on Tracks for Online Multi-Object Tracking, Kyujin Shim, Kangwook Ko, Yujin Yang, Changick Kim
* GRAE-3DMOT: Geometry Relation-Aware Encoder for Online 3D Multi-Object Tracking, Hyunseop Kim, Hyo-Jun Lee, Yonguk Lee, Jinu Lee, Hanul Kim, Yeong Jun Koh
* MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking, Haolin Qin, Tingfa Xu, Tianhao Li, Zhenxiang Chen, Tao Feng, Jianan Li
* All-Day Multi-Camera Multi-Target Tracking, Huijie Fan, Yu Qiao, Yihao Zhen, Tinghui Zhao, Baojie Fan, Qiang Wang
* Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking, You Wu, Xucheng Wang, Xiangyang Yang, Mengyuan Liu, Dan Zeng, Hengzhou Ye, Shuiwang Li
* Mono3DVLT: Monocular-Video-Based 3D Visual Language Tracking, Hongkai Wei, Yang Yang, Shijie Sun, Mingtao Feng, Xiangyu Song, Qi Lei, Hongli Hu, Rong Wang, Huansheng Song, Naveed Akhtar, Ajmal Saeed Mian
* Dynamic Updates for Language Adaptation in Visual-Language Tracking, Xiaohai Li, Bineng Zhong, Qihua Liang, Zhiyi Mo, Jian Nong, Shuxiang Song

## **通用分割 (Segmentation)**


* **\[Oral\] Towards Explicit Geometry-Reflectance Collaboration for Generalized LiDAR Segmentation in Adverse Weather**, Longyu Yang, Ping Hu, Shangbo Yuan, Lu Zhang, Jun Liu, Hengtao Shen, Xiaofeng Zhu
* **\[Oral\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation**, Minhyeok Lee, Suhwan Cho, Jungho Lee, Sunghun Yang, Heeseung Choi, Ig-Jae Kim, Sangyoun Lee
* **\[Oral\] SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images**, Kaiyu Li, Ruixun Liu, Xiangyong Cao, Xueru Bai, Feng Zhou, Deyu Meng, Zhi Wang
* **\[Oral\] Keep the Balance: A Parameter-Efficient Symmetrical Framework for RGB+X Semantic Segmentation**, Jiaxin Cai, Jingze Su, Qi Li, Wenjie Yang, Shu Wang, Tiesong Zhao, Shengfeng He, Wenxi Liu
* *\[Highlight\] LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation*, Vladan Stojnić, Yannis Kalantidis, Jiří Matas, Giorgos Tolias
* Universal Domain Adaptation for Semantic Segmentation, Seun-An Choe, Keon-Hee Park, Jinwoo Choi, Gyeong-Moon Park
* Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation, Reza Qorbani, Gianluca Villani, Theodoros Panagiotakopoulos, Marc Botet Colomer, Linus Härenstam-Nielsen, Mattia Segu, Pier Luigi Dovesi, Jussi Karlgren, Daniel Cremers, Federico Tombari, Matteo Poggi
* Guidance for Semantic Segmentation, Hritam Basak, Zhaozheng Yin
* Weakly Supervised Semantic Segmentation via Progressive Confidence Region Expansion, Xiangfeng Xu, Pinyi Zhang, Wenxuan Huang, Yunhang Shen, Haosheng Chen, Jingzhong Lin, Wei Li, Gaoqi He, Jiao Xie, Shaohui Lin
* Beyond Background Shift: Rethinking Instance Replay in Continual Semantic Segmentation, Hongmei Yin, Tingliang Feng, Fan Lyu, Fanhua Shang, Hongying Liu, Wei Feng, Liang Wan
* 3D-AVS: LIDAR-based 3D Auto-Vocabulary Segmentation, Weijie Wei, Osman Ülger, Fatemeh Karimi Nejadasl, Theo Gevers, Martin R. Oswald
* Foveated Instance Segmentation, Hongyi Zeng, Wenxuan Liu, Tianhua Xia, Jinhui Chen, Ziyun Li, Sai Qian Zhang
* Zero-Shot 4D Lidar Panoptic Segmentation, Yushan Zhang, Aljoša Ošep, Laura Leal-Taixé, Tim Meinhardt
* Scene-Centric Unsupervised Panoptic Segmentation, Oliver Hahn, Christoph Reich, Nikita Araslanov, Daniel Cremers, Christian Rupprecht, Stefan Roth
* SUM Parts: Benchmarking Part-Level Semantic Segmentation of Urban Meshes, Weixiao Gao, Liangliang Nan, Hugo Ledoux
* Exploring Simple Open-Vocabulary Semantic Segmentation, Zihang Lai
* POPEN: Preference-Based Optimization and Ensemble for LVLM-Based Reasoning Segmentation, Lanyun Zhu, Tianrun Chen, Qianxiong Xu, Xuanyi Liu, Deyi Ji, Haiyang Wu, De Wen Soh, Jun Liu
* Towards Continual Universal Segmentation, Zihan Lin, Zilei Wang, Xu Wang
* Segment This Thing: Foveated Tokenization for Efficient Point-Prompted Segmentation, Tanner Schmidt, Richard Newcombe
* Segment Anything, Even Occluded, Wei-En Tai, Yu-Lin Shih, Cheng Sun, Yu-Chiang Frank Wang, Hwann-Tzong Chen
* D^3CTTA: Domain-Dependent Decorrelation for Continual Test-Time Adaption of 3D LiDAR Segmentation, Jichun Zhao, Haiyong Jiang, Haoxuan Song, Jun Xiao, Dong Gong
* Spotting the Unexpected (STU): A 3D LIDAR Dataset for Anomaly Segmentation in Autonomous Driving, Alexey Nekrasov, Malcolm Burdorf, Stewart Worrall, Bastian Leibe, Julie Stephany Berrio Perez
* Generative Map Priors for Collaborative BEV Semantic Segmentation, Jiahui Fu, Yue Gong, Luting Wang, Shifeng Zhang, Xu Zhou, Si Liu
* SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion, Xiyue Guo, Jiarui Hu, Junjie Hu, Hujun Bao, Guofeng Zhang
* Three Cars Approaching within 100m\! Enhancing Distant Geometry by Tri-Axis Voxel Scanning for Camera-based Semantic Scene Completion, Jongseong Bae, Junwoo Ha, Ha Young Kim
* Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation, Junha Lee, Chunghyun Park, Jaesung Choe, Yu-Chiang Frank Wang, Jan Kautz, Minsu Cho
* PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding, Hongjia Zhai, Hai Li, Zhenzhe Li, Xiaokun Pan, Yijia He, Guofeng Zhang
* Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation, Yongkang Li, Tianheng Cheng, Bin Feng, Wenyu Liu, Xinggang Wang
* Parameter-efficient Fine-tuning in Hyperspherical Space for Open-vocabulary Semantic Segmentation, Zelin Peng, Zhengqin Xu, Zhilin Zeng, Yu Huang, Yaoming Wang, Wei Shen
* Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation, Chanyoung Kim, Dayun Ju, Woojung Han, Ming-Hsuan Yang, Seong Jae Hwang
* FisherTune: Fisher-Guided Robust Tuning of Vision Foundation Models for Domain Generalized Segmentation, Dong Zhao, Jinlong Li, Shuang Wang, Mengyao Wu, Qi Zang, Nicu Sebe, Zhun Zhong
* FALCON: Fairness Learning via Contrastive Attention Approach to Continual Semantic Scene Understanding, Thanh-Dat Truong, Utsav Prabhu, Bhiksha Raj, Jackson Cothren, Khoa Luu
* Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Models, Zhaochong An, Guolei Sun, Yun Liu, Runjia Li, Junlin Han, Ender Konukoglu, Serge Belongie
* VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving, Haiming Zhang, Wending Zhou, Yiyao Zhu, Xu Yan, Jiantao Gao, Dongfeng Bai, Yingjie Cai, Bingbing Liu, Shuguang Cui, Zhen Li
* SOAP: Vision-Centric 3D Semantic Scene Completion with Scene-Adaptive Decoder and Occluded Region-Aware View Projection, Hyo-Jun Lee, Yeong Jun Koh, Hanul Kim, Hyunseop Kim, Yonguk Lee, Jinu Lee
* PolarNext: Rethink Instance Segmentation with Polar Representation, Jiacheng Sun, Xinghong Zhou, Yiqiang Wu, Bin Zhu, Jiaxuan Lu, Yu Qin, Xiaomao Li
* SAM2Object: Consolidating View Consistency via SAM2 for Zero-Shot 3D Instance Segmentation, Jihuai Zhao, Junbao Zhuo, Jiansheng Chen, Huimin Ma
* 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning, Yuncong Yang, Han Yang, Jiachen Zhou, Peihao Chen, Hongxin Zhang, Yilun Du, Chuang Gan
* Dual Semantic Guidance for Open Vocabulary Semantic Segmentation, Zhengyang Wang, Tingliang Feng, Fan Lyu, Fanhua Shang, Wei Feng, Liang Wan
* Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation, Zhiwei Yang, Yucong Meng, Kexue Fu, Feilong Tang, Shuo Wang, Zhijian Song
* Improving Semi-Supervised Semantic Segmentation with Sliced-Wasserstein Feature Alignment and Uniformity, Chen-Yi Lu, Kasra Derakhshandeh, Somali Chaterji
* Soft Self-labeling and Potts Relaxations for Weakly-supervised Segmentation, Zhongwen Zhang, Yuri Boykov
* Towards Efficient Foundation Model for Zero-shot Amodal Segmentation, Zhaochen Liu, Limeng Qiao, Xiangxiang Chu, Lin Ma, Tingting Jiang
* VP Lab: a PEFT-Enabled Visual Prompting Laboratory for Semantic Segmentation, Thomas Frick, Niccolo Avogaro, Yagmur G. Cinar, Daniel Caraballo, Cezary Skura, Filip M. Janicki, Piotr Kluska, Brown Ebouky, Nicola Farronato, Florian Scheidegger, Cristiano Malossi, Konrad Schindler, Andrea Bartezzaghi, Roy Assaf, Mattia Rigotti
* Efficient Segmentation for Edge Devices, Xin Li, Shuai Zhang

## **图像分割 (Image Segmentation)**


* **\[Oral\] VidSeg: Training-free Video Semantic Segmentation based on Diffusion Models**, Qian Wang, Abdelrahman Eldesokey, Mohit Mendiratta, Fangneng Zhan, Adam Kortylewski, Christian Theobalt, Peter Wonka
* *\[Highlight\] Advancing Manga Analysis: Comprehensive Segmentation Annotations for the Manga109 Dataset*, Minshan Xie, Jian Lin, Hanyuan Liu, Chengze Li, Tien-Tsin Wong
* A Dataset for Semantic Segmentation in the Presence of Unknowns, Zakaria Laskar, Tomas Vojir, Matej Grcic, laroslav Melekhov, Shankar Gangisetty, Juho Kannala, Jiri Matas, Giorgos Tolias, C.V. Jawahar
* Segment Any-Quality Images with Generative Latent Space Enhancement, Guangqian Guo, Yong Guo, Xuehui Yu, Wenbo Li, Yaoxing Wang, Shan Gao
* Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking, Phuc Nguyen, Minh Luu, Anh Tran, Cuong Pham, Khoi Nguyen
* Rethinking Query-based Transformer for Continual Image Segmentation, Yuchen Zhu, Cheng Shi, Dingyou Wang, Jiajin Tang, Zhengxuan Wei, Yu Wu, Guanbin Li, Sibei Yang
* The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation, Yuhan Liu, Yixiong Zou, Yuhua Li, Ruixuan Li
* Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation in Hyperbolic Space, Zelin Peng, Zhengqin Xu, Zhilin Zeng, Changsong Wen, Yu Huang, Menglin Yang, Feilong Tang, Wei Shen
* Scaling up Image Segmentation across Data and Tasks, Pei Wang, Zhaowei Cai, Hao Yang, Ashwin Swaminathan, R. Manmatha, Stefano Soatto
* DSV-LFS: Unifying LLM-Driven Semantic Cues with Visual Features for Robust Few-Shot Segmentation, Amin Karimi, Charalambos Poullis
* Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation, Jiho Choi, Seonho Lee, Minhyun Lee, Seungho Lee, Hyunjung Shim
* Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation, Jiahao Lu, Jiacheng Deng
* Sketchy Bounding-box Supervision for 3D Instance Segmentation, Qian Deng, Le Hui, Jin Xie, Jian Yang
* HyperSeg: Hybrid Segmentation Assistant with Fine-grained Visual Perceiver, Cong Wei, Yujie Zhong, Haoxian Tan, Yong Liu, Jie Hu, Dengjie Li, Zheng Zhao, Yujiu Yang
* An End-to-End Robust Point Cloud Semantic Segmentation Network with Single-Step Conditional Diffusion Models, Wentao Qu, Jing Wang, YongShun Gong, Xiaoshui Huang, Liang Xiao
* Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation, Chuandong Liu, Xingxing Weng, Shuguo Jiang, Pengcheng Li, Lei Yu, Gui-Song Xia
* NightAdapter: Learning a Frequency Adapter for Generalizable Night-time Scene Segmentation, Qi Bi, Jingjun Yi, Huimin Huang, Hao Zheng, Haolan Zhan, Yawen Huang, Yuexiang Li, Xian Wu, Yefeng Zheng
* Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection, Feng Yan, Xiaoheng Jiang, Yang Lu, Jiale Cao, Dong Chen, Mingliang Xu
* DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation, Ziyu Zhao, Xiaoguang Li, Lingjia Shi, Nasrin Imanpour, Song Wang
* Golden Cudgel Network for Real-Time Semantic Segmentation, Guoyu Yang, Yuan Wang, Daming Shi, Yanzhong Wang
* WISH: Weakly Supervised Instance Segmentation using Heterogeneous Labels, Hyeokjun Kweon, Kuk-Jin Yoon
* FFR: Frequency Feature Rectification for Weakly Supervised Semantic Segmentation, Ziqian Yang, Xinqiao Zhao, Xiaolei Wang, Quan Zhang, Jimin Xiao
* BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis, Weiguang Zhao, Rui Zhang, Qiufeng Wang, Guangliang Cheng, Kaizhu Huang
* SCSegamba: Lightweight Structure-Aware Vision Mamba for Crack Segmentation in Structures, Hui Liu, Chen Jia, Fan Shi, Xu Cheng, Shengyong Chen
* Hybrid Global-Local Representation with Augmented Spatial Guidance for Zero-Shot Referring Image Segmentation, Ting Liu, Siyuan Li
* CamPoint: Boosting Point Cloud Segmentation with Virtual Camera, Jianhui Zhang, Yizhi Luo, Zicheng Zhang, Xuecheng Nie, Bonan Li
* Hyperbolic Uncertainty-Aware Few-Shot Incremental Point Cloud Segmentation, Tanuj Sur, Samrat Mukherjee, Kaizer Rahaman, Subhasis Chaudhuri, Muhammad Haris Khan, Biplab Banerjee
* Exact: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation, Hao Zhu, Yan Zhu, Jiayu Xiao, Tianxiang Xiao, Yike Ma, Yucheng Zhang, Feng Dai
* Convex Combination Star Shape Prior for Data-driven Image Semantic Segmentation, Xinyu Zhao, Jun Xie, Shengzhe Chen, Jun Liu
* Insightful Instance Features for 3D Instance Segmentation, Wonseok Roh, Hwanhee Jung, Giljoo Nam, Dong In Lee, Hyeongcheol Park, Sang Ho Yoon, Jungseock Joo, Sangpil Kim
* DocSAM: Unified Document Image Segmentation via Query Decomposition and Heterogeneous Mixed Learning, Xiao-Hui Li, Fei Yin, Cheng-Lin Liu
* POT: Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation, Jian Wang, Tianhong Dai, Bingfeng Zhang, Siyue Yu, Eng Gee Lim, Jimin Xiao
* WISNet: Pseudo Label Generation on Unbalanced and Patch Annotated Waste Images, Shifan Zhang, Hongzi Zhu, Yinan He, Minyi Guo, Ziyang Lou, Shan Chang
* Minimizing Labeled, Maximizing Unlabeled: An Image-Driven Approach for Video Instance Segmentation, Fangyun Wei, Jinjing Zhao, Kun Yan, Chang Xu
* The Impact Label Noise and Choice of Threshold has on Cross-Entropy and Soft-Dice in Image Segmentation, Marcus Nordström, Atsuto Maki, Henrik Hult

## **视频目标分割 (Video Object Segmentation)**


* *\[Highlight\] SAMWISE: Infusing Wisdom in SAM2 for Text-Driven Video Segmentation*, Claudia Cuttano, Gabriele Trivigno, Gabriele Rosi, Carlo Masone, Giuseppe Averta
* High Temporal Consistency through Semantic Similarity Propagation in Semi-Supervised Video Semantic Segmentation for Autonomous Flight, Cédric Vincent, Taehyoung Kim, Henri Meeß
* Segment Any Motion in Videos, Nan Huang, Wenzhao Zheng, Chenfeng Xu, Kurt Keutzer, Shanghang Zhang, Angjoo Kanazawa, Qianqian Wang
* SAM-12V: Upgrading SAM to Support Promptable Video Segmentation with Less than 0.2% Training Cost, Haiyang Mei, Pengyu Zhang, Mike Zheng Shou
* RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety, Andrei Dumitriu, Florin Tatui, Florin Miron, Aakash Ralhan, Radu Tudor lonescu, Radu Timofte
* GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation, Lang Lin, Xueyang Yu, Ziqi Pang, Yu-Xiong Wang
* LIVOS: Light Video Object Segmentation with Gated Linear Matching, Qin Liu, Jianfeng Wang, Zhengyuan Yang, Linjie Li, Kevin Lin, Marc Niethammer, Lijuan Wang
* Using Diffusion Priors for Video Amodal Segmentation, Kaihua Chen, Deva Ramanan, Tarasha Khurana
* EntitySAM: Segment Everything in Video, Mingqiao Ye, Seoung Wug Oh, Lei Ke, Joon-Young Lee
* The Devil is in Temporal Token: High Quality Video Reasoning Segmentation, Sitong Gong, Yunzhi Zhuge, Lu Zhang, Zongxin Yang, Pingping Zhang, Huchuan Lu
* M^3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation, Zixuan Chen, Jiaxin Li, Junxuan Liang, Liming Tan, Yejie Guo, Cewu Lu, Yong-Lu Li
* Decoupled Motion Expression Video Segmentation, Hao Fang, Runmin Cong, Xiankai Lu, Xiaofei Zhou, Sam Kwong, Wei Zhang
* Semantic and Sequential Alignment for Referring Video Object Segmentation, Feiyu Pan, Hao Fang, Fangkai Li, Yanyu Xu, Yawei Li, Luca Benini, Xiankai Lu
* Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos, Vadim Tschernezki, Diane Larlus, Iro Laina, Andrea Vedaldi

## **医学图像分割 (Medical Image Segmentation)**


* **\[Oral\] ✩ Segmenting Maxillofacial Structures in CBCT Volumes**, Federico Bolelli, Kevin Marchesini, Niels van Nistelrooij, Luca Lumetti, Vittorio Pipoli, Elisa Ficarra, Shankeeth Vinayahalingam, Costantino Grana
* **\[Oral\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation**, Aishik Konwer, Zhijian Yang, Erhan Bas, Cao Xiao, Prateek Prasanna, Parminder Bhatia, Taha Kass-Hout
* *\[Highlight\] EffiDec3D: An Optimized Decoder for High-Performance and Efficient 3D Medical Image Segmentation*, Md Mostafijur Rahman, Radu Marculescu
* *\[Highlight\] Annotation Ambiguity Aware Semi-Supervised Medical Image Segmentation*, Suruchi Kumari, Pravendra Singh
* Steady Progress Beats Stagnation: Mutual Aid of Foundation and Conventional Models in Mixed Domain Semi-Supervised Medical Image Segmentation, Qinghe Ma, Jian Zhang, Zekun Li, Lei Qi, Qian Yu, Yinghuan Shi
* Revisiting MAE Pre-training for 3D Medical Image Segmentation, Tassilo Wald, Constantin Ulrich, Stanislav Lukyanenko, Andrei Goncharov, Alberto Paderno, Maximilian Miller, Leander Maerkisch, Paul Jaeger, Klaus Maier-Hein
* SuperLightNet: Lightweight Parameter Aggregation Network for Multimodal Brain Tumor Segmentation, Feng Yu, Jiacheng Cao, Li Liu, Minghua Jiang
* EchoONE: Segmenting Multiple Echocardiography Planes in One Model, Jiongtong Hu, Wufeng Xue, Jun Cheng, Yingying Liu, Wei Zhuo, Dong Ni
* Unified Medical Lesion Segmentation via Self-referring Indicator, Shijie Chang, Xiaoqi Zhao, Lihe Zhang, Tiancheng Wang
* Minding Fuzzy Regions: A Data-driven Alternating Learning Paradigm for Stable Lesion Segmentation, Lexin Fang, Yunyang Xu, Xiang Ma, Xuemei Li, Caiming Zhang
* Learning Dynamic Collaborative Network for Semi-supervised 3D Vessel Segmentation, Jiao Xu, Xin Chen, Lihe Zhang
* Boost the Inference with Co-training: A Depth-guided Mutual Learning Framework for Semi-supervised Medical Polyp Segmentation, Yuxin Li, Zihao Zhu, Yuxiang Zhang, Yifan Chen, Zhibin Yu
* 3D Dental Model Segmentation with Geometrical Boundary Preserving, Shufan Xi, Zexian Liu, Junlin Chang, Hongyu Wu, Xiaogang Wang, Aimin Hao
* A Semantic Knowledge Complementarity based Decoupling Framework for Semi-supervised Class-imbalanced Medical Image Segmentation, Zheng Zhang, Guanchun Yin, Bo Zhang, Wu Liu, Xiuzhuang Zhou, Wendong Wang
* Advancing Generalizable Tumor Segmentation with Anomaly-Aware Open-Vocabulary Attention Maps and Frozen Foundation Diffusion Models, Yankai Jiang, Peng Zhang, Donglin Yang, Yuan Tian, Hai Lin, Xiaosong Wang
* beta-FFT: Nonlinear Interpolation and Differentiated Training Strategies for Semi-Supervised Medical Image Segmentation, Ming Hu, Jianfu Yin, Zhuangzhuang Ma, Jianheng Ma, Feiyu Zhu, Bingbing Wu, Ya Wen, Meng Wu, Cong Hu, Bingliang Hu, Quan Wang
* DyCON: Dynamic Uncertainty-aware Consistency and Contrastive Learning for Semi-supervised Medical Image Segmentation, Maregu Assefa, Muzammal Naseer, lyyakutti lyappan Ganapathi, Syed Sadaf Ali, Mohamed L Seghier, Naoufel Werghi
* Rethinking Decoder Design: Improving Biomarker Segmentation Using Depth-to-Space Restoration and Residual Linear Attention, Saad Wazir, Daeyoung Kim
* LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging, Maximilian Rokuss, Yannick Kirchhoff, Seval Akbal, Balint Kovacs, Saikat Roy, Constantin Ulrich, Tassilo Wald, Lukas T. Rotkopf, Heinz-Peter Schlemmer, Klaus Maier-Hein
* Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation, Qingchen Tang, Lei Fan, Maurice Pagnucco, Yang Song
* CSC-PA: Cross-image Semantic Correlation via Prototype Attentions for Single-network Semi-supervised Breast Tumor Segmentation, Zhenhui Ding, Guilian Chen, Qin Zhang, Huisi Wu, Jing Qin
* Take the Bull by the Horns: Learning to Segment Hard Samples, Yuan Guo, Jingyu Kong, Yu Wang, Yuping Duan
* KMD: Koopman Multi-modality Decomposition for Generalized Brain Tumor Segmentation under Incomplete Modalities, Tianyi Liu, Haochuan Jiang, Kaizhu Huang
* DeNVER: Deformable Neural Vessel Representations for Unsupervised Video Vessel Segmentation, Chun-Hung Wu, Shih-Hong Chen, Chih-Yao Hu, Hsin-Yu Wu, Kai-Hsin Chen, Yu-You Chen, Chih-Hai Su, Chih-Kuo Lee, Yu-Lun Liu
* Test-Time Domain Generalization via Universe Learning: A Multi-Graph Matching Approach for Medical Image Segmentation, Xingguo Lv, Xingbo Dong, Liwen Wang, Jiewen Yang, Lei Zhao, Bin Pu, Zhe Jin, Xuejun Li
* Show and Segment: Universal Medical Image Segmentation via In-Context Learning, Yunhe Gao, Di Liu, Zhuowei Li, Yunsheng Li, Dongdong Chen, Mu Zhou, Dimitris N. Metaxas
* nnWNet: Rethinking the Use of Transformers in Biomedical Image Segmentation and Calling for a Unified Evaluation Benchmark, Yanfeng Zhou, Lingrui Li, Le Lu, Minfeng Xu
* VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging, Yufan He, Pengfei Guo, Yucheng Tang, Andriy Myronenko, Vishwesh Nath, Ziyue Xu, Dong Yang, Can Zhao, Benjamin Simon, Mason Belue, Stephanie Harmon, Baris Turkbey, Daguang Xu, Wenqi Li
* vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation, Bastian Wittmann, Yannick Wattenberg, Tamaz Amiranashvili, Suprosanna Shit, Bjoern Menze

## **交互式分割 (Interactive Segmentation)**


* NTClick: Achieving Precise Interactive Segmentation With Noise-tolerant Clicks, Chenyi Zhang, Ting Liu, Xiaochao Qu, Luoqi Liu, Yao Zhao, Yunchao Wei
* OnlineAnySeg: Online Zero-Shot 3D Segmentation by Visual Foundation Model Guided 2D Mask Merging, Yijie Tang, Jiazhao Zhang, Yuqing Lan, Yulan Guo, Dezun Dong, Chenyang Zhu, Kai Xu
* Repurposing Stable Diffusion Attention for Training-Free Unsupervised Interactive Segmentation, Markus Karmann, Onay Urfalioglu
* SAM-REF: Introducing Image-Prompt Synergy during Interaction for Detail Enhancement in the Segment Anything Model, Chongkai Yu, Ting Liu, Anqi Li, Xiaochao Qu, Chengjing Wu, Luoqi Liu, Xiaolin Hu
* UNICL-SAM: Uncertainty-Driven In-Context Segmentation with Part Prototype Discovery, Dianmo Sheng, Dongdong Chen, Zhentao Tan, Qiankun Liu, Qi Chu, Tao Gong, Bin Liu, Jing Han, Wenbin Tu, Shengwei Xu, Nenghai Yu
* Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline, Junlong Cheng, Bin Fu, Jin Ye, Guoan Wang, Tianbin Li, Haoyu Wang, Ruoyu Li, He Yao, Junren Cheng, Jingwen Li, Yanzhou Su, Min Zhu, Junjun He

## **视频理解 (Video Understanding)**


* **\[Oral\] VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection**, Songhao Han, Wei Huang, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, Si Liu
* **\[Oral\] SEAL: Semantic Attention Learning for Long Video Representation**, Lan Wang, Yujia Chen, Du Tran, Vishnu Naresh Boddeti, Wen-Sheng Chu
* *\[Highlight\] ExpertAF: Expert Actionable Feedback from Video*, Kumar Ashutosh, Tushar Nagarajan, Georgios Pavlakos, Kris Kitani, Kristen Grauman
* *\[Highlight\] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation*, Yuying Ge, Yizhuo Li, Yixiao Ge, Ying Shan
* *\[Highlight\] MLVU: Benchmarking Multi-task Long Video Understanding*, Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, Zheng Liu
* VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video Spatiotemporal Augmentation, Weiming Ren, Huan Yang, Jie Min, Cong Wei, Wenhu Chen
* VideoICL: Confidence-based Iterative In-context Learning for Out-of-Distribution Video Understanding, Kangsan Kim, Geon Park, Youngwan Lee, Woongyeong Yeo, Sung Ju Hwang
* Online Video Understanding: OVBench and VideoChat-Online, Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, Limin Wang
* DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding, Yudong Han, Qingpei Guo, Liyuan Pan, Liu Liu, Yu Guan, Ming Yang
* HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding, Shehreen Azad, Vibhav Vineet, Yogesh S Rawat
* Re-thinking Temporal Search for Long-Form Video Understanding, Jinhui Ye, Zihan Wang, Haosen Sun, Keshigeyan Chandrasegaran, Zane Durante, Cristobal Eyzaguirre, Yonatan Bisk, Juan Carlos Niebles, Ehsan Adeli, Li Fei-Fei, Jiajun Wu, Manling Li
* Towards Universal Soccer Video Understanding, Jiayuan Rao, Haoning Wu, Hao Jiang, Ya Zhang, Yanfeng Wang, Weidi Xie
* MotionBench: Benchmarking and Improving Fine-grained Video Motion Understanding for Vision Language Models, Wenyi Hong, Yean Cheng, Zhuoyi Yang, Weihan Wang, Lefan Wang, Xiaotao Gu, Shiyu Huang, Yuxiao Dong, Jie Tang
* VideoAutoArena: An Automated Arena for Evaluating Large Multimodal Models in Video Analysis through User Simulation, Ziyang Luo, Haoning Wu, Dongxu Li, Jing Ma, Mohan Kankanhalli, Junnan Li
* MMVU: Measuring Expert-Level Multi-Discipline Video Understanding, Yilun Zhao, Haowei Zhang, Lujing Xie, Tongyan Hu, Guo Gan, Yitao Long, Zhiyuan Hu, Weiyuan Chen, Chuhan Li, Zhijian Xu, Chengye Wang, Ziyao Shangguan, Zhenwen Liang, Yixin Liu, Chen Zhao, Arman Cohan
* Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding, Duo Zheng, Shijia Huang, Liwei Wang
* Change3D: Revisiting Change Detection and Captioning from A Video Modeling Perspective, Duowang Zhu, Xiaohu Huang, Haiyan Huang, Hao Zhou, Zhenfeng Shao
* DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification, Darryl Ho, Samuel Madden
* SF2T: Self-supervised Fragment Finetuning of Video-LLMs for Fine-Grained Understanding, Yangliu Hu, Zikai Song, Na Feng, Yawei Luo, Junqing Yu, Yi-Ping Phoebe Chen, Wei Yang
* Adaptive Keyframe Sampling for Long Video Understanding, Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, Qixiang Ye
* SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding, Chenkai Zhang, Yiming Lei, Zeming Liu, Haitao Leng, ShaoGuo Liu, Tingting Gao, Qingjie Liu, Yunhong Wang
* VideoWorld: Exploring Knowledge Learning from Unlabeled Videos, Zhongwei Ren, Yunchao Wei, Xun Guo, Yao Zhao, Bingyi Kang, Jiashi Feng, Xiaojie Jin
* Unified Dense Prediction of Video Diffusion, Lehan Yang, Lu Qi, Xiangtai Li, Sheng Li, Varun Jampani, Ming-Hsuan Yang
* Adapting Pre-trained 3D Models for Point Cloud Video Understanding via Cross-frame Spatio-temporal Perception, Baixuan Lv, Yaohua Zha, Tao Dai, Xue Yuerong, Ke Chen, Shu-Tao Xia
* Learning from Streaming Video with Orthogonal Gradients, Tengda Han, Dilara Gokay, Joseph Heyward, Chuhan Zhang, Daniel Zoran, Viorica Patraucean, Joao Carreira, Dima Damen, Andrew Zisserman
* Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-grained View-invariant Video Representations, Jungin Park, Jiyoung Lee, Kwanghoon Sohn
* On the Consistency of Video Large Language Models in Temporal Comprehension, Minjoon Jung, Junbin Xiao, Byoung-Tak Zhang, Angela Yao
* ReWind: Understanding Long Videos with Instructed Learnable Memory, Anxhelo Diko, Tinghuai Wang, Wassim Swaileh, Shiyan Sun, loannis Patras
* STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding, Zichen Liu, Kunlun Xu, Bing Su, Xu Zou, Yuxin Peng, Jiahuan Zhou
* Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity, Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Xiaonan Huang, Changxin Gao, Shanjun Zhang, Li Yu, Nong Sang
* OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?, Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang
* DrVideo: Document Retrieval Based Long Video Understanding, Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, Jianfei Cai
* Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMS, Lucas Ventura, Antoine Yang, Cordelia Schmid, Gül Varol
* DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models, Keda Tao, Can Qin, Haoxuan You, Yang Sui, Huan Wang
* Understanding Multi-Task Activities from Single-Task Videos, Yuhan Shen, Ehsan Elhamifar

## **动作识别 (Action Recognition)**


* **\[Oral\] Temporal Alignment-Free Video Matching for Few-shot Action Recognition**, SuBeen Lee, WonJun Moon, Hyun Seok Seong, Jae-Pil Heo
* TAMT: Temporal-Aware Model Tuning for Cross-Domain Few Shot Action Recognition, Yilong Wang, Zilin Gao, Qilong Wang, Zhaofeng Chen, Peihua Li, Qinghua Hu
* Neuron: Learning Context-Aware Evolving Representations for Zero-Shot Skeleton Action Recognition, Yang Chen, Jingcai Guo, Song Guo, Dacheng Tao
* H-MoRe: Learning Human-centric Motion Representation for Action Analysis, Zhanbo Huang, Xiaoming Liu, Yu Kong
* Are Spatial-Temporal Graph Convolution Networks for Human Action Recognition Over-Parameterized?, Jianyang Xie, Yitian Zhao, Yanda Meng, He Zhao, Anh Nguyen, Yalin Zheng
* DIGIT: Multi-Dilated Gated Encoder and Central-Adjacent Region Integrated Decoder for Temporal Action Detection Transformer, Ho-Joong Kim, Yearang Lee, Jung-Ho Hong, Seong-Whan Lee
* Revealing Key Details to See Differences: A Novel Prototypical Perspective for Skeleton-based Action Recognition, Hongda Liu, Yunfan Liu, Min Ren, Hao Wang, Yunlong Wang, Zhenan Sun
* Semantic-guided Cross-Modal Prompt Learning for Skeleton-based Zero-shot Action Recognition, Angi Zhu, Jingmin Zhu, James Bailey, Mingming Gong, Qiuhong Ke
* FSboard: Over 3 Million Characters of ASL Fingerspelling Collected via Smartphones, Manfred Georg, Garrett Tanzer, Esha Uboweja, Saad Hassan, Maximus Shengelia, Sam Sepah, Sean Forbes, Thad Starner
* Boosting Point-Supervised Temporal Action Localization through Integrating Query Reformation and Optimal Transport, Mengnan Liu, Le Wang, Sanping Zhou, Kun Xia, Xiaolong Sun, Gang Hua
* Action Detail Matters: Refining Video Recognition with Local Action Queries, Mengmeng Wang, Zeyi Huang, Xiangjie Kong, Guojiang Shen, Guang Dai, Jingdong Wang, Yong Liu
* Heterogeneous Skeleton-Based Action Representation Learning, Hongsong Wang, Xiaoyan Ma, Jidong Kuang, Jie Gui
* Event-Driven ASL Recognition: Building a DVS Dataset for Neuromorphic Systems, Arshia Eslami, James (Blake) Seekings, Peyton Chandarana, Ramtin Zand

## **视频-语言 (Video-Language)**


* **\[Oral\] Learning Audio-guided Video Representation with Gated Attention for Video-Text Retrieval**, Boseung Jeong, Jicheol Park, Sungyeon Kim, Suha Kwak
* *\[Highlight\] Question-Aware Gaussian Experts for Audio-Visual Question Answering*, Hongyeob Kim, Inyoung Jung, Dayoon Suh, Youjia Zhang, Sangmin Lee, Sungeun Hong
* VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos, Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, Mohit Bansal
* STEP: Enhancing Video-LLMs' Compositional Reasoning by Spatio-Temporal Graph-guided Self-Training, Haiyi Qiu, Minghe Gao, Long Qian, Kaihang Pan, Qifan Yu, Juncheng Li, Wenjie Wang, Siliang Tang, Yueting Zhuang, Tat-Seng Chua
* Localizing Events in Videos with Multimodal Queries, Gengyuan Zhang, Mang Ling Ada Fok, Jialu Ma, Yan Xia, Daniel Cremers, Philip Torr, Volker Tresp, Jindong Gu
* SALOVA: Segment-Augmented Long Video Assistant for Targeted Retrieval and Routing in Long-Form Video Analysis, Junho Kim, Hyunjun Kim, Hosu Lee, Yong Man Ro
* EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering, Sheng Zhou, Junbin Xiao, Qingyun Li, Yicong Li, Xun Yang, Dan Guo, Meng Wang, Tat-Seng Chua, Angela Yao
* VideoGEM: Training-free Action Grounding in Videos, Felix Vogel, Walid Bousselham, Anna Kukleva, Nina Shvetsova, Hilde Kuehne
* STPro: Spatial and Temporal Progressive Learning for Weakly Supervised Spatio-Temporal Grounding, Aaryan Garg, Akash Kumar, Yogesh S Rawat
* LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant, Wei Li, Bing Hu, Rui Shao, Leyang Shen, Liqiang Nie
* AVQACL: A Novel Benchmark for Audio-Visual Question Answering Continual Learning, Kaixuan Wu, Xinde Li, Xinling Li, Chuanfei Hu, Guoliang Wu
* Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning, Huabin Liu, Filip Ilievski, Cees G. M. Snoek
* Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation, Yudi Shi, Shangzhe Di, Qirui Chen, Weidi Xie
* AdaCM^2: On Understanding Extremely Long-Term Video with Adaptive Cross-Modality Memory Reduction, Yuanbin Man, Ying Huang, Chengming Zhang, Bingzhe Li, Wei Niu, Miao Yin
* Video Language Model Pretraining with Spatio-temporal Masking, Yue Wu, Zhaobo Qi, Junshu Sun, Yaowei Wang, Qingming Huang, Shuhui Wang
* Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models, Zhihang Liu, Chen-Wei Xie, Pandeng Li, Liming Zhao, Longxiang Tang, Yun Zheng, Chuanbin Liu, Hongtao Xie
* LLAVA-ST: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding, Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xiaoming Wei, Si Liu
* V^2Dial: Unification of Video and Visual Dialog via Multimodal Experts, Adnen Abdessaied, Anna Rohrbach, Marcus Rohrbach, Andreas Bulling
* Narrating the Video: Boosting Text-Video Retrieval via Comprehensive Utilization of Frame-Level Captions, Chan Hur, Jeong-hun Hong, Dong-hun Lee, Dabin Kang, Semin Myeong, Sang-hyo Park, Hyeyoung Park
* Cross-modal Causal Relation Alignment for Video Question Grounding, Weixing Chen, Yang Liu, Binglin Chen, Jiandong Su, Yongsen Zheng, Liang Lin
* Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos, Sagnik Majumder, Tushar Nagarajan, Ziad Al-Halah, Reina Pradhan, Kristen Grauman
* ReSpec: Relevance and Specificity Grounded Online Filtering for Learning on Video-Text Data Streams, Chris Dongjoo Kim, Jihwan Moon, Sangwoo Moon, Heeseung Yun, Sihaeng Lee, Aniruddha Kembhavi, Soonyoung Lee, Gunhee Kim, Sangho Lee, Christopher Clark
* Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks, Nina Shvetsova, Arsha Nagrani, Bernt Schiele, Hilde Kuehne, Christian Rupprecht
* VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models, Dahun Kim, AJ Piergiovanni, Ganesh Mallya, Anelia Angelova
* Flexible Frame Selection for Efficient Video Reasoning, Shyamal Buch, Arsha Nagrani, Anurag Arnab, Cordelia Schmid
* LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale, Joya Chen, Ziyun Zeng, Yiqi Lin, Wei Li, Zejun Ma, Mike Zheng Shou
* BIMBA: Selective-Scan Compression for Long-Range Video Question Answering, Md Mohaiminul Islam, Tushar Nagarajan, Huiyu Wang, Gedas Bertasius, Lorenzo Torresani
* Efficient Transfer Learning for Video-language Foundation Models, Haoxing Chen, Zizheng Huang, Yan Hong, Yanshuo Wang, Zhongcai Lyu, Zhuoer Xu, Jun Lan, Zhangxuan Gu
* EventGPT: Event Stream Understanding with Multimodal Large Language Models, Shaoyu Liu, Jianing Li, Guanghui Zhao, Yunjian Zhang, Xin Meng, Fei Richard Yu, Xiangyang Ji, Ming Li
* HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation, Trong-Thuan Nguyen, Pha Nguyen, Jackson Cothren, Alper Yilmaz, Khoa Luu
* DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation, Mu Chen, Liulei Li, Wenguan Wang, Yi Yang
* Progress-Aware Video Frame Captioning, Zihui Xue, Joungbin An, Xitong Yang, Kristen Grauman
* SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction, Enrico Pallotta, Sina Mokhtarzadeh Azar, Shuai Li, Olga Zatsarynna, Juergen Gall
* SVLTA: Benchmarking Vision-Language Temporal Alignment via Synthetic Video Situation, Hao Du, Bo Wu, Yan Lu, Zhendong Mao
* VELOCITI: Benchmarking Video-Language Compositional Reasoning with Strict Entailment, Darshana Saravanan, Varun Gupta, Darshan Singh, Zeeshan Khan, Vineet Gandhi, Makarand Tapaswi
* VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM, Yuqian Yuan, Hang Zhang, Wentong Li, Zesen Cheng, Boqiang Zhang, Long Li, Xin Li, Deli Zhao, Wenqiao Zhang, Yueting Zhuang, Jianke Zhu, Lidong Bing
* ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos, Tanveer Hannan, Md Mohaiminul Islam, Jindong Gu, Thomas Seidl, Gedas Bertasius
* ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation, Ali Athar, Xueqing Deng, Liang-Chieh Chen
* VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos, Shehan Munasinghe, Hanan Gani, Wengi Zhu, Jiale Cao, Eric Xing, Fahad Shahbaz Khan, Salman Khan
* Video-CoIBERT: Contextualized Late Interaction for Text-to-Video Retrieval, Arun Reddy, Alexander Martin, Eugene Yang, Andrew Yates, Kate Sanders, Kenton Murray, Reno Kriz, Celso M. de Melo, Benjamin Van Durme, Rama Chellappa
* DiscoVLA: Discrepancy Reduction in Vision, Language, and Alignment for Parameter-Efficient Video-Text Retrieval, Leqi Shen, Guoqiang Gong, Tianxiang Hao, Tao He, Yifeng Zhang, Pengzhang Liu, Sicheng Zhao, Jungong Han, Guiguang Ding

# **五、人体与人脸技术 (Human & Face Technology)**
## **人体分析 (Human-related)**


* *\[Highlight\] Pippo: High-Resolution Multi-View Humans from a Single Image*, Yash Kant, Ethan Weber, Jin Kyu Kim, Rawal Khirodkar, Su Zhaoen, Julieta Martinez, Igor Gilitschenski, Shunsuke Saito, Timur Bagautdinov
* Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization, Junying Wang, Jingyuan Liu, Xin Sun, Krishna Kumar Singh, Zhixin Shu, He Zhang, Jimei Yang, Nanxuan Zhao, Tuanfeng Y. Wang, Simon S. Chen, Ulrich Neumann, Jae Shin Yoon
* The Language of Motion: Unifying Verbal and Non-verbal Language of 3D Human Motion, Changan Chen, Juze Zhang, Shrinidhi K. Lakshmikanth, Yusu Fang, Ruizhi Shao, Gordon Wetzstein, Li Fei-Fei, Ehsan Adeli
* Is this Generated Person Existed in Real-world? Fine-grained Detecting and Calibrating Abnormal Human-body, Zeqing Wang, Qingyang Ma, Wentao Wan, Haojie Li, Keze Wang, Yonghong Tian
* Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling, Nannan Li, Kevin J. Shih, Bryan A. Plummer
* A Focused Human Body Model for Accurate Anthropometric Measurements Extraction, Shuhang Chen, Xianliang Huang, Zhizhou Zhong, Juhong Guan, Shuigeng Zhou
* Ego40: Egocentric Human Motion Capture and Understanding from Multi-Modal Input, Jian Wang, Rishabh Dabral, Diogo Luvizon, Zhe Cao, Lingjie Liu, Thabo Beeler, Christian Theobalt
* Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction, Dong Li, Wenqi Zhong, Wei Yu, Yingwei Pan, Dingwen Zhang, Ting Yao, Junwei Han, Tao Mei
* VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction, Zijian He, Yuwei Ning, Yipeng Qin, Guangrun Wang, Sibei Yang, Liang Lin, Guanbin Li
* BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training, Xuanpu Zhang, Dan Song, Pengxin Zhan, Tianyu Chang, Jianhao Zeng, Qingguo Chen, Weihua Luo, An-An Liu
* SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing, Xueting Li, Ye Yuan, Shalini De Mello, Gilles Daviet, Jonathan Leaf, Miles Macklin, Jan Kautz, Umar Iqbal
* Disco4D: Disentangled 4D Human Generation and Animation from a Single Image, Hui En Pang, Shuai Liu, Zhongang Cai, Lei Yang, Tianwei Zhang, Ziwei Liu
* AnyDressing: Customizable Multi-Garment Virtual Dressing via Latent Diffusion Models, Xinghui Li, Qichao Sun, Pengze Zhang, Fulong Ye, Zhichao Liao, Wanquan Feng, Songtao Zhao, Qian He
* Controllable Human Image Generation with Personalized Multi-Garments, Yisol Choi, Sangkyung Kwak, Sihyun Yu, Hyungwon Choi, Jinwoo Shin
* ITA-MDT: Image-Timestep-Adaptive Masked Diffusion Transformer Framework for Image-Based Virtual Try-On, Ji Woo Hong, Tri Ton, Trung X. Pham, Gwanhyeong Koo, Sunjae Yoon, Chang D. Yoo
* MEAT: Multiview Diffusion Model for Human Generation on Megapixels with Mesh Attention, Yuhan Wang, Fangzhou Hong, Shuai Yang, Liming Jiang, Wayne Wu, Chen Change Loy
* RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance, Yuheng Jiang, Zhehao Shen, Chengcheng Guo, Yu Hong, Zhuo Su, Yingliang Zhang, Marc Habermann, Lan Xu
* Certified Human Trajectory Prediction, Mohammadhossein Bahari, Saeed Saadatnejad, Amirhossein Askari Farsangi, Seyed-Mohsen Moosavi-Dezfooli, Alexandre Alahi
* On Denoising Walking Videos for Gait Recognition, Dongyang Jin, Chao Fan, Jingzhe Ma, Jingkai Zhou, Weihua Chen, Shiqi Yu
* SapiensID: Foundation for Human Recognition, Minchul Kim, Dingqiang Ye, Yiyang Su, Feng Liu, Xiaoming Liu
* FreeLunch Enhancements for Multi-modal Crowd Counting, Haoliang Meng, Xiaopeng Hong, Zhengqin Lai, Miao Shang
* Reconstructing Animals and the Wild, Peter Kulits, Michael J. Black, Silvia Zuffi
* Re-HOLD: Video Hand Object Interaction Reenactment via adaptive Layout-instructed Diffusion Model, Yingying Fan, Quanwei Yang, Kaisiyuan Wang, Hang Zhou, Yingying Li, Haocheng Feng, Errui Ding, Yu Wu, Jingdong Wang
* Visual Persona: Foundation Model for Full-Body Human Customization, Jisu Nam, Soowon Son, Zhan Xu, Jing Shi, Difan Liu, Feng Liu, Seungryong Kim, Yang Zhou

## **姿态估计 (Pose Estimation)**


* *\[Highlight\] Co-op: Correspondence-based Novel Object Pose Estimation*, Sungphill Moon, Hyeontae Son, Dongcheol Hur, Sangwook Kim
* DynPose: Largely Improving the Efficiency of Human Pose Estimation by a Simple Dynamic Framework, Yalong Xu, Lin Zhao, Chen Gong, Guangyu Li, Di Wang, Nannan Wang
* Rethinking Correspondence-based Category-Level Object Pose Estimation, Huan Ren, Wenfei Yang, Shifeng Zhang, Tianzhu Zhang
* UA-Pose: Uncertainty-Aware 6D Object Pose Estimation and Online Object Completion with Partial References, Ming-Feng Li, Xin Yang, Fu-En Wang, Hritam Basak, Yuyin Sun, Shreekant Gayaka, Min Sun, Cheng-Hao Kuo
* RefPose: Leveraging Reference Geometric Correspondences for Accurate 6D Pose Estimation of Unseen Objects, Jaeguk Kim, Jaewoo Park, Keuntek Lee, Nam Ik Cho
* One2Any: One-Reference 6D Pose Estimation for Any Object, Mengya Liu, Siyuan Li, Ajad Chhatkuli, Prune Truong, Luc Van Gool, Federico Tombari
* Unsupervised Discovery of Facial Landmarks and Head Pose, Satyajit Tourani, Siddharth Tourani, Arif Mahmood, Muhammad Haris Khan
* Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation, Junjie Chen, Weilong Chen, Yifan Zuo, Yuming Fang
* SCFlow2: Plug-and-Play Object Pose Refiner with Shape-Constraint Scene Flow, Qingyuan Wang, Rui Song, Jiaojiao Li, Kerui Cheng, David Ferstl, Yinlin Hu
* GIVEPose: Gradual Intra-class Variation Elimination for RGB-based Category-Level Object Pose Estimation, Ziqin Huang, Gu Wang, Chenyangguang Zhang, Ruida Zhang, Xiu Li, Xiangyang Ji
* UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image, Xingyu Liu, Gu Wang, Ruida Zhang, Chenyangguang Zhang, Federico Tombari, Xiangyang Ji
* ProbPose: A Probabilistic Approach to 2D Human Pose Estimation, Miroslav Purkrabek, Jiri Matas
* GCE-Pose: Global Context Enhancement for Category-level Object Pose Estimation, Weihang Li, Hongli XU, Junwen Huang, Hyunjun Jung, Peter KT Yu, Nassir Navab, Benjamin Busam
* MVDoppler-Pose: Multi-Modal Multi-View mmWave Sensing for Long-Distance Self-Occluded Human Walking Pose Estimation, Jaeho Choi, Soheil Hor, Shubo Yang, Amin Arbabian
* Probabilistic Prompt Distribution Learning for Animal Pose Estimation, Jiyong Rao, Brian Nlong Zhao, Yu Wang
* MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation, Aviral Chharia, Wenbo Gou, Haoye Dong
* Multi-View Pose-Agnostic Change Localization with Zero Labels, Chamuditha Jayanga Galappaththige, Jason Lai, Lloyd Windrim, Donald Dansereau, Niko Sunderhauf, Dimity Miller
* Structure-Aware Correspondence Learning for Relative Pose Estimation, Yihan Chen, Wenfei Yang, Huan Ren, Shifeng Zhang, Tianzhu Zhang, Feng Wu
* Any6D: Model-free 6D Pose Estimation of Novel Object, Taeyeop Lee, Bowen Wen, Minjun Kang, Gyuree Kang, In So Kweon, Kuk-Jin Yoon
* CRISP: Object Pose and Shape Estimation with Test-Time Adaptation, Jingnan Shi, Rajat Talak, Harry Zhang, David Jin, Luca Carlone
* CAP-Net: A Unified Network for 6D Pose and Size Estimation of Categorical Articulated Parts from a Single RGB-D Image, Jingshun Huang, Haitao Lin, Tianyu Wang, Yanwei Fu, Xiangyang Xue, Yi Zhu
* UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation, Yinqiao Wang, Hao Xu, Pheng-Ann Heng, Chi-Wing Fu
* Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation, Zhuoran Zhao, Linlin Yang, Pengzhan Sun, Pan Hui, Angela Yao
* PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation, Uyoung Jeong, Jonathan Freer, Seungryul Baek, Hyung Jin Chang, Kwang In Kim
* M3GYM: A Large-Scale Multimodal Multi-view Multi-person Pose Dataset for Fitness Activity Understanding in Real-world Settings, Qingzheng Xu, Ru Cao, Xin Shen, Heming Du, Sen Wang, Xin Yu
* Can Generative Video Models Help Pose Estimation?, Ruojin Cai, Jason Y. Zhang, Philipp Henzler, Zhengqi Li, Noah Snavely, Ricardo Martin-Brualla
* Pos3R: 6D Pose Estimation for Unseen Objects Made Easy, Weijian Deng, Dylan Campbell, Chunyi Sun, Jiahao Zhang, Shubham Kanitkar, Matt E. Shaffer, Stephen Gould
* ONDA-Pose: Occlusion-Aware Neural Domain Adaptation for Self-Supervised 6D Object Pose Estimation, Tao Tan, Qiulei Dong
* Leveraging Global Stereo Consistency for Category-Level Shape and 6D Pose Estimation from Stereo Images, Junning Qiu, Minglei Lu, Fei Wang, Yu Guo, Yonggen Ling
* 3D-Pose-Based Evaluation of the Risk of Sarcopenia, Yu-Hsuan Chiu, Gee-Sern Jison Hsu, Jiunn-Horng Kang, Jie-Syuan Wu

## **人-物交互 (Human-Object Interaction)**


* **\[Oral\] TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization**, Liang Pan, Zeshi Yang, Zhiyang Dou, Wenjia Wang, Buzhen Huang, Bo Dai, Taku Komura, Jingbo Wang
* **\[Oral\] ✩ G3Flow: Generative 3D Semantic Flow for Pose-aware and Generalizable Object Manipulation**, Tianxing Chen, Yao Mu, Zhixuan Liang, Zanxin Chen, Shijia Peng, Qiangyu Chen, Mingkun Xu, Ruizhen Hu, Hongyuan Zhang, Xuelong Li, Ping Luo
* *\[Highlight\] How Do I Do That? Synthesizing 3D Hand Motion and Contacts for Everyday Interactions*, Aditya Prakash, Benjamin Lundell, Dmitry Andreychuk, David Forsyth, Saurabh Gupta, Harpreet Sawhney
* CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement, Yun Liu, Chengwen Zhang, Ruofan Xing, Bingda Tang, Bowen Yang, Li Yi
* PICO: Reconstructing 3D People In Contact with Objects, Alpár Cseke, Shashank Tripathi, Sai Kumar Dwivedi, Arjun S. Lakshmipathy, Agniv Chatterjee, Michael J. Black, Dimitrios Tzionas
* ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions, Jeonghwan Kim, Jisoo Kim, Jeonghyeon Na, Hanbyul Joo
* EasyHOI: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild, Yumeng Liu, Xiaoxiao Long, Zemin Yang, Yuan Liu, Marc Habermann, Christian Theobalt, Yuexin Ma, Wenping Wang
* InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation, Sirui Xu, Dongting Li, Yucheng Zhang, Xiyan Xu, Qi Long, Ziyin Wang, Yunzhi Lu, Shuchang Dong, Hezi Jiang, Akshat Gupta, Yu-Xiong Wang, Liang-Yan Gui
* HOIGPT: Learning Long-Sequence Hand-Object Interaction with Language Models, Mingzhen Huang, Fu-Jen Chu, Bugra Tekin, Kevin J. Liang, Haoyu Ma, Weiyao Wang, Xingyu Chen, Pierre Gleize, Hongfei Xue, Siwei Lyu, Kris Kitani, Matt Feiszli, Hao Tang
* HSI-GPT: A General-Purpose Large Scene-Motion-Language Model for Human Scene Interaction, Yuan Wang, Yali Li, Xiang Li, Shengjin Wang
* Guiding Human-Object Interactions with Rich Geometry and Relations, Mengqing Xue, Yifei Liu, Ling Guo, Shaoli Huang, Changxing Ding
* PersonaHOI: Effortlessly Improving Face Personalization in Human-Object Interaction Generation, Xinting Hu, Haoran Wang, Jan Eric Lenssen, Bernt Schiele
* HORP: Human-Object Relation Priors Guided HOI Detection, Pei Geng, Jian Yang, Shanshan Zhang
* TASTE-Rob: Advancing Video Generation of Task-Oriented Hand-Object Interaction for Generalizable Robotic Manipulation, Hongxiang Zhao, Xingchen Liu, Mutian Xu, Yiming Hao, Weikai Chen, Xiaoguang Han
* End-to-End HOI Reconstruction Transformer with Graph-based Encoding, Zhenrong Wang, Qi Zheng, Sihan Ma, Maosheng Ye, Yibing Zhan, Dongjiang Li
* ChainHOI: Joint-based Kinematic Chain Modeling for Human-Object Interaction Generation, Ling-An Zeng, Guohong Huang, Yi-Lin Wei, Shengbo Gu, Yu-Ming Tang, Jingke Meng, Wei-Shi Zheng
* InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions, Sirui Xu, Hung Yu Ling, Yu-Xiong Wang, Liang-Yan Gui
* An Image-like Diffusion Method for Human-Object Interaction Detection, Xiaofei Hui, Haoxuan Qu, Hossein Rahmani, Jun Liu
* Locality-Aware Zero-Shot Human-Object Interaction Detection, Sanghyun Kim, Deunsol Jung, Minsu Cho
* LatentHOI: On the Generalizable Hand Object Motion Generation with Latent Hand Diffusion, Muchen Li, Sammy Christen, Chengde Wan, Yujun Cai, Renjie Liao, Leonid Sigal, Shugao Ma
* Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions, Boran Wen, Dingbang Huang, Zichen Zhang, Jiahong Zhou, Jianbin Deng, Jingyu Gong, Yulong Chen, Lizhuang Ma, Yong-Lu Li

## **行人重识别 (Person Re-identification)**


* AG-VPReID: A Challenging Large-Scale Benchmark for Aerial-Ground Video-based Person Re-Identification, Huy Nguyen, Kien Nguyen, Akila Pemasiri, Feng Liu, Sridha Sridharan, Clinton Fookes
* DKC: Differentiated Knowledge Consolidation for Cloth-Hybrid Lifelong Person Re-identification, Zhenyu Cui, Jiahuan Zhou, Yuxin Peng
* From Laboratory to Real World: A New Benchmark Towards Privacy-Preserved Visible-Infrared Person Re-Identification, Yan Jiang, Hao Yu, Xu Cheng, Haoyu Chen, Zhaodong Sun, Guoying Zhao
* Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification, Jiayu Jiang, Changxing Ding, Wentao Tan, Junhong Wang, Jin Tao, Xiangmin Xu
* SeCap: Self-Calibrating and Adaptive Prompts for Cross-view Person Re-Identification in Aerial-Ground Networks, Shining Wang, Yunlong Wang, Ruiqi Wu, Bingliang Jiao, Wenxuan Wang, Peng Wang
* From Poses to Identity: Training-Free Person Re-Identification via Feature Centralization, Chao Yuan, Guiwei Zhang, Changxiao Ma, Tianyi Zhang, Guanglin Niu
* Person De-reidentification: A Variation-guided Identity Shift Modeling, Yi-Xing Peng, Yu-Ming Tang, Kun-Yu Lin, Qize Yang, Jingke Meng, Xihan Wei, Wei-Shi Zheng
* DIFFER: Disentangling Identity Features via Semantic Cues for Clothes-Changing Person Re-ID, Xin Liang, Yogesh S Rawat
* Mixture of Submodules for Domain Adaptive Person Search, Minsu Kim, Seungryong Kim, Kwanghoon Sohn
* Identity-Clothing Similarity Modeling for Unsupervised Clothing Change Person Re-Identification, Zhiqi Pang, Junjie Wang, Lingling Zhao, Chunyu Wang
* Cheb-GR: Rethinking K-nearest Neighbor Search in Re-ranking for Person Re-identification, Jinxi Yang, He Li, Bo Du, Mang Ye

## **人体网格恢复 (Human Mesh Recovery)**


* **\[Oral\] MEGA: Masked Generative Autoencoder for Human Mesh Recovery**, Guénolé Fiche, Simon Leglaive, Xavier Alameda-Pineda, Francesc Moreno-Noguer
* **\[Oral\] Reconstructing Humans with a Biomechanically Accurate Skeleto**, Yan Xia, Xiaowei Zhou, Etienne Vouga, Qixing Huang, Georgios Pavlakos
* PromptHMR: Promptable Human Mesh Recovery, Yufu Wang, Yu Sun, Priyanka Patel, Kostas Daniilidis, Michael J. Black, Muhammed Kocabas
* DiSRT-In-Bed: Diffusion-Based Sim-to-Real Transfer Framework for In-Bed Human Mesh Recovery, Jing Gao, Ce Zheng, Laszlo A. Jeni, Zackory Erickson
* HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery, Yuto Matsubara, Ko Nishino
* BLADE: Single-view Body Mesh Estimation through Accurate Depth Estimation, Shengze Wang, Jiefeng Li, Tianye Li, Ye Yuan, Henry Fuchs, Koki Nagano, Shalini De Mello, Michael Stengel
* Pose-Guided Temporal Enhancement for Robust Low-Resolution Hand Reconstruction, Kaixin Fan, Pengfei Ren, Jingyu Wang, Haifeng Sun, Qi Qi, Zirui Zhuang, Jianxin Liao
* Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera, Zhengdi Yu, Stefanos Zafeiriou, Tolga Birdal
* PI-HMR: Towards Robust In-bed Temporal Human Shape Reconstruction with Contact Pressure Sensing, Ziyu Wu, Yufan Xiong, Mengting Niu, Fangting Xie, Quan Wan, Qijun Ying, Boyan Liu, Xiaohui Cai
* D^3-Human: Dynamic Disentangled Digital Human from Monocular Video, Honghu Chen, Bo Peng, Yunfan Tao, Juyong Zhang
* Thin-Shell-SfT: Fine-Grained Monocular Non-rigid 3D Surface Tracking with Neural Deformation Fields, Navami Kairanda, Marc Habermann, Shanthika Naik, Christian Theobalt, Vladislav Golyanik
* SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens, Chi Su, Xiaoxuan Ma, Jiajun Su, Yizhou Wang

## **人脸技术 (Face)**


* *\[Highlight\] FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly Inference Strategy*, Xingchao Yang, Takafumi Taketomi, Yuki Endo, Yoshihiro Kanamori
* *\[Highlight\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness*, Yiming Zhong, Qi Jiang, Jingyi Yu, Yuexin Ma
* *\[Highlight\] EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing*, Gaoxiang Cong, Jiadong Pan, Liang Li, Yuankai Qi, Yuxin Peng, Anton van den Hengel, Jian Yang, Qingming Huang
* *\[Highlight\] From Faces to Voices: Learning Hierarchical Representations for High-quality Video-to-Speech*, Ji-Hoon Kim, Jeongsoo Choi, Jaehun Kim, Chaeyoung Jung, Joon Son Chung
* Electromyography-Informed Facial Expression Reconstruction for Physiological-Based Synthesis and Analysis, Tim Büchner, Christoph Anders, Orlando Guntinas-Lichius, Joachim Denzler
* GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding, Yuki Kawana, Shintaro Shiba, Quan Kong, Norimasa Kobori
* De^2Gaze: Deformable and Decoupled Representation Learning for 3D Gaze Estimation, Yunfeng Xiao, Xiaowei Bai, Baojun Chen, Hao Su, Hao He, Liang Xie, Erwei Yin
* ControlFace: Harnessing Facial Parametric Control for Face Rigging, Wooseok Jang, Youngjun Hong, Geonho Cha, Seungryong Kim
* Face Forgery Video Detection via Temporal Forgery Cue Unraveling, Zonghui Guo, Yingjie Liu, Jie Zhang, Haiyong Zheng, Shiguang Shan
* SVFR: A Unified Framework for Generalized Video Face Restoration, Zhiyao Wang, Xu Chen, Chengming Xu, Junwei Zhu, Xiaobin Hu, Jiangning Zhang, Chengjie Wang, Yuqi Liu, Yiyi Zhou, Rongrong Ji
* FaceBench: A Multi-View Multi-Level Facial Attribute VOA Dataset for Benchmarking Face Perception MLLMS, Xiaoqin Wang, Xusen Ma, Xianxu Hou, Meidan Ding, Yudong Li, Junliang Chen, Wenting Chen, Xiaoyang Peng, Linlin Shen
* Enhancing Facial Privacy Protection via Weakening Diffusion Purification, Ali Salar, Qing Liu, Yingli Tian, Guoying Zhao
* VTON-HandFit: Virtual Try-on for Arbitrary Hand Pose Guided by Hand Priors Embedding, Yujie Liang, Xiaobin Hu, Boyuan Jiang, Donghao Luo, Xu Peng, Kai Wu, Chengming Xu, Wenhui Han, Taisong Jin, Chengjie Wang, Rongrong Ji
* T-FAKE: Synthesizing Thermal Images for Facial Landmarking, Philipp Flotho, Moritz Piening, Anna Kukleva, Gabriele Steidl
* OFER: Occluded Face Expression Reconstruction, Pratheba Selvaraju, Victoria Fernandez Abrevaya, Timo Bolkart, Rick Akkerman, Tianyu Ding, Faezeh Amjadi, Ilya Zharkov
* Dynamic Stereotype Theory Induced Micro-expression Recognition with Oriented Deformation, Bohao Zhang, Xuejiao Wang, Changbo Wang, Gaoqi He
* Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios, Hang Shao, Lei Luo, Jianjun Qian, Mengkai Yan, Shuo Chen, Jian Yang
* Generalizing Deepfake Video Detection with Plug-and-Play: Video-Level Blending and Spatiotemporal Adapter Tuning, Zhiyuan Yan, Yandan Zhao, Shen Chen, Mingyi Guo, Xinghe Fu, Taiping Yao, Shouhong Ding, Yunsheng Wu, Li Yuan
* S^3-Face: SSS-Compliant Facial Reflectance Estimation via Diffusion Priors, Xingyu Ren, Jiankang Deng, Yuhao Cheng, Wenhan Zhu, Yichao Yan, Xiaokang Yang, Stefanos Zafeiriou, Chao Ma
* Real-time Facial Expression Recognition For Intuitive Robot Coaches, Peyton Chandarana, Mohammadreza Mohammadi, Hasti Zanganeh, Ramtin Zand

## **人脸识别 (Face Recognition)**


* *\[Highlight\] Towards Explainable and Unprecedented Accuracy in Matching Challenging Finger Crease Patterns*, Zhenyu Zhou, Chengdong Dong, Ajay Kumar
* *\[Highlight\] CryptoFace: End-to-End Encrypted Face Recognition*, Wei Ao, Vishnu Naresh Boddeti
* Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation, Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang
* GIF: Generative Inspiration for Face Recognition at Scale, Saeed Ebrahimi, Sahar Rahimi, Ali Dabouei, Srinjoy Das, Jeremy M. Dawson, Nasser M. Nasrabadi
* ProjAttacker: A Configurable Physical Adversarial Attack for Face Recognition via Projector, Yuanwei Liu, Hui Wei, Chengyu Jia, Ruqi Xiao, Weijian Ruan, Xingxing Wei, Joey Tianyi Zhou, Zheng Wang
* UMFN: Unified Multi-Domain Face Normalization for Joint Cross-domain Prototype Learning and Heterogeneous Face Recognition, Meng Pang, Wenjun Zhang, Nanrun Zhou, Shengbo Chen, Hong Rao

## **人脸生成 (Face Generation)**


* **\[Oral\] CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models**, Felix Taubner, Ruihang Zhang, Mathieu Tuli, David B. Lindell
* **\[Oral\] Award candidate paper**
* **\[Oral\] IM-Portrait: Learning 3D-aware Video Diffusion for Photorealistic Talking Heads from Monocular VideosC**, Yuan Li, Ziqian Bai, Feitong Tan, Zhaopeng Cui, Sean Fanello, Yinda Zhang
* *\[Highlight\] FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation*, Kefan Chen, Chaerin Min, Linguang Zhang, Shreyas Hampali, Cem Keskin, Srinath Sridhar
* Prosody-Enhanced Acoustic Pre-training and Acoustic-Disentangled Prosody Adapting for Movie Dubbing, Zhedong Zhang, Liang Li, Chenggang Yan, Chunshan Liu, Anton van den Hengel, Yuankai Qi
* Wav2Sem: Plug-and-Play Audio Semantic Decoupling for 3D Speech-Driven Facial Animation, Hao Li, Ju Dai, Xin Zhao, Feng Zhou, Junjun Pan, Lei Li
* Sonic: Shifting Focus to Global Audio Perception in Portrait Animation, Xiaozhong Ji, Xiaobin Hu, Zhihong Xu, Junwei Zhu, Chuming Lin, Qingdong He, Jiangning Zhang, Donghao Luo, Yi Chen, Qin Lin, Qinglin Lu, Chengjie Wang
* Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture, Xuanchen Li, Jianyu Wang, Yuhao Cheng, Yikun Zeng, Xingyu Ren, Wenhan Zhu, Weiming Zhao, Yichao Yan
* High-Fidelity Relightable Monocular Portrait Animation with Lighting-Controllable Video Diffusion Model, Mingtao Guo, Guanyu Xing, Yanli Liu
* GPAvatar: High-fidelity Head Avatars by Learning Efficient Gaussian Projections, Wei-Qi Feng, Dong Han, Ze-Kang Zhou, Shunkai Li, Xiaoqiang Liu, Pengfei Wan, Di Zhang, Miao Wang
* HERA: Hybrid Explicit Representation for Ultra-Realistic Head Avatars, Hongrui Cai, Yuting Xiao, Xuan Wang, Jiafei Li, Yudong Guo, Yanbo Fan, Shenghua Gao, Juyong Zhang
* GASP: Gaussian Avatars with Synthetic Priors, Jack Saunders, Charlie Hewitt, Yanan Jian, Marek Kowalski, Tadas Baltrusaitis, Yiye Chen, Darren Cosker, Virginia Estellers, Nicholas Gydé, Vinay P. Namboodiri, Benjamin E. Lundell
* FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images, Rong Wang, Fabian Prada, Ziyan Wang, Zhongshi Jiang, Chengxiang Yin, Junxuan Li, Shunsuke Saito, Igor Santesteban, Javier Romero, Rohan Joshi, Hongdong Li, Jason Saragih, Yaser Sheikh
* DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh, Jingyu Zhuang, Di Kang, Linchao Bao, Liang Lin, Guanbin Li
* GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior, Zichen Tang, Yuan Yao, Miaomiao Cui, Liefeng Bo, Hongyu Yang
* SynthLight: Portrait Relighting with Diffusion Model by Learning to Re-render Synthetic Faces, Sumit Chaturvedi, Mengwei Ren, Yannick Hold-Geoffroy, Jingyuan Liu, Julie Dorsey, Zhixin Shu
* Exploring Timeline Control for Facial Motion Generation, Yifeng Ma, Jinwei Qi, Chaonan Ji, Peng Zhang, Bang Zhang, Zhidong Deng, Liefeng Bo
* Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency, Yutong Wang, Jiajie Teng, Jiajiong Cao, Yuming Li, Chenguang Ma, Hongteng Xu, Dixin Luo
* Let's Chorus: Partner-aware Hybrid Song-Driven 3D Head Animation, Xiumei Xie, Zikai Huang, Wenhao Xu, Peng Xiao, Xuemiao Xu, Huaidong Zhang
* KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation, Antoni Bigata, Michał Stypułkowski, Rodrigo Mira, Stella Bounareli, Konstantinos Vougioukas, Zoe Landgraf, Nikita Drobyshev, Maciej Zieba, Stavros Petridis, Maja Pantic
* EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation, Rang Meng, Xingyu Zhang, Yuming Li, Chenguang Ma
* X-Dyna: Expressive Dynamic Human Image Animation, Di Chang, Hongyi Xu, You Xie, Yipeng Gao, Zhengfei Kuang, Shengqu Cai, Chenxu Zhang, Guoxian Song, Chao Wang, Yichun Shi, Zeyuan Chen, Shijie Zhou, Linjie Luo, Gordon Wetzstein, Mohammad Soleymani
* Lux Post Facto: Learning Portrait Performance Relighting with Conditional Video Diffusion and a Hybrid Dataset, Yiqun Mei, Mingming He, Li Ma, Julien Philip, Wenqi Xian, David M George, Xueming Yu, Gabriel Dedic, Ahmet Levent Taşel, Ning Yu, Vishal M. Patel, Paul Debevec
* Monocular and Generalizable Gaussian Talking Head Animation, Shengjie Gong, Haojie Li, Jiapeng Tang, Dongming Hu, Shuangping Huang, Hao Chen, Tianshui Chen, Zhuoman Liu
* FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video, Jiawei Zhang, Zijian Wu, Zhiyang Liang, Yicheng Gong, Dongfang Hu, Yao Yao, Xun Cao, Hao Zhu
* GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-view Diffusion, Jiapeng Tang, Davide Davoli, Tobias Kirschstein, Liam Schoneveld, Matthias Nießner
* Vid2Avatar-Pro: Authentic Avatar from Videos in the Wild via Universal Prior, Chen Guo, Junxuan Li, Yash Kant, Yaser Sheikh, Shunsuke Saito, Chen Cao
* SinGS: Animatable Single-Image Human Gaussian Splats with Kinematic Priors, Yufan Wu, Xuanhong Chen, Wen Li, Shunran Jia, Hualiang Wei, Kairui Feng, Jialiang Chen, Yuhan Li, Ang He, Weimin Zhang, Bingbing Ni, Wenjun Zhang
* EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting, Suzhen Wang, Weijie Chen, Wei Zhang, Minda Zhao, Lincheng Li, Rongsheng Zhang, Zhipeng Hu, Xin Yu
* Learning Person-Specific Animatable Face Models from In-the-Wild Images via a Shared Base Model, Yuxiang Mao, Zhenfeng Fan, Zhilie Zhang, Zhiheng Zhang, Shihong Xia
* HiFi-Portrait: Zero-shot Identity-preserved Portrait Generation with High-fidelity Multi-face Fusion, Yifang Xu, Benxiang Zhai, Yunzhuo Sun, Ming Li, Yang Li, Sidan Du
* Omni-ID: Holistic Identity Representation Designed for Generative Tasks, Guocheng Qian, Kuan-Chieh Wang, Or Patashnik, Negin Heravi, Daniil Ostashev, Sergey Tulyakov, Daniel Cohen-Or, Kfir Aberman
* DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations, Ziqiao Peng, Yanbo Fan, Haoyu Wu, Xuan Wang, Hongyan Liu, Jun He, Zhaoxin Fan
* Perceptually Accurate 3D Talking Head Generation: New Definitions, Speech-Mesh Representation, and Evaluation Metrics, Lee Chae-Yeon, Oh Hyun-Bin, Han EunGi, Kim Sung-Bin, Suekyeong Nam, Tae-Hyun Oh
* Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation, Dingcheng Zhen, Shunshun Yin, Shiyang Qin, Hou Yi, Ziwei Zhang, Siyuan Liu, Gan Qi, Ming Tao
* Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Video Diffusion Transformer, Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, Siyu Zhu
* StableAnimator: High-Quality Identity-Preserving Human Image Animation, Shuyuan Tu, Zhen Xing, Xintong Han, Zhi-Qi Cheng, Qi Dai, Chong Luo, Zuxuan Wu
* 3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations, Yating Wang, Xuan Wang, Ran Yi, Yanbo Fan, Jichen Hu, Jingcheng Zhu, Lizhuang Ma
* LUCAS: Layered Universal Codec Avatars, Di Liu, Teng Deng, Giljoo Nam, Yu Rong, Stanislav Pidhorskyi, Junxuan Li, Jason Saragih, Dimitris N. Metaxas, Chen Cao
* GeoAvatar: Geometrically-Consistent Multi-Person Avatar Reconstruction from Sparse Multi-View Videos, Soohyun Lee, Seoyeon Kim, HeeKyung Lee, Won-Sik Jeong, Joo Ho Lee
* AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction, Lingteng Qiu, Shenhao Zhu, Qi Zuo, Xiaodong Gu, Yuan Dong, Junfei Zhang, Chao Xu, Zhe Li, Weihao Yuan, Liefeng Bo, Guanying Chen, Zilong Dong
* TAGA: Self-supervised Learning for Template-free Animatable Gaussian Articulated Model, Zhichao Zhai, Guikun Chen, Wenguan Wang, Dong Zheng, Jun Xiao
* DRIVE: Diffusion-based Rigging Empowers Generation of Versatile and Expressive Characters, Mingze Sun, Junhao Chen, Junting Dong, Yurun Chen, Xinyu Jiang, Shiwei Mao, Puhua Jiang, Jingbo Wang, Bo Dai, Ruqi Huang
* MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling, Yifang Men, Yuan Yao, Miaomiao Cui, Liefeng Bo
* Data Synthesis with Diverse Styles for Face Recognition via 3DMM-Guided Diffusion, Yuxi Mi, Zhizhou Zhong, Yuge Huang, Qiuyang Yuan, Xuan Zhao, Jianqing Xu, Shouhong Ding, Shaoming Wang, Rizen Guo, Shuigeng Zhou
* EmotiveTalk: Expressive Talking Head Generation through Audio Information Decoupling and Emotional Video Diffusion, Haotian Wang, Yuzhe Weng, Yueyan Li, Zilu Guo, Jun Du, Shutong Niu, Jiefeng Ma, Shan He, Xiaoyan Wu, Qiming Hu, Bing Yin, Cong Liu, Qingfeng Liu
* Synergizing Motion and Appearance: Multi-Scale Compensatory Codebooks for Talking Head Video Generation, Shuling Zhao, Fa-Ting Hong, Xiaoshui Huang, Dan Xu
* MVPortrait: Text-Guided Motion and Emotion Control for Multi-view Vivid Portrait Animation, Yukang Lin, Hokit Fung, Jianjin Xu, Zeping Ren, Adela S.M. Lau, Guosheng Yin, Xiu Li
* Free-viewpoint Human Animation with Pose-correlated Reference Selection, Fa-Ting Hong, Zhan Xu, Haiyang Liu, Qinjie Lin, Luchuan Song, Zhixin Shu, Yang Zhou, Duygu Ceylan, Dan Xu
* DiffPortrait360: Consistent Portrait Diffusion for 360 View Synthesis, Yuming Gu, Phong Tran, Yujian Zheng, Hongyi Xu, Heyuan Li, Adilbek Karmanov, Hao Li
* MEGA: Hybrid Mesh-Gaussian Head Avatar for High-Fidelity Rendering and Head Editing, Cong Wang, Di Kang, Heyi Sun, Shenhan Qian, Zixuan Wang, Linchao Bao, Song-Hai Zhang
* HRAvatar: High-Quality and Relightable Gaussian Head Avatar, Dongbin Zhang, Yunfei Liu, Lijian Lin, Ye Zhu, Kangjie Chen, Minghan Qin, Yu Li, Haoqian Wang
* Real-time High-fidelity Gaussian Human Avatars with Position-based Interpolation of Spatially Distributed MLPs, Youyi Zhan, Tianjia Shao, Yin Yang, Kun Zhou
* IDOL: Instant Photorealistic 3D Human Creation from a Single Image, Yiyu Zhuang, Jiaxi Lv, Hao Wen, Qing Shuai, Ailing Zeng, Hao Zhu, Shifeng Chen, Yujiu Yang, Xun Cao, Wei Liu
* StdGEN: Semantic-Decomposed 3D Character Generation from Single Images, Yuze He, Yanning Zhou, Wang Zhao, Zhongkai Wu, Kaiwen Xiao, Wei Yang, Yong-Jin Liu, Xiao Han
* Diff-Palm: Realistic Palmprint Generation with Polynomial Creases and Intra-Class Variation Controllable Diffusion Models, Jianlong Jin, Chenglong Zhao, Ruixin Zhang, Sheng Shang, Jianqing Xu, Jingyun Zhang, ShaoMing Wang, Yang Zhao, Shouhong Ding, Wei Jia, Yunsheng Wu
* GBC-Splat: Generalizable Gaussian-Based Clothed Human Digitalization under Sparse RGB Cameras, Hanzhang Tu, Zhanfeng Liao, Boyao Zhou, Shunyuan Zheng, Xilong Zhou, Liuxin Zhang, QianYing Wang, Yebin Liu
* SFDM: Robust Decomposition of Geometry and Reflectance for Realistic Face Rendering from Sparse-view Images, Daisheng Jin, Jiangbei Hu, Baixin Xu, Yuxin Dai, Chen Qian, Ying He
* Black Hole-Driven Identity Absorbing in Diffusion Models, Muhammad Shaheryar, Jong Taek Lee, Soon Ki Jung
* INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations, Yongming Zhu, Longhao Zhang, Zhengkun Rong, Tianshu Hu, Shuang Liang, Zhipeng Ge
* InsTaG: Learning Personalized 3D Talking Head from Few-Second Video, Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Jun Zhou, Lin Gu
* RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars, Linzhou Li, Yumeng Li, Yanlin Weng, Youyi Zheng, Kun Zhou
* AvatarArtist: Open-Domain 4D Avatarization, Hongyu Liu, Xuan Wang, Ziyu Wan, Yue Ma, Jingye Chen, Yanbo Fan, Yujun Shen, Yibing Song, Qifeng Chen
* Arc2Avatar: Generating Expressive 3D Avatars from a Single Image via ID Guidance, Dimitrios Gerogiannis, Foivos Paraperas Papantoniou, Rolandos Alexandros Potamias, Alexandros Lattas, Stefanos Zafeiriou
* Make-It-Animatable: An Efficient Framework for Authoring Animation-Ready 3D Characters, Zhiyang Guo, Jinxu Xiang, Kai Ma, Wengang Zhou, Houqiang Li, Ran Zhang
* PhysAnimator: Physics-Guided Generative Cartoon Animation, Tianyi Xie, Yiwei Zhao, Ying Jiang, Chenfanfu Jiang
* Zero-Shot Head Swapping in Real-World Scenarios, Taewoong Kang, Sohyun Jeong, Hyojin Jang, Jaegul Choo
* CaricatureBooth: Data-Free Interactive Caricature Generation in a Photo Booth, Zhiyu Qu, Yunqi Miao, Zhensong Zhang, Jifei Song, Jiankang Deng, Yi-Zhe Song
* DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models, Radu Alexandru Rosu, Keyu Wu, Yao Feng, Youyi Zheng, Michael J. Black
* HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis, Mengtian Li, Jinshu Chen, Wanquan Feng, Bingchuan Li, Fei Dai, Songtao Zhao, Qian He
* VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis, Enric Corona, Andrei Zanfir, Eduard Gabriel Bazavan, Nikos Kolotouros, Thiemo Alldieck, Cristian Sminchisescu
* HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animations, Zunnan Xu, Zhentao Yu, Zixiang Zhou, Jun Zhou, Xiaoyu Jin, Fa-ting Hong, Xiaozhong Ji, Junwei Zhu, Chengfei Cai, Shiyu Tang, Qin Lin, Xiu Li, Qinglin Lu
* MobilePortrait: Real-Time One-Shot Neural Head Avatars on Mobile Devices, Jianwen Jiang, Gaojie Lin, Zhengkun Rong, Chao Liang, Yongming Zhu, Jiaqi Yang, Tianyun Zhong
* Gaussian Eigen Models for Human Heads, Wojciech Zielonka, Timo Bolkart, Thabo Beeler, Justus Thies
* Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion, Zhenglin Zhou, Fan Ma, Hehe Fan, Tat-Seng Chua
* PERSE: Personalized 3D Generative Avatars from A Single Portrait, Hyunsoo Cha, Inhee Lee, Hanbyul Joo
* WildAvatar: Learning In-the-wild 3D Avatars from the Web, Zihao Huang, Shoukang Hu, Guangcong Wang, Tianqi Liu, Yuhang Zang, Zhiguo Cao, Wei Li, Ziwei Liu
* FreeCloth: Free-form Generation Enhances Challenging Clothed Human Modeling, Hang Ye, Xiaoxuan Ma, Hai Ci, Wentao Zhu, Yizhou Wang
* MagicArticulate: Make Your 3D Models Articulation-Ready, Chaoyue Song, Jianfeng Zhang, Xiu Li, Fan Yang, Yiwen Chen, Zhongcong Xu, Jun Hao Liew, Xiaoyang Guo, Fayao Liu, Jiashi Feng, Guosheng Lin
* GroomLight: Hybrid Inverse Rendering for Relightable Human Hair Appearance Modeling, Yang Zheng, Menglei Chai, Delio Vicini, Yuxiao Zhou, Yinghao Xu, Leonidas Guibas, Gordon Wetzstein, Thabo Beeler

## **活体检测/反欺诈 (Anti-Spoofing)**


* *\[Highlight\] Where the Devil Hides: Deepfake Detectors Can No longer Be Trusted*, Shuaiwei Yuan, Junyu Dong, Yuezun Li
* Al-Face: A Million-Scale Demographically Annotated Al-Generated Face Dataset and Fairness Benchmark, Li Lin, Santosh Santosh, Mingyang Wu, Xin Wang, Shu Hu
* Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of Al-generated Images, Tai D. Nguyen, Aref Azizpour, Matthew C. Stamm
* OmniGuard: Hybrid Manipulation Localization via Augmented Versatile Deep Image Watermarking, Xuanyu Zhang, Zecheng Tang, Zhipei Xu, Runyi Li, Youmin Xu, Bin Chen, Feng Gao, Jian Zhang
* Community Forensics: Using Thousands of Generators to Train Fake Image Detectors, Jeongsoo Park, Andrew Owens
* Beyond Generation: A Diffusion-based Low-level Feature Extractor for Detecting Al-generated Images, Nan Zhong, Haoyu Chen, Yiran Xu, Zhenxing Qian, Xinpeng Zhang
* FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing, Hossein Kashiani, Niloufar Alipour Talemi, Fatemeh Afghah
* Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model, Yue-Hua Han, Tai-Ming Huang, Kai-Lung Hua, Jun-Cheng Chen
* D^3: Scaling Up Deepfake Detection by Learning from Discrepancy, Yongqi Yang, Zhihao Qian, Ye Zhu, Olga Russakovsky, Yu Wu
* Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing, Zhuowei Li, Tianchen Zhao, Xiang Xu, Zheng Zhang, Zhihua Li, Xuanbai Chen, Qin Zhang, Alessandro Bergamo, Anil K. Jain, Yifan Xing
* FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning, Gaojian Wang, Feng Lin, Tong Wu, Zhenguang Liu, Zhongjie Ba, Kui Ren
* 12VGuard: Safeguarding Images against Misuse in Diffusion-based Image-to-Video Models, Dongnan Gui, Xun Guo, Wengang Zhou, Yan Lu
* Silence is Golden: Leveraging Adversarial Examples to Nullify Audio Control in LDM-based Talking-Head Generation, Yuan Gan, Jiaxu Miao, Yunze Wang, Yi Yang
* Secret Lies in Color: Enhancing Al-Generated Images Detection with Color Distribution Analysis, Zexi Jia, Chuanwei Huang, Yeshuang Zhu, Hongyan Fei, Xiaoyue Duan, Zhiqiang Yuan, Ying Deng, Jiapei Zhang, Jinchao Zhang, Jie Zhou
* CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by Al, Siyuan Cheng, Lingjuan Lyu, Zhenting Wang, Xiangyu Zhang, Vikash Sehwag
* Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection, Jikang Cheng, Zhiyuan Yan, Ying Zhang, Li Hao, Jiaxin Ai, Qin Zou, Chen Li, Zhongyuan Wang
* A Bias-Free Training Paradigm for More General Al-generated Image Detection, Fabrizio Guillaro, Giada Zingarini, Ben Usman, Avneesh Sud, Davide Cozzolino, Luisa Verdoliva
* Any-Resolution Al-Generated Image Detection by Spectral Learning, Dimitrios Karageorgiou, Symeon Papadopoulos, loannis Kompatsiaris, Efstratios Gavves
* Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning, Stefan Smeu, Dragos-Alexandru Boldisor, Dan Oneata, Elisabeta Oneata
* Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection, Xinjie Cui, Yuezun Li, Ao Luo, Jiaran Zhou, Junyu Dong
* Towards General Visual-Linguistic Face Forgery Detection, Ke Sun, Shen Chen, Taiping Yao, Ziyin Zhou, Jiayi Ji, Xiaoshuai Sun, Chia-Wen Lin, Rongrong Ji

# **六、底层视觉与图像处理 (Low-Level Vision & Image Processing)**
## **图像复原 (Image Restoration)**


* A Regularization-Guided Equivariant Approach for Image Restoration, Yulu Bai, Jiahong Fu, Qi Xie, Deyu Meng
* Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond), Tomer Garber, Tom Tirer
* SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration, Jianyi Wang, Zhijie Lin, Meng Wei, Yang Zhao, Ceyuan Yang, Chen Change Loy, Lu Jiang
* Reversing Flow for Image Restoration, Haina Qin, Wenyang Luo, Libin Wang, Dandan Zheng, Jingdong Chen, Ming Yang, Bing Li, Weiming Hu
* Navigating Image Restoration with VAR's Distribution Alignment Prior, Siyang Wang, Naishan Zheng, Jie Huang, Feng Zhao
* URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration, Rui Xu, Yuzhen Niu, Yuezhou Li, Huangbiao Xu, Wenxi Liu, Yuzhong Chen
* Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement, Hesong Li, Ziqi Wu, Ruiwen Shao, Tao Zhang, Ying Fu
* JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration, Yunlong Lin, Zixu Lin, Haoyu Chen, Panwang Pan, Chenxin Li, Sixiang Chen, Kairun Wen, Yeying Jin, Wenbo Li, Xinghao Ding
* Blind Bitstream-corrupted Video Recovery via Metadata-guided Diffusion Model, Shuyun Wang, Hu Zhang, Xin Shen, Dadong Wang, Xin Yu
* UHD-processer: Unified UHD Image Restoration with Progressive Frequency Learning and Degradation-aware Prompts, Yidi Liu, Dong Li, Xueyang Fu, Xin Lu, Jie Huang, Zheng-Jun Zha
* FiRe: Fixed-points of Restoration Priors for Solving Inverse Problems, Matthieu Terris, Ulugbek S. Kamilov, Thomas Moreau
* Adapting Text-to-Image Generation with Feature Difference Instruction for Generic Image Restoration, Chao Wang, Hehe Fan, Huichen Yang, Sarvnaz Karimi, Lina Yao, Yi Yang
* Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration, Yudong Mao, Hao Luo, Zhiwei Zhong, Peilin Chen, Zhijiang Zhang, Shiqi Wang
* VolFormer: Explore More Comprehensive Cube Interaction for Hyperspectral Image Restoration and Beyond, Dabing Yu, Zheng Gao
* One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion, Chunyang Cheng, Tianyang Xu, Zhenhua Feng, Xiaojun Wu, Zhangyong Tang, Hui Li, Zeyang Zhang, Sara Atito, Muhammad Awais, Josef Kittler
* Continuous Adverse Weather Removal via Degradation-Aware Distillation, Xin Lu, Jie Xiao, Yurui Zhu, Xueyang Fu
* Degradation-Aware Feature Perturbation for All-in-One Image Restoration, Xiangpeng Tian, Xiangyu Liao, Xiao Liu, Meng Li, Chao Ren
* DarkIR: Robust Low-Light Image Restoration, Daniel Feijoo, Juan C. Benito, Alvaro Garcia, Marcos V. Conde
* A Universal Scale-Adaptive Deformable Transformer for Image Restoration across Diverse Artifacts, Xuyi He, Yuhui Quan, Ruotao Xu, Hui Ji
* Complexity Experts are Task-Discriminative Learners for Any Image Restoration, Eduard Zamfir, Zongwei Wu, Nancy Mehta, Yuedong Tan, Danda Pani Paudel, Yulun Zhang, Radu Timofte
* OSDFace: One-Step Diffusion Model for Face Restoration, Jingkai Wang, Jue Gong, Lin Zhang, Zheng Chen, Xing Liu, Hong Gu, Yutong Liu, Yulun Zhang, Xiaokang Yang
* Hazy Low-Quality Satellite Video Restoration Via Learning Optimal Joint Degradation Patterns and Continuous-Scale Super-Resolution Reconstruction, Ning Ni, Libao Zhang
* Dynamic Content Prediction with Motion-aware Priors for Blind Face Video Restoration, Lianxin Xie, Bingbing Zheng, Si Wu, Hau San Wong
* LP-Diff: Towards Improved Restoration of Real-World Degraded License Plate, Haoyan Gong, Zhenrong Zhang, Yuzheng Feng, Anh Nguyen, Hongbin Liu
* ACL: Activating Capability of Linear Attention for Image Restoration, Yubin Gu, Yuan Meng, Jiayi Ji, Xiaoshuai Sun
* From Zero to Detail: Deconstructing Ultra-High-Definition Image Restoration from Progressive Spectral Perspective, Chen Zhao, Zhizhou Chen, Yunzhe Xu, Enxuan Gu, Jian Li, Zili Yi, Qian Wang, Jian Yang, Ying Tai
* UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior, I-Hsiang Chen, Wei-Ting Chen, Yu-Wei Liu, Yuan-Chun Chiang, Sy-Yen Kuo, Ming-Hsuan Yang

## **底层视觉 (Low-level Vision)**

* **\[Oral\] Removing Reflections from RAW Photos**, Eric Kee, Adam Pikielny, Kevin Blackburn-Matzen, Marc Levoy
* **\[Oral\] Learned Binocular-Encoding Optics for RGBD Imaging Using Joint Stereo and Focus Cues**, Yuhui Liu, Liangxun Ou, Qiang Fu, Hadi Amata, Wolfgang Heidrich, Yifan Peng
* **\[Oral\] Opportunistic Single-Photon Time of Flight**, Sotiris Nousias, Mian Wei, Howard Xiao, Maxx Wu, Shahmeer Athar, Kevin J. Wang, Anagh Malik, David A. Barmherzig, David B. Lindell, Kyros N. Kutulakos
* PolarFree: Polarization-based Reflection-Free Imaging, Mingde Yao, Menglu Wang, King-Man Tam, Lingen Li, Tianfan Xue, Jinwei Gu
* OpticalNet: An Optical Imaging Dataset and Benchmark Beyond the Diffraction Limit, Benquan Wang, Ruyi An, Jin-Kyu So, Sergei Kurdiumov, Eng Aik Chan, Giorgio Adamo, Yuhan Peng, Yewen Li, Bo An
* A Physics-Informed Blur Learning Framework for Imaging Systems, Liqun Chen, Yuxuan Li, Jun Dai, Jinwei Gu, Tianfan Xue
* MaDCoW: Marginal Distortion Correction for Wide-Angle Photography with Arbitrary Objects, Kevin Zhang, Jia-Bin Huang, Jose Echevarria, Stephen DiVerdi, Aaron Hertzmann
* S2D-LFE: Sparse-to-Dense Light Field Event Generation, Yutong Liu, Wenming Weng, Yueyi Zhang, Zhiwei Xiong
* Image Reconstruction from Readout-Multiplexed Single-Photon Detector Arrays, Shashwath Bharadwaj, Ruangrawee Kitichotkul, Akshay Agarwal, Vivek K Goyal
* Spk2SRImgNet: Super-Resolve Dynamic Scene from Spike Stream via Motion Aligned Collaborative Filtering, Yuanlin Wang, Yiyang Zhang, Ruiqin Xiong, Jing Zhao, Jian Zhang, Xiaopeng Fan, Tiejun Huang
* EventPSR: Surface Normal and Reflectance Estimation from Photometric Stereo Using an Event Camera, Bohan Yu, Jin Han, Boxin Shi, Imari Sato
* RORem: Training a Robust Object Remover with Human-in-the-Loop, Ruibin Li, Tao Yang, Song Guo, Lei Zhang
* DL2G: Degradation-guided Local-to-Global Restoration for Eyeglass Reflection Removal, Zhilv Yi, Xiao Lu, Hong Ding, Jingbo Hu, Zhi Jiang, Chunxia Xiao
* Improving Visual and Downstream Performance of Low-Light Enhancer with Vision Foundation Models Collaboration, Yuxuan Gu, Haoxuan Wang, Pengyang Ling, Zhixiang Wei, Huaian Chen, Yi Jin, Enhong Chen
* PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution, Shuangfan Zhou, Chu Zhou, Youwei Lyu, Heng Guo, Zhanyu Ma, Boxin Shi, Imari Sato
* UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion, Zixuan Chen, Yujin Wang, Xin Cai, Zhiyuan You, Zheming Lu, Fan Zhang, Shi Guo, Tianfan Xue
* LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene, Xiaoyu Zhang, Weihong Pan, Chong Bao, Xiyu Zhang, Xiaojun Xiang, Hanqing Jiang, Hujun Bao
* Sea-ing in Low-light, Nisha Varghese, A. N. Rajagopalan
* Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention for Region-aware Exposure Correction, Donggoo Jung, Daehyun Kim, Guanghui Wang, Tae Hyun Kim
* Do Computer Vision Foundation Models Learn the Low-level Characteristics of the Human Visual System?, Yancheng Cai, Fei Yin, Dounia Hammou, Rafal Mantiuk
* A Snapshot Low-Light Depth from Defocus System, Wei Xu, Charles James Wagner, Junjie Luo, Qi Guo
* Focal Split: Untethered Snapshot Depth from Differential Defocus, Junjie Luo, John Mamish, Alan Fu, Thomas Concannon, Josiah Hester, Emma Alexander, Qi Guo
* Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries, Wei Xu, Charles James Wagner, Junjie Luo, Qi Guo
* LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting, Xiaoyan Xing, Konrad Groh, Sezer Karaoglu, Theo Gevers, Anand Bhattad
* IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images, Chih-Hao Lin, Jia-Bin Huang, Zhengqin Li, Zhao Dong, Christian Richardt, Tuotuo Li, Michael Zollhöfer, Johannes Kopf, Shenlong Wang, Changil Kim
* Differentiable Inverse Rendering with Interpretable Basis BRDFs, Hoon-Gyu Chung, Seokjun Choi, Seung-Hwan Baek
* TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering, Chun Gu, Xiaofei Wei, Li Zhang, Xiatian Zhu
* Explicit Depth-Aware Blurry Video Frame Interpolation Guided by Differential Curves, Zaoming Yan, Pengcheng Lei, Tingting Wang, Faming Fang, Junkang Zhang, Yaomin Huang, Haichuan Song
* EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation, Zihao Zhang, Haoran Chen, Haoyu Zhao, Guansong Lu, Yanwei Fu, Hang Xu, Zuxuan Wu
* HVI: A New Color Space for Low-light Image Enhancement, Qingsen Yan, Yixu Feng, Cheng Zhang, Guansong Pang, Kangbiao Shi, Peng Wu, Wei Dong, Jinqiu Sun, Yanning Zhang
* Flash-Split: 2D Reflection Removal with Flash Cues and Latent Diffusion Separation, Tianfu Wang, Mingyang Xie, Haoming Cai, Sachin Shah, Christopher A. Metzler
* ScribbleLight: Single Image Indoor Relighting with Scribbles, Jun Myeong Choi, Annie Wang, Pieter Peers, Anand Bhattad, Roni Sengupta
* Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes, JunYong Choi, Min-cheol Sagong, Seokyeong Lee, Seung-Won Jung, Ig-Jae Kim, Junghyun Cho
* BIM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions, Wonyong Seo, Jihyong Oh, Munchurl Kim
* Integral Fast Fourier Color Constancy, Wenjun Wei, Yanlin Qian, Huaian Chen, Junkang Dai, Yi Jin
* Reversible Decoupling Network for Single Image Reflection Removal, Hao Zhao, Mingjia Li, Qiming Hu, Xiaojie Guo
* Stabilizing and Accelerating Autofocus with Expert Trajectory Regularized Deep Reinforcement Learning, Shouhang Zhu, Chenglin Li, Yuankun Jiang, Li Wei, Nuowen Kan, Ziyang Zheng, Wenrui Dai, Junni Zou, Hongkai Xiong
* SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal, Xinrui Wang, Langing Guo, Xiyu Wang, Siyu Huang, Bihan Wen
* MetaShadow: Object-Centered Shadow Detection, Removal, and Synthesis, Tianyu Wang, Jianming Zhang, Haitian Zheng, Zhihong Ding, Scott Cohen, Zhe Lin, Wei Xiong, Chi-Wing Fu, Luis Figueroa, Soo Ye Kim
* Towards Smart Point-and-Shoot Photography, Jiawan Li, Fei Zhou, Zhipeng Zhong, Jiongzhi Lin, Guoping Qiu

## **超分辨率 (Super-Resolution)**


* *\[Highlight\] Volume Tells: Dual Cycle-Consistent Diffusion for 3D Fluorescence Microscopy De-noising and Super-Resolution*, Zelin Li, Chenwei Wang, Zhaoke Huang, Yiming Ma, Cunming Zhao, Zhongying Zhao, Hong Yan
* Decoupling Fine Detail and Global Geometry for Compressed Depth Map Super-Resolution, Huan Zheng, Wencheng Han, Jianbing Shen
* VideoGigaGAN: Towards Detail-rich Video Super-Resolution, Yiran Xu, Taesung Park, Richard Zhang, Yang Zhou, Eli Shechtman, Feng Liu, Jia-Bin Huang, Difan Liu
* Progressive Focused Transformer for Single Image Super-Resolution, Wei Long, Xingyu Zhou, Leheng Zhang, Shuhang Gu
* HIIF: Hierarchical Encoding based Implicit Image Function for Continuous Super-resolution, Yuxuan Jiang, Ho Man Kwan, Tianhao Peng, Ge Gao, Fan Zhang, Xiaoqing Zhu, Joel Sole, David Bull
* Augmenting Perceptual Super-Resolution via Image Quality Predictors, Fengjia Zhang, Samrudhdhi B. Rangrej, Tristan Aumentado-Armstrong, Afsaneh Fazly, Alex Levinshtein
* Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach, Lingchen Sun, Rongyuan Wu, Zhiyuan Ma, Shuaizheng Liu, Qiaosi Yi, Lei Zhang
* Latent Space Super-Resolution for Higher-Resolution Image Generation with Diffusion Models, Jinho Jeong, Sangmin Han, Jinwoo Kim, Seon Joo Kim
* Self-supervised ControlNet with Spatio-Temporal Mamba for Real-world Video Super-resolution, Shijun Shi, Jing Xu, Lijing Lu, Zhihang Li, Kai Hu
* Adaptive Dropout: Unleashing Dropout across Layers for Generalizable Image Super-Resolution, Hang Xu, Jie Huang, Wei Yu, Jiangtong Tan, Zhen Zou, Feng Zhao
* DiflISR: A Diffusion Model with Gradient Guidance for Infrared Image Super-Resolution, Xingyuan Li, Zirui Wang, Yang Zou, Zhixin Chen, Jun Ma, Zhiying Jiang, Long Ma, Jinyuan Liu
* ADD: Attribution-Driven Data Augmentation Framework for Boosting Image Super-Resolution, Ze-Yu Mi, Yu-Bin Yang
* AutoLUT: LUT-Based Image Super-Resolution with Automatic Sampling and Adaptive Residual Learning, Yuheng Xu, Shijie Yang, Xin Liu, Jie Liu, Jie Tang, Gangshan Wu
* The Power of Context: How Multimodality Improves Image Super-Resolution, Kangfu Mei, Hossein Talebi, Mojtaba Ardakani, Vishal M. Patel, Peyman Milanfar, Mauricio Delbracio
* TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution, Linwei Dong, Qingnan Fan, Yihong Guo, Zhonghao Wang, Qi Zhang, Jinwei Chen, Yawei Luo, Changqing Zou
* BF-STVSR: B-Splines and Fourier--Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution, Eunjin Kim, Hyeonjin Kim, Kyong Hwan Jin, Jaejun Yoo
* FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution, Junyang Chen, Jinshan Pan, Jiangxin Dong
* Efficient Video Super-Resolution for Real-time Rendering with Decoupled G-buffer Guidance, Mingjun Zheng, Long Sun, Jiangxin Dong, Jinshan Pan
* EvEnhancer: Empowering Effectiveness, Efficiency and Generalizability for Continuous Space-Time Video Super-Resolution with Events, Shuoyan Wei, Feng Li, Shengeng Tang, Yao Zhao, Huihui Bai
* PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-wise Video Super-Resolution, Shian Du, Menghan Xia, Chang Liu, Xintao Wang, Jing Wang, Pengfei Wan, Di Zhang, Xiangyang Ji
* CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution, Xin Liu, Jie Liu, Jie Tang, Gangshan Wu
* Auto-Encoded Supervision for Perceptual Image Super-Resolution, MinKyu Lee, Sangeek Hyun, Woojin Jun, Jae-Pil Heo
* Live freeway traffic state super-resolution, Junyi Ji, Alex Richardson, Derek Gloudemans, Gergely Zachár, Matthew Nice, William Barbour, Jonathan Sprinkle, Benedetto Piccoli, Daniel B. Work

## **图像去噪 (Denoising)**


* Rethinking Reconstruction and Denoising in the Dark: New Perspective, General Architecture and Beyond, Tengyu Ma, Long Ma, Ziye Li, Yuetong Wang, Jinyuan Liu, Chengpei Xu, Risheng Liu
* Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable, Xin Jin, Simon Niklaus, Zhoutong Zhang, Zhihao Xia, Chunle Guo, Yuting Yang, Jiawen Chen, Chongyi Li
* Noise Modeling in One Hour: Minimizing Preparation Efforts for Self-supervised Low-Light RAW Image Denoising, Feiran Li, Haiyang Jiang, Daisuke Iso
* Zero-Shot Blind-spot Image Denoising via Implicit Neural Sampling, Yuhui Quan, Tianxiang Zheng, Zhiyuan Ma, Hui Ji
* DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Denoising, Sidi Yang, Binxiao Huang, Yulun Zhang, Dahai Yu, Yujiu Yang, Ngai Wong
* All-Optical Nonlinear Diffractive Deep Network for Ultrafast Image Denoising, Xiaoling Zhou, Zhemg Lee, Wei Ye, Rui Xie, Wenbo Zhang, Guanju Peng, Zongze Li, Shikun Zhang
* Rotation-Equivariant Self-Supervised Method in Image Denoising, Hanze Liu, Jiahong Fu, Qi Xie, Deyu Meng
* Complementary Advantages: Exploiting Cross-Field Frequency Correlation for NIR-Assisted Image Denoising, Yuchen Wang, Hongyuan Wang, Lizhi Wang, Xin Wang, Lin Zhu, Wanxuan Lu, Hua Huang
* Positive2Negative: Breaking the Information-Lossy Barrier in Self-Supervised Single Image Denoising, Tong Li, Lizhi Wang, Zhiyuan Xu, Lin Zhu, Wanxuan Lu, Hua Huang

## **图像去模糊 (Deblurring)**


* Diffusion-based Event Generation for High-Quality Image Deblurring, Xinan Xie, Qing Zhang, Wei-Shi Zheng
* Quad-Pixel Image Defocus Deblurring: A New Benchmark and Model, Hang Chen, Yin Xie, Xiaoxiu Peng, Lihu Sun, Wenkai Su, Xiaodong Yang, Chengming Liu
* Parameterized Blur Kernel Prior Learning for Local Motion Deblurring, Zhenxuan Fang, Fangfang Wu, Tao Huang, Le Dong, Weisheng Dong, Xin Li, Guangming Shi
* Gyro-based Neural Single Image Deblurring, Heemin Yang, Jaesung Rim, Seungyong Lee, Seung-Hwan Baek, Sunghyun Cho
* Efficient Visual State Space Model for Image Deblurring, Lingshun Kong, Jiangxin Dong, Jinhui Tang, Ming-Hsuan Yang, Jinshan Pan

## **图像去雾 (Dehazing)**


* Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images, Jiuchen Chen, Xinyu Yan, Qizhi Xu, Kaiqi Li
* Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing, Ruiyi Wang, Yushuo Zheng, Zicheng Zhang, Chunyi Li, Shuaicheng Liu, Guangtao Zhai, Xiaohong Liu
* CoA: Towards Real Image Dehazing via Compression-and-Adaptation, Long Ma, Yuxin Feng, Yan Zhang, Jinyuan Liu, Weimin Wang, Guang-Yong Chen, Chengpei Xu, Zhuo Su
* Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing, Jiayi Fu, Siyu Liu, Zikun Liu, Chun-Le Guo, Hyunhee Park, Ruiqi Wu, Guoqing Wang, Chongyi Li

## **图像压缩 (Image Compression)**


* *\[Highlight\] Balanced Rate-Distortion Optimization in Learned Image Compression*, Yichi Zhang, Zhihao Duan, Yuning Huang, Fengqing Zhu
* *\[Highlight\] Multirate Neural Image Compression with Adaptive Lattice Vector Quantization*, Hao Xu, Xiaolin Wu, Xi Zhang
* *\[Highlight\] Good*, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion, Jona Ballé, Luca Versari, Emilien Dupont, Hyunjik Kim, Matthias Bauer
* Towards Lossless Implicit Neural Representation via Bit Plane Decomposition, Woo Kyoung Han, Byeonghun Lee, Hyunmin Cho, Sunghoon Im, Kyong Hwan Jin
* Bridging the Gap between Gaussian Diffusion Models and Universal Quantization for Image Compression, Lucas Relic, Roberto Azevedo, Yang Zhang, Markus Gross, Christopher Schroers
* SINR: Sparsity Driven Compressed Implicit Neural Representations, Dhananjaya Jayasundara, Sudarshan Rajagopalan, Yasiru Ranasinghe, Trac D. Tran, Vishal M. Patel
* Test-Time Fine-Tuning of Image Compression Models for Multi-Task Adaptability, Unki Park, Seongmoon Jeong, Youngchan Jang, Gyeong-Moon Park, Jong Hwan Ko
* High Dynamic Range Video Compression: A Large-Scale Benchmark Dataset and A Learned Bit-depth Scalable Compression Algorithm, Zhaoyi Tian, Feifeng Wang, Shiwei Wang, Zihao Zhou, Yao Zhu, Liquan Shen
* ECVC: Exploiting Non-Local Correlations in Multiple Frames for Contextual Video Compression, Wei Jiang, Junru Li, Kai Zhang, Li Zhang
* Linear Attention Modeling for Learned Image Compression, Donghui Feng, Zhengxue Cheng, Shen Wang, Ronghua Wu, Hongwei Hu, Guo Lu, Li Song
* Fitted Neural Lossless Image Compression, Zhe Zhang, Zhenzhong Chen, Shan Liu
* TopNet: Transformer-Efficient Occupancy Prediction Network for Octree-Structured Point Cloud Geometry Compression, Xinjie Wang, Yifan Zhang, Ting Liu, Xinpu Liu, Ke Xu, Jianwei Wan, Yulan Guo, Hanyun Wang
* FLAVC: Learned Video Compression with Feature Level Attention, Chun Zhang, Heming Sun, Jiro Katto
* PICD: Versatile Perceptual Image Compression with Diffusion Rendering, Tongda Xu, Jiahao Li, Bin Li, Yan Wang, Ya-Qin Zhang, Yan Lu
* WISE: A Framework for Gigapixel Whole-Slide-Image Lossless Compression, Yu Mao, Jun Wang, Nan Guan, Chun Jason Xue
* HUNet: Homotopy Unfolding Network for Image Compressive Sensing, Feiyang Shen, Hongping Gan
* Frequency-Biased Synergistic Design for Image Compression and Compensation, Jiaming Liu, Qi Zheng, Zihao Liu, Yilian Zhong, Peiye Liu, Tao Liu, Shusong Xu, Yanheng Lu, Sicheng Li, Dimin Niu, Yibo Fan
* Learned Image Compression with Dictionary-based Entropy Model, Jingbo Lu, Leheng Zhang, Xingyu Zhou, Mu Li, Wen Li, Shuhang Gu
* Decouple Distortion from Perception: Region Adaptive Diffusion for Extreme-low Bitrate Perception Image Compression, Jinchang Xu, Shaokang Wang, Jintao Chen, Zhe Li, Peidong Jia, Fei Zhao, Guoqing Xiang, Zhijian Hao, Shanghang Zhang, Xiaodong Xie
* Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing, Chen Liao, Yan Shen, Dan Li, Zhongli Wang
* Perceptual Video Compression with Neural Wrapping, Muhammad Umar Karim Khan, Aaron Chadha, Mohammad Ashraful Anam, Yiannis Andreopoulos
* Plug-and-Play Versatile Compressed Video Enhancement, Huimin Zeng, Jiacheng Li, Zhiwei Xiong
* HIRISE: High-Resolution Image Scaling for Edge ML via In-Sensor Compression and Selective ROI, Brendan Reidy, Peyton Chandarana, Ramtin Zand

## **图像质量评价 (Image Quality Assessment)**


* *\[Highlight\] FineVQ: Fine-Grained User Generated Content Video Quality Assessment*, Huiyu Duan, Qiang Hu, Jiarui Wang, Liu Yang, Zitong Xu, Lu Liu, Xiongkuo Min, Chunlei Cai, Tianxiao Ye, Xiaoyun Zhang, Guangtao Zhai
* Distilling Spatially-Heterogeneous Distortion Perception for Blind Image Quality Assessment, Xudong Li, Wenjie Nie, Yan Zhang, Runze Hu, Ke Li, Xiawu Zheng, Liujuan Cao
* Rethinking Personalized Aesthetics Assessment: Employing Physique Aesthetics Assessment as An Exemplification, Haobin Zhong, Shuai He, Anlong Ming, Huadong Ma
* KVQ: Boosting Video Quality Assessment via Saliency-guided Local Perception, Yunpeng Qu, Kun Yuan, Qizhi Xie, Ming Sun, Chao Zhou, Jian Wang
* Image Quality Assessment: From Human to Machine Preference, Chunyi Li, Yuan Tian, Xiaoyue Ling, Zicheng Zhang, Haodong Duan, Haoning Wu, Ziheng Jia, Xiaohong Liu, Xiongkuo Min, Guo Lu, Weisi Lin, Guangtao Zhai
* Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment, Fatemeh Behrad, Tinne Tuytelaars, Johan Wagemans
* HybridMQA: Exploring Geometry-Texture Interactions for Colored Mesh Quality Assessment, Armin Shafiee Sarvestani, Sheyang Tang, Zhou Wang
* Exploring Semantic Feature Discrimination for Perceptual Image Super-Resolution and Opinion-Unaware No-Reference Image Quality Assessment, Guanglu Dong, Xiangyu Liao, Mingyang Li, Guihuan Guo, Chao Ren
* Toward Generalized Image Quality Assessment: Relaxing the Perfect Reference Quality Assumption, Du Chen, Tianhe Wu, Kede Ma, Lei Zhang
* Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution, Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, Chao Dong
* Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference, Wenhao Shen, Mingliang Zhou, Yu Chen, Xuekai Wei, Yong Feng, Huayan Pu, Weijia Jia
* AIGV-Assessor: Benchmarking and Evaluating the Perceptual Quality of Text-to-Video Generation with LMM, Jiarui Wang, Huiyu Duan, Guangtao Zhai, Juntong Wang, Xiongkuo Min

## **视频去雨 (Video Deraining)**
* **\[Oral\] Semi-Supervised State-Space Model with Dynamic Stacking Filter for Real-World Video Deraining**, Shangquan Sun, Wenqi Ren, Juxiang Zhou, Shu Wang, Jianhou Gan, Xiaochun Cao



# **七、特定应用领域 (Specific Application Domains)**
## **具身智能 (Embodied AI)**


* **\[Oral\] GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill**, Jieming Cui, Tengyu Liu, Ziyu Meng, Jiale Yu, Ran Song, Wei Zhang, Yixin Zhu, Siyuan Huang
* **\[Oral\] PDFactor: Learning Tri-Perspective View Policy Diffusion Field for Multi-Task Robotic Manipulation**, Jingyi Tian, Le Wang, Sanping Zhou, Sen Wang, Jiayi Li, Haowen Sun, Wei Tang
* **\[Oral\] Viewpoint Rosetta Stone: Unlocking Unpaired Ego-Exo Videos for View-invariant Representation Learning**, Mi Luo, Zihui Xue, Alex Dimakis, Kristen Grauman
* RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete, Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, Xinda Xue, Qinghang Su, Huaihai Lyu, Xiaolong Zheng, Jiaming Liu, Zhongyuan Wang, Shanghang Zhang
* CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos, Xinhao Liu, Jintong Li, Yicheng Jiang, Niranjan Sujay, Zhicheng Yang, Juexiao Zhang, John Abanes, Jing Zhang, Chen Feng
* BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligence, Xuewu Lin, Tianwei Lin, Lichao Huang, Hongyu Xie, Zhizhong Su
* Ges3ViG: Incorporating Pointing Gestures into Language-Based 3D Visual Grounding for Embodied Reference Understanding, Atharv Mahesh Mane, Dulanga Weerakoon, Vigneshwaran Subbaraju, Sougata Sen, Sanjay E. Sarma, Archan Misra
* Universal Actions for Enhanced Embodied Foundation Models, Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, Xianyuan Zhan
* TANGO: Training-free Embodied Al Agents for Open-world Tasks, Filippo Ziliotto, Tommaso Campari, Luciano Serafini, Lamberto Ballan
* RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation, Mingfei Han, Liang Ma, Kamila Zhumakhanova, Ekaterina Radionova, Jingyi Zhang, Xiaojun Chang, Xiaodan Liang, Ivan Laptev
* Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration, Lizheng Zu, Lin Lin, Song Fu, Na Zhao, Pan Zhou
* Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach, Steeven Janny, Hervé Poirier, Leonid Antsfeld, Guillaume Bono, Gianluca Monaci, Boris Chidlovskii, Francesco Giuliari, Alessio Del Bue, Christian Wolf
* ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting, Shaofei Cai, Zihao Wang, Kewei Lian, Zhancun Mu, Xiaojian Ma, Anji Liu, Yitao Liang
* IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments, Can Zhang, Gim Hee Lee
* A Data-Centric Revisit of Pre-Trained Vision Models for Robot Leaming, Xin Wen, Bingchen Zhao, Yilun Chen, Jiangmiao Pang, Xiaojuan Qi
* Robotic Visual Instruction, Yanbang Li, Ziyang Gong, Haoyang Li, Xiaoqi Huang, Haolan Kang, Guangping Bai, Xianzheng Ma
* DynScene: Scalable Generation of Dynamic Robotic Manipulation Scenes for Embodied Al, Sangmin Lee, Sungyong Park, Heewon Kim
* UniGraspTransformer: Simplified Policy Distillation for Scalable Dexterous Robotic Grasping, Wenbo Wang, Fangyun Wei, Lei Zhou, Xi Chen, Lin Luo, Xiaohan Yi, Yizhong Zhang, Yaobo Liang, Chang Xu, Yan Lu, Jiaolong Yang, Baining Guo
* ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping, Youxin Pang, Ruizhi Shao, Jiajun Zhang, Hanzhang Tu, Yun Liu, Boyao Zhou, Hongwen Zhang, Yebin Liu
* Hand-held Object Reconstruction from RGB Video with Dynamic Interaction, Shijian Jiang, Qi Ye, Rengan Xie, Yuchi Huo, Jiming Chen
* Feature4X: Bridging Any Monocular Video to 4D Agentic Al with Versatile Gaussian Feature Fields, Shijie Zhou, Hui Ren, Yijia Weng, Shuwang Zhang, Zhen Wang, Dejia Xu, Zhiwen Fan, Suya You, Zhangyang Wang, Leonidas Guibas, Achuta Kadambi
* g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks, Zihan Wang, Gim Hee Lee
* MobileH2R: Learning Generalizable Human to Mobile Robot Handover Exclusively from Scalable and Diverse Synthetic Data, Zifan Wang, Ziqing Chen, Junyu Chen, Jilong Wang, Yuxin Yang, Yunze Liu, Xueyi Liu, He Wang, Li Yi
* OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints, Mingjie Pan, Jiyao Zhang, Tianshu Wu, Yinghao Zhao, Wenlong Gao, Hao Dong
* Generating 6DoF Object Manipulation Trajectories from Action Description in Egocentric Vision, Tomoya Yoshida, Shuhei Kurita, Taichi Nishimura, Shinsuke Mori
* Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation, Yu Qi, Yuanchen Ju, Tianming Wei, Chi Chu, Lawson L.S. Wong, Huazhe Xu
* Spatial-Temporal Graph Diffusion Policy with Kinematic Modeling for Bimanual Robotic Manipulation, Qi Lv, Hao Li, Xiang Deng, Rui Shao, Yinchuan Li, Jianye Hao, Longxiang Gao, Michael Yu Wang, Liqiang Nie
* UniGoal: Towards Universal Zero-shot Goal-oriented Navigation, Hang Yin, Xiuwei Xu, Linqing Zhao, Ziwei Wang, Jie Zhou, Jiwen Lu
* Visual Agentic Al for Spatial Reasoning with a Dynamic API, Damiano Marsili, Rohun Agrawal, Yisong Yue, Georgia Gkioxari
* GenECA: A Generalizable Framework for Real-Time Multimodal Embodied Conversational Agents with Emotion-Sensitive Interaction, Santosh Patapati, Trisanth Srinivasan
* AR2D2: Training a Robot Without A Robot, Abhimanyu Saighal, Jiafei Duan, Ranjay Krishna, Dieter Fox
* VIZ: Virtual and Physical Navigation System for the Visually Impaired, Trisanth Srinivasan, Santosh Patapati

## **自动驾驶 (Autonomous Driving)**


* **\[Oral\] Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models**, Zhejun Zhang, Peter Karkus, Maximilian Igl, Wenhao Ding, Yuxiao Chen, Boris Ivanovic, Marco Pavone
* **\[Oral\] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving**, Xuesong Chen, Linjiang Huang, Tao Ma, Rongyao Fang, Shaoshuai Shi, Hongsheng Li
* **\[Oral\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-stage Fusion**, Yiran Wang, Jiaqi Li, Chaoyi Hong, Ruibo Li, Liusheng Sun, Xiao Song, Zhe Wang, Zhiguo Cao, Guosheng Lin
* HiLoTs: High-Low Temporal Sensitive Representation Learning for Semi-Supervised LiDAR Segmentation in Autonomous Driving, R.D. Lin, Pengcheng Weng, Yinqiao Wang, Han Ding, Jinsong Han, Fei Wang
* BEVDiffuser: Plug-and-Play Diffusion Model for BEV Denoising with Ground-Truth Guidance, Xin Ye, Burhaneddin Yaman, Sheng Cheng, Feng Tao, Abhirup Mallik, Liu Ren
* Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction, Dubing Chen, Huan Zheng, Jin Fang, Xingping Dong, Xianfei Li, Wenlong Liao, Tao He, Pai Peng, Jianbing Shen
* STCOcc: Sparse Spatial-Temporal Cascade Renovation for 3D Occupancy and Scene Flow Prediction, Zhimin Liao, Ping Wei, Shuaijia Chen, Haoxuan Wang, Ziyang Ren
* LIDAR-RT: Gaussian-based Ray Tracing for Dynamic LiDAR Re-simulation, Chenxu Zhou, Lvchang Fu, Sida Peng, Yunzhi Yan, Zhanhua Zhang, Yong Chen, Jiazhi Xia, Xiaowei Zhou
* Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation, Ziyang Xie, Zhizheng Liu, Zhenghao Peng, Wayne Wu, Bolei Zhou
* GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving, Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, Wei Yin
* ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling, Zikang Zhou, Hengjian Zhou, Haibo Hu, Zihao Wen, Jianping Wang, Yung-Hui Li, Yu-Kai Huang
* Adapting to Observation Length of Trajectory Prediction via Contrastive Learning, Ruiqi Qiu, Jun Gong, Xinyu Zhang, Siqi Luo, Bowen Zhang, Yi Cen
* DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes, Chensheng Peng, Chengwei Zhang, Yixiao Wang, Chenfeng Xu, Yichen Xie, Wenzhao Zheng, Kurt Keutzer, Masayoshi Tomizuka, Wei Zhan
* JiSAM: Alleviate Labeling Burden and Corner Case Problems in Autonomous Driving via Minimal Real-World Data, Runjian Chen, Wenqi Shao, Bo Zhang, Shaoshuai Shi, Li Jiang, Ping Luo
* Rethinking Lanes and Points in Complex Scenarios for Monocular 3D Lane Detection, Yifan Chang, Junjie Huang, Xiaofeng Wang, Yun Ye, Zhujin Liang, Yi Shan, Dalong Du, Xingang Wang
* SceneCrafter: Controllable Multi-View Driving Scene Editing, Zehao Zhu, Yuliang Zou, Chiyu Max Jiang, Bo Sun, Vincent Casser, Xiukun Huang, Jiahao Wang, Zhenpei Yang, Ruiqi Gao, Leonidas Guibas, Mingxing Tan, Dragomir Anguelov
* Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map, Xinyuan Chang, Maixuan Xue, Xinran Liu, Zheng Pan, Xing Wei
* CoSDH: Communication-Efficient Collaborative Perception via Supply-Demand Awareness and Intermediate-Late Hybridization, Junhao Xu, Yanan Zhang, Zhi Cai, Di Huang
* Generating Multimodal Driving Scenes via Next-Scene Prediction, Yanhao Wu, Haoyang Zhang, Tianwei Lin, Lichao Huang, Shujie Luo, Rui Wu, Congpei Qiu, Wei Ke, Tong Zhang
* Bridging Past and Future: End-to-End Autonomous Driving with Historical Prediction and Planning, Bozhou Zhang, Nan Song, Xin Jin, Li Zhang
* UCM-VeID V2: A Richer Dataset and A Pre-training Method for UAV Cross-Modality Vehicle Re-Identification, Xingyue Liu, Jiahao Qi, Chen Chen, KangCheng Bin, Ping Zhong
* Hyperdimensional Uncertainty Quantification for Multimodal Uncertainty Fusion in Autonomous Vehicles Perception, Luke Chen, Junyao Wang, Trier Mortlock, Pramod Khargonekar, Mohammad Abdullah Al Faruque
* Omni-Scene: Omni-Gaussian Representation for Ego-Centric Sparse-View Scene Reconstruction, Dongxu Wei, Zhiqi Li, Peidong Liu
* Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation, David T. Hoffmann, Syed Haseeb Raza, Hanqiu Jiang, Denis Tananaev, Steffen Klingenhoefer, Martin Meinke
* Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting, Jingyi Xu, Xieyuanli Chen, Junyi Ma, Jiawei Huang, Jintao Xu, Yue Wang, Ling Pei
* Rectification-specific Supervision and Constrained Estimator for Online Stereo Rectification, Rui Gong, Kim-Hui Yap, Weide Liu, Xulei Yang, Jun Cheng
* Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction, Xiaolu Liu, Ruizi Yang, Song Wang, Wentong Li, Junbo Chen, Jianke Zhu
* Continuous Locomotive Crowd Behavior Generation, Inhwan Bae, Junoh Lee, Hae-Gon Jeon
* Don't Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving, Ziying Song, Caiyan Jia, Lin Liu, Hongyu Pan, Yongchang Zhang, Junming Wang, Xingyu Zhang, Shaoqing Xu, Lei Yang, Yadan Luo
* SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction, Kai Chen, Xiaodong Zhao, Yujie Huang, Guoyu Fang, Xiao Song, Ruiping Wang, Ziyuan Wang
* Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling, Guillem Capellera, Antonio Rubio, Luis Ferraz, Antonio Agudo
* EvOcc: Accurate Semantic Occupancy for Automated Driving Using Evidence Theory, Jonas Kälble, Sascha Wirges, Maxim Tatarchenko, Eddy Ilg
* GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction, Yuanhui Huang, Amonnut Thammatadatrakoon, Wenzhao Zheng, Yunpeng Zhang, Dalong Du, Jiwen Lu
* SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field for Autonomous Driving, Su Sun, Cheng Zhao, Zhuoyang Sun, Yingjie Victor Chen, Mei Chen
* DriveGEN: Generalized and Robust 3D Detection in Driving via Controllable Text-to-Image Diffusion Generation, Hongbin Lin, Zilu Guo, Yifan Zhang, Shuaicheng Niu, Yafeng Li, Ruimao Zhang, Shuguang Cui, Zhen Li
* Graph-Based 3D Lane Detection from Monocular Images, Halil İbrahim Öztürk, Muhammet Esat Kalfaoğlu, Ozsel Kilinc
* UrbanCAD: Towards Highly Controllable and Photorealistic 3D Vehicles for Urban Scene Simulation, Yichong Lu, Yichi Cai, Shangzhan Zhang, Hongyu Zhou, Haoji Hu, Huimin Yu, Andreas Geiger, Yiyi Liao
* DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation, Tianyi Yan, Dongming Wu, Wencheng Han, Junpeng Jiang, Xia Zhou, Kun Zhan, Cheng-zhong Xu, Jianbing Shen
* Causal Composition Diffusion Model for Closed-loop Traffic Generation, Haohong Lin, Xin Huang, Tung Phan, David Hayden, Huan Zhang, Ding Zhao, Siddhartha Srinivasa, Eric Wolff, Hongge Chen
* Towards Autonomous Micromobility through Scalable Urban Simulation, Wayne Wu, Honglin He, Chaoyuan Zhang, Jack He, Seth Z. Zhao, Ran Gong, Quanyi Li, Bolei Zhou
* Towards Generalizable Trajectory Prediction using Dual-Level Representation Learning and Adaptive Prompting, Kaouther Messaoud, Matthieu Cord, Alexandre Alahi
* Multi-modal Knowledge Distillation-based Human Trajectory Forecasting, Jaewoo Jeong, Seohee Lee, Daehee Park, Giwon Lee, Kuk-Jin Yoon
* UniScene: Unified Occupancy-centric Driving Scene Generation, Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang, Tiancai Wang, Shuchang Zhou, Li Zhang, Xiaojuan Qi, Hao Zhao, Mu Yang, Wenjun Zeng, Xin Jin
* SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment, Katrin Renz, Long Chen, Elahe Arani, Oleg Sinavski
* FreeSim: Toward Free-viewpoint Camera Simulation in Driving Scenes, Lue Fan, Hao Zhang, Qitai Wang, Hongsheng Li, Zhaoxiang Zhang
* TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception, Zhiying Song, Lei Yang, Fuxi Wen, Jun Li
* MPDrive: Improving Spatial Understanding with Marker-Based Prompt Learning for Autonomous Driving, Zhiyuan Zhang, Xiaofan Li, Zhihao Xu, Wenjie Peng, Zijian Zhou, Miaojing Shi, Shuangping Huang
* ZeroVO: Visual Odometry with Minimal Assumptions, Lei Lai, Zekai Yin, Eshed Ohn-Bar
* InteractionMap: Improving Online Vectorized HDMap Construction with Interaction, Kuang Wu, Chuan Yang, Zhanbin Li
* T2SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving, Changsheng Lv, Mengshi Qi, Liang Liu, Huadong Ma
* Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments, Luke Rowe, Roger Girgis, Anthony Gosselin, Liam Paull, Christopher Pal, Felix Heide
* Leveraging SD Map to Augment HD Map-based Trajectory Prediction, Zhiwei Dong, Ran Ding, Wei Li, Peng Zhang, Guobin Tang, Jia Guo
* Enduring, Efficient and Robust Trajectory Prediction Attack in Autonomous Driving via Optimization-Driven Multi-Frame Perturbation Framework, Yi Yu, Weizhen Han, Libing Wu, Bingyi Liu, Enshu Wang, Zhuangzhuang Zhang
* CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-Scale Reinforcement Learning in Autonomous Driving, Dongkun Zhang, Jiaming Liang, Ke Guo, Sha Lu, Qi Wang, Rong Xiong, Zhenwei Miao, Yue Wang
* DriveGPT4-V2: Harnessing Large Language Model Capabilities for Enhanced Closed-Loop Autonomous Driving, Zhenhua Xu, Yan Bai, Yujia Zhang, Zhuoling Li, Fei Xia, Kwan-Yee K. Wong, Jianqiang Wang, Hengshuang Zhao
* Sim-to-Real Causal Transfer: A Metric Learning Approach to Causally-Aware Interaction Representations, Ahmad Rahimi, Po-Chien Luan, Yuejiang Liu, Frano Rajič, Alexandre Alahi
* MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation, Yuxiang Fu, Qi Yan, Lele Wang, Ke Li, Renjie Liao
* RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives, Chirag Parikh, Deepti Rawat, Rakshitha R. T., Tathagata Ghosh, Ravi Kiran Sarvadevabhatla
* Hybrid Rendering for Multimodal Autonomous Driving: Merging Neural and Physics-Based Simulation, Máté Tóth, Péter Kovács, Zoltán Bendefy, Zoltán Hortsin, Tamás Matuszka

## **医学影像 (Medical Image)**


* **\[Oral\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation**, Aishik Konwer, Zhijian Yang, Erhan Bas, Cao Xiao, Prateek Prasanna, Parminder Bhatia, Taha Kass-Hout
* *\[Highlight\] Multi-modal Vision Pre-training for Medical Image Analysis*, Shaohao Rui, Lingzhi Chen, Zhenyu Tang, Lilong Wang, Mianxin Liu, Shaoting Zhang, Xiaosong Wang
* AeSPa: Attention-guided Self-supervised Parallel Imaging for MRI Reconstruction, Jinho Joo, Hyeseong Kim, Hyeyeon Won, Deukhee Lee, Taejoon Eo, Dosik Hwang
* SACB-Net: Spatial-awareness Convolutions for Medical Image Registration, Xinxing Cheng, Tianyang Zhang, Wenqi Lu, Qingjie Meng, Alejandro F. Frangi, Jinming Duan
* CXPMRG-Bench: Pre-training and Benchmarking for X-ray Medical Report Generation on CheXpert Plus Dataset, Xiao Wang, Fuling Wang, Yuehang Li, Qingchuan Ma, Shiao Wang, Bo Jiang, Jin Tang
* SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding, Ying Chen, Guoan Wang, Yuanfeng Ji, Yanjun Li, Jin Ye, Tianbin Li, Ming Hu, Rongshan Yu, Yu Qiao, Junjun He
* Learning Heterogeneous Tissues with Mixture of Experts for Gigapixel Whole Slide Images, Junxian Wu, Minheng Chen, Xinyi Ke, Tianwang Xun, Xiaoming Jiang, Hongyu Zhou, Lizhi Shao, Youyong Kong
* Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis, Yousef Yeganeh, Azade Farshad, loannis Charisiadis, Marta Hasny, Martin Hartenberger, Björn Ommer, Nassir Navab, Ehsan Adeli
* Unraveling Normal Anatomy via Fluid-Driven Anomaly Randomization, Peirong Liu, Ana Lawry Aguila, Juan E. Iglesias
* Blood Flow Speed Estimation with Optical Coherence Tomography Angiography Images, Wensheng Cheng, Zhenghong Li, Jiaxiang Ren, Hyomin Jeong, Congwu Du, Yingtian Pan, Haibin Ling
* Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation, Kang Liu, Zhuoqi Ma, Xiaolu Kang, Yunan Li, Kun Xie, Zhicheng Jiao, Qiguang Miao
* CPath-Omni: A Unified Multimodal Foundation Model for Patch and Whole Slide Image Analysis in Computational Pathology, Yuxuan Sun, Yixuan Si, Chenglu Zhu, Xuan Gong, Kai Zhang, Pingyi Chen, Ye Zhang, Zhongyi Shui, Tao Lin, Lin Yang
* BioX-CPath: Biologically-driven Explainable Diagnostics for Multistain IHC Computational Pathology, Amaya Gallagher-Syed, Henry Senior, Omnia Alwazzan, Elena Pontarini, Michele Bombardieri, Costantino Pitzalis, Myles J. Lewis, Michael R. Barnes, Luca Rossi, Gregory Slabaugh
* Robust Multimodal Survival Prediction with Conditional Latent Differentiation Variational AutoEncoder, Junjie Zhou, Jiao Tang, Yingli Zuo, Peng Wan, Daoqiang Zhang, Wei Shao
* Neuro-3D: Towards 3D Visual Decoding from EEG Signals, Zhanqiang Guo, Jiamin Wu, Yonghao Song, Jiahui Bu, Weijian Mai, Qihao Zheng, Wanli Ouyang, Chunfeng Song
* OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection, Max Gutbrod, David Rauber, Danilo Weber Nunes, Christoph Palm
* CheXwhatsApp: A Dataset for Exploring Challenges in the Diagnosis of Chest X-rays through Mobile Devices, Mariamma Antony, Rajiv Porana, Sahil M Lathiya, Siva Teja Kakileti, Chiranjib Bhattacharyya
* Bringing CLIP to the Clinic: Dynamic Soft Labels and Negation-Aware Learning for Medical Analysis, Hanbin Ko, Chang-Min Park
* Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation, Shahad Albastaki, Anabia Sohail, lyyakutti lyappan Ganapathi, Basit Alawode, Asim Khan, Sajid Javed, Naoufel Werghi, Mohammed Bennamoun, Arif Mahmood
* ODA-GAN: Orthogonal Decoupling Alignment GAN Assisted by Weakly-supervised Learning for Virtual Immunohistochemistry Staining, Tong Wang, Mingkang Wang, Zhongze Wang, Hongkai Wang, Qi Xu, Fengyu Cong, Hongming Xu
* STINR: Deciphering Spatial Transcriptomics via Implicit Neural Representation, Yisi Luo, Xile Zhao, Kai Ye, Deyu Meng
* Boltzmann Attention Sampling for Image Analysis with Small Objects, Theodore Zhao, Sid Kiblawi, Naoto Usuyama, Ho Hin Lee, Sam Preston, Hoifung Poon, Mu Wei
* Boosting the Dual-Stream Architecture in Ultra-High Resolution Segmentation with Resolution-Biased Uncertainty Estimation, Rong Qin, Xingyu Liu, Jinglei Shi, Liang Lin, Jufeng Yang
* DAMM-Diffusion: Learning Divergence-Aware Multi-Modal Diffusion Model for Nanoparticles Distribution Prediction, Junjie Zhou, Shouju Wang, Yuxia Tang, Qi Zhu, Daoqiang Zhang, Wei Shao
* DeformCL: Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image, Ziwei Zhao, Zhixing Zhang, Yuhang Liu, Zhao Zhang, Haojun Yu, Dong Wang, Liwei Wang
* MultiMorph: On-demand Atlas Construction, S. Mazdak Abulnaga, Andrew Hoopes, Neel Dey, Malte Hoffmann, Bruce Fischl, John Guttag, Adrian Dalca
* Anatomical Consistency and Adaptive Prior-informed Transformation for Multi-contrast MR Image Synthesis via Diffusion Model, Yejee Shin, Yeeun Lee, Hanbyol Jang, Geonhui Son, Hyeongyu Kim, Dosik Hwang
* A Unified Model for Compressed Sensing MRI Across Undersampling Patterns, Armeet Singh Jatyani, Jiayun Wang, Aditi Chandrashekar, Zihui Wu, Miguel Liu-Schiaffini, Bahareh Tolooshams, Anima Anandkumar
* CARL: A Framework for Equivariant Image Registration, Hastings Greer, Lin Tian, François-Xavier Vialard, Roland Kwitt, Raul San Jose Estepar, Marc Niethammer
* Domain Adaptive Diabetic Retinopathy Grading with Model Absence and Flowing Data, Wenxin Su, Song Tang, Xiaofeng Liu, Xiaojing Yi, Mao Ye, Chunxiao Zu, Jiahao Li, Xiatian Zhu
* HistoFS: Non-IID Histopathologic Whole Slide Image Classification via Federated Style Transfer with Rol-Preserving, Farchan Hakim Raswa, Chun-Shien Lu, Jia-Ching Wang
* Alignment, Mining and Fusion: Representation Alignment with Hard Negative Mining and Selective Knowledge Fusion for Medical Visual Question Answering, Yuanhao Zou, Zhaozheng Yin
* MedUnifier: Unifying Vision-and-Language Pre-training on Medical Data with Vision Generation Task using Discrete Visual Representations, Ziyang Zhang, Yang Yu, Yucheng Chen, Xulei Yang, Si Yong Yeo
* Multi-modal Medical Diagnosis via Large-small Model Collaboration, Wanyi Chen, Zihua Zhao, Jiangchao Yao, Ya Zhang, Jiajun Bu, Haishuai Wang
* Towards All-in-One Medical Image Re-Identification, Yuan Tian, Kaiyuan Ji, Rongzhao Zhang, Yankai Jiang, Chunyi Li, Xiaosong Wang, Guangtao Zhai
* FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models, Alice Heiman, Xiaoman Zhang, Emma Chen, Sung Eun Kim, Pranav Rajpurkar
* Interactive Medical Image Analysis with Concept-based Similarity Reasoning, Ta Duc Huy, Sen Kim Tran, Phan Nguyen, Nguyen Hoang Tran, Tran Bao Sam, Anton van den Hengel, Zhibin Liao, Johan W. Verjans, Minh-Son To, Vu Minh Hieu Phan
* BOE-VIT: Boosting Orientation Estimation with Equivariance in Self-Supervised 3D Subtomogram Alignment, Runmin Jiang, Jackson Daggett, Shriya Pingulkar, Yizhou Zhao, Priyanshu Dhingra, Daniel Brown, Qifeng Wu, Xiangrui Zeng, Xingjian Li, Min Xu
* DIN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels, Erjian Guo, Zhen Zhao, Zicheng Wang, Tong Chen, Yunyi Liu, Luping Zhou
* Q-PART: Quasi-Periodic Adaptive Regression with Test-time Training for Pediatric Left Ventricular Ejection Fraction Regression, Jie Liu, Tiexin Qin, Hui Liu, Yilei Shi, Lichao Mou, Xiao Xiang Zhu, Shiqi Wang, Haoliang Li
* OralXrays-9: Towards Hospital-Scale Panoramic X-ray Anomaly Detection via Personalized Multi-Object Query-Aware Mining, Bingzhi Chen, Sisi Fu, Xiaocheng Fang, Jieyi Cai, Boya Zhang, Minhua Lu, Yishu Liu
* DART: Disease-aware Image-Text Alignment and Self-correcting Re-alignment for Trustworthy Radiology Report Generation, Sang-Jun Park, Keun-Soo Heo, Dong-Hee Shin, Young-Han Son, Ji-Hye Oh, Tae-Eui Kam
* FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole Slide Image Classification, Zhengrui Guo, Conghao Xiong, Jiabo Ma, Qichen Sun, Lishuang Feng, Jinzhuo Wang, Hao Chen
* MERGE: Multi-faceted Hierarchical Graph-based GNN for Gene Expression Prediction from Whole Slide Histopathology Images, Aniruddha Ganguly, Debolina Chatterjee, Wentao Huang, Jie Zhang, Alisa Yurovsky, Travis Steele Johnson, Chao Chen
* STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection, Divya Velayudhan, Abdelfatah Ahmed, Mohamad Alansari, Neha Gour, Abderaouf Behouch, Taimur Hassan, Syed Talal Wasim, Nabil Maalej, Muzammal Naseer, Juergen Gall, Mohammed Bennamoun, Ernesto Damiani, Naoufel Werghi
* MEXD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification, Jianwei Zhao, Xin Li, Fan Yang, Qiang Zhai, Ao Luo, Yang Zhao, Hong Cheng, Huazhu Fu
* Advancing Multiple Instance Learning with Continual Learning for Whole Slide Imaging, Xianrui Li, Yufei Cui, Jun Li, Antoni B. Chan
* Multi-modal Topology-embedded Graph Learning for Spatially Resolved Genes Prediction from Pathology Images with Prior Gene Similarity Information, Hang Shi, Changxi Chi, Peng Wan, Daoqiang Zhang, Wei Shao
* BIOMEDICA: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature, Alejandro Lozano, Min Woo Sun, James Burgess, Liangyu Chen, Jeffrey J. Nirschl, Jeffrey Gu, Ivan Lopez, Josiah Aklilu, Anita Rau, Austin Wolfgang Katzer, Yuhui Zhang, Collin Chiu, Xiaohan Wang, Alfred Seunghoon Song, Robert Tibshirani, Serena Yeung-Levy

## **遥感影像 (Remote Sensing)**


* *\[Highlight\] Open-Canopy: Towards Very High Resolution Forest Monitoring*, Fajwel Fogel, Yohann Perron, Nikola Besic, Laurent Saint-André, Agnès Pellissier-Tanon, Martin Schwartz, Thomas Boudras, Ibrahim Fayad, Alexandre d'Aspremont, Loic Landrieu, Philippe Ciais
* The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generationf, Yanis Benidir, Nicolas Gonthier, Clement Mallet
* Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method, Pan Yin, Kaiyu Li, Xiangyong Cao, Jing Yao, Lei Liu, Xueru Bai, Feng Zhou, Deyu Meng
* ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object, Zhe Shan, Yang Liu, Lei Zhou, Cheng Yan, Heng Wang, Xia Xie
* RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark, Xin Zhang, Xue Yang, Yuxuan Li, Jian Yang, Ming-Ming Cheng, Xiang Li
* RobSense: A Robust Multi-modal Foundation Model for Remote Sensing with Static, Temporal, and Incomplete Data Adaptability, Minh Kha Do, Kang Han, Phu Lai, Khoa T. Phan, Wei Xiang
* Satellite to GroundScape \- Large-scale Consistent Ground View Generation from Satellite Views, Ningli Xu, Rongjun Qin
* Benchmarking Object Detectors under Real-World Distribution Shifts in Satellite Imagery, Sara A. Al-Emadi, Yin Yang, Ferda Ofli
* Cross-Rejective Open-Set SAR Image Registration, Shasha Mao, Shiming Lu, Zhaolong Du, Licheng Jiao, Shuiping Gou, Luntian Mou, Xuequan Lu, Lin Xiong, Yimeng Zhang
* HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery, Jingtao Li, Yingyi Liu, Xinyu Wang, Yunning Peng, Chen Sun, Shaoyu Wang, Zhendong Sun, Tian Ke, Xiao Jiang, Tangwei Lu, Anran Zhao, Yanfei Zhong
* U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for PAN-Sharpening, Sungpyo Kim, Jeonghyeok Do, Jaehyup Lee, Munchurl Kim
* Satellite Observations Guided Diffusion Model for Accurate Meteorological States at Arbitrary Resolution, Siwei Tu, Ben Fei, Weidong Yang, Fenghua Ling, Hao Chen, Zili Liu, Kun Chen, Hang Fan, Wanli Ouyang, Lei Bai
* Automatic Spectral Calibration of Hyperspectral Images: Method, Dataset and Benchmark, Zhuoran Du, Shaodi You, Cheng Cheng, Shikui Wei
* Feature Spectrum Learning for Remote Sensing Change Detection, Qi Zang, Dong Zhao, Shuang Wang, Dou Quan, Zhun Zhong
* Dual-Granularity Semantic Guided Sparse Routing Diffusion Model for General Pansharpening, Yinghui Xing, Litao Qu, Shizhou Zhang, Di Xu, Yingkun Yang, Yanning Zhang
* Hyperspectral Pansharpening via Diffusion Models with Iteratively Zero-Shot Guidance, Jin-Liang Xiao, Ting-Zhu Huang, Liang-Jian Deng, Guang Lin, Zihan Cao, Chao Li, Qibin Zhao
* Effective Cloud Removal for Remote Sensing Images by an Improved Mean-Reverting Denoising Model with Elucidated Design Space, Yi Liu, Wengen Li, Jihong Guan, Shuigeng Zhou, Yichao Zhang
* Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model, Jian Zhu, He Wang, Yang Xu, Zebin Wu, Zhihui Wei
* Adaptive Rectangular Convolution for Remote Sensing Pansharpening, Xueyang Wang, Zhixin Zheng, Jiandong Shao, Yule Duan, Liang-Jian Deng
* AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities, Guillaume Astruc, Nicolas Gonthier, Clément Mallet, Loic Landrieu

## **文字识别 (OCR)**


* Hierarchical Adaptive Filtering Network for Text Image Specular Highlight Removal, Zhi Jiang, Jingbo Hu, Ling Zhang, Gang Fu, Chunxia Xiao
* Linguistics-aware Masked Image Modeling for Self-supervised Scene Text Recognition, Yifei Zhang, Chang Liu, Jin Wei, Xiaomeng Yang, Yu Zhou, Can Ma, Xiangyang Ji
* SemiETS: Integrating Spatial and Content Consistencies for Semi-Supervised End-to-end Text Spotting, Dongliang Luo, Hanshen Zhu, Ziyang Zhang, Dingkang Liang, Xudong Xie, Yuliang Liu, Xiang Bai
* MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning, Wenhao Gu, Li Gu, Chingyee Yee Suen, Yang Wang
* CLIP is Almost All You Need: Towards Parameter-Efficient Scene Text Retrieval without OCR, Xugong Qin, Peng Zhang, Jun Jie Ou Yang, Gangyan Zeng, Yubo Li, Yuanyuan Wang, Wanqian Zhang, Pengwen Dai
* DreamText: High Fidelity Scene Text Synthesis, Yibin Wang, Weizhong Zhang, Honghui Xu, Cheng Jin
* OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations, Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, Jin Shi, Fan Wu, Pei Chu, Minghao Liu, Zhenxiang Li, Chao Xu, Bo Zhang, Botian Shi, Zhongying Tu, Conghui He
* Towards Natural Language-Based Document Image Retrieval: New Dataset and Benchmark, Hao Guo, Xugong Qin, Jun Jie Ou Yang, Peng Zhang, Gangyan Zeng, Yubo Li, Hailun Lin
* Accurate Scene Text Recognition with Efficient Model Scaling and Cloze Self-Distillation, Andrea Maracani, Savas Ozkan, Sijun Cho, Hyowon Kim, Eunchung Noh, Jeongwon Min, Cho Jung Min, Dookun Park, Mete Ozay
* Image Over Text: Transforming Formula Recognition Evaluation with Character Detection Matching, Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Botian Shi, Bo Zhang, Conghui He

# **八、学习范式与模型优化 (Learning Paradigms & Model Optimization)**
## **自监督学习 (Self-Supervised Learning)**

* Self-Supervised Spatial Correspondence Across Modalities, Ayush Shrivastava, Andrew Owens
* PSA-SSL: Pose and Size-aware Self-Supervised Learning on LIDAR Point Clouds, Barza Nisar, Steven L. Waslander
* Scaling Vision Pre-Training to 4K Resolution, Baifeng Shi, Boyi Li, Han Cai, Yao Lu, Sifei Liu, Marco Pavone, Jan Kautz, Song Han, Trevor Darrell, Pavlo Molchanov, Hongxu Yin
* Multimodal Autoregressive Pre-training of Large Vision Encoders, Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor G. Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua Susskind, Alaaeldin El-Nouby

## **对比学习 (Contrastive Learning)**

* *\[Highlight\] Breaking the Memory Barrier of Contrastive Loss via Tile-Based Strategy*, Zesen Cheng, Hang Zhang, Kehan Li, Sicong Leng, Zhiqiang Hu, Fei Wu, Deli Zhao, Xin Li, Lidong Bing
* Adapting to Observation Length of Trajectory Prediction via Contrastive Learning, Ruiqi Qiu, Jun Gong, Xinyu Zhang, Siqi Luo, Bowen Zhang, Yi Cen
* Link-based Contrastive Learning for One-Shot Unsupervised Domain Adaptation, Yue Zhang, Mingyue Bin, Yuyang Zhang, Zhongyuan Wang, Zhen Han, Chao Liang
* Perceptual Inductive Bias Is What You Need Before Contrastive Learning, Junru Zhao, Tianqin Li, Dunhan Jiang, Shenghao Wu, Alan Ramirez, Tai Sing Lee
* A Tale of Two Classes: Adapting Supervised Contrastive Learning to Binary Imbalanced Datasets, David Mildenberger, Paul Hager, Daniel Rueckert, Martin J. Menten

## **持续学习 (Continual Learning)**

* Unsupervised Continual Domain Shift Learning with Multi-Prototype Modeling, Haopeng Sun, Yingwei Zhang, Lumin Xu, Sheng Jin, Ping Luo, Chen Qian, Wentao Liu, Yiqiang Chen
* ProtoDepth: Unsupervised Continual Depth Completion with Prototypes, Patrick Rim, Hyoungseob Park, S. Gangopadhyay, Ziyao Zeng, Younjoon Chung, Alex Wong
* Boosting Domain Incremental Learning: Selecting the Optimal Parameters is All You Need, Qiang Wang, Xiang Song, Yuhang He, Jizhou Han, Chenhao Ding, Xinyuan Gao, Yihong Gong
* Ferret: An Efficient Online Continual Learning Framework under Varying Memory Constraints, Yuhao Zhou, Yuxin Tian, Jindi Lv, Mingjia Shi, Yuanxi Li, Qing Ye, Shuhao Zhang, Jiancheng Lv
* Learning Conditional Space-Time Prompt Distributions for Video Class-Incremental Learning, Xiaohan Zou, Wenchao Ma, Shu Zhao
* Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning, Takuma Fukuda, Hiroshi Kera, Kazuhiko Kawamoto
* Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping, Guannan Lai, Yujie Li, Xiangkun Wang, Junbo Zhang, Tianrui Li, Xin Yang
* Do Your Best and Get Enough Rest for Continual Learning, Hankyul Kang, Gregor Seifer, Donghyun Lee, Jongbin Ryu
* Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning, Huiyi Wang, Haodong Lu, Lina Yao, Dong Gong
* Task-Agnostic Guided Feature Expansion for Class-Incremental Learning, Bowen Zheng, Da-Wei Zhou, Han-Jia Ye, De-Chuan Zhan
* Reducing Class-wise Confusion for Incremental Learning with Disentangled Manifolds, Huitong Chen, Yu Wang, Yan Fan, Guosong Jiang, Qinghua Hu
* Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration, Yiyang Chen, Tianyu Ding, Lei Wang, Jing Huo, Yang Gao, Wenbin Li

## **联邦学习 (Federated Learning)**

* *\[Highlight\] Handling Spatial-Temporal Data Heterogeneity for Federated Continual Learning via Tail Anchor*, Hao Yu, Xin Yang, Le Zhang, Hanlin Gu, Tianrui Li, Lixin Fan, Qiang Yang
* Federated Learning with Domain Shift Eraser, Zheng Wang, Zihui Wang, Zheng Wang, Xiaoliang Fan, Cheng Wang
* AFL: A Single-Round Analytic Approach for Federated Learning with Pre-trained Models, Run He, Kai Tong, Di Fang, Han Sun, Ziqian Zeng, Haoran Li, Tianyi Chen, Huiping Zhuang
* Fortifying Federated Learning Towards Trustworthiness via Auditable Data Valuation and Verifiable Client Contribution, K Naveen Kumar, Ranjeet Ranjan Jha, C Krishna Mohan, Ravindra Babu Tallamraju
* Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised Learning Across Pseudo-Mismatch, Yijie Liu, Xinyi Shang, Yiqun Zhang, Yang Lu, Chen Gong, Jing-Hao Xue, Hanzi Wang
* Population Normalization for Federated Learning, Zhuoyao Wang, Fan Yi, Peizhu Gong, Caitou He, Cheng Jin, Weizhong Zhang

## **主动学习 (Active Learning)**

* Instance-wise Supervision-level Optimization in Active Learning, Shinnosuke Matsuo, Riku Togashi, Ryoma Bise, Seiichi Uchida, Masahiro Nomura
* Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach, Chen-Chen Zong, Sheng-Jun Huang
* Towards Cost-Effective Learning: A Synergy of Semi-Supervised and Active Learning, Tianxiang Yin, Ningzhong Liu, Han Sun

## **小样本学习 (Few-Shot Learning)**

* Single Domain Generalization for Few-Shot Counting via Universal Representation Matching, Xianing Chen, Si Huo, Borui Jiang, Hailin Hu, Xinghao Chen
* UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning, Long Zhou, Fereshteh Shakeri, Aymen Sadraoui, Mounir Kaaniche, Jean-Christophe Pesquet, Ismail Ben Ayed
* Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation, Zhaoyang Li, Yuan Wang, Wangkai Li, Tianzhu Zhang, Xiang Liu

## **知识蒸馏 (Knowledge Distillation)**

* **\[Oral\] Autoregressive Distillation of Diffusion Transformers**, Yeongmin Kim, Sotiris Anagnostidis, Yuming Du, Edgar Schönfeld, Jonas Kohler, Markos Georgopoulos, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu
* DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture, Qianlong Xiang, Miao Zhang, Yuzhang Shang, Jianlong Wu, Yan Yan, Liqiang Nie
* Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification, Dongseob Kim, Hyunjung Shim
* Efficient ANN-Guided Distillation: Aligning Rate-based Features of Spiking Neural Networks through Hybrid Block-wise Replacement, Shu Yang, Chengting Yu, Lei Liu, Hanzhi Ma, Aili Wang, Erping Li
* Temporal Separation with Entropy Regularization for Knowledge Distillation in Spiking Neural Networks, Kairong Yu, Chengting Yu, Tianqing Zhang, Xiaochen Zhao, Shu Yang, Hongwei Wang, Qiang Zhang, Qi Xu

## **模型量化 (Quantization)**

* *\[Highlight\] Multirate Neural Image Compression with Adaptive Lattice Vector Quantization*, Hao Xu, Xiaolin Wu, Xi Zhang
* Quantization without Tears, Minghao Fu, Hao Yu, Jie Shao, Junjie Zhou, Ke Zhu, Jianxin Wu
* Efficient Personalization of Quantized Diffusion Model without Backpropagation, Hoigi Seo, Wongi Jeong, Kyungryeol Lee, Se Young Chun

## **网络剪枝 (Network Pruning)**

* Flexible Group Count Enables Hassle-Free Structured Pruning, Jiamu Zhang, Shaochen Zhong, Andrew Ye, Zirui Liu, Sebastian Zhao, Kaixiong Zhou, Li Li, Soo-Hyun Choi, Rui Chen, Xia Hu, Shuai Xu, Vipin Chaudhary
* ICP: Immediate Compensation Pruning for Mid-to-high Sparsity, Xin Luo, Xueming Fu, Zihang Jiang, S. Kevin Zhou

## **网络架构搜索 (NAS)**

* L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers, Sofia Casarin, Sergio Escalera, Oswald Lanz
* NADER: Neural Architecture Design via Multi-Agent Collaboration, Zekang Yang, Wang Zeng, Sheng Jin, Chen Qian, Ping Luo, Wentao Liu
* NN-Former: Rethinking Graph Structure in Neural Architecture Representation, Ruihan Xu, Haokui Zhang, Yaowei Wang, Wei Zeng, Shiliang Zhang

## **轻量化模型 (Lightweight Models)**

* Binarized Neural Network for Multi-spectral Image Fusion, Junming Hou, Xiaoyu Chen, Ran Ran, Xiaofeng Cong, Xinyang Liu, Jian Wei You, Liang-Jian Deng
* Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing, Shiyang Zhou, Haijin Zeng, Yunfan Lu, Tong Shao, Ke Tang, Yongyong Chen, Jie Liu, Jingyong Su

# **九、其他 (Others)**

## **对抗性攻击与防御 (Adversarial Attack & Defense)**

* *\[Highlight\] Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems*, Song Xia, Yi Yu, Wenhan Yang, Meiwen Ding, Zhuo Chen, Ling-Yu Duan, Alex C. Kot, Xudong Jiang
* NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary, Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang
* Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients, Li Lun, Kunyu Feng, Qinglong Ni, Ling Liang, Yuan Wang, Ying Li, Dunshan Yu, Xiaoxin Cui
* PatchDEMUX: A Certifiably Robust Framework for Multi-label Classifiers Against Adversarial Patches, Dennis Jacob, Chong Xiang, Prateek Mittal
* Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning, Hasin Us Sami, Swapneel Sen, Amit K. Roy-Chowdhury, Srikanth V. Krishnamurthy, Basak Guler
* Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis, Jeonghwan Park, Niall McLaughlin, Ihsen Alouani
* Doppelgängers and Adversarial Vulnerability, George Kamberov
* PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection, Wei Li, Pin-Yu Chen, Sijia Liu, Ren Wang
* Rethinking the Adversarial Robustness of Multi-Exit Neural Networks in an Attack-Defense Game, Keyizhi Xu, Chỉ Zhang, Zhan Chen, Zhongyuan Wang, Chunxia Xiao, Chao Liang
* MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework, Ping Guo, Cheng Gong, Xi Lin, Fei Liu, Zhichao Lu, Qingfu Zhang, Zhenkun Wang
* A3: Few-shot Prompt Learning of Unlearnable Examples with Cross-Modal Adversarial Feature Alignment, Xuan Wang, Xitong Gao, Dongping Liao, Tianrui Qin, Yu-liang Lu, Cheng-zhong Xu

## **其他 (Others)**


* **\[Oral\] Do We Always Need the Simplicity Bias? Looking for Optimal Inductive Biases in the Wild**, Damien Teney, Liangze Jiang, Florin Gogianu, Ehsan Abbasnejad
* **\[Oral\] The PanAf-FGBG Dataset: Understanding the Impact of Backgrounds in Wildlife Behaviour Recognition**, Otto Brookes, Maksim Kukushkin, Majid Mirmehdi, Colleen Stephens, Paula Dieguez, Thurston C. Hicks, Sorrel Jones, Kevin Lee, Maureen S. McCarthy, Amelia Meier, Emmanuelle Normand, Erin G. Wessling, Roman M. Wittig, Kevin Langergraber, Klaus Zuberbühler, Lukas Boesch, Thomas Schmid, Mimi Arandjelovic, Hjalmar Kühl, Tilo Burghardt
* **\[Oral\] Temporally Consistent Object-Centric Learning by Contrasting Slots**, Anna Manasyan, Maximilian Seitzer, Filip Radovic, Georg Martius, Andrii Zadaianchuk
* *\[Highlight\] Seeing More with Less: Human-like Representations in Vision Models*, Andrey Gizdov, Shimon Ullman, Daniel Harari
* *\[Highlight\] DiffCAM: Data-Driven Saliency Maps by Capturing Feature Differences*, Xingjian Li, Qiming Zhao, Neelesh Bisht, Mostofa Rafid Uddin, Jin Yu Kim, Bryan Zhang, Min Xu
* Accurate Differential Operators for Hybrid Neural Fields, Aditya Chetan, Guandao Yang, Zichen Wang, Steve Marschner, Bharath Hariharan
* Learning Extremely High Density Crowds as Active Matters, Feixiang He, Jiangbei Yue, Jialin Zhu, Armin Seyfried, Dan Casas, Julien Pettré, He Wang
* Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset, Zhao Dong, Ka Chen, Zhaoyang Lv, Hong-Xing Yu, Yunzhi Zhang, Cheng Zhang, Yufeng Zhu, Stephen Tian, Zhengqin Li, Geordie Moffatt, Sean Christofferson, James Fort, Xiaqing Pan, Mingfei Yan, Jiajun Wu, Carl Yuheng Ren, Richard Newcombe
* A New Statistical Model of Star Speckles for Learning to Detect and Characterize Exoplanets in Direct Imaging Observations, Théo Bodrito, Olivier Flasseur, Julien Mairal, Jean Ponce, Maud Langlois, Anne-Marie Lagrange
* METASCENES: Towards Automated Replica Creation for Real-world 3D Scans, Huangyue Yu, Baoxiong Jia, Yixin Chen, Yandan Yang, Puhao Li, Rongpeng Su, Jiaxin Li, Qing Li, Wei Liang, Song-Chun Zhu, Tengyu Liu, Siyuan Huang
* GraphMimic: Graph-to-Graphs Generative Modeling from Videos for Policy Learning, Guangyan Chen, Te Cui, Meiling Wang, Chengcai Yang, Mengxiao Hu, Haoyang Lu, Yao Mu, Zicai Peng, Tianxing Zhou, Xinran Jiang, Yi Yang, Yufeng Yue
* Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes, Yiming Dou, Wonseok Oh, Yuqing Luo, Antonio Loquercio, Andrew Owens
* Rashomon Sets for Prototypical-Part Networks: Editing Interpretable Models in Real-Time, Jon Donnelly, Zhicheng Guo, Alina Jade Barnett, Hayden McTavish, Chaofan Chen, Cynthia Rudin
* Learning-enabled abled Polynomial Lyapunov Function Fun Synthesis via High-Accuracy Counterexample-Guided Framework, Hanrui Zhao, Niuniu Qi, Mengxin Ren, Banglong Liu, Shuming Shi, Zhengfeng Yang
* MIRE: Matched Implicit Neural Representations, Dhananjaya Jayasundara, Heng Zhao, Demetrio Labate, Vishal M. Patel
* Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry, Rui Wang, Shaocheng Jin, Ziheng Chen, Xiaoqing Luo, Xiao-Jun Wu
