# SSE-CortexEmbed
Code for ISBI 2025 paper "Using Structural Similarity and Kolmogorov-Arnold Networks for Anatomical Embedding of Cortical Folding Patterns"
## Introduction
### 3-hinge Gyrus
The cortical folding patterns contain underlying mechanisms of brain organization.
While the major cortical regions are largely consistent across individuals, the local shapes and patterns within these regions vary significantly, posing challenges for the quantitative and efficient characterization of cortical folding.
Recently, a finer scale cortical folding pattern, known as the 3-hinge gyrus (3HG), has been identified and defined as the junction where three gyri converge from different directions.
Notably, the 3HG is evolutionarily conserved across multiple primate species and remains stable in the human brain, regardless of population differences or brain conditions. 
This cortical folding pattern exhibits strong consistency within species while varying among individuals, with unique features such as the thickest corteces, higher DTI-derived fibers density,  and greater connectivity diversity across structural and functional domains compared with other gyral regions. This suggests that 3HGs are more like hubs in the cortical-cortical connection network and play a vital role in the global structural and functional networks of humans. 
And a recent study revealed that a finer-scale brain connectome based on 3HG can better capture the intricate patterns of Alzheimer's Disease.
### Contributions of this work
<img src="/img/fig1.png" width="600px"/>

In this paper, we propose a self-supervised framework for anatomical feature embedding of the 3HG based on our initial study, [Cortex2vector](https://academic.oup.com/cercor/article/33/10/5851/6880883).
We introduce structural similarity between independent nodes to enhance the hierarchical multi-hop encoding strategy.
To further improve the representation ability of the network  while keeping it lightweight, we adopt Kolmogorov–Arnold Networks (KAN), a recently proposed neural network inspired by the Kolmogorov-Arnold representation theorem, for anatomical feature encoding of 3HGs. In addition, we propose a new loss function -- selective reconstruction loss, which penalizes reconstruction errors in non-zero elements, thereby enhancing the representational capacity of the embedding vector.
Our experimental results show that the learned embeddings can accurately establish cross-subject correspondences in complex cortical landscapes, while also maintaining the commonality and variability inherent in 3HGs.
## Citation
If you use this code for your research, please cite our paper:
```
@article{chen2024using,
  title={Using Structural Similarity and Kolmogorov-Arnold Networks for Anatomical Embedding of 3-hinge Gyrus},
  author={Chen, Minheng and Cao, Chao and Chen, Tong and Zhuang, Yan and Zhang, Jing and Lyu, Yanjun and Yu, Xiaowei and Zhang, Lu and Liu, Tianming and Zhu, Dajiang},
  journal={arXiv preprint arXiv:2410.23598},
  year={2024}
}
```
