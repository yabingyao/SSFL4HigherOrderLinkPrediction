# SSFL
**Yao Y, Ti Z, Xu Z, et al.**[Subgraph Structure Feature Learning for Triangle Clique Prediction in Complex Networks](10.1109/TNSE.2025.3566227)[J].IEEE Transactions on Network Science and Engineering, 2025.
## Abstract
Link prediction is a critical task in network analysis, widely used to infer potential relationships between nodes.
While traditional methods focus on pairwise interactions, rea-world networks often exhibit higher-order interactions involving multiple nodes, such as 3-clique (triangle), which play a crucial role in understanding tightly-knit groups and complex network dynamics. In this paper, we propose a triangle clique prediction method based on Subgraph Structure Feature Learning (SSFL), which focuses on triangle structures in a network for prediction. In detail, it extracts the one-hop neighborhood around a target 3-clique, encodes it as a enclosing subgraph, and represents its structural features as a vector. These feature vectors are then processed using a fully connected neural network to predict 3-clique formations effectively. Experimental results show that the proposed method outperforms similarity-based link prediction methods and demonstrates comparable performance to embedding-based and machine learning-based approaches across various datasets. Our work can not only directly predict 3-clique structures in a network, but also provides insights into better understanding the evolution mechanism of networks.
## Method Overview
![ccc_00](https://github.com/user-attachments/assets/c3d7bd7f-0d81-4f6a-aba7-ae4c9492ac67)
Figure：Overall framework of SSFL for link prediction.
## Citing
If you find MCAS useful in your research, please consider citing the following paper：
```bibtex
@article{SSFL2025Yao,
  title={Subgraph Structure Feature Learning for Triangle Clique Prediction in Complex Networks},
  author={Yao, Yabing and Mao, Zhiheng and He, Yangyang and Xu, Zhipeng and Ti, Ziyu and Guo, Pingxia and Nian, Fuzhong and Ma, Ning},
  journal={IEEE Transactions on Network Science and Engineering},
  year={2025},
  publisher={IEEE},
  doi={10.1109/TNSE.2025.3566227},
  url={https://ieeexplore.ieee.org/document/10985838}
}
