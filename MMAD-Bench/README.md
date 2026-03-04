---
license: cc-by-nc-sa-4.0
task_categories:
- question-answering
tags:
- Anomaly Detection
- MLLM
size_categories:
- 10K<n<100K
dataset_info:
# - config_name: viewer
#   features:
#   - name: question
#     dtype: string
#   - name: options
#     dtype: string
#   - name: answer
#     dtype: string
#   - name: query_image
#     dtype: image
#   - name: template_image
#     dtype: image
#   - name: mask
#     dtype: image
configs:
- config_name: viewer
  data_files: "metadata.csv"
---

# MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2410.09453) 
[![github](https://img.shields.io/badge/Code-Github-blue)](https://github.com/jam-cc/MMAD)

## 💡 This dataset is the full version of MMAD
- **Content**：Containing both questions, images, and captions.
  
- **Questions**: All questions are presented in a multiple-choice format with manual verification, including options and answers.
  
- **Images**：Images are collected from the following links：
[DS-MVTec](https://huggingface.co/datasets/DefectSpectrum/Defect_Spectrum/tree/main/DS-MVTec)
, [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
, [MVTec-LOCO](https://www.mvtec.com/company/research/datasets/mvtec-loco)
, [VisA](https://github.com/amazon-science/spot-diff)
, [GoodsAD](https://github.com/jianzhang96/GoodsAD).
We retained the mask format of the ground truth to facilitate future evaluations of the segmentation performance of multimodal large language models.

- **Captions**：Most images have a corresponding text file with the same name in the same folder, which contains the associated caption. Since this is not the primary focus of this benchmark, we did not perform manual verification. Although most captions are of good quality, please use them with caution.
  
## 👀 Overview
In the field of industrial inspection, Multimodal Large Language Models (MLLMs) have a high potential to renew the paradigms in practical applications due to their robust language capabilities and generalization abilities. However, despite their impressive problem-solving skills in many domains, MLLMs' ability in industrial anomaly detection has not been systematically studied. To bridge this gap, we present MMAD, the first-ever full-spectrum MLLMs benchmark in industrial Anomaly Detection. We defined seven key subtasks of MLLMs in industrial inspection and designed a novel pipeline to generate the MMAD dataset with 39,672 questions for 8,366 industrial images. With MMAD, we have conducted a comprehensive, quantitative evaluation of various state-of-the-art MLLMs.

 Our benchmark responds to the following questions:
- How well are current MLLMs performing as industrial quality inspectors?
- Which MLLM performs the best in industrial anomaly detection? 
- What are the key challenges in industrial anomaly detection for MLLMs?

## 🕹️ How to evaluate
Please refer to the ['evaluation/examples'](https://github.com/jam-cc/MMAD/tree/main/evaluation/examples) folder in our [GitHub repository](https://github.com/jam-cc/MMAD).

## 🥹 BibTex Citation

If you find this paper and repository useful for your study, please cite our paper☺️.

```bibtex
@inproceedings{Jiang2024MMADTF,
  title={MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection},
  author={Xi Jiang and Jian Li and Hanqiu Deng and Yong Liu and Bin-Bin Gao and Yifeng Zhou and Jialin Li and Chengjie Wang and Feng Zheng},
  year={2024},
  journal={arXiv preprint arXiv:2410.09453},
}
```