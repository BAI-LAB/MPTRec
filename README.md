# Efficient Multi-task Prompt Tuning for Recommendation

This repository contains the source code for the paper "Efficient Multi-task Prompt Tuning for Recommendation." The project aims to improve the performance of multi-task learning in recommendation systems and enable rapid generalization to new tasks by leveraging knowledge from previously learned tasks. The code related to multi-task learning is placed in the `two_task` branch, while the code for new task generalization is in the `three_task` branch.

## Environment Requirements
- Python Version: 3.10
- Other Library Versions: See [requirements.txt](requirements.txt)

## Running Instructions

### STEP 1: Prepare Dataset
1. Create a `dataset` directory in the root folder, and then create a `Census-income` directory inside it to store the Census-income dataset.
2. Download the dataset from [UCI Machine Learning Repository](http://archive.ics.uci.edu/dataset/117/census+income+kdd).
3. Preprocess the data using `multitaskrec/data_preprocessing/CensusIncome_process.py`.

### STEP 2: Multi-task Learning (Two Tasks)
1. Switch to the `two_task` branch:
   ```bash
   git checkout two_task
   ```
2. Run the `censusincome_main.py` script:
   ```bash
   python censusincome_main.py
   ```

### STEP 3: New Task Generalization (Pre-train on Two Tasks and Generalize to a New Task)
1. Switch to the three_task branch:
   ```bash
   git checkout three_task
   ```
2. Run the `CensusIncome_NewTask.py` script:
   ```bash
   python CensusIncome_NewTask.py
   ```

<span style="color: rgb(153, 204, 255);">**Other datasets and baselines follow a similar approach.**</span>

## Citation
If you find this research helpful, please cite our paper:
```bibtex
@article{bai2024efficient,
  title={Efficient Multi-task Prompt Tuning for Recommendation},
  author={Bai, Ting and Huang, Le and Yu, Yue and Yang, Cheng and Hou, Cheng and Zhao, Zhe and Shi, Chuan},
  journal={arXiv preprint arXiv:2408.17214},
  year={2024}
}
```
