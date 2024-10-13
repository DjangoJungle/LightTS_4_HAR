# Light TS Models for Human Activity Recognition

## Project Overview

This project aims to evaluate & develop lightweight time series models for Human Activity Recognition (HAR) tasks. 

HAR is a machine learning task that is widely used in health monitoring, smart home, sports analysis and other fields. It collects user activity data through wearable devices or sensors and identifies the user's current activity status based on these data. 

Given that the data of HAR tasks are usually high-frequency and multi-dimensional time series data, this project explores a series of efficient time series models, striving to strike a balance between accuracy and computational efficiency.

The models in the project implement the latest time series modeling methods. The project includes data processing, model training, testing and performance evaluation modules to facilitate the study and comparison of the performance of different models.



## List of TS Models

| Model        | Done? | arXiv                                                        | Year | Conference/Journal |
| ------------ | ----- | ------------------------------------------------------------ | ---- | ------------------ |
| TimeMixer    | ✅     | [TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting \| OpenReview](https://openreview.net/forum?id=7oLshfEIC2) | 2024 | ICLR               |
| ImputeFormer | ✅     | [[2312.01728] ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation (arxiv.org)](https://arxiv.org/abs/2312.01728) | 2024 | KDD                |
| TimesNet     | ✅     | [[2210.02186] TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (arxiv.org)](https://arxiv.org/abs/2210.02186) | 2023 | ICLR               |
| MICN         |       |                                                              | 2023 | ICLR               |
| RevIN_SCINet |       |                                                              | 2022 | ICLR               |
| Pyraformer   |       |                                                              | 2022 | ICLR               |
| Informer     |       |                                                              | 2021 | AAAI               |
| BTTF         |       |                                                              | 2021 | TPAMI              |
| GRU-D        |       |                                                              | 2018 | Sci. Rep.          |
| LOCF/NOCB    |       |                                                              | -    | Naive              |
| Mean         |       |                                                              | -    | Naive              |
| Median       |       |                                                              | -    | Naive              |

## How to Run

1. Before training the model, remember to place the UCI-HAR dataset in the `data/` folder.

   You can use `download_data/py`in the folder to download the dataset.

2. Train the model using:

   ```bash
   python experiments/train_mhnn.py
   ```

3. Test the model using:

   ```bash
   python test/data_analysis.py
   ```

4. There is a more detailed README file in each folder for a specific model.
