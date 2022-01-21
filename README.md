## SAM: Self-Augmentation Mechanism for COVID-19 Detection using Chest X-Ray Images 
#### Authors: Usman Muhammad, Md. Ziaul Hoque, Mourad Oussalah, Anja Keskinarkaus, Tapio Seppänen and Pinaki Sarder
#### Journal: Knowledge Based Systems - Elsevier [Special Issue on Deep Learning]
##

### Highlights
• Feature augmentation is introduced to mitigate the current lack of sufficient annotated data.

• A combined CNN-BiLSTM is employed for the diagnosis of COVID-19 in a robust manner.

• Experimental results demonstrate state-of-the-art performance on three COVID-19 databases.

• PCA and t-SNE feature visualization has been utilized for the explainability of the proposed learning model.

### Abstract
COVID-19 is a rapidly spreading viral disease and has affected over 100 countries worldwide. The numbers of casualties and cases of infection have escalated particularly in countries with weakened healthcare systems. Recently, reverse transcription-polymerase chain reaction (RT-PCR) is the test of choice for diagnosing COVID-19. However, current evidence suggests that COVID-19 infected patients are mostly stimulated from a lung infection after coming in contact with this virus. Therefore, chest X-ray (i.e., radiography) and chest CT can be a surrogate in some countries where PCR is not readily available. This has forced the scientific community to detect COVID-19 infection from X-ray images and recently proposed machine learning methods offer great promise for fast and accurate detection. Deep learning with convolutional neural networks (CNNs) has been successfully applied to radiological imaging for improving the accuracy of diagnosis. However, the performance remains limited due to the lack of representative X-ray images available in public benchmark datasets. To alleviate this issue, we propose a self-augmentation mechanism for data augmentation in the feature space rather than in the data space using reconstruction independent component analysis (RICA). Specifically, a unified architecture is proposed which contains a deep convolutional neural network (CNN), a feature augmentation mechanism, and a bidirectional LSTM (BiLSTM). The CNN provides the high-level features extracted at the pooling layer where the augmentation mechanism chooses the most relevant features and generates low-dimensional augmented features. Finally,  BiLSTM is used to classify the processed sequential information. We conducted experiments on three publicly available databases to show that the proposed approach achieves the state-of-the-art results with accuracy of 97%, 84% and 98%. Explainability analysis has been carried out using feature visualization through PCA projection and t-SNE plots.

#### This research is made available to the research community. If you are using this code please cite the following paper:                              
Usman Muhammad, Md. Ziaul Hoque, Mourad Oussalah, Anja Keskinarkaus, Tapio Seppänen, Pinaki Sarder, SAM: Self-augmentation mechanism for COVID-19 detection using chest X-ray images, Knowledge-Based Systems, 2022, 108207, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2022.108207.

### BibTeX
@article{MUHAMMAD2022108207,
title = {SAM: Self-augmentation mechanism for COVID-19 detection using chest X-ray images},
journal = {Knowledge-Based Systems},
pages = {108207},
year = {2022},
issn = {0950-7051},
author = {Usman Muhammad and Md. Ziaul Hoque and Mourad Oussalah and Anja Keskinarkaus and Tapio Seppänen and Pinaki Sarder}, publisher={Elsevier}
}

### Links
• https://doi.org/10.1016/j.knosys.2022.108207

• https://www.sciencedirect.com/science/article/pii/S0950705122000545

• https://doi.org/10.36227/techrxiv.16574990.v1
