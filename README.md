**Read ME**

The folder contains:

|1|model|Contains the GSMRecIT model structure|
| :- | :- | :- |
|2|Data files|Contains Game, Twitch dataset files, and experimental results|
|3|Experimental code|2 folders respectively for the main program + Baseline experiment .ipynb file of Game, Twitch dataset|


**Data Files**

1. The original data files in the Game folder, a total of 6 files are preprocessed by the original author. The files starting with te and tr are the test dataset and training set divided by the author. In the experiment, we will merge them and reshuffle them. The file contents are the items corresponding to the user, interval (item-delta-time), duration (item-duration).



After preprocessing in the preprocess.ipynb in the data preprocessing folder, you can get the input data of the model experiment. There are three files, namely the items in the user sequence, Duration, Interval.



2. The Twitch folder contains 100,000 original data files of users: 100k_a.csv. After preprocessing with preprocessing.ipynb, you can get the input data of the model experiment. There are three files, namely the items in the user sequence, Duration, Interval.

**Experimental Code**

There are two folders corresponding to the Game and Twitch datasets, each with a main model + 7 Baselines for a total of 8 files. More detailed settings are written in the comments of the code. The current program parameter setting is the best parameter. If you need to perform sensitivity analysis, search for that parameter in the program execution file and directly change the number.

Import file

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.004.png)

Result display

The model output results are as shown in the figure below. It will generate Accuracy, HR@K, MRR@K, NDCG@K and other evaluation indicators for the three prediction tasks (Item, Duration, Interval). Among them, _it is the item, _du is duration, _int is interval.
![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.005.png)

