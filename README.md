**Read ME**

The folder contains:

|1|model|Contains the GSMRecIT model structure|
| :- | :- | :- |
|2|Data files|Contains Game, Twitch dataset files, and experimental results|
|3|Experimental code|2 folders respectively for the main program + Baseline experiment .ipynb file of Game, Twitch dataset|
|4|Environment|Environment file|


**Data Files**

1. The original data files in the Game folder, a total of 6 files are preprocessed by the original author. The files starting with te and tr are the test dataset and training set divided by the author. In the experiment, we will merge them and reshuffle them. The file contents are the items corresponding to the user, interval (item-delta-time), duration (item-duration).

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.001.png)

After preprocessing in the preprocess.ipynb in the data preprocessing folder, you can get the input data of the model experiment. There are three files, namely the items in the user sequence, Duration, Interval.

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.002.png)

2. The Twitch folder contains 100,000 original data files of users: 100k_a.csv. After preprocessing with preprocessing.ipynb, you can get the input data of the model experiment. There are three files, namely the items in the user sequence, Duration, Interval.![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.003.png)

**Experimental Code**

There are two folders corresponding to the Game and Twitch datasets, each with a main model + 7 Baselines for a total of 8 files. More detailed settings are written in the comments of the code. The current program parameter setting is the best parameter. If you need to perform sensitivity analysis, search for that parameter in the program execution file and directly change the number.

Import file

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.004.png)

Result display

The model output results are as shown in the figure below. It will generate Accuracy, HR@K, MRR@K, NDCG@K and other evaluation indicators for the three prediction tasks (Item, Duration, Interval). Among them, _it is the item, _du is duration, _int is interval.
![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.005.png)

**Environment**

- Experimental environment: environment.yml

`	`Usage: Anaconda -> import -> environment.yml

- Program usage instructions: Open Anaconda, select environment (pytorch1), select IDE (jupyter), enter the root directory, select the program and open it directly
- Program environment setup instructions:
  - Steps for using the computer in Professor Yanliang’s laboratory are as follows:
1. Open Anaconda 
2. Click Environments

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.006.png)

3. Click import at the bottom 
4. Select environment.yml in the **Environment** folder
5. Take the environment name
6. After setting up the environment, select the newly created environment name in the drop-down menu of Applications on at the top

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.007.png)

7. Select Jupyter program and click Launch to open, enter the following screen

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.008.png)

8. Click Upload, enter the following screen, go to the code folder to select the ipynb file to be executed

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.009.png)



9. Select the ipynb file to be imported, after importing, it will be in the list, click Upload

   ![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.010.png)

10. Click the file in the list to open, enter the following screen and you can start running the program

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.011.png)

11. Click Cell->Run All to execute (you can also press shift+enter to execute each cell to view the program of that cell)

   ![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.012.png)

Contact information:

Student: 王茂田

Phone: 0932173776

LINE: arvin0503

Email: arvinwang806@gmail.com
