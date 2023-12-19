**Read ME**

資料夾包含：

|一|model|內容為GSMRecIT模型程式架構|
| :- | :- | :- |
|二|資料檔案|內容包含Game、Twitch資料集檔案，和實驗結果|
|三|實驗程式碼|2個資料夾分別為Game、Twitch資料集的主程式+Baseline實驗.ipynb檔案|
|四|環境|環境檔|


**資料檔案**

1. Game資料夾中的資料原始檔案，一共6個檔案為原作者前處理完的檔案，其中te以及tr開頭的檔案為作者切分的測試資料集和訓練集，在實驗中我們會將其合併重新進行打亂，檔案內容分別為用戶對應的項目(item)，interval(item-delta-time)，duration(item-duration)。

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.001.png)

經過資料前處理資料夾中的preprocess.ipynb處理後，可以得到模型實驗的輸入資料共三個檔案分別為用戶序列中的項目、Duration、Interval。

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.002.png)

2. Twitch資料夾包含10萬筆用戶的原始資料檔案: 100k\_a.csv、經過資料前處理preprocessing.ipynb可以得到模型實驗的輸入資料共三個檔案分別為用戶序列中的項目、Duration、Interval。![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.003.png)

**實驗程式碼**

總共兩個資料夾分別對應Game、Twitch資料集，皆為一個主模型+7個Baseline總計8個檔案。更詳細的細節設置則寫在程式碼的註解。現在的程式參數設定為最佳參數，若需要進行敏感度分析，則在程式執行檔案中搜尋到那個參數，直接對數字進行更改即可。

導入檔案

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.004.png)

結果顯示

模型輸出結果如下圖所示，會針對三項預測任務(Item、Duration 、Interval)產生Accuracy、HR@K、MRR@K、NDCG@K等評估指標，其中\_it為項目，\_du為duration，\_int為interval。

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.005.png)

**環境**

- 實驗環境：environment.yml

`	`使用方式：Anaconda -> import -> environment.yml

- 程式使用說明：開Anaconda、選環境(pytorch1)、選IDE(jupyter)、進入根目錄、選擇程式直接開啟即可
- 程式環境建立說明：
  - 使用彥良老師實驗室內電腦步驟如下：
1. 開Anaconda 
2. 點Environments

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.006.png)

3. 點下面的import 
4. 選**環境**資料夾中的environment.yml
5. 取環境名稱
6. 建立好環境後，在最上面Applications on的下拉選單選擇剛建立的環境名稱

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.007.png)

7. 選Jupyter程式點Launch開啟，進入下圖畫面

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.008.png)

8. 點選Upload，進入下圖畫面，到程式碼資料夾中選擇要執行的ipynb檔

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.009.png)



9. 選擇要匯入的ipynb檔，匯入後會在列表中，按下Upload

   ![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.010.png)

10. 點選列表中的檔案開啟，進入下圖畫面即可開始執行程式

![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.011.png)

11. 點選Cell->Run All執行 (也可以針對各個cell按shift+enter 執行以檢視該cell的程式)

   ![](Aspose.Words.3cf5bdcf-dc8c-4701-81f6-2681ed0bf7fa.012.png)

聯絡方式：

學生：王茂田

手機：0932173776

LINE：arvin0503

Email：arvinwang806@gmail.com
