
# Project overview

The aim of this project was to design a software that distinguishes between at least 5 different music genres. 

The GTZAN music dataset was used in this project (http://marsyas.info/downloads/datasets.html). 

The project was accomplished by extracting mel frequency cepstral coefficients (MFCC) from short music files. No signal processing libraries were used to solve this task. A naive Bayes classifier is trained with the derived data. It then determines a validated audio's genre.

# Results

Total of 7 different music genres from the dataset were selected. Classifier's performance was measured with confusion matrix for songs that were selected from 4, 5, 6 and 7 different genres of music. The results are shown below.

<p align="center">
  <img src="https://i.gyazo.com/8de151f8546dcea5362bea12a7a7f84f.png">
</p>


