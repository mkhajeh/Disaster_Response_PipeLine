# Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Contributors](#Contributors)
6. [Licensing & Copyright](#licensing)
7. [Acknowledgements](#Acknowledgements)
---
## Installation<a name="installation"></a>
Here is the list of main libraries that I have used for this project:
* Numpy
* Pandas
* sqlite 3
* sklearn
* re
* nltk
* matplotlib

more detailed about modules and functions used from each of these libraries could be found in *required libraries* section of python files.

The code should run with no issues using Python versions "3.8.3".

---
## Project Motivation<a name="motivation"></a>
Although Data Engineering and Data Science are two different job titles but it doesn't necessarily means that Data Scientist don't need to know anything about what Data Engineers are doing and vice versa. They have lots of overlaps tasks. In this project, I had this chance to not only challenge myself with ETL and ML pipelines but also I learned A LOT about NLP pipeline and also I got familiar with some important tools that Data Engineers are working with them. More specifically, I had this chance to work with FLASK and plotly.


---
## File Descriptions <a name="files"></a>
There are two initial **csv** files available for this project:
* disaster_categoris.csv
* disaster_messages.csv

in the first part of this project ETL pipeline has been implemented to Extract, Transfer and Load data (in SQL data base). the full description of steps and all codes/functions written for this phase has been summarized in **process_data.py**. in the second phase NLP and ML pipelines has been implemented for modelling. full description of this phase has been summarized in **train_classifier.py**. At the end the project was deployed in web app using FLASK and plotly

---
## Results <a name="results"></a>
The final outcome of this project was a dashboard developed in FLASK to show the plots of distribution of each of labels considered in this project (in total there are 36 lables but here only 4 of them will be displayed.)


---
## Contributors <a name="Contributors"></a>
Mehdi Khajeh
* email: <khajeh@ualberta.ca>
* cellphone: 403-667-8048
---
## License & Copyright <a name="licensing"></a>

&copy; Mehdi Khajeh 2017
Licensed under [MIT License](License)

---
## Acknowledgements <a name="Acknowledgements"></a>
Thanks to Udacity for defining this great project. As a result of this project, I forced myself to learn more about Machine Learning and for more than 80 hrs., I kept myself busy with practicing different ML algorithms. This project was among the most comprehensive projects that I did so far with Udacity NanoDegree program. I trained myself with lots of subjects which were not necessarily required to complete this project but I thought it is definitely worthwhile to consider some time for this purpose. Thank You Udacity!


---
