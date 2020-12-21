# Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [How to run Python scripts and web app](#run)
5. [Results](#results)
6. [Contributors](#Contributors)
7. [Licensing & Copyright](#licensing)
8. [Acknowledgements](#Acknowledgements)
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
There are two main reasons that I found this project so interesting:
- **Applicability of the outcome**

  following disaster typically there are millions of communications via different sources including social media and response organization has least capacity to filter and pull out the message which are not so important. different organization are responsible for specific issue (e.g. one department is in charge of flood response and the other department is in charge of water or block-roads or other kind of problems). Using Machine Learning Algorithms, the outcome of this project is categorize each message and label the message to the related issue. This will be so beneficial for quick response to the people affected by several kind of disasters.

- **Technical skillsets learning**

  Although Data Engineering and Data Science are two different job titles but it doesn't necessarily means that Data Scientist don't need to know anything about what Data Engineers are doing and vice versa. They have lots of overlaps tasks. In this project, I had this chance to not only challenge myself with ETL and ML pipelines but also I learned A LOT about NLP pipeline and also I got familiar with some important tools that Data Engineers are working with them. More specifically, I had this chance to work with FLASK and plotly.


---
## File Descriptions <a name="files"></a>
There are two initial **csv** files available for this project:
* disaster_categoris.csv
* disaster_messages.csv

in the first part of this project ETL pipeline has been implemented to Extract, Transfer and Load data (in SQL data base). the full description of steps and all codes/functions written for this phase has been summarized in **process_data.py**. in the second phase NLP and ML pipelines has been implemented for modelling. full description of this phase has been summarized in **train_classifier.py**. At the end the project was deployed in web app using FLASK and plotly.

## How to run Python scripts and web app? <a name="run"></a>
This project has three main parts:
- **Part 1: ETL pipeline**

  Two data sets (.csv) files has been provided as an input data: *disaster_categories.csv* and *disaster_messages.csv*. Using ETL pipe line, data will be extracted from these csv files and converted to pandas dataframe, cleaned up/transformed properly  and at the end loaded in database for second phase of the project. all the codes with enough information about each steps has been summarized in **data/process_data.py** file in this repository. The final output is database with the table of cleaned data inside of that database.

- **Part 2: NLP and ML pipelines**

  in the second phase, NLP (for further cleaning of text messages and extraction of features from each text message) and ML pipeline (for modelling) have been implemented on the output data coming from phase one. All the codes with detailed explanation of each part of of the code has been summarized in **models/train_classifier.py** of this repository. The output of this part is a pickle file used for web app in phase 3.

- **Part 3: Visualization of the outcome via web app**

  First make sure you have installed FLAST and Plotly. in your terminal navigate to the folder which includes **run.py** file. then in terminal write `python run.py`
  in separate terminal type `env|grep WORK`. The output of this command is the SPACEID.

  In final step, open new web browser and type the following:
  https://SPACEID-3001.SPACEDOMAIN and instead of SPACEID write what you have gotten from `env|grep WORK`
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
