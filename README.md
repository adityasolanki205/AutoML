# AutoML

This is a repository for **learning and implementation** of AutoML service on Google Cloud. In this repository we will learn to train and deploy an AutoML model for tabular data. The complete process is divided into43 parts:

1. **Dataset Creation**
2. **AutoML model Training**
3. **AutoML model Deployment**
4. **AutoML model Testing**


## Motivation
For the last few years, I have been part of a great learning curve wherein I have upskilled to move into a Machine Learning and Cloud Computing. This project was practice project for all the learnings I have had. This is one of the many more to come. 
 

## Libraries/framework used

<b>Built with</b>
- [AutoML - Tabular](https://cloud.google.com/vertex-ai/docs/tabular-data/overview)


## Repo Cloning

```bash
    git clone https://github.com/adityasolanki205/AUTOML.git
```

## Implementation

Below are the steps to setup the enviroment and run the models:

### Step 1 - Dataset Creation

-  **Data Selection**: First we will need a data for classification. Therefore We have selected Census income dataset which can downloaded from [UCI Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income). In this dataset we have to predict the income of an individual by looking a various attributes. 

-  **Train and Test dataset creation**: Now we will divide the dataset into Train and Test for evaluation(Training and Testing while model creation will be done by GCP). For testing we have around 32250 instances in the dataset. We will take the last 2250 instances separately and treat it as testing dataset for model evaluation. This data can be found [here](https://github.com/adityasolanki205/AUTOML/tree/main/Data) . We will create a Google cloud storage bucket by the name **automl-testing-table** in the region will create the model.

-  **Dataset Creation**: 
    
    i. Go to the Vertex AI tab and click on "Create Dataset"
    
    ii. Select Tabular Datatype
    
    iii. Select Regression/Classification objective
    
    iv. Click on Create
    
    v. Now we can upload csv file from local machine or pull a csv file from Cloud storage or pull data from bigquery table. We will pull csv file from local machine.
    
    vi. Now we will also select the Cloud Storage where the csv file will exist for the process. Here it will be **automl-testing-table**
    
    vii. Click on continue. This will upload the file in Datasets tab in Vertex AI

https://user-images.githubusercontent.com/56908240/211058129-78335d90-bc92-4b3b-8308-fc570aa8b6a9.mp4


### Step 2 - AutoML model Training

-  **Model Training**: Now we will try to train the model.

    i. Click on Train new model button.
    
    ii. Select Classification for objective.
    
    iii. In Model details select "Salary" column as target. 
    
    iv. Leave training options as is.
    
    v. Provide Maximum node hours as per the requirement and enable early stopping and Click on Start Training.
    
    vi. With this dataset Training takes around 6 to 8 hours. 

https://user-images.githubusercontent.com/56908240/211068315-23dbcfbb-3176-473d-b8b0-beb7d973d657.mp4

### Step 3 - AutoML model Deployment

-  **Endpoint Creation**:  

    i. Goto model repository.
    
    ii. Click on the Trained Model and its relavant version.
    
    iii. Click on Deploy and Test button.
    
    iv. Click on Deploy on endpoint.
    
    v. Fill in all the details as per the requirement.

https://user-images.githubusercontent.com/56908240/211163203-d713330e-d6cb-4097-bd32-715b8fb0497c.mp4

### Step 4 - AutoML model Testing
    
-  **Online Prediction**:
    
    i. Goto new endpoint created.
    
    ii. Below the endpoint we will see test your model tab with prefilled details. Either prefilled values could be used  to test the model, or we can provide our own values. There is a way to test through API as well. 
    
    iii. As soon as we click on Predict an output will be displayed along with confidence percentage

https://user-images.githubusercontent.com/56908240/211163159-c54d2ec2-6958-4364-b8d0-4856330d63fe.mp4

-  **Batch Prediction**:

https://user-images.githubusercontent.com/56908240/211206708-84aa8ec8-a812-4774-991c-d8d5f38357b9.mp4

## Credits
1. David Sandberg's facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)
2. Tim Esler's Git repo:[https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
3. Akash Nimare's README.md: https://gist.github.com/akashnimare/7b065c12d9750578de8e705fb4771d2f#file-readme-md
4. [Machine learning mastery](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)
