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

1. **Data Selection**: First we will need a data for classification. Therefore We have selected Census income dataset which can downloaded from [UCI Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income). In this dataset we have to predict the income of an individual by looking a various attributes. 

2. **Train and Test dataset creation**: Now we will divide the dataset into Train and Test for evaluation(Training and Testing while model creation will be done by GCP). For testing we have around 32250 instances in the dataset. We will take the last 2250 instances separately and treat it as testing dataset for model evaluation. This data can be found [here](https://github.com/adityasolanki205/AUTOML/tree/main/Data) . We will create a Google cloud storage bucket by the name **automl-testing-table** in the region will create the model.

3. **Dataset Creation**: 
    
    i. Go to the Vertex AI tab and click on "Create Dataset"
    
    ii. Select Tabular Datatype
    
    iii. Select Regression/Classification objective
    
    iv. Click on Create
    
    v. Now we can upload csv file from local machine or pull a csv file from Cloud storage or pull data from bigquery table. We will pull csv file from local machine.
    
    vi. Now we will also select the Cloud Storage where the csv file will exist for the process. Here it will be **automl-testing-table**
    
    vii. Click on continue. This will upload the file in Datasets tab in Vertex AI

https://user-images.githubusercontent.com/56908240/211058129-78335d90-bc92-4b3b-8308-fc570aa8b6a9.mp4


### Step 2 - AutoML model Training

1. **Model Training**: Now we will try to train the model.

<<<<<<< Updated upstream
    i. Click on 

https://user-images.githubusercontent.com/56908240/211066038-ccfd253b-f580-4c16-a6ad-e9907d21555b.mp4
=======
    i. Click on Train new model button.
    
    ii. Select Classification for objective.
    
    iii. In Model details select "Salary" column as target. 
    
    iv. Leave training options as is.
    
    v. Provide Maximum node hours as per the requirement and enable early stopping and Click on Start Training.
    
    vi. With this dataset Training takes around 8 hours.

>>>>>>> Stashed changes

### Step 3 - AutoML model Deployment

3. **Face Embeddings**: After face extraction we will fetch the face embedding using [FaceNet](https://github.com/davidsandberg/facenet). Downloaded the model [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn). After running this code for all the faces in train and test folders, we can save the embeddings using [np.saves_compressed](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)

```python
    # The Dimension of the input has to be increased as the model expects input in the form (Sample size, 160, 160,3)
    samples = np.expand_dims(image_pixels, axis = 0)
    
    # Use the Predict method to find the Embeddings in the face. Output would be 1D vector of 128 embeddings of that face
    embeddings = model.predict(samples)
```

### Step 4 - AutoML model Testing

4. **Training the SVM model on these Embeddings**:  Now we will train SVM model over the embeddings to predict the face of a person.

```python
    # We will use Linear SVM model to train over the embeddings
    model = SVC(kernel = 'linear', probability=True).fit(X_train,y_train)
```

5. **Predict the Face**: After the training of SVM model we will predict the face over test dataset.

```python
    # Preprocessing of the test photos have to be done like we did for Train and Validation photos
    image = np.asarray(image.convert('RGB'))
    
    # Now extract the face
    faces = MTCNN.detect_faces(image)
    
    # Extract embeddings
    embeddings = model.predict(samples)
    
    # At last we will predict the face embeddings
    SVM_model.predict(X_test)
```

## Tests
To test the code we need to do the following:

    1. Copy the photo to be tested in 'Test' subfolder of 'Data' folder. 
    Here I have used a photo of Elton John and Madonna
![](data/test/singers.jpg)
    
    2. Goto the 'Predict face in a group' folder.
    
    3. Open the 'Predict from a group of faces.ipynb'
    
    4. Goto filename variable and provide the path to your photo. Atlast run the complete code. 
    The recognised faces would have been highlighted and a photo would be saved by the name 'Highlighted.jpg'
![](final.jpg)

**Note**: The boundary boxes are color coded:

    1. Aditya Solanki  : Yellow
    2. Ben Affleck      : Blue   
    3. Elton John      : Green
    4. Jerry Seinfield : Red
    5. Madonna         : Aqua
    6. Mindy Kaling    : White
    
## How to use?
To run the complete code, follow the process below:

    1. Create Data Folder. 
    
    2. Create Sub folders as Training and Validation Dataset
    
    3. Create all the celebrity folders with all the required photos in them. 
    
    4. Run the Train and Test Data.ipynb file under Training Data Creation folder
    
    5. Save the output as numpy arrays
    
    6. Run the Face embedding using FaceNet.ipynb under the same folder name. This will create training data for SVM model
    
    7. Run the Predict from a group of faces.ipynb to recognise a familiar face

## Credits
1. David Sandberg's facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)
2. Tim Esler's Git repo:[https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
3. Akash Nimare's README.md: https://gist.github.com/akashnimare/7b065c12d9750578de8e705fb4771d2f#file-readme-md
4. [Machine learning mastery](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)
