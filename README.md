# Anomaly Detection Algorithm for Keystroke Dynamics

## Background

In 2009, Killourhy and Maxion published a paper titled 'Comparing Anomaly-Detection Algorithms for Keystroke Dynamics' [1], which analysises typing rhythms to verify the identity of the person producing the keystrokes. It offers the potential to improve password-based authentication methods by introducing a biometric component, which could allow for the detection of imposters. The paper performs an analysis of various of various anomoly-detection algorithms using the same keystroke-dynamics data set. 

It is believed that by looking at a user's behaviour as they type a password, you can identify imposters (external or internal) just by looking at their keyboard actions. It is possible to create a machine learning (ML) model which is capable of recognising whether the password entered is actually by user based on their keystrokes. This concept falls into the class of anomaly detection as it is the identification of events which differ from the rest of the data. It is expected that genuine login attempts by the same user will be similar to other login attempts by the same user, but attempts by an imposter will register as an outlier.

What is being performed here, can be classed as One Class Classification (OOC) for anomaly detection. Anomalies are rare examples (imposters) that do not fit with the rest of the data. The aim is to build an algorithm that attempts to model 'normal' examples in order to then classify new examples as either normal or abnormal (anomaly). The aim of the model is to fit a model on normal data, and predict whether new data is normal or an anomaly. 

### Dataset

The dataset used in the aforementioned paper is from 51 subjects (users) typing 400 passwords each. The same password is typed by all subjects. The password, `.tie5Roanl` is a 10-character password containing letters, numbers, and punctuation. In subsequent years, this dataset has been referred to as the DSL2009 benchmark dataset [2], and will be referred to as such here. It is worth noting, that nowadays, when users are asked to generate a password, it is now common to include a special character, e.g. '$', '%', '#' etc.

The data was collected as follows: A laptop was set up with an external keyboard, and a Windows application was developed that prompts the subject to type the password. The application displayed the password in a screen with a text-entry field. The subject must type the password correctly and then press enter. If any errors are detected, the user is asked to re-type the password. The subject must type the password correctly 50 times to complete a data-collection session. Each subject completed 8 data-collection sessions, for a total of 400 password-typing samples. There was at least a day between sessions to capture day-to-day variation between each subject's typing.

Whenever a subject presses or releases a key, the application records the event (i.e keydown or keyup). An external reference clock with an accuracy of $ \pm $ 200 microseconds was used.

The subjects consisted of 30 males and 21 females. 8 were left-handed, and 43 were right-handed. The median age for the grroup was 31-40, the youngest was 18-20 and the oldest was 61-70. The subject's sessions took between 1.25 - 11 minutes, which the median session taking 3 minutes.

The dataset contains 34 features, of which 31 features are floats, 2 are ints, and 1 is an object. Inspection of the data shows there are no non-null (i.e. missing) entries. There is a total of 20400 entries (400 * 51). Please see `dataset/dataset_features.txt` for more in-depth discussion of the dataset.

### Method

The aim of the task is to implement and evaluate a ML approach for anomaly dectection using the aforemention dataset, and coded in Python. The evaluation setup needs to conform to the description at `www.cs.cmu.edu/~keystroke/`. For each of the 51 subjects, the first 200 typing attempts should be used to train a model, and the remaining 200*51 attempts should be used to compute the Equal Error Rate (EER) as a measure of performance. 

However, in Ref.[2], it is mentioned that training with the first 200 passwords for a user, has the disadvantage that there is strong intraclass variance that comes when the user gains more experience when typing the same password repeatedly, which will negatively affect the results. In real life, a person is not typing the same password 50 times in quick succession. A subject's ability to type the same password on the same computer will get better and better within a data collection session. To try and minimise this bias, it was elected to take a random sampling of a subject's 200 password typing attempts (i.e collecting data from across the 8 sessions rather than the first 4) as the training set, and the remaining 200 password attempts as the testing set. 

Training and testing on 100% imbalanced datasets (as the user is considered genuine), using traditional binary or multi-class classication leads to biases to the class with the larger number of instances, and so modelling and detecting instances of an imposter is very difficult [3]. Hence, OOC is an alternative approach to detect abnormal (imposter) data compared to genuine user data. Typically, the negative class is considered the normal class, and the positive class the abnormal class. However, here in the ML model that is used that is switched, and the negative class is considered the imposter class, and the positive class the genuine user class. 

One problem with anomaly detection is that typically there are severe class imbalances, as the number of positive classes (imposters) is much much smaller than the negative class (genuine users). In this instance, OCC can be used. In this type of binary classification, the ML model analyses the instances of only one class, which is usually the class of interest. The ML model trains and tests on a single class (genuine user), and then predicts on the anomalous (imposter) class.

Before anything, a number of plots are made of each timing variable in the dataset. There are 3 features in the dataset which are objects and not numbers, two of these are dropped later as they are not relevant for training. One of the features identifies the user in terms of 's0XX' where XX represents a number, and this column of data is converted to a number, with each subject getting a unique number. For example user 's002' is translated to 0. Although the user's identification is not trained on, it could be useful for any future work to have the user identification as a number.

The entire dataset is also normalised to be between 0-1. There are two ways to go out scaling all the numbers between 0-1. Either scale the entire (global) dataset, or scale each individual dataset that is a subset of the global dataset that gets created. It was felt it was better to scale the entire dataset. Lets say you have a human and a bot (imposter) each typing the same password, the bot will do its best to mimic the human. As such, the bot will try to have the same typing speeds as a human, and for the most part the bot could probably replicate that. However, when one wants to use a special character or a capital, the human user needs to hit the shift key, whereas a bot does not need to do that, it can just select the appropriate character from its 'library'. Hence, the time it takes a bot to enter a special character or capital takes less time. A clever bot would recognise the need to wait a little longer before selecting the appropriate character to mimic the time delay of a human. But it is likely to not be able to mimic this time delay as well as it takes a human to hit the shift key + character. The bot could over or underestimate. Subtle differences such as these help to tell a human apart from a bot. Normalising the entire dataset means that the bot/imposter will have the same distribution of times as the human, but the bots absolute time to enter the password will be different from the human. If I were to normalise each user's dataset, I would be normalising away the absolute time that makes a bot stick out compared to a human.

For each user, training, testing and prediction (EER) is performed. The full dataset is divided up into 51 datasets, grouped by the user (called subject in the dataset), hence, there are 51 datasets containing 400 passwords. Each user dataset is then further divided equally into two at random, with one considered the training dataset and the other the testing dataset. 

The training dataset is passed through a neural network (NN) which only contains one hidden layer. It was found that adding several hidden layers and including drop out (at rate 0.2) saw little to no improvement in the EER result. Hence, the network was kept simple. The input layer consists of 31 nodes, which is equal to the 31 timing-data features in the dataset. The hidden layer contains XX nodes. The output layer contains a single node. The input and hidden layer both use the same activation function (ReLu) and the output layer uses sigmoid. An activation function, $f(x)$ defines how the weighted sum of inputs is transformed into an output from a node or nodes in a layer in the network. Many activation functions are non-linear adding non-linearity to the network. Generally, all hidden layers of a NN will use the same activation function, with the output layer using a different activation function dependent on the prediction required by the model. Although there are many choices for activation function, the Rectified Linear Unit, or ReLu is one of the most popular for Neural Networks (NNs). It is least susceptible to vanishing gradients (although that should not really be a worry here as there are not many layers in the NN), and introduces non-linearity to the network. As we are using the ReLu activation function, the input data the NN has to be scaled to be between 0-1. The function returns 0 if the linear combination of inputs is negative, and for any positive value it returns that value back. This function is given by:

```math
$f(x) = max(0,x)$
```

As it is quite similar to calculate the function itself and its derivative, it is speeds up the training and testing process in comparision to other functions.

The output layer uses the sigmoid activation function takes any real value as input and outputs values between 0-1. The larger (more positive) the input, the closer the value is to 1, and similarly, the smaller the input (more negative), the closer the output will be to 0.0. If using binary cross entropy as the loss function, then the output layer must have one node and a sigmoid function to predict the probability for loss

The sigmoid function is given by:

$f(x) = \frac{1}{1+ e^{-x}}$

As such the NN will output an anomaly score between 0-1 for each user. Values closer to 1 are likely to be genuine users, and values closer to 0 are likely to be seen as imposters. 

As a test, the softmax activation function was also used in the output layer, however the results were comparable with sigmoid. However, softmax is mostly used with multi-label classification, which would also mean the loss function would need to be changed to categorical_crossentropy and the number of node outputs can be increased. But, categorial cross entropy is binary cross entropy when there is just 2 classes.



### Equal Error Rate (EER)

The equal error rate (EER) is defined as the intersection of the false accept rate (FAR) and the false reject rate (FRR), where FRR = 1 - TPR (true positive rate) and FRR = FPR (false positive rate). 

### Plots

Several different sets of plots are plotted.

TAKING THE ORIGINAL DATASET AND MAKING NO CHANGES TO IT:

1. Swarm Plots: Swarm plots of each user vs. variable is plotted as a swarm plot and a swarm plot with box plot overlayed. This shows the distribution of each variable for each user. Outliers can be seen, and averages can be ascertained.

2. 2D Plots: For each letter PPD vs RPD where PPD = H.key1 + UD.key1.key2  RPD = UD.key1 is plotted. It is expected that these are linearly related.

AFTER TRAINING, TESTING, PREDICTING...

3. ROC Curve: For each subject (user) a ROC curve is plotted of the true positive rate vs. false positive rate.

4. Equal Error Rate (EER): The EER per user is plotted. It plots both the False Accept Rate (FAR) and False Reject Rate (FRR). The rate at which the FAR and FRR are equal is known as the EER. Each user has their own EER assessed.

5. Loss Plot: For each user the loss and accuracy is plotted per epoch.

6. EER across users: A single EER bar chart is produced, which shows the EER for each user on one plot.

Looking at the loss plots helped with knowing how to adjust some of the training parameters.

 
### Using the Dataset with an Anomoly Detector

The raw typing data (e.g. keystrokes and timestamps) cannot be directly used by an anomoly detector. A set of timing features has been extracted from the raw data. The features are organised into a vector of times, called a `timing vector'. 


# Data Cleaning

All but one of the features can be trained on 'as is', except the feature 'subject' as this is an object, rather than an int or float. Each unique subject is replaced by an integer, e.g subject with ID s002 is replaced by 0.

Three data columns are dropped during training, subject, sessionIndex, and rep. Subject is the user ID, sessionIndex details which session the data was collected in (1-8), and rep, details the ith attempt at entering the password (1-50). Therefore, the NN only trains on time data.

No more data cleaning needs to be performed on this dataset. The data has already been cleaned (if it even needed to be) by the authors of the dataset.

# Training and Evaluating steps

There are four main steps to discriminate a single subject (designated as a genuine user) from the other 50 subjects (designated as imposters). After evaluating for a single subject, the four steps are repeated for each subject in the data set, so that each subject in turn will have been 'attacked' by each of the other 50 subjects.

Step One (training): Select at random 200 password timining features by a genuine user. Use a NN to build a detection model for the user's typing.

Step Two (genuine-user testing): Take the remaining 200 password timing features by a genuine user. Use the anomaly detections scoring system (sigmoid) function, and the dectection model from step one, to generate anomaly scores for these password-typing times. Record these anomly scores as `user scores`.

Step Three (imposter-user testing): Take the first 5 password timing features from each of the other 50 users (i.e all subjects other than the genuine user). Use the anomaly detectors scoring function and the detection model (step 1) to generate anomaly scores for these password-typing times. Records these anomoly scores as `imposter scores`.

Step Four (assessing performance): Employ the user scores and impostor scores to generate an ROC curve for the genuine user. Calculate, from the ROC curve, an equal-error rate, that is, the error rate corresponding to the point on the curve where the false-alarm (false-positive) rate and the miss (false-negative) rate are equal.

This process is then repeated, designating each of the other subjects as the genuine user in turn.

# Training Model

The model is as follows:

```
def nn_model(input_dim, output_dim=1, nodes=31):
	model = keras.Sequential()
	model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
	#model.add(Dropout(0.02))
	#model.add(Dense(nodes, activation='relu'))
	#model.add(Dropout(0.02))
	#model.add(Dense(nodes, activation='relu'))
	model.add(Dense(output_dim, activation='sigmoid'))
	optimiser = keras.optimizers.Adam(learning_rate = 0.00001) #default parameters used except for lr.
	model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy', tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives()])
	return model

model = nn_model(n_features, 1, 31)
history = model.fit(np.array(df_train_dict[subject]), np.zeros(df_train_dict[subject].shape[0]), epochs=200, batch_size=5)  
```
Each user has their own training dataset, and is labelled as normal data. This means passing in the training dataset (see np.array(df_train_dict[subject])) and an array of 0s which is equal to the rows in the training dataset. This is telling the model, all the training data can be considered as normal data (i.e contains no outliers/anomalies), please go learn this. 

It is a little clunky, but as a subject (user) is commonly denoted with the label 0, and an imposter with the label 1, the metrics returned mean that on the training data the number of true negatives should be 200 per training and the others (tpr, tnr, fnr) should be 0. The accuracy should also very quickly rise to 1... 

### Thoughts 

Subjects ability to type the same password will become easier and quicker the more often a password is typed. Eventually muscle memory will kick in and the typing of the password will be quicker.

At the start of a session a user is likely to type a password more slower than at other stages of the data-collection session, as they will get into a rythm as time goes on

Each user was using the same keyboard each time. In reality, people could be using different keyboards (e.g. at work, at home, on a laptop, on a computer, on a external keyboard) when entering the password, so this affect is not taken into account. In a more personal comment, I use both an external keyboard and the keyboard attached to my laptop. I have two external keyboards depending on where I am (Germany or the UK), and although my German keyboard has the physical layout of a German keyboard the input is actually British, so sometimes I press the wrong key because I forget which key is actually what as they are not the same.

If I am looking at my keyboard rather than screen, I am less likely to make a mistake when typing.

### Data Setup

Although it is requested to use the first 200 feature vectors of each subject (user) as training data, I have decided against this. This is because if you take the first 200 typing samples, there is likely to be strong intraclass variances that comes when the user is typing the same password repeatedly, and this will negatively affect the results, as they are likely to get quicker and more accurate at typing the same password repeatedly. Instead what I have done is taken 200 feature vectors for each subject at random from the 400 feature vectors that are available for each subject. The remaining 200 feature vectors are used as positive test data for each user. Then the first 5 samples from the remaining 50 subjects are used to form 250 negative feature vectors as imposters for the authentication phase for this user. The imposter's samples are never seen during the training time. As such there are 51 sets of trainings/testings performed, each using a different subjects data for training and testing. 

The authentication accuracy is evaluated using the equal error rate (EER), where the miss rate and false alarm rate are equal. The evaluation is performed for each subject. The mean and standard deviation of the EERs for the 51 subjects is also reported.

### Neural Network

The training data is quite limited, which could lead to problems with overfitting. As such, it is better to keep the network small to avoid the NN memorising the training set. The first layer in the NN is the input layer, which has 31 nodes, corresponding to the 31 features being trained on. The hidden layer consists of just 8 nodes, with the output layer consisting of a single node. 


### Activation Function

An activation function, $f(x)$ defines how the weighted sum of inputs is transformed into an output from a node or nodes in a layer in the network. Many activation functions are non-linear adding non-linearity to the network. Generally, all hidden layers of a NN will use the same activation function, with the output layer using a different activation function dependent on the prediction required by the model

There are two activation functions used here: ReLu and Sigmoid.

Although there are many choices for activation function, the Rectified Linear Unit, or ReLu is one of the most popular for Neural Networks (NNs). It is least susceptible to vanishing gradients (although that should not really be a worry here as there are not many layers in the NN), and introduces non-linearity to the network. As we are using the ReLu activation function, the input data the NN has to be scaled to be between 0-1. 

The function returns 0 if the linear combination of inputs is negative, and for any positive value it returns that value back. This function is given by:

$f(x) = max(0,x)$

As it is quite similar to calculate the function itself and its derivative, it is speeds up the training and testing process in comparision to other functions.

The output layer uses the sigmoid activation function takes any real value as input and outputs values between 0-1. The larger (more positive) the input, the closer the value is to 1, and similarly, the smaller the input (more negative), the closer the output will be to 0.0. If using binary cross entropy as the loss function, then the output layer must have one node and a sigmoid function to predict the probability for loss

The sigmoid function is given by:

$f(x) = \frac{1}{1+ e^{-x}}$

As such the NN will output an anomaly score between 0-1 for each user. Values closer to 1 are likely to be genuine users, and values closer to 0 are likely to be seen as imposters. 

As a test, the softmax activation function was also used in the output layer, however the results were comparable with sigmoid. However, softmax is mostly used with multi-label classification, which would also mean the loss function would need to be changed to categorical_crossentropy and the number of node outputs can be increased. But, categorial cross entropy is binary cross entropy when there is just 2 classes.




### Loss

NNs are trained using stochastic gradient descent optimisation algorithm. The error for the current state of the model must be estimated repeatedly so that weights can be updated to reduce the loss on the next evaluation. The type of loss function is dependent on whether we are doing regression or classification, and here we are doing the latter. We are trying classify whether a user is an imposter or not. The output layer must also be configured to be compatible with the loss function.

I have elected to use binary cross entropy here. Normally when using an entropy loss function then the correct (actual) labels, in this case the test dataset must be encoded as floating numbers, one hot, or an array of integers. The predicted labels must then be presented as a probability distribution. As we are using the sigmoid function in the last layer, the predicted labels are automatically converted to a probabilitiy distribution so I do not need to explicitly do this.


### Accuracy

Unsurprisingly the accuracy is 100% for the most part, as all the data being trained on has the same label, and so this is a very misleading metric, as each training session is completely imbalanced. Again, also pointlessly, one can easily guess what the tpr, tnr, fpr, and fnr are going to be, but for the sake of it, their values are still given, and the confusion matrix calculated for each training and plotted at the end.



### Improvements / Food for Thought


Validation dataset?
Regression?
LSTM? Not sure what I think of this. Ideally each row of data is supposed to be independent, but in reality it is not, and an LSTM would probably pick up on this, which isn't what you want.
Balanced labelled data? (i.e datasets labelled with 0 and 1 (although 1 is rare...))
Probably not relevant for current model with the datasets as they are, but back propagation? Drop out?


























### References
[1] Kevin S. Killourhy and Roy A. Maxion. "Comparing Anomaly Detectors for Keystroke Dynamics," in Proceedings of the 39th Annual International Conference on Dependable Systems and Networks (DSN-2009), pages 125-134, Estoril, Lisbon, Portugal, June 29-July 2, 2009. IEEE Computer Society Press, Los Alamitos, California, 2009.

[2] Su, Gordon. "Analysis of Keystroke Dynamics Algorithms with Feedforward Neural Networks", The Cooper Union for the Advancement of Science and Art, Albert Nerken School of Engineering, December 8 2020
"
[3] Naeem Seliya, Azadeh Abdollah Zadeh, Taghi M. Khosgoftaar. "A literature review on one-class classification and its potential applications in big data" in Journal of Big Data, 122 (2021)
