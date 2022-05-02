# Anomaly Detection Algorithm for Keystroke Dynamics

## Background

In 2009, Killourhy and Maxion published a paper titled 'Comparing Anomaly-Detection Algorithms for Keystroke Dynamics'[1], which analysises typing rhythms to verify the identity of the person producing the keystrokes. It offers the potential to improve password-based authentication methods by introducing a biometric component, which could allow for the detection of imposters. The paper performs an analysis of various of various anomoly-detection algorithms using the same keystroke-dynamics data set. 

It is believed that by looking at a user's behaviour as they type a password, you can identify imposters (external or internal) just by looking at their keyboard actions. It is possible to create a machine learning (ML) model which is capable of recognising whether the password entered is actually by user based on their keystrokes. This concept falls into the class of anomaly detection as it is the identification of events which differ from the rest of the data. It is expected that genuine login attempts by the same user will be similar to other login attempts by the same user, but attempts by an imposter will register as an outlier.

### Dataset

The dataset used in the aforementioned paper is from 51 subjects typing 400 passwords each. The same password is typed by all subjects. The password, `.tie5Roanl` is a 10-character password containing letters, numbers, and punctuation. In subsequent years, this dataset has been referred to as the DSL2009 benchmark dataset, and will be referred to as such here. It is worth noting, that nowadays, when users are asked to generate a password, it is now common to include a special character, e.g. '$', '%', '#' etc.

The data was collected as follows: A laptop was set up with an external keyboard, and a Windows application was developed that prompts the subject to type the password. The application displayed the password in a screen with a text-entry field. The subject must type the password correctly and then press enter. If any errors are detected, the user is asked to re-type the password. The subject must type the password correctly 50 times to complete a data-collection session. Each subject completed 8 data-collection sessions, for a total of 400 password-typing samples. There was at least a day between sessions to capture day-to-day variation between each subject's typing.

Whenever a subject presses or releases a key, the application records the event (i.e keydown or keyup). An external reference clock with an accuracy of $\pm$200 microsecons was used.

The subjects consisted of 30 males and 21 females. 8 were left-handed, and 43 were right-handed. The median age froup was 31-40, the youngest was 18-20 and the oldest was 61-70. The subject's sessions took between 1.25 - 11 minutes, which the median session taking 3 minutes.

The dataset contains 34 features, of which 31 features are floats, 2 are ints, and 1 is an object. Inspection of the data shows there are no non-null (i.e. missing) entries. There is a total of 20400 entries (400 * 51). Please see dataset/dataset_features.txt for more in-depth discussion of the dataset.

### Plots

Before making any changes to the dataset, each variable has been plotted. Please note that producing these plots is slow, so these plots have been made once, stored in the 'plots' directory and then this part of the code has been commented out. Afterall, these plots only need to be made once, not everytime the code is run. 


### Using the Dataset with an Anomoly Detector

The raw typing data (e.g. keystrokes and timestamps) cannot be directly used by an anomoly detector. A set of timing features has been extracted from the raw data. The features are organised into a vector of times, called a `timing vector'. 


# Data Cleaning

All but one of the features can be trained on 'as is', except the feature 'subject' as this is an object, rather than an int or float. Each unique subject is replaced by an integer, e.g subject with ID s002 is replaced by 0.

Three data columns are dropped during training, subject, sessionIndex, and rep. Subject is the user ID, sessionIndex details which session the data was collected in (1-8), and rep, details the ith attempt at entering the password (1-50). Therefore, the NN only trains on time data.


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

The output layer uses the sigmoid activation function takes any real value as input and outputs values between 0-1. The larger (more positive) the input, the closer the value is to 1, and similarly, the smaller the input (more negative), the closer the output will be to 0.0.

The sigmoid function is given by:

$f(x) = \frac{1}{1+ e^{-x}}$

As such the NN will output an anomaly score between 0-1 for each user. Values closer to 1 are likely to be genuine users, and values closer to 0 are likely to be seen as imposters. 






























### References
[1] Kevin S. Killourhy and Roy A. Maxion. "Comparing Anomaly Detectors for Keystroke Dynamics," in Proceedings of the 39th Annual International Conference on Dependable Systems and Networks (DSN-2009), pages 125-134, Estoril, Lisbon, Portugal, June 29-July 2, 2009. IEEE Computer Society Press, Los Alamitos, California, 2009.
