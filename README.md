# Anomaly Detection Algorithm for Keystroke Dynamics

## Background

In 2009, Killourhy and Maxion published a paper titled 'Comparing Anomaly-Detection Algorithms for Keystroke Dynamics'[1], which analysises typing rhythms to verify the identity of the person producing the keystrokes. It offers the potential to improve password-based authentication methods by introducing a biometric component, which could allow for the detection of imposters. The paper performs an analysis of various of various anomoly-detection algorithms using the same keystroke-dynamics data set. 

It is believed that by looking at a user's behaviour as they type a password, you can identify imposters (external or internal) just by looking at their keyboard actions. It is possible to create a machine learning (ML) model which is capable of recognising whether the password entered is actually by user based on their keystrokes. This concept falls into the class of anomaly detection as it is the identification of events which differ from the rest of the data. It is expected that genuine login attempts by the same user will be similar to other login attempts by the same user, but attempts by an imposter will register as an outlier.

### Dataset

The dataset used in the aforementioned paper is from 51 subjects typing 400 passwords each. The same password is typed by all subjects. The password, `.tie5Roanl` is a 10-character password containing letters, numbers, and punctuation. In subsequent years, this dataset has been referred to as the DSL2009 benchmark dataset, and will be referred to as such here. It is worth noting, that nowadays, when users are asked to generate a password, it is now common to include a special character, e.g. '$', '%', '#' etc.

The data was collected as follows: A laptop was set up with an external keyboard, and a Windows application was developed that prompts the subject to type the password. The application displayed the password in a screen with a text-entry field. The subject must type the password correctly and then press enter. If any errors are detected, the user is asked to re-type the password. The subject must type the password correctly 50 times to complete a data-collection session. Each subject completed 8 data-collection sessions, for a total of 400 password-typing samples. There was at least a day between sessions to capture day-to-day variation between each subject's typing.

Whenever a subject presses or releases a key, the application records the event (i.e keydown or keyup). An external reference clock with an accuracy of $\pm$200 microsecons was used.

The subjects consisted of 30 males and 21 females. 8 were left-handed, and 43 were right-handed. The median age froup was 31-40, the youngest was 18-20 and the oldest was 61-70. The subject's sessions took between 1.25 - 11 minutes, which the median session taking 3 minutes.

The dataset contains 34 features, of which 31 features are floats, 2 are ints, and 1 is an object. Inspection of the data shows there are no non-null (i.e. missing) entries. There is a total of 20400 entries (400 * 51).

RangeIndex: 20400 entries, 0 to 20399
Data columns (total 34 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   subject          20400 non-null  object 
 1   sessionIndex     20400 non-null  int64  
 2   rep              20400 non-null  int64  
 3   H.period         20400 non-null  float64
 4   DD.period.t      20400 non-null  float64
 5   UD.period.t      20400 non-null  float64
 6   H.t              20400 non-null  float64
 7   DD.t.i           20400 non-null  float64
 8   UD.t.i           20400 non-null  float64
 9   H.i              20400 non-null  float64
 10  DD.i.e           20400 non-null  float64
 11  UD.i.e           20400 non-null  float64
 12  H.e              20400 non-null  float64
 13  DD.e.five        20400 non-null  float64
 14  UD.e.five        20400 non-null  float64
 15  H.five           20400 non-null  float64
 16  DD.five.Shift.r  20400 non-null  float64
 17  UD.five.Shift.r  20400 non-null  float64
 18  H.Shift.r        20400 non-null  float64
 19  DD.Shift.r.o     20400 non-null  float64
 20  UD.Shift.r.o     20400 non-null  float64
 21  H.o              20400 non-null  float64
 22  DD.o.a           20400 non-null  float64
 23  UD.o.a           20400 non-null  float64
 24  H.a              20400 non-null  float64
 25  DD.a.n           20400 non-null  float64
 26  UD.a.n           20400 non-null  float64
 27  H.n              20400 non-null  float64
 28  DD.n.l           20400 non-null  float64
 29  UD.n.l           20400 non-null  float64
 30  H.l              20400 non-null  float64
 31  DD.l.Return      20400 non-null  float64
 32  UD.l.Return      20400 non-null  float64
 33  H.Return         20400 non-null  float64
dtypes: float64(31), int64(2), object(1)


### Using the Dataset with an Anomoly Detector

The raw typing data (e.g. keystrokes and timestamps) cannot be directly used by an anomoly detector. A set of timing features has been extracted from the raw data. The features are organised into a vector of times, called a `timing vector'. 



### Thoughts 

Subjects ability to type the same password will become easier and quicker the more often a password is typed. Eventually muscle memory will kick in and the typing of the password will be quicker.

At the start of a session a user is likely to type a password more slower than at other stages of the data-collection session, as they will get into a rythm as time goes on

Each user was using the same keyboard each time. In reality, people could be using different keyboards (e.g. at work, at home, on a laptop, on a computer, on a external keyboard) when entering the password, so this affect is not taken into account. In a more personal comment, I use both an external keyboard and the keyboard attached to my laptop. I have two external keyboards depending on where I am (Germany or the UK), and although my German keyboard has the physical layout of a German keyboard the input is actually British, so sometimes I press the wrong key because I forget which key is actually what as they are not the same.

If I am looking at my keyboard rather than screen, I am less likely to make a mistake when typing.

































### References
[1] Kevin S. Killourhy and Roy A. Maxion. "Comparing Anomaly Detectors for Keystroke Dynamics," in Proceedings of the 39th Annual International Conference on Dependable Systems and Networks (DSN-2009), pages 125-134, Estoril, Lisbon, Portugal, June 29-July 2, 2009. IEEE Computer Society Press, Los Alamitos, California, 2009.
