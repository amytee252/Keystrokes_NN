# Anomaly Detection Algorithm for Keystroke Dynamics

## Background

In 2009, Killourhy and Maxion published a paper titled 'Comparing Anomaly-Detection Algorithms for Keystroke Dynamics'[1], which analysises typing rhythms to verify the identity of the person producing the keystrokes. It offers the potential to improve password-based authentication methods by introducing a biometric component, which could allow for the detection of imposters. The paper performs an analysis of various of various anomoly-detection algorithms using the same keystroke-dynamics data set. 

### Dataset

The dataset used in the aforementioned paper is from 51 subjects typing 400 passwords each. The same password is typed by all subjects. The password, `.tie5Roanl' is a 10-character password containing letters, numbers, and punctuation. In subsequent years, this dataset has been referred to as the DSL2009 benchmark dataset, and will be referred to as such here. It is worth noting, that nowadays, when users are asked to generate a password, it is now common to include a special character, e.g. '$', '%', '#' etc.

The data was collected as follows: A laptop was set up with an external keyboard, and a Windows application was developed that prompts the subject to type the password. The application displayed the password in a screen with a text-entry field. The subject must type the password correctly and then press enter. If any errors are detected, the user is asked to re-type the password. The subject must type the password correctly 50 times to complete a data-collection session. Each subject completed 8 data-collection sessions, for a total of 400 password-typing samples. There was at least a day between sessions to capture day-to-day variation between each subject's typing.

The subjects consisted of 30 males and 21 females. 8 were left-handed, and 43 were right-handed. The median age froup was 31-40, the youngest was 18-20 and the oldest was 61-70. The subject's sessions took between 1.25 - 11 minutes, which the median session taking 3 minutes.



### Thoughts 

Subjects ability to type the same password will become easier and quicker the more often a password is typed. Eventually muscle memory will kick in and the typing of the password will be quicker

At the start of a session a user is likely to type a password more slower than at other stages of the data-collection session, as they will get into a rythm as time goes on

Each user was using the same keyboard each time. In reality, people could be using different keyboards (e.g. at work, at home, on a laptop, on a computer, on a external keyboard) when entering the password, so this affect is not taken into account. In a more personal comment, I use both an external keyboard and the keyboard attached to my laptop. I have two external keyboards depending on where I am (Germany or the UK), and although my German keyboard has the physical layout of a German keyboard the input is actually British, so sometimes I press the wrong key because I forget which key is actually what as they are not the same.

































### References
[1] Kevin S. Killourhy and Roy A. Maxion. "Comparing Anomaly Detectors for Keystroke Dynamics," in Proceedings of the 39th Annual International Conference on Dependable Systems and Networks (DSN-2009), pages 125-134, Estoril, Lisbon, Portugal, June 29-July 2, 2009. IEEE Computer Society Press, Los Alamitos, California, 2009.
