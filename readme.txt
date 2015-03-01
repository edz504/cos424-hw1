###################
## For project 1
###################


1. Download data from:

isun.cs.princeton.edu/P1Data.tar.gz

2. Move data and script to a folder

3. Uncompress data:

> gunzip P1Data.tar.gz
> tar xvf P1Data.tar

this will create Train/ and Test/ folders with Spam/ and NotSpam/
directories in each of them. Unpacking these files will take a few
minutes and require about 1.5G space on your computer.

4. Build dictionary from training set and convert each email to a
bag-of-words representation (the first of which takes approximately 50
minutes on my iMac):

> python email_process.py -p <your folder path>/Train -o train_emails 

You need to, unfortunately, copy the vocabulary file to the /Test
folder at this point

> python email_process.py -p <your folder path>/Test -o test_emails -v train_emails_vocab.txt

From here you will have the following files:

<your folder path>/train_emails_bag_of_words_<vocab_cutoff>.dat

(the list of emails with the word count vector, in the same order as
the vocabulary. Note that this is a binary file because it is so big;
we have included a python function to let you read in this binary file
in the same script.)

<your folder path>/train_emails_samples_class_<vocab_cutoff>.txt

(the list of emails in the same order as in *bag_of_words* and
*classes* files)

<your folder path>/train_emails_classes_<vocab_cutoff>.txt

(the list of classifiers---Spam or NotSpam---for each email)

<your folder path>/train_emails_vocab_<vocab_cutoff>.txt

(the list of vocabulary words. Please look through and modify or adapt
script or vocab list as you see necessary.)

Note: please look over email_process.py and modify or add features,
change cutoffs for word counts, adapt dictionaries, etc. if you would
like.

5. Build classifiers based on the training data

6. Test classifiers on the test data

7. Use the dictionary to determine which words are most predictive of
spam/no-spam

8. Look at the text of the emails in the data folders to find new
features, illustrate difficult types of emails to classify, or
evaluate classifier assumptions.
