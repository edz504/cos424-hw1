
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time
from nltk.stem.porter import *

# special characters
special = ["~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "{", "}", "[", "]", "<", ">", "/", "\\", "-", "_", "+", "="]
# punctuation
punctuation = [".", ",", "!", "?", ";", ":", "'", "\""]

chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


def get_files(mypath):
    return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def get_dirs(mypath):
    return [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]

# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile, numofemails=10000):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords

def tokenize_corpus(path, train=True):
    porter = nltk.PorterStemmer() # also lancaster stemmer
    wnl = nltk.WordNetLemmatizer()
    stopWords = stopwords.words("english")
    classes = []
    samples = []
    docs = []
    dirs = get_dirs(path)
    if train == True:
        words = {}
    for dir in dirs:
        files = get_files(path+"/"+dir)
        for f in files:
            classes.append(dir)
            samples.append(f)
            inf = open(path+'/'+dir+'/'+f,'r')
            raw = inf.read().decode('latin1') # or ascii or utf8 or utf16
            # remove noisy characters; tokenize
            # raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
            # tokens = word_tokenize(raw)
            # tokens = raw.split()
            # convert to lower case
            # tokens = [w.lower() for w in tokens]
            # tokens = [w for w in tokens if w not in stopWords]
            # tokens = [wnl.lemmatize(t) for t in tokens]
            # tokens = [porter.stem(t) for t in tokens]
            # if train == True:
            #     for t in tokens: 
            #         # this is a hack but much faster than lookup each
            #         # word within many dict keys
            #         try:
            #             words[t] = words[t]+1
            #         except:
            #             words[t] = 1
            # docs.append(tokens)
            docs.append(raw)
    if train == True:
        return(docs, classes, samples, words)
    else:
        return(docs, classes, samples)
        

def wordcount_filter(words, num=5):
    keepset = []
    for k in words.keys():
        if(words[k] > num):
            keepset.append(k)
    print len(keepset)
    return(sorted(set(keepset)))


def count_lego(bow):

  hapax_lego_count = 0
  hapax_dislego_count = 0

  for i in range(len(bow[0])):
    if bow[i] == 1:
      hapax_lego_count += 1
    elif bow[i] == 2:
      hapax_dislego_count += 1

  return (hapax_lego_count, hapax_dislego_count)

def string_features(s):

  features = []

  global special
  global punctuation

  special_features = [0] * len(special)
  punctuation_features = [0] * len(punctuation)

  total_chars = 0
  special_count = 0
  punctuation_count = 0
  punctuation_gap_count = 0
  upper_count = 0
  lower_count = 0
  digit_count = 0
  alpha_count = 0
  camel_count = 0
  all_upper_count = 0
  all_lower_count = 0

  has_upper = 0
  has_lower = 0

  total_chars = len(s)

  # types of characters
  for l in list(s):

    if l in special:
      special_count += 1
      special_features[special.index(l)] += 1

    if l in punctuation:
      punctuation_count += 1
      punctuation_features[punctuation.index(l)] += 1
      

    if l.isupper():
      has_upper = 1
      upper_count += 1
    elif l.islower():
      has_lower = 1
      lower_count += 1
    elif l.isdigit():
      digit_count += 1
    elif l.isalpha():
      alpha_count += 1
  # end parsing characters -----------------------

  # word capitalization
  if has_upper and has_lower:
    camel_count += 1
  elif has_upper:
    all_upper_count += 1
  elif has_lower:
    all_lower_count += 1

  punctuation_features = [float(x) / total_chars for x in punctuation_features]
  special_features = [float(x) / total_chars for x in special_features]
  features += punctuation_features
  features += special_features

  upper_ratio = float(upper_count) / total_chars
  lower_ratio = float(lower_count) / total_chars
  alpha_ratio = float(alpha_count) / total_chars
  digit_ratio = float(digit_count) / total_chars
  special_ratio = float(special_count) / total_chars
  punctuation_ratio = float(punctuation_count) / total_chars


  character_features = [upper_ratio, lower_ratio, alpha_ratio, \
  digit_ratio, special_ratio, punctuation_ratio, total_chars]

  features += character_features

  print len(features)
  return features

  
content_types = {}
content_counter = 0

def calc_doc_features(doc, vocab):

  global content_types
  global content_counter
  stemmer = PorterStemmer()

  num_features = 20
  # features = numpy.zeros(shape=(1, num_features), dtype = numpy.uint8)
  features = []

  vocabIndex={}
  # bagofwords = numpy.zeros(shape=(1,len(vocab)), dtype=numpy.uint8)

  # for i in range(len(vocab)):
  #    vocabIndex[vocab[i]]=i

  # for t in doc:
  #    index_t=vocabIndex.get(t)
  #    if index_t>=0:
  #       bagofwords[i,index_t]=bagofwords[i,index_t]+1

  # hapax_count = count_lego(bagofwords) # tuple of lego and dislego

  upper_count = 0
  lower_count = 0
  all_upper_count = 0
  all_lower_count = 0
  camel_count = 0
  alpha_count = 0
  digit_count = 0
  special_count = 0

  total_chars = 0
  total_words = 0
  punctuation_count = 0

  global special
  global punctuation 

  special_features = [0] * len(special)
  punctuation_features = [0] * len(punctuation)

  punctuation_gap_count = 0
  last_punctuation = 0

  has_upper = 0
  has_lower = 0

  found_from = 0
  found_date = 0
  found_subject = 0

  from_address = ""
  subject = ""

  lines = doc.split("\n")
  first_line = lines[0]

  days_of_week = {"Sun": 0, "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6}

  hour = -1
  minute = -1
  second = -1
  day = "" 
  from_address = "" 
  time = -1
  time_match = re.search("[0-2][0-9]:[0-5][0-9]:[0-5][0-9]", first_line)
  if time_match:
    time = time_match.group(0)
    # print time
    # print "match" + str(time)
  day_match = re.search("(Mon|Tue|Wed|Thu|Fri|Sat|Sun)", first_line)
  if day_match:
    day = day_match.group(0)
    day = days_of_week.get(day)

  # print day
  # print time

  first_line_split = first_line.split()

  from_name = ""
  from_domain = ""
  for x in first_line_split:
    from_address_match = re.search(r"([^@]+@[^@]+\.[^@]+)", x)
    if from_address_match:
      from_address = from_address_match.group(0).split("@")

      from_name = from_address[0]
      from_domain = from_address[1]
      break
  # print from_address

  # print first_line
  time_split = str(time).split(":")
  hour = int(time_split[0])
  minute = int(time_split[1])
  second = int(time_split[2])
  # print from_address
  # print hour
  # print day
  # print first_line


  found_subject = 0
  subject = ""
  content_type = -1
  found_content_type = 0

  for line in lines:
    if not found_subject and re.match("Subject", line):
      subject = (" ").join(line.split()[1:])
      found_subject = 1
    elif not found_content_type and re.match("Content-Type", line):
      content_type_string = re.split(" |;", line)[1].lower()
      # print content_type_string
      found_content_type = 1
      if not content_type_string in content_types:
        content_types[content_type_string] = content_counter
        content_type = content_counter
        content_counter += 1
      else:
        content_type = content_types.get(content_type_string)
  # print content_type



###################################################
# BEGIN DOCUMENT PARSING ##########################
###################################################

  # store number of words of that length. 1 - 19 and 20+
  length_histogram = [0] * 20

  # look at each word in the doc
  doc = doc.split()
  for t in doc:

    total_words += 1

    # word count hash table
    word_counts = {}
    stem = stemmer.stem(t)
    stem = stem.lower()
    if stem in word_counts:
      word_counts[stem] = word_counts[stem] + 1
    else:
      word_counts[stem] = 1

    this_word_length = len(t)

    total_chars += this_word_length
    if this_word_length > 19:
      length_histogram[19] += 1
    else:
      length_histogram[this_word_length - 1] += 1

    # types of characters
    for l in list(t):

      # special_index = special.index(l)
      # punctuation_index = punctuation.index(l)

      if l in special:
        special_count += 1
        special_features[special.index(l)] += 1

      if l in punctuation:
        punctuation_count += 1
        punctuation_features[punctuation.index(l)] += 1
        last_punctuation = 1
        if not last_punctuation:
          punctuation_gap_count += 1 # found a new "sentence"
      else:
        last_punctuation = 0

      if l.isupper():
        has_upper = 1
        upper_count += 1
      if l.islower():
        has_lower = 1
        lower_count += 1
      elif l.isdigit():
        digit_count += 1

      if l.isalpha():
        alpha_count += 1
    # end parsing characters -----------------------

    # word capitalization
    if has_upper and has_lower:
      camel_count += 1
    if has_upper:
      all_upper_count += 1
    if has_lower:
      all_lower_count += 1

  # end parsing words ------------------
  hlego_count = 0
  hdislego_count = 0

  for x in word_counts:
    if word_counts[x] == 1:
      hlego_count += 1
    elif word_counts[x] == 2:
      hdislego_count += 1

  # features.append(hapax_count)
  features.append(float(hlego_count) / total_words)
  # features.append(float(hdislego_count) / total_words)
  punctuation_features = [float(x) / total_chars for x in punctuation_features]
  special_features = [float(x) / total_chars for x in special_features]
  features += punctuation_features
  features += special_features

  upper_ratio = float(upper_count) / total_chars
  lower_ratio = float(lower_count) / total_chars
  alpha_ratio = float(alpha_count) / total_chars
  digit_ratio = float(digit_count) / total_chars
  special_ratio = float(special_count) / total_chars
  punctuation_ratio = float(punctuation_count) / total_chars

  # all_upper_ratio = float(all_upper_count) / total_words
  # all_lower_ratio = float(all_lower_count) / total_words
  # camel_ratio = float(camel_count)  / total_words

  character_features = [upper_ratio, lower_ratio, alpha_ratio, \
  digit_ratio, special_ratio, punctuation_ratio, total_chars, 0]

  features += character_features

  # print len(string_features(subject))
  # if len(subject) < 1:
  #   subject_features = [0] * 37
  # else:
  #   subject_features = string_features(subject)
  # if len(from_name) < 1:
  #   from_name_features = [0] * 37
  # else:
  #   from_name_features = string_features(from_name)
  # if len(from_domain) < 1:
  #   from_domain_features = [0] * 37
  # else:
  #   from_domain_features = string_features(from_domain)
  features += [content_type]

  time_features = [hour, minute, second, day]
  # print time_features
  features += time_features

  length_histogram = [float(x) / total_words for x in length_histogram]
  features += length_histogram

  # print features

  # print "actual number of features is: " + str(len(features))
  return features

def create_custom_features(docs, vocab):
  num_features = 64

  # print "len docs: " + str(len(docs))
  features = numpy.zeros(shape=(len(docs), num_features), dtype = numpy.float16)

  for i in range(len(docs)):
    # if i == 30:
    #   print docs[i]
    new_features = calc_doc_features(docs[i], vocab)
    # print "len new features: " + str(len(new_features))
    for j in range(len(new_features)):
      features[i][j] = new_features[j]
    # features[i] = calc_doc_features(docs[i], vocab)
    # print "features " + str(len(features[i]))

    # doc = doc[i]
    # tag and chunk doc
    # tagged = nltk.pos_tag(doc)
    # entities = nltk.chunk.ne_chunk(tagged)

    # pos_bigrams = []

    # # calculate pos bigrams
    # pos_index = 1
    # for i in range(len(tagged) - 1):
    #   first_pos = tagged[i][pos_index]
    #   second_pos = tagged[i+1][pos_index]

    #   new_bigram = first_pos + " " + second_pos



  return features




def find_wordcounts(docs, vocab):
    bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
    vocabIndex={}
    for i in range(len(vocab)):
       vocabIndex[vocab[i]]=i

    for i in range(len(docs)):
        doc = docs[i]

        for t in doc:
           index_t=vocabIndex.get(t)
           if index_t>=0:
              bagofwords[i,index_t]=bagofwords[i,index_t]+1

    print "Finished find_wordcounts for : "+str(len(docs))+"  docs"
    return(bagofwords)


# path should have one folder for each class. Class folders should
# contain text documents that are labeled with the class label (folder
# name). Bag of words representation, vocabulary will be output to
# <outputfile>_*.dat files.
def main(argv):
   path = ''
   outputf = ''
   vocabf = ''
   start_time = time.time()

   try:
      opts, args = getopt.getopt(argv,"p:o:v:",["path=","ofile=","vocabfile="])
   except getopt.GetoptError:
      print 'python text_process.py -p <path> -o <outputfile> -v <vocabulary>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'text_process.py -p <path> -o <outputfile> -v <vocabulary>'
         sys.exit()
      elif opt in ("-p", "--path"):
         path = arg
      elif opt in ("-o", "--ofile"):
         outputf = arg
      elif opt in ("-v", "--vocabfile"):
         vocabf = arg
   
   print 'Path is "', path
   print 'Output file name is "', outputf
   print 'vocabulary file is "', vocabf
   if (not vocabf):
      (docs, classes, samples, words) = tokenize_corpus(path, train=True)
      word_count_threshold = 200
      vocab = wordcount_filter(words, num=word_count_threshold)
   else:
      vocabfile = open(path+vocabf, 'r')
      vocab = [line.rstrip('\n') for line in vocabfile]
      vocabfile.close()
      (docs, classes, samples) = tokenize_corpus(path, train=False)

   # bow = find_wordcounts(docs, vocab)
   # print "AHA"
   custom_features = create_custom_features(docs, vocab) 

   numpy.savetxt("custom_features_train.csv", custom_features, delimiter=",")
   print "num docs " + str(len(custom_features))
   print "num features " + str(len(custom_features[0]))
   print custom_features[0]
   
   #sum over docs to see any zero word counts, since that would stink.
   # x = numpy.sum(bow, axis=1) 
   # print "doc with smallest number of words in vocab has: "+str(min(x))
   # # print out files
   # if (vocabf):
   #    word_count_threshold = 0   
   # else:
   #    #outfile= open(path+"/"+outputf+"_vocab_"+str(word_count_threshold)+".txt", 'w')
   #    outfile= codecs.open(path+"/"+outputf+"_vocab_"+str(word_count_threshold)+".txt", 'w',"utf-8-sig")
   #    outfile.write("\n".join(vocab))
   #    outfile.close()
   # #write to binary file for large data set
   # bow.tofile(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".dat")

   # #write to text file for small data set
   # #bow.tofile(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".txt", sep=",", format="%s")
   # outfile= open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
   # outfile.write("\n".join(classes))
   # outfile.close()
   # outfile= open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
   # outfile.write("\n".join(samples))
   # outfile.close()
   # print str(time.time() - start_time)

if __name__ == "__main__":
   main(sys.argv[1:])


