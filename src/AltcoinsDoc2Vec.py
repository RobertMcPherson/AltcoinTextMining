#Import all the dependencies
import subprocess
import gensim
import numpy
import os
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join

dataDirectory = os.getcwd()

#use shell script to remove text from pdf files. Store shell script one directory level above
subprocess.call([dataDirectory + '/convertmyfiles.sh'])
print(dataDirectory + '/convertmyfiles.sh')

#now create a list that contains the name of all the text file in your data #folder
docLabels = []
docLabels = [f for f in listdir(dataDirectory) if 
 f.endswith('.txt')]
#create a list data that stores the content of all text files in order of their names in docLabels
data = []
for doc in docLabels:
  data.append(open(dataDirectory + '/' + doc, encoding="ISO-8859-1").read())

#implement NLTK
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
#This function does all cleaning of data using two objects above
def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

#Create an class to return iterator object
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    
[self.labels_list[idx]])

#clean up data with nlp_clean method
data = nlp_clean(data)

#iterator returned over all documents
it = LabeledLineSentence(data, docLabels)

#create Doc2Vec model
model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)

#training of model
for epoch in range(100):
#  print 'iteration ' + str(epoch+1)
  model.train(it, total_examples=model.corpus_count, epochs=model.iter)
  model.alpha -= 0.002
  model.min_alpha = model.alpha
  model.train(it, total_examples=model.corpus_count, epochs=model.iter)

#saving the created model
model.save('doc2vec.model')
#print 'model saved'

#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')

#start testing

#printing the vector of document at index 1 in docLabels
#docvec = d2v_model.docvecs[1]
#print(docvec)

#printing the vector of the file using its name
#docvec = d2v_model.docvecs['1.txt'] #if string tag used in training
#print(docvec)

#to get most similar document with similarity scores using document-index
#similar_doc = d2v_model.docvecs.most_similar(14) 
#print(similar_doc)

#to get most similar document with similarity scores using document- name
#sims = d2v_model.docvecs.most_similar('1.txt')
#print(sims)

#to get vector of document that are not present in corpus 
#docvec = d2v_model.docvecs.infer_vector('war.txt')
#print(docvec)

#numpy.savetxt("model.csv", d2v_model.docvecs, delimiter=",")
#numpy.savetxt("lables.csv", d2v_model.doclabels, delimiter=",")

#df = pandas.DataFrame(d2v_model.docvecs)
#df.to_csv("model2.csv", header=False)

#d2v_model.save_word2vec_format(fname, doctag_vec=False, word_vec=True, prefix='*dt_', fvocab=None, binary=False)

#DocvecsArray(docvecs_mapfile)

#print(docLabels)


lbls = pd.DataFrame(docLabels)
#lbls.to_csv('lbls.csv')

#d2vs = pd.DataFrame(d2v_model.docvecs)
#d2vs.to_csv('d2vs.csv')

d2vs = pd.DataFrame(d2v_model.docvecs.doctag_syn0)
#d2vs.to_csv('d2vs.csv')

labels_and_vectors = pd.concat([lbls, d2vs], axis=1)

labels_and_vectors.columns = ['name','vector0','vector1','vector2','vector3','vector4','vector5','vector6','vector7','vector8','vector9','vector10','vector11','vector12','vector13','vector14','vector15','vector16','vector17','vector18','vector19','vector20','vector21','vector22','vector23','vector24','vector25','vector26','vector27','vector28','vector29','vector30','vector31','vector32','vector33','vector34','vector35','vector36','vector37','vector38','vector39','vector40','vector41','vector42','vector43','vector44','vector45','vector46','vector47','vector48','vector49','vector50','vector51','vector52','vector53','vector54','vector55','vector56','vector57','vector58','vector59','vector60','vector61','vector62','vector63','vector64','vector65','vector66','vector67','vector68','vector69','vector70','vector71','vector72','vector73','vector74','vector75','vector76','vector77','vector78','vector79','vector80','vector81','vector82','vector83','vector84','vector85','vector86','vector87','vector88','vector89','vector90','vector91','vector92','vector93','vector94','vector95','vector96','vector97','vector98','vector99','vector100','vector101','vector102','vector103','vector104','vector105','vector106','vector107','vector108','vector109','vector110','vector111','vector112','vector113','vector114','vector115','vector116','vector117','vector118','vector119','vector120','vector121','vector122','vector123','vector124','vector125','vector126','vector127','vector128','vector129','vector130','vector131','vector132','vector133','vector134','vector135','vector136','vector137','vector138','vector139','vector140','vector141','vector142','vector143','vector144','vector145','vector146','vector147','vector148','vector149','vector150','vector151','vector152','vector153','vector154','vector155','vector156','vector157','vector158','vector159','vector160','vector161','vector162','vector163','vector164','vector165','vector166','vector167','vector168','vector169','vector170','vector171','vector172','vector173','vector174','vector175','vector176','vector177','vector178','vector179','vector180','vector181','vector182','vector183','vector184','vector185','vector186','vector187','vector188','vector189','vector190','vector191','vector192','vector193','vector194','vector195','vector196','vector197','vector198','vector199','vector200','vector201','vector202','vector203','vector204','vector205','vector206','vector207','vector208','vector209','vector210','vector211','vector212','vector213','vector214','vector215','vector216','vector217','vector218','vector219','vector220','vector221','vector222','vector223','vector224','vector225','vector226','vector227','vector228','vector229','vector230','vector231','vector232','vector233','vector234','vector235','vector236','vector237','vector238','vector239','vector240','vector241','vector242','vector243','vector244','vector245','vector246','vector247','vector248','vector249','vector250','vector251','vector252','vector253','vector254','vector255','vector256','vector257','vector258','vector259','vector260','vector261','vector262','vector263','vector264','vector265','vector266','vector267','vector268','vector269','vector270','vector271','vector272','vector273','vector274','vector275','vector276','vector277','vector278','vector279','vector280','vector281','vector282','vector283','vector284','vector285','vector286','vector287','vector288','vector289','vector290','vector291','vector292','vector293','vector294','vector295','vector296','vector297','vector298','vector299']

labels_and_vectors.to_csv('labels_and_vectors.csv')

#The raw vectors array of words in a Word2Vec or Doc2Vec model is available in model.wv.syn0. The list of words in the index-order of that array is in model.wv.index2word.

#Raw doc-vectors for a Doc2Vec model will be in model.docvecs.doctag_syn0. If you used string doctags as the keys for document vectors, and only string doctags, a list of string doctags in that array's index-order is in model.docvecs.offset2doctag. (If you used only plain-int doctags, that list will be empty. If you used a mix of plain-ints and strings, all the ints appear first, then the offset2doctag doc-vectors appear after the model.docvecs.max_rawint slot.)

#call the shell script that runs the jar files that make the logistic regression prediction, based on the POJO model exported from H2O
subprocess.call([dataDirectory + '/model_commands.sh'])
print(dataDirectory + '/model_commands.sh')

output = pd.read_csv('output.csv')
#print(output)

print(lbls)
predictions = pd.concat((lbls, output), axis=1)
print(predictions)

predictions.to_csv('predictions.csv')
