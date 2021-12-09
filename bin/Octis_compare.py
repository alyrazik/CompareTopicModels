from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

dataset = Dataset()
dataset.fetch_dataset('20NewsGroup')
from octis.models.LDA import  LDA
from octis.models.HDP import HDP
from octis.models.NMF import NMF
from octis.models.LSI import LSI
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from octis.models.ProdLDA import ProdLDA
from octis.models.NeuralLDA import NeuralLDA

model_LDA = LDA(num_topics=20).train_model(dataset)
model_HDP = HDP().train_model(dataset)
model_NMF = NMF(num_topics=20).train_model(dataset)
model_LSI = LSI(num_topics=20).train_model(dataset)
model_CTM = CTM(num_topics=20).train_model(dataset)
model_ETM = ETM(num_topics=20).train_model(dataset)
model_ProdLDA = ProdLDA(num_topics=20).train_model(dataset)
model_NeuralLDA = NeuralLDA(num_topics=20).train_model(dataset)

cv = Coherence(texts=dataset.get_corpus(),topk=10, measure='c_v')
topic_diversity = TopicDiversity(topk=10)

print("Coherence LDA: ", str(cv.score(model_LDA)))
print("Topic diversity: "+str(topic_diversity.score(model_LDA)))
print("Coherence HDP:", str(cv.score(model_HDP)))
print("Topic diversity: "+str(topic_diversity.score(model_HDP)))
print("Coherence NMF: ", str(cv.score(model_NMF)))
print("Topic diversity: "+str(topic_diversity.score(model_NMF)))
print("Coherence LSI: ", str(cv.score(model_LSI)))
print("Topic diversity: "+str(topic_diversity.score(model_LSI)))
print("Coherence CTM: ", str(cv.score(model_CTM)))
print("Topic diversity: "+str(topic_diversity.score(model_CTM)))
print("Coherence ETM: ", str(cv.score(model_ETM)))
print("Topic diversity: "+str(topic_diversity.score(model_ETM)))
print("Coherence ProdLDA: ", str(cv.score(model_ProdLDA)))
print("Topic diversity: "+str(topic_diversity.score(model_ProdLDA)))
print("Coherence NeuralLDA: ", str(cv.score(model_NeuralLDA)))
print("Topic diversity: "+str(topic_diversity.score(model_NeuralLDA)))