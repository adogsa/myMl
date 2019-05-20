from gensim import corpora, models, similarities
corpus = [
        ['carrot', 'salad', 'tomato', 'highway'],
        ['carrot', 'salad', 'dish'],
        ['tomato', 'dish','highway'],
        ['tomato', 'salad'],
        ['car', 'break', 'highway'],
        ['highway', 'accident', 'car'],
        ['비타민A', '비타민B', '철분', '지방'],
        ['아연', '칼슘', '철분'],
        ['노니', '양배추', '레몬밤'],
        ['레몬밤', '탄수화물', '지방'],
        ['moto', 'break', 'break', 'break', 'break', 'break'],
        ['accident', 'moto', 'car']
    ]
dictionary = corpora.Dictionary(corpus)
print(dictionary.token2id)
nb_passes = 1
nb_topics = 7
lda_model = models.LdaModel(
                                   num_topics = nb_topics,\
                                   id2word = dictionary,\
                                   passes = nb_passes)

print(lda_model.get_topics())

import pyLDAvis.gensim as gensimvis
import pyLDAvis
corpus = [dictionary.doc2bow(doc) for doc in corpus]
vis_data = gensimvis.prepare(lda_model,corpus,dictionary)
pyLDAvis.display(vis_data)
pyLDAvis.save_html(vis_data, "./jjwhtml.html")