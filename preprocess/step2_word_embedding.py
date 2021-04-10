import os
import numpy as np
import pickle
import tqdm
import fire
from gensim.models import Word2Vec
import gensim.downloader as api

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

class Processor:
    def __init__(self):
        print('get word embeddings..')

    def run(self, root='YOUR_DATA_PATH'):
        tags = np.load(os.path.join(root, 'tags.npy'))

        tag_to_google_emb = {}
        google_wv = api.load('word2vec-google-news-300')
        for tag in tqdm.tqdm(tags):
            sub_tags = tag.split("_")
            sub_tag = sub_tags[0]
            
            emb = google_wv.get_vector(sub_tag)
            for sub_tag in sub_tags:
                if not sub_tag in stopwords:
                    emb += google_wv.get_vector(sub_tag)                
            tag_to_google_emb[tag] = emb
        pickle.dump(tag_to_google_emb, open(os.path.join(root, 'google_emb.pkl'), 'wb'))

        tag_to_music_emb = {}
        music_wv = Word2Vec.load(os.path.join(root, 'music_w2v', 'model_semeval_trigrams_300.model'))
        for tag in tqdm.tqdm(tags):
            emb = music_wv.wv.get_vector(tag)
            tag_to_music_emb[tag] = emb
        pickle.dump(tag_to_music_emb, open(os.path.join(root, 'music_emb.pkl'), 'wb'))
        print('done!')

if __name__ == '__main__':
    p = Processor()
    fire.Fire({'run': p.run})
