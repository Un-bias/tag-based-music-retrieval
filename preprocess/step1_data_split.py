import os
import tqdm
import fire
import pickle
import random
import glob
import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.models import Word2Vec
from collections import Counter


class Processor:
    def __init__(self):
        random.seed(0)

    def run(self, root='YOUR_DATA_PATH', top_n=100, threshold=100):
        df = self.get_dataframe(root)
        df, ids, binaries, tags = self.get_existing(root, df, top_n)
        train_ix, valid_ix, test_ix = self.split(df, ids, binaries, tags, threshold)
        self.save(root, ids, binaries, tags, train_ix, valid_ix, test_ix)

    def get_dataframe(self, root):
            # load data
            print('load data...')
            filename = os.path.join(root, 'track_tags.tsv')
            df = pd.read_csv(filename, sep='\t', names=['id', 'artist', 'tag', 'merged', 'type', 'score'])
            #w2v = api.load('word2vec-google-news-300')
            model = Word2Vec.load(os.path.join(root, 'music_w2v', 'model_semeval_trigrams_300.model'))
            w2v = model.wv

            # score threshold
            df = df[df.score>=50]

            # get available tags with w2v embedding
            print('get available tags...')
            tags = list(set(df.merged))
            available_tags = []
            for tag in tags:
                try:
                    v = w2v.get_vector(tag)
                    available_tags.append(tag)
                except KeyError:
                    continue
            df = df[df.merged.isin(available_tags)]

            '''
            # add artist information to the table
            print('add artist information...')
            song_to_artist = pickle.load(open(os.path.join(root, 'song_to_artist.pkl'), 'rb'))
            df.insert(1, 'artist', ['unknown' for _ in range(len(df))], True)
            for song_id in tqdm.tqdm(set(df.id)):
                    artist_id = song_to_artist[song_id]
                    df.loc[df.id==song_id, 'artist'] = artist_id
            df = df[~df.artist.isin(['unknown'])]
            '''
            # save dataframe
            df.to_csv(os.path.join(root, 'processed_df.tsv'), sep='\t', header=None)
            return df

    def get_existing(self, root, df, top_n):
        # get only exsisting item
        print('get available song ids...')
        existing_path = glob.glob(os.path.join(root, 'npy/*.npy'))
        existing_ids = [line.split('/')[-1][:-4] for line in existing_path]
        #df = df[df.id.isin(existing_ids)]

        # get top 100 tags
        top100 = Counter(df.merged).most_common(top_n)
        tags = [line[0] for line in top100]
        for tag in tags:
                print(tag)
        df = df[df.merged.isin(tags)]

        # id to binary
        print('get binaries...')
        tags = [tag.lower() for tag in tags]
        tags.sort()
        tag_to_index = {tags[i]: i for i in range(len(tags))}

        ids = [msd_id for msd_id in set(df.id)]
        ids.sort()
        ids = np.array(ids)
        np.save(open(os.path.join(root, 'existing_ids.npy'), 'wb'), ids)


        '''
        binaries = np.zeros((len(ids), len(tags)))
        i = 0
        for msd_id in tqdm.tqdm(ids):
                annotations = list(df[df.id==msd_id].merged)
                for tag in annotations:
                        binaries[i, tag_to_index[tag.lower()]] = 1
                i += 1
        '''
        binaries = df[["id","tag"]].copy()
        binaries["value"] = 1
        binaries = pd.pivot_table(binaries, index="id",columns=["tag"],values='value').fillna(0)
        binaries = binaries.sort_values(ascending=True, by="id").values

        np.save(open(os.path.join(root, 'binaries.npy'), 'wb'), binaries)
        np.save(open(os.path.join(root, 'tags.npy'), 'wb'), tags)
        return df, ids, binaries, tags

    def split(self, df, ids, binaries, tags, threshold):
        id_to_index = {ids[i]: i for i in range(len(ids))}
        is_run = True
        while is_run:
            artists = list(set(df.artist))
            np.random.shuffle(artists)
            train_artist = artists[:int(len(artists)*0.7)]
            valid_artist = artists[int(len(artists)*0.7):int(len(artists)*0.85)]
            test_artist = artists[int(len(artists)*0.85):]

            # validate
            train_songs = list(set(df[df.artist.isin(train_artist)].id))
            valid_songs = list(set(df[df.artist.isin(valid_artist)].id))
            test_songs = list(set(df[df.artist.isin(test_artist)].id))
            train_ix = [id_to_index[song] for song in train_songs]
            valid_ix = [id_to_index[song] for song in valid_songs]
            test_ix = [id_to_index[song] for song in test_songs]
            train_count = binaries[train_ix].sum(axis=0)
            valid_count = binaries[valid_ix].sum(axis=0)
            test_count = binaries[test_ix].sum(axis=0)
            print(tags[train_count.argmin()], train_count.min())
            print(tags[valid_count.argmin()], valid_count.min())
            print(tags[test_count.argmin()], test_count.min())
            if np.min([train_count.min(), valid_count.min(), test_count.min()]) < threshold:
                is_run = True
            else:
                is_run = False
        return train_ix, valid_ix, test_ix

    def save(self, root, ids, binaries, tags, train_ix, valid_ix, test_ix):
        train_ids = ['%s//%s'%(ix, ids[ix]) for ix in train_ix]
        valid_ids = ['%s//%s'%(ix, ids[ix]) for ix in valid_ix]
        test_ids = ['%s//%s'%(ix, ids[ix]) for ix in test_ix]

        print('save items...')
        np.save(open(os.path.join(root, 'train_ids.npy'), 'wb'), train_ids)
        np.save(open(os.path.join(root, 'valid_ids.npy'), 'wb'), valid_ids)
        np.save(open(os.path.join(root, 'test_ids.npy'), 'wb'), test_ids)

        # get tag-to-item dictionary (train set only)
        train_tag_to_ix = {tag: [] for tag in tags}
        for ix in train_ix:
            binary = binaries[ix]
            for ti, is_tag in enumerate(binary):
                if is_tag:
                    train_tag_to_ix[tags[ti]].append('%s//%s'%(ix, ids[ix]))
        pickle.dump(train_tag_to_ix, open(os.path.join(root, 'train_tag_to_ix.pkl'), 'wb'))
        print('done!')

if __name__ == '__main__':
    p = Processor()
    fire.Fire({'run': p.run})
