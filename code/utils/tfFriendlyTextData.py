import pandas as pd

def makeDataset(inputDirectory, outputDirectory):
    """creation of a dataset looking like : https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory"""
    PATH = inputDirectory
    df = pd.read_json(PATH+'train.json').set_index('Id')
    labels = pd.read_csv(PATH+'train_label.csv', index_col=0)
    df = pd.concat([df, labels], axis=1).drop('gender', axis=1)

    PATH = outputDirectory
    for index in df.index:
        cat = int(df.loc[index, 'Category'])
        text = df.loc[index, 'description']
        fp = open(PATH+f'train/{cat}/{index}.txt', "w", encoding="utf-8")
        fp.write(text)
        fp.close()



if __name__ == "__main__":
    # makeDataset('/Users/antoineadam/git/NLPChallenge2021/data/', '/Users/antoineadam/git/NLPChallenge2021/data/')