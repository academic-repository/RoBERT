from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from tqdm import tqdm

labels=[]
path='adv_result_new/textbugger_attack/'
with open(os.path.join(path, 'DMSC_transformer6_normal_representation_result.txt'), 'r',encoding='utf8') as fout:
    for line in fout.readlines():
        line_=line.strip().split(' ',1)
        #print(line_)
        labels.append(line_[0])

representation=np.load(os.path.join(path, 'DMSC_transformer6_normal_representation_result.npy'))
print(representation.shape)
representation=representation.tolist()
# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, method='barnes_hut')
tsne=TSNE()
low_dim_embs=tsne.fit_transform(representation)
#plt.figure(figsize=(100, 100))  # in inches
for i, label in tqdm(enumerate(labels)):
    x, y = low_dim_embs[i, :]
    if label=='0':
        plt.scatter(x, y,c='r')
    else:
        plt.scatter(x, y,c='b')
plt.xticks([]) 
plt.yticks([])
plt.axis('off')
filename='figure.png'
plt.tight_layout()
plt.savefig(filename,bbox_inches='tight')