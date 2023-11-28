from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_codes(codes, y, save=True, path='../result/graph/tsne', title='tsne'):
    perplexity = 10
    n_components = 2
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_result = tsne.fit_transform(codes)
    # plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    normal_samples_x, normal_samples_y = tsne_result[1-y][:, 0], tsne_result[1-y][:, 1]
    anomalous_samples_x, anomalous_samples_y = tsne_result[y][:, 0], tsne_result[y][:, 1]
    #print(tsne_result)
    #print(normal_samples_x.shape)
    ax.scatter(normal_samples_x, normal_samples_y, marker="+", alpha=0.2).set_label('Normal data')
    ax.scatter(anomalous_samples_x, anomalous_samples_y, marker="x", alpha=0.2).set_label('Anomalous data')
    ax.plot()
    
    i = 1
    if save:
        filename = title
        pathfile = os.path.normpath(os.path.join(path, filename+str(i)))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile+'.png'):
            pathfile = os.path.normpath(os.path.join(path,filename+str(i)))
            i+=1
        fig.savefig(pathfile, bbox_inches='tight')
    