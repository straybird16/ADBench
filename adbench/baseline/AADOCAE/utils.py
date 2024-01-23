from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_codes(codes, labels, save=True, path='../result/graph/tsne', title='tsne'):
    perplexity = 10
    n_components = 2
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    #print(type(codes), codes.shape)
    tsne_result = tsne.fit_transform(codes)
    # plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    normal_samples_x, normal_samples_y = tsne_result[1-labels][:, 0], tsne_result[1-labels][:, 1]
    anomalous_samples_x, anomalous_samples_y = tsne_result[labels][:, 0], tsne_result[labels][:, 1]
    x, y = tsne_result.T
    #print(type(x), type(y), x[:5] ,y[:5])
    ax.scatter(x=x[labels==0], y=y[labels==0], alpha=0.8).set_label('Normal data')
    ax.scatter(x=x[labels==1], y=y[labels==1], marker="X", alpha=0.8).set_label('Anomalous data')
    ax.legend()
    #ax.scatter(normal_samples_x, normal_samples_y, marker="+", alpha=0.2).set_label('Normal data')
    #ax.scatter(anomalous_samples_x, anomalous_samples_y, marker="x", alpha=0.2).set_label('Anomalous data')
    ax.plot()
    
    i = 1
    if save:
        filename = title
        pathfile = os.path.normpath(os.path.join(path, filename+'({})'.format(i)))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile+'.png'):
            pathfile = os.path.normpath(os.path.join(path,filename+'({})'.format(i)))
            i+=1
        fig.savefig(pathfile+'.png', bbox_inches='tight')
        plt.close(fig)
    