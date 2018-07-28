from pyfasttext import FastText
from sklearn import cluster
from sklearn import metrics
import numpy as np


def main():
    model = FastText('model_text8.bin')

    target_word = 'deep'    
    
    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    print('Embedding shape:', target_word_embedding.shape)
    print('Embedding:', target_word_embedding[0:15], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
    
        
    target_word = 'president'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'self'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'insult'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      

        
    target_word = 'general'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'inclined'
    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'property'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'international'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
    target_word = 'many'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'imprisoned'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
    target_word = 'branches'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'communist'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
    target_word = 'france'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
    target_word = 'strict'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'earthly'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
    terget_word = "zero"

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
    target_word = 'feminism'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
         
    target_word = 'ideas'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'theory'

     # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      
        
    target_word = 'writings'

     # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    #print('Embedding shape:', target_word_embedding.shape)
    #print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    closest_word_embeddings = []
    numw = 0
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)
        closest_word_embeddings.append(model.get_numpy_vector(word))
           
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(closest_word_embeddings)
    labels = kmeans.labels_
    print ('Cluster id labels for inputted data')
    print (labels)
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    
    for i in range(0,15):
      if labels[i] == 0:
          cluster1.append(closest_words[i][0]) 
          
      if labels[i] == 1:
          cluster2.append(closest_words[i][0])
          
      if labels[i] == 2:
          cluster3.append(closest_words[i][0])
      
    print("cluster #1 : ", cluster1)
    print("cluster #2 : ", cluster2)
    print("cluster #3 : ", cluster3)
      


if __name__ == '__main__':
    main()