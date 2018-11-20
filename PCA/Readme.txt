Written by Qingyuan Jin, q1jin@ucsd.edu, 11/19/2018

Cause we have 18-dimensional variable in row data, it is simply waste of time to deal with all of them. 
Thus we should detect the data internally to find our principle components, which involves PCA(Principle component analysis).
 
What PCA does here: Finding the directions of maximum variance in high-dimensional data and project it onto a smaller dimensional subspace while retaining most of the information.

I write down every annotation in each part of code. Please feel free to read them and tell me if there is any doubt.

So far, the input is raw data and out put is a new variable subspace and each eigenvalue, 
eigenvector in each  dimension by sequence. 

We can directly use new subspace of variable or only derive the sequence of relevance to help us obtain information more efficiently.
