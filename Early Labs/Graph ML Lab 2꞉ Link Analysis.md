---
attachments: [Clipboard_2022-12-07-15-31-44.png, Clipboard_2022-12-07-15-49-38.png]
tags: [Graph ML]
title: 'Graph ML Lab 2: Link Analysis'
created: '2022-12-07T21:23:50.926Z'
modified: '2022-12-07T22:11:51.777Z'
---

# Graph ML Lab 2: Link Analysis

## Part 1:
### Users:
1. User A, whose interests are represented by the teleport set {1,2,3}
2. User B, whose interests are represented by the teleport set {3,4,5}
3. User C, whose interests are represented by the teleport set {1,4,5}
4. User D, whose interests are represented by the teleport set {1}.

### Five node graph with some edges:
![](@attachment/Clipboard_2022-12-07-15-49-38.png)

#### Equations:
  - $R(1) = \frac{R(4)}{L(4)}$
  - $R(2) = \frac{R(1)}{L(1)} + \frac{R(3)}{L(3)} + \frac{R(5)}{L(5)}$
  - $R(3) = \frac{R(1)}{L(1)} + \frac{R(4)}{L(4)} + \frac{R(5)}{L(5)}$
  - $R(4) = \frac{R(5)}{L(5)}$
  - $R(5) = \frac{R(4)}{L(4)}$

#### Matrix Equation: $\beta = 0.85$
$$
  \begin{bmatrix}
  r_1\\
  r_2\\ 
  r_3\\ 
  r_4\\ 
  r_5\\ 
  \end{bmatrix}
  =\beta
  \begin{bmatrix}
  0 & \frac{1}{3} & \frac{1}{3} & 0 & 0\\
  0 & 0 & 0 & 0 & 0\\
  0 & \frac{1}{3} & 0 & 0 & 0\\
  1 & 0 & \frac{1}{3} & 0 & 1\\
  0 & \frac{1}{3} & \frac{1}{3} & 1 & 0\\
  \end{bmatrix}
  \begin{bmatrix}
  \frac{1}{5}\\
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \end{bmatrix}
  + \frac{1-\beta}{5}*p
$$  
Note that $p$ is the personalized teleport vector (what people want to look at)


### Define your initial PageRank vector, your stochastic adjacency matrix, and teleport vectors for each user.
- Initial Vector: 
  - start by giving an even amount to each document: 
  $$
  \frac{1}{5} = 
  \begin{bmatrix}
  \frac{1}{5}\\
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \end{bmatrix}
  $$
- Adjacency Matrix
  $$
  \begin{bmatrix}
  0 & \frac{1}{3} & \frac{1}{3} & 0 & 0\\
  0 & 0 & 0 & 0 & 0\\
  0 & \frac{1}{3} & 0 & 0 & 0\\
  1 & 0 & \frac{1}{3} & 0 & 1\\
  0 & \frac{1}{3} & \frac{1}{3} & 1 & 0\\
  \end{bmatrix}
  $$
- Teleport Vector $p$
  - User A:
  $$
  \begin{bmatrix}
  \frac{1}{3}\\
  \frac{1}{3}\\ 
  \frac{1}{3}\\ 
  0\\ 
  0\\ 
  \end{bmatrix}
  $$
  - User B:
  $$
  \begin{bmatrix}
  0\\
  0\\ 
  \frac{1}{3}\\ 
  \frac{1}{3}\\ 
  \frac{1}{3}\\ 
  \end{bmatrix}
  $$
  - User C:
  $$
  \begin{bmatrix}
  \frac{1}{3}\\
  0\\ 
  0\\ 
  \frac{1}{3}\\ 
  \frac{1}{3}\\ 
  \end{bmatrix}
  $$
  - User D:
  $$
  \begin{bmatrix}
  1\\
  0\\ 
  0\\ 
  0\\ 
  0\\ 
  \end{bmatrix}
  $$
### Without looking at the graph or actually running the PageRank algorithm, can you compute the personalized PageRank vectors for the following users? If so, how? If not, why not and how would you make it work?
Yes this is possible. Just run through the equation a few times until the value converges.
#### Matrix Equation:
$$
  \begin{bmatrix}
  r_1\\
  r_2\\ 
  r_3\\ 
  r_4\\ 
  r_5\\ 
  \end{bmatrix}
  =0.85
  \begin{bmatrix}
  0 & \frac{1}{3} & \frac{1}{3} & 0 & 0\\
  0 & 0 & 0 & 0 & 0\\
  0 & \frac{1}{3} & 0 & 0 & 0\\
  1 & 0 & \frac{1}{3} & 0 & 1\\
  0 & \frac{1}{3} & \frac{1}{3} & 1 & 0\\
  \end{bmatrix}
  \begin{bmatrix}
  \frac{1}{5}\\
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \frac{1}{5}\\ 
  \end{bmatrix}
  + 0.03*p
$$  
### Compute the Personalized PageRank values for your graph. You can start with the code provided in the PageRank notebook. You can use the custom implementation or Networkx. If you are using Networkx check out the “personalized” parameter. Include a copy of your implementation and sample results.



## 2: write out flow equns for user A



























