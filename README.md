# This is solution to CS188(fa2022)
Completed all 6 projects on Feb 21, 2024
(Although it doesn't mean I got all credits...)

## Project 1: Search
Handout: <https://inst.eecs.berkeley.edu/~cs188/fa22/projects/proj1/>
Autograder feedback:
```
Provisional grades
==================
Question q1: 3/3
Question q2: 3/3
Question q3: 3/3
Question q4: 3/3
Question q5: 3/3
Question q6: 2/3
Question q7: 3/4
Question q8: 3/3
------------------
Total: 23/25
```
For question 6 and question 7, I failed to find a 'good' heuristic fuction
to get full credits. **TODO**

## Project 2: Multi-Agent Search
Handout: <https://inst.eecs.berkeley.edu/~cs188/fa22/projects/proj2/>
Autograder feedback:
```
Provisional grades
==================
Question q1: 3/4
Question q2: 5/5
Question q3: 5/5
Question q4: 5/5
Question q5: 5/6
------------------
Total: 23/25
```
A pitfall: do not call self.evaluationFunction() before max depth achieved!
or you will recieve the exception and fail autograder test.
```
FAIL: Exception raised: getScore() called on non-terminal state or \
before maximum depth achieved.
```

## Project 3: Reinforcement Learning
Handout: <https://inst.eecs.berkeley.edu/~cs188/fa22/projects/proj3/>
Autograder Feedback:
```
Provisional grades
==================
Question q1: 6/6
Question q2: 5/5
Question q3: 6/6
Question q4: 2/2
Question q5: 2/2
Question q6: 4/4
------------------
Total: 25/25
```
Q-Learning wins!

## Project 4: Ghostbusters
Handout: <https://inst.eecs.berkeley.edu/~cs188/fa22/projects/proj4/>
Autograder Feedback:
```
Provisional grades
==================
Question q1: 2/2
Question q2: 3/3
Question q3: 2/2
Question q4: 2/2
Question q5: 1/1
Question q6: 2/2
Question q7: 2/2
Question q8: 1/1
Question q9: 1/1
Question q10: 2/2
Question q11: 2/2
Question q12: 1/1
Question q13: 2/2
Question q14: 2/2
------------------
Total: 25/25
```
Mainly about Particle Filters and Dynamic Bayes Nets. 

## Project 5: Machine Learning
Handout: <https://inst.eecs.berkeley.edu/~cs188/fa22/projects/proj5/>
Autograder feedback:
```
Provisional grades
==================
Question q1: 6/6
Question q2: 6/6
Question q3: 6/6
Question q4: 7/7
------------------
Total: 25/25
```
The most exciting part of cs188- you will train a neuro network to do 
digit-classification and language-classfication!

# Next will be solution to CS189(fa 23)
## Homewok 3
In folder `hw3_files`.  
[Handout](https://eecs189.org/docs/hw3.pdf)  
Due to time constraints, I only completed task 2 and 3 in it.  
In task 2, you will implement a simple neuron network using `numpy`.  
Task 3 is an introduction to [pytorch](https://pytorch.org) and [Google Colab](https://colab.research.google.com/) where **free** GPU is available.
Autograder of this course is unfortunately **locked**(so grade to my solution will not be shown)...

## Homewok 7
In folder `hw7_files`.  
[Handout](https://eecs189.org/docs/hw7.pdf)  
I completed task 2 of it which is using [Pytorch Geometric Library](https://pytorch-geometric.readthedocs.io/en/latest/) to train a graph neuron network(a.k.a GNN). SInce it will take hours to train, be patient and leave enough time for it!

# Where to Go Next
Computer science is really hard. Be careful of what to choose.  
The **end**.
