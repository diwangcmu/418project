## Go AI
Project for 15-418 Fall 2017, by Di Wang (diw2) and Bo Gao (bgao).
### Summary		

We are going to implement a Go AI using tree search algorithm on CUDA in the GHC clusters, and report a detailed performance analysis with respect to different factors including board size, CUDA block size, memory access frequency.

### Background

Go: 
Go is an abstract strategy board-game for two players, in which the aim is to surround more territory than the opponent. Our interactive application is the artificial intelligence creating a computer program that plays the game with users. 

Alpha-beta pruning:
The idea for tree search is Alpha-beta pruning which is an adversarial search algorithm used commonly for machine playing of two-player games. It is a search algorithm that seeks to decrease the number of nodes that are evaluated by the minimax algorithm in its search tree. It stops completely evaluating a move when at least one possibility has been found that proves the move to be worse than a previously examined move. Such moves need not be evaluated further. 

[Benson’s algorithm](https://en.wikipedia.org/wiki/Benson%27s_algorithm_(Go)):
In the game Go, Benson's algorithm (named after David B. Benson) can be used to determine the stones which are safe from capture no matter how many turns in a row the opposing player gets, i.e. unconditionally alive.

### Challenge

First of all, the workload of this problem is too large for not only there are a huge number of possible points to make a current move, but even more potential conditions on the following sequence of moves. It’s really hard to achieve considerable speedup on such large problem size. 

Additionally, we cannot use tree search alone for the best decision of each move, because of the great number of possibilities that makes exhausting every one of them impossible. Therefore we have to incorporate the use of expert knowledge for a more clever searching algorithm. The task to take these heuristics, formalize them into computer code, and utilize pattern matching algorithms to recognize these general rules is challenging to implement. 

### Resources

The resources we use are basically GHC cluster computers, and we will start from scratch. Also, we will reference to Monte-Carlo Go Developments (B. Bouzy and B. Helmstetter), a paper that explores an intermediate approach in which a Go program performs a global search using very little knowledge.

We also need to find resources on detailed pattern matching algorithms as well as specific implementations, but are still in search of those. 

### Goals and Deliverables

The first thing that we plan to achieve is a full UI of Go that is able to interact with the user. 
The application should follow the general rules of Go.

And we also plan to achieve a life and death problem solver for Go. This solver should provide the correct sequence of moves to the problem, and can respond to any user’s incorrect moves. This requires the solver to know in advance which is the optimal sequence of moves given the current board state, which closely resembles that in an actual Go game.

We hope to achieve a competitive Go AI that can beat amateur players. The AI should be able to do the basics achievable to any human player, to evaluate which player is in an advantage given any board state, and to what extent. It can also project if a piece is dead or alive. 

The demo we plan to show is a graphical, interactive interface. It has two modes, one as the life and death problem solver, and the other as a Go playing program which can play reasonably with the user. 

We will analyze on whether parallelism helps in the search algorithm, and if so, to what extent. We would also analyze the importance of granularity in the parallel algorithm. Also, we are curious if correctness will be affected when the factors of the parallel algorithm change, including the block size and thread number.

### Platform choice: 

We decide to use CUDA as the platform for our main algorithm. This means that we are using the Shared Address Space model, which has many benefits over Message Passing and Data Parallel. 

Our model will not need to have communications between the threads, because the decision for the best move only depends on the current board state, but not the sequence of moves that reaches it. Therefore making any form of communication between threads would only incur unnecessary overhead.

Also, data parallelism is not necessary because there is really no data that we can parallelize on.

Our idea right now is to dynamically make the threads proceed with a work queue of the different possibilities. The system would still perform the calculations even when the next move by the player is yet to be placed. When the move is finalized, we intend to make all the invalid possibilities (i.e. board states that could no longer happen) disappear. 

### Schedule

Week 1: 
  - Implement the Go UI using Python Pygame 
  - Begin reading materials for Alpha-beta pruning tree and Monte-Carlo tree
  
Week 2: 
  - Implement the sequential tree search algorithm for life and death problems
  - Implement the parallel version of tree search algorithm 
  - Analyze the performance
  
Week 3:
  - Apply Alpha-beta algorithm on small board UI
  - Apply Monte-Carlo algorithm on small board UI
  - Begin reading materials for pattern matching

Week 4: 
  - Apply pattern matching to small board
  - Try to implement neural network 
  - Try to apply the UI on larger board size
  - Testing and refinement


