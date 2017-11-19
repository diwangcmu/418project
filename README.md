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
  
### Checkpoint Report  

The work done so far:
We successfully implemented the Go UI, along with the capture rules and ending rules. The UI in Java prompts up a yellow board, and allows the users to place black and white stones in order. If a move is prohibited then the user cannot place the move, but can still place a valid move afterwards. 
In terms of the life and death problem AI, we first debated on the model and algorithm behind the AI. Originally we decided on using Cuda and Alpha-Beta pruning, but then found that it is very hard to share messages between Cuda threads. 
One approach is to declare shared memory arrays in a Cuda block, which we started to implement. We also borrowed an idea from a previous project, which is to first run one branch of Alpha-Beta pruning sequentially, and then use the pruning results to run the rest in parallel. However, the bottleneck to this approach is that it is hard to find a useful and accurate heuristic for Go. Some small changes in the board state in one area can affect the entire balance of the board. Therefore the Alpha-Beta pruning returned very poor results. We decided to not go along that approach.
After the unsatisfactory results with Alpha-Beta tree, we decided to implement a simple Tree search in parallel. We still used Cuda, but this time the threads are divided with respect to the combination of the first two possible moves given the current board state (e.g. black places at (3,3), and white places at (3,4)). Then the search will continue until we reached the ending stages.
Here are some screenshots of the AI’s decision of whether a piece is alive or dead:

Goals and Deliverables:
We believe that our original goal of implementing an interactive tsumego (life and death problem) AI is achievable. However, the full Go game interactive AI might be a stretch. It is hard to exhaust every possibility, since there are around 3^(19*19) possibilities, resulting from each point on the board being either unoccupied, black or white. Also, we found that it is hard to implement a parallel tree search algorithm given a full game board. Therefore pattern matching is required to reduce the possibilities to search for, but we have not started on that. Therefore this becomes a “nice to have”. 
The revised list of goals is as follows:
Fully implemented Go life and death problem programs that respond to a user’s attempts
Go life and death solver that gives the correct solution or indicates if there is no solution
Small board Go AI that allows for board state evaluation

Poster Session:
We plan to show the above interactive game demos at the poster session. 

Schedule for coming weeks:
- Week of 11/19 - 11/25:
	First Half: Di: Implement the interactive tsumego program. Bo: Implement ending solver
	Second Half: Di & Bo: Implement the parallel tree search solver
- Week of 11/26 - 12/2:
	Di: Adapt the solver to be the underlying AI for the program. Bo: Implement the heuristic of evaluating the current board state.
- Week of 12/3 - 12/9:
  Di & Bo: Implement the game AI for board size of 9*9; possibly the “nice to haves” done successfully.


