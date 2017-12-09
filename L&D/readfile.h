#ifndef READFILE_H
#define READFILE_H

#include <vector>

struct stone_pos {
    int row;
    int col;
};

struct Stone;

struct Chain{
	int state;
	Stone* stones[81];
	int chain_size;
};

struct Stone{
	int row;
	int col;
	int state;
	Chain* chain;
};

struct Grid{
	int size;
	Stone* stones[9][9];
	int player;
	int player_reverse;
	Grid* next_grid[81];
};

struct GameBoard{
	int size;
	int current_player_state;
	Grid* initial_grid;
	int black_count;
	int white_count;
	int range_count;
	stone_pos* black_stones[81];
	stone_pos* white_stones[81];
	stone_pos* range_stones[81];
};

#endif