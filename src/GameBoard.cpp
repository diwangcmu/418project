#include "Grid.cpp"

class GameBoard{
	public:

	int size;
	int current_player_state; // 1 for black, 0 for white
	Grid* initial_grid;

	int black_count;
	int white_count;
	int range_count;

	stone_pos* black_stones[81];
	stone_pos* white_stones[81];
	stone_pos* range_stones[81];

	void construct(int s, int black, stone_pos* blacks[], int white, stone_pos* whites[], int range, stone_pos* ranges[]){
		this->size = s;
		this->initial_grid = new Grid;
		this->initial_grid->construct(s, 0);
		this->current_player_state = 0;

		this->black_count = black;
		for (int i=0; i<black; i++){
			this->initial_grid->addStone(blacks[i]->row, blacks[i]->col, 1);
			this->black_stones[i] = blacks[i];
		}

		this->white_count = white;
		for (int i=0; i<white; i++){
			this->initial_grid->addStone(whites[i]->row, whites[i]->col, 0);
			this->white_stones[i] = whites[i];
		}

		this->range_count = range;
		for (int i=0; i<range; i++){
			this->range_stones[i] = ranges[i];
		}
	}

	int search(){
		this->initial_grid->printboard();
		return this->initial_grid->search_first(this->range_count, this->range_stones, this->black_count, this->black_stones);
	}
};