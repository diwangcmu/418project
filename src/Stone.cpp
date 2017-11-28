#include <iostream>

class Chain;

class Stone{
	public:

	int row;
	int col;
	int state;
	Chain* chain;


	void construct(int row, int col, int state){
		this->state = state;
		this->row = row;
		this->col = col;
		this->chain = NULL;
	}
};