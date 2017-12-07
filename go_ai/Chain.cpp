#include <vector>

struct Stone;

struct Chain{
	int state;
	int chain_size;
	Stone* stones[81];
};

void chain_addStone(Chain* this_chain, Stone* s){
	s->chain = this_chain;
	this_chain->stones[this_chain->chain_size] = s;
	this_chain->chain_size += 1;
}

void chain_join(Chain* this_chain, Chain* chain){
	for (int i = 0; i != chain->chain_size; ++i){
		chain_addStone(this_chain, chain->stones[i]);
	}
}

