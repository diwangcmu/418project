#include <vector>

class Stone;

class Chain{
	public:

	int state;
	std::vector <Stone*> stones;

	void construct(){
		// do nothing
	}

	void addStone(Stone* s){
		s->chain = this;
		this->stones.push_back(s);
	}

	void join(Chain* chain){
		for (int i = 0; i != chain->stones.size(); ++i){
			this->addStone(chain->stones[i]);
		}
	}



};