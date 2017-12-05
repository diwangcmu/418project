package go;

import go.GameBoard.State;
import java.util.HashSet;

/**
 * Provides game logic.
 *
 *
 */
public class Grid {

    private final int SIZE;
    /**
     * [row][column]
     */
    private Stone[][] stones;

    public Grid(int size) {
        SIZE = size;
        stones = new Stone[SIZE][SIZE];
    }

    /**
     * Adds Stone to Grid.
     *
     * @param row
     * @param col
     * @param black
     */
    public int addStone(int row, int col, State state) {
        Stone newStone = new Stone(row, col, state);
        stones[row][col] = newStone;
        // Check neighbors
        Stone[] neighbors = new Stone[4];
        neighbors[0] = null;
        neighbors[1] = null;
        neighbors[2] = null;
        neighbors[3] = null;

        // Don't check outside the board
        if (row > 0) {
            neighbors[0] = stones[row - 1][col];
        }
        if (row < SIZE - 1) {
            neighbors[1] = stones[row + 1][col];
        }
        if (col > 0) {
            neighbors[2] = stones[row][col - 1];
        }
        if (col < SIZE - 1) {
            neighbors[3] = stones[row][col + 1];
        }

        // Prepare Chain for this new Stone

        Chain current_chain = new Chain();
        current_chain.addStone(newStone);

        int flag = 0;
        for (Stone neighbor : neighbors) {
            // do nothing if no adjacent Stone
            if (neighbor != null) {
//                newStone.liberties -= 1;
//                neighbor.liberties -= 1;

                if (neighbor.state != newStone.state) {
                    if (checkStone(neighbor) == 1) {
                        flag = 1;
                    }
                } else {
                    if (neighbor.chain != newStone.chain) {
                        neighbor.chain.join(newStone.chain);
                    }
                }
            }
        }

        if (getLiberties(newStone.chain) == 0 && flag == 0){
           stones[row][col] = null;
           return 0;
        }

        return 1;
        //current_chain.addStone(newStone);
    }

    public int getLiberties(Chain cur_chain){
        HashSet h = new HashSet();
        int pos;

        for (Stone s : cur_chain.stones){

            int row = s.row;
            int col = s.col;

            Stone[] neighbors = new Stone[4];
            // Don't check outside the board
            if (row > 0) {
                neighbors[0] = stones[row - 1][col];
                pos = (row-1) * SIZE + col;
                if (neighbors[0] == null && !h.contains(pos)){
                    h.add(pos);
                }
            }
            if (row < SIZE - 1) {
                neighbors[1] = stones[row + 1][col];
                pos = (row+1) * SIZE + col;
                if (neighbors[1] == null && !h.contains(pos)){
                    h.add(pos);
                }
            }
            if (col > 0) {
                neighbors[2] = stones[row][col - 1];
                pos = row * SIZE + col - 1;
                if (neighbors[2] == null && !h.contains(pos)){
                    h.add(pos);
                }
            }
            if (col < SIZE - 1) {
                neighbors[3] = stones[row][col + 1];
                pos = row * SIZE + col + 1;
                if (neighbors[3] == null && !h.contains(pos)){
                    h.add(pos);
                }
            }
        }
        return h.size();
    }


    /**
     * Check liberties of Stone
     *
     * @param stone
     */
    public int checkStone(Stone stone) {
        // Every Stone is part of a Chain so we check total liberties

        int flag = 0;
        if (getLiberties(stone.chain) == 0) {
            flag = 1;
            for (Stone s : stone.chain.stones) {
                stones[s.row][s.col] = null;
            }
        }
        return flag;
    }


    /**
     * Returns true if given position is occupied by any stone
     *
     * @param row
     * @param col
     * @return true if given position is occupied
     */
    public boolean isOccupied(int row, int col) {
        return stones[row][col] != null;
    }

    /**
     * Returns State (black/white) of given position or null if it's unoccupied.
     * Needs valid row and column.
     *
     * @param row
     * @param col
     * @return
     */
    public State getState(int row, int col) {
        Stone stone = stones[row][col];
        if (stone == null) {
            return null;
        } else {
            // System.out.println("getState != null");
            return stone.state;
        }
    }
}
