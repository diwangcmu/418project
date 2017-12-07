package go;

import go.GameBoard.State;

/**
 * Basic game element.
 *
 */
public class Stone {

    public Chain chain;
    public State state;
    public int liberties;
    // Row and col are need to remove (set to null) this Stone from Grid
    public int row;
    public int col;

    public Stone(int row, int col, State state) {
        chain = null;
        this.state = state;
        this.row = row;
        this.col = col;

        if ((row == 0 && col == 0) || (row == 18 && col == 18) || (row == 0 && col == 18) || (row == 18 && col == 0)){
            liberties = 2;
        } else if ((row == 0) || (row == 18) || (col == 0) || (col == 18)){
            liberties = 3;
        } else {
            liberties = 4;
        }
    }
}
