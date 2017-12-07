package go;

public class Decision {

    public native boolean start(int size);
    public native int getResponseMove(int move);
    
    public boolean init_AI(int size) {
        return start(size);
    }

    public int placeMove_AI(int move) {
        return getResponseMove(move);
    }

    static {
        System.loadLibrary("go_Decision");
    }

}