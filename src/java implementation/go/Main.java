package go;

import java.awt.BorderLayout;
import java.awt.Color;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JPanel;

/**
 * Builds UI and starts the game.
 *
 */
public class Main {

    public static final String TITLE = "Parallel Go UI";
    public static final int BORDER_SIZE = 10;

    public static void main(String[] args) {
        new Main().init();
    }

    public static void mymain() {
        init();
    }

    private static void init() {
        System.out.println("Yes!");
        System.out.println("Ohh!");

        JFrame f = new JFrame();
        System.out.println("No!");
        f.setTitle(TITLE);

        JPanel container = new JPanel();
        container.setBackground(Color.GRAY);
        container.setLayout(new BorderLayout());
        f.add(container);
        container.setBorder(BorderFactory.createEmptyBorder(BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE * 10));
        System.out.println("Yes!");

        GameBoard board = new GameBoard();
        container.add(board);

        System.out.println("Yes!");
        f.pack();
        f.setResizable(true);
        f.setLocationByPlatform(true);
        f.setVisible(true);
    }}
