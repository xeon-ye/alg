package zju.dspf;

import Jama.Matrix;
import zju.devmodel.MapObject;
import zju.dsmodel.Feeder;
import zju.dsmodel.GeneralBranch;

import java.util.Map;

/**
 * Created by zjq on 2017/4/26.
 */
public class YMatrix {
    private Matrix[] yMatrix;
    private int size;
    private Map<MapObject, GeneralBranch> allBranches;


    public YMatrix(Map<MapObject, GeneralBranch> branches, int size) {
        this.allBranches = branches;
        this.size = size;
        yMatrix = new Matrix[3];
    }

    public void formInitYMatrix() {
        Matrix yMatrixPhaseA = new Matrix(size, 2 * size, 0);
        Matrix yMatrixPhaseB = new Matrix(size, 2 * size, 0);
        Matrix yMatrixPhaseC = new Matrix(size, 2 * size, 0);

        for (MapObject i : allBranches.keySet()) {
            String branchName = i.getName();
            String[] nodeNames = branchName.split("-");
            int[] nodes = new int[2];
            //nodename从1开始编号
            nodes[0] = Integer.parseInt(nodeNames[0]) - 1;
            nodes[1] = Integer.parseInt(nodeNames[1]) - 1;

            Feeder feeder = (Feeder) allBranches.get(i);
            double[][] z_real = feeder.getZ_real();
            double[][] z_image = feeder.getZ_imag();

            //A相Y矩阵
            double y_real = z_real[0][0] / (z_real[0][0] * z_real[0][0] + z_image[0][0] * z_image[0][0]);
            double y_imag = -z_image[0][0] / (z_real[0][0] * z_real[0][0] + z_image[0][0] * z_image[0][0]);
            yMatrixPhaseA.set(nodes[0], 2 * nodes[0], yMatrixPhaseA.get(nodes[0], 2 * nodes[0]) + y_real);
            yMatrixPhaseA.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseA.get(nodes[0], 2 * nodes[0] + 1) + y_imag);
            yMatrixPhaseA.set(nodes[1], 2 * nodes[1], yMatrixPhaseA.get(nodes[1], 2 * nodes[1]) + y_real);
            yMatrixPhaseA.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseA.get(nodes[1], 2 * nodes[1] + 1) + y_imag);
            yMatrixPhaseA.set(nodes[0], 2 * nodes[1], yMatrixPhaseA.get(nodes[0], 2 * nodes[1]) - y_real);
            yMatrixPhaseA.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseA.get(nodes[0], 2 * nodes[1] + 1) - y_imag);
            yMatrixPhaseA.set(nodes[1], 2 * nodes[0], yMatrixPhaseA.get(nodes[1], 2 * nodes[0]) - y_real);
            yMatrixPhaseA.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseA.get(nodes[1], 2 * nodes[0] + 1) - y_imag);
            yMatrix[0] = yMatrixPhaseA;

            //B相Y矩阵
            y_real = z_real[1][1] / (z_real[1][1] * z_real[1][1] + z_image[1][1] * z_image[1][1]);
            y_imag = -z_image[1][1] / (z_real[1][1] * z_real[1][1] + z_image[1][1] * z_image[1][1]);
            yMatrixPhaseB.set(nodes[0], 2 * nodes[0], yMatrixPhaseB.get(nodes[0], 2 * nodes[0]) + y_real);
            yMatrixPhaseB.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseB.get(nodes[0], 2 * nodes[0] + 1) + y_imag);
            yMatrixPhaseB.set(nodes[1], 2 * nodes[1], yMatrixPhaseB.get(nodes[1], 2 * nodes[1]) + y_real);
            yMatrixPhaseB.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseB.get(nodes[1], 2 * nodes[1] + 1) + y_imag);
            yMatrixPhaseB.set(nodes[0], 2 * nodes[1], yMatrixPhaseB.get(nodes[0], 2 * nodes[1]) - y_real);
            yMatrixPhaseB.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseB.get(nodes[0], 2 * nodes[1] + 1) - y_imag);
            yMatrixPhaseB.set(nodes[1], 2 * nodes[0], yMatrixPhaseB.get(nodes[1], 2 * nodes[0]) - y_real);
            yMatrixPhaseB.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseB.get(nodes[1], 2 * nodes[0] + 1) - y_imag);
            yMatrix[1] = yMatrixPhaseB;

            //C相Y矩阵
            y_real = z_real[2][2] / (z_real[2][2] * z_real[2][2] + z_image[2][2] * z_image[2][2]);
            y_imag = -z_image[2][2] / (z_real[2][2] * z_real[2][2] + z_image[2][2] * z_image[2][2]);
            yMatrixPhaseC.set(nodes[0], 2 * nodes[0], yMatrixPhaseC.get(nodes[0], 2 * nodes[0]) + y_real);
            yMatrixPhaseC.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseC.get(nodes[0], 2 * nodes[0] + 1) + y_imag);
            yMatrixPhaseC.set(nodes[1], 2 * nodes[1], yMatrixPhaseC.get(nodes[1], 2 * nodes[1]) + y_real);
            yMatrixPhaseC.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseC.get(nodes[1], 2 * nodes[1] + 1) + y_imag);
            yMatrixPhaseC.set(nodes[0], 2 * nodes[1], yMatrixPhaseC.get(nodes[0], 2 * nodes[1]) - y_real);
            yMatrixPhaseC.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseC.get(nodes[0], 2 * nodes[1] + 1) - y_imag);
            yMatrixPhaseC.set(nodes[1], 2 * nodes[0], yMatrixPhaseC.get(nodes[1], 2 * nodes[0]) - y_real);
            yMatrixPhaseC.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseC.get(nodes[1], 2 * nodes[0] + 1) - y_imag);
            yMatrix[2] = yMatrixPhaseC;
        }
    }

    //不改变allbranches中的内容
    public void deleteBranch(String branchName) {
        String[] nodeNames = branchName.split("-");
        int[] nodes = new int[2];
        nodes[0] = Integer.parseInt(nodeNames[0]) - 1;
        nodes[1] = Integer.parseInt(nodeNames[1]) - 1;
        double[][] z_real = null;
        double[][] z_image = null;

        for (MapObject i : allBranches.keySet()) {
            if (i.getName().equals(branchName)) {
                Feeder feeder = (Feeder) allBranches.get(i);
                z_real = feeder.getZ_real();
                z_image = feeder.getZ_imag();
                break;
            }
        }

        //A相Y矩阵
        Matrix yMatrixPhaseA = yMatrix[0];
        double y_real = z_real[0][0] / (z_real[0][0] * z_real[0][0] + z_image[0][0] * z_image[0][0]);
        double y_imag = -z_image[0][0] / (z_real[0][0] * z_real[0][0] + z_image[0][0] * z_image[0][0]);
        yMatrixPhaseA.set(nodes[0], 2 * nodes[0], yMatrixPhaseA.get(nodes[0], 2 * nodes[0]) - y_real);
        yMatrixPhaseA.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseA.get(nodes[0], 2 * nodes[0] + 1) - y_imag);
        yMatrixPhaseA.set(nodes[1], 2 * nodes[1], yMatrixPhaseA.get(nodes[1], 2 * nodes[1]) - y_real);
        yMatrixPhaseA.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseA.get(nodes[1], 2 * nodes[1] + 1) - y_imag);
        yMatrixPhaseA.set(nodes[0], 2 * nodes[1], yMatrixPhaseA.get(nodes[0], 2 * nodes[1]) + y_real);
        yMatrixPhaseA.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseA.get(nodes[0], 2 * nodes[1] + 1) + y_imag);
        yMatrixPhaseA.set(nodes[1], 2 * nodes[0], yMatrixPhaseA.get(nodes[1], 2 * nodes[0]) + y_real);
        yMatrixPhaseA.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseA.get(nodes[1], 2 * nodes[0] + 1) + y_imag);


        //B相Y矩阵
        Matrix yMatrixPhaseB = yMatrix[1];
        y_real = z_real[1][1] / (z_real[1][1] * z_real[1][1] + z_image[1][1] * z_image[1][1]);
        y_imag = -z_image[1][1] / (z_real[1][1] * z_real[1][1] + z_image[1][1] * z_image[1][1]);
        yMatrixPhaseB.set(nodes[0], 2 * nodes[0], yMatrixPhaseB.get(nodes[0], 2 * nodes[0]) - y_real);
        yMatrixPhaseB.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseB.get(nodes[0], 2 * nodes[0] + 1) - y_imag);
        yMatrixPhaseB.set(nodes[1], 2 * nodes[1], yMatrixPhaseB.get(nodes[1], 2 * nodes[1]) - y_real);
        yMatrixPhaseB.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseB.get(nodes[1], 2 * nodes[1] + 1) - y_imag);
        yMatrixPhaseB.set(nodes[0], 2 * nodes[1], yMatrixPhaseB.get(nodes[0], 2 * nodes[1]) + y_real);
        yMatrixPhaseB.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseB.get(nodes[0], 2 * nodes[1] + 1) + y_imag);
        yMatrixPhaseB.set(nodes[1], 2 * nodes[0], yMatrixPhaseB.get(nodes[1], 2 * nodes[0]) + y_real);
        yMatrixPhaseB.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseB.get(nodes[1], 2 * nodes[0] + 1) + y_imag);


        //C相Y矩阵
        Matrix yMatrixPhaseC = yMatrix[2];
        y_real = z_real[2][2] / (z_real[2][2] * z_real[2][2] + z_image[2][2] * z_image[2][2]);
        y_imag = -z_image[2][2] / (z_real[2][2] * z_real[2][2] + z_image[2][2] * z_image[2][2]);
        yMatrixPhaseC.set(nodes[0], 2 * nodes[0], yMatrixPhaseC.get(nodes[0], 2 * nodes[0]) - y_real);
        yMatrixPhaseC.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseC.get(nodes[0], 2 * nodes[0] + 1) - y_imag);
        yMatrixPhaseC.set(nodes[1], 2 * nodes[1], yMatrixPhaseC.get(nodes[1], 2 * nodes[1]) - y_real);
        yMatrixPhaseC.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseC.get(nodes[1], 2 * nodes[1] + 1) - y_imag);
        yMatrixPhaseC.set(nodes[0], 2 * nodes[1], yMatrixPhaseC.get(nodes[0], 2 * nodes[1]) + y_real);
        yMatrixPhaseC.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseC.get(nodes[0], 2 * nodes[1] + 1) + y_imag);
        yMatrixPhaseC.set(nodes[1], 2 * nodes[0], yMatrixPhaseC.get(nodes[1], 2 * nodes[0]) + y_real);
        yMatrixPhaseC.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseC.get(nodes[1], 2 * nodes[0] + 1) + y_imag);
    }
    //不改变allbranches中的内容
    public void addBranch(String branchName) {
        String[] nodeNames = branchName.split("-");
        int[] nodes = new int[2];
        nodes[0] = Integer.parseInt(nodeNames[0]) - 1;
        nodes[1] = Integer.parseInt(nodeNames[1]) - 1;
        double[][] z_real = null;
        double[][] z_image = null;

        for (MapObject i : allBranches.keySet()) {
            if (i.getName().equals(branchName)) {
                Feeder feeder = (Feeder) allBranches.get(i);
                z_real = feeder.getZ_real();
                z_image = feeder.getZ_imag();
                break;
            }
        }

        //A相Y矩阵
        Matrix yMatrixPhaseA = yMatrix[0];
        double y_real = z_real[0][0] / (z_real[0][0] * z_real[0][0] + z_image[0][0] * z_image[0][0]);
        double y_imag = -z_image[0][0] / (z_real[0][0] * z_real[0][0] + z_image[0][0] * z_image[0][0]);
        yMatrixPhaseA.set(nodes[0], 2 * nodes[0], yMatrixPhaseA.get(nodes[0], 2 * nodes[0]) + y_real);
        yMatrixPhaseA.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseA.get(nodes[0], 2 * nodes[0] + 1) + y_imag);
        yMatrixPhaseA.set(nodes[1], 2 * nodes[1], yMatrixPhaseA.get(nodes[1], 2 * nodes[1]) + y_real);
        yMatrixPhaseA.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseA.get(nodes[1], 2 * nodes[1] + 1) + y_imag);
        yMatrixPhaseA.set(nodes[0], 2 * nodes[1], yMatrixPhaseA.get(nodes[0], 2 * nodes[1]) - y_real);
        yMatrixPhaseA.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseA.get(nodes[0], 2 * nodes[1] + 1) - y_imag);
        yMatrixPhaseA.set(nodes[1], 2 * nodes[0], yMatrixPhaseA.get(nodes[1], 2 * nodes[0]) - y_real);
        yMatrixPhaseA.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseA.get(nodes[1], 2 * nodes[0] + 1) - y_imag);


        //B相Y矩阵
        Matrix yMatrixPhaseB = yMatrix[1];
        y_real = z_real[1][1] / (z_real[1][1] * z_real[1][1] + z_image[1][1] * z_image[1][1]);
        y_imag = -z_image[1][1] / (z_real[1][1] * z_real[1][1] + z_image[1][1] * z_image[1][1]);
        yMatrixPhaseB.set(nodes[0], 2 * nodes[0], yMatrixPhaseB.get(nodes[0], 2 * nodes[0]) + y_real);
        yMatrixPhaseB.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseB.get(nodes[0], 2 * nodes[0] + 1) + y_imag);
        yMatrixPhaseB.set(nodes[1], 2 * nodes[1], yMatrixPhaseB.get(nodes[1], 2 * nodes[1]) + y_real);
        yMatrixPhaseB.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseB.get(nodes[1], 2 * nodes[1] + 1) + y_imag);
        yMatrixPhaseB.set(nodes[0], 2 * nodes[1], yMatrixPhaseB.get(nodes[0], 2 * nodes[1]) - y_real);
        yMatrixPhaseB.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseB.get(nodes[0], 2 * nodes[1] + 1) - y_imag);
        yMatrixPhaseB.set(nodes[1], 2 * nodes[0], yMatrixPhaseB.get(nodes[1], 2 * nodes[0]) - y_real);
        yMatrixPhaseB.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseB.get(nodes[1], 2 * nodes[0] + 1) - y_imag);


        //C相Y矩阵
        Matrix yMatrixPhaseC = yMatrix[2];
        y_real = z_real[2][2] / (z_real[2][2] * z_real[2][2] + z_image[2][2] * z_image[2][2]);
        y_imag = -z_image[2][2] / (z_real[2][2] * z_real[2][2] + z_image[2][2] * z_image[2][2]);
        yMatrixPhaseC.set(nodes[0], 2 * nodes[0], yMatrixPhaseC.get(nodes[0], 2 * nodes[0]) + y_real);
        yMatrixPhaseC.set(nodes[0], 2 * nodes[0] + 1, yMatrixPhaseC.get(nodes[0], 2 * nodes[0] + 1) + y_imag);
        yMatrixPhaseC.set(nodes[1], 2 * nodes[1], yMatrixPhaseC.get(nodes[1], 2 * nodes[1]) + y_real);
        yMatrixPhaseC.set(nodes[1], 2 * nodes[1] + 1, yMatrixPhaseC.get(nodes[1], 2 * nodes[1] + 1) + y_imag);
        yMatrixPhaseC.set(nodes[0], 2 * nodes[1], yMatrixPhaseC.get(nodes[0], 2 * nodes[1]) - y_real);
        yMatrixPhaseC.set(nodes[0], 2 * nodes[1] + 1, yMatrixPhaseC.get(nodes[0], 2 * nodes[1] + 1) - y_imag);
        yMatrixPhaseC.set(nodes[1], 2 * nodes[0], yMatrixPhaseC.get(nodes[1], 2 * nodes[0]) - y_real);
        yMatrixPhaseC.set(nodes[1], 2 * nodes[0] + 1, yMatrixPhaseC.get(nodes[1], 2 * nodes[0] + 1) - y_imag);
    }

    public Matrix[] clone() {
        Matrix[] matrixArray = new Matrix[yMatrix.length];
        for (int i = 0; i < yMatrix.length; i++) {
            matrixArray[i] = yMatrix[i].copy();
        }
        return matrixArray;
    }

    public void print(int w,int d){
        System.out.println("A相导纳矩阵：");
        yMatrix[0].print(w,d);
        System.out.println("B相导纳矩阵：");
        yMatrix[1].print(w,d);
        System.out.println("C相导纳矩阵：");
        yMatrix[2].print(w,d);
    }

    public Matrix[] getyMatrix() {
        return yMatrix;
    }
}
