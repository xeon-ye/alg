package zju.se;

import Jama.Matrix;

import java.util.List;

/**
 * Created by he on 2017/4/7.
 */
public class Parameter {
    private static List<Matrix> matrixList;

    public static List<Matrix> getMatrixList() {
        return matrixList;
    }

    public static void setMatrixList(List<Matrix> matrixList) {
        Parameter.matrixList = matrixList;
    }
}
