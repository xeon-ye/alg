package zju.se;

import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import junit.framework.TestCase;
import zju.ieeeformat.DefaultIcfParser;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.measure.*;
import zju.util.*;

/**
 * @author <Dong Shufeng>
 * @version 1.0
 * @since <pre>12/12/2007</pre>
 */
public class SeTest_case4 extends TestCase implements SeConstants, MeasTypeCons {

    double tolerance = 1e-3;

    static IEEEDataIsland island;

    static YMatrixGetter y;

    static SystemMeasure sm;

    static {
        island = new DefaultIcfParser().parse(SeTest_case4.class.getResourceAsStream("/ieeefiles/case4.txt"));
        y = new YMatrixGetter(island);
        y.formYMatrix();
    }

    public SeTest_case4(String name) {
        super(name);
        sm = DefaultMeasParser.parse(this.getClass().getResourceAsStream("/measfiles/case4_meas.txt"));
    }

    public void testSE_wls_no_chaintrs() {
        sm.getLine_from_p().get("1").setValue(-0.4179);
        SeTest_IeeeCase seTest = new SeTest_IeeeCase("seTest");
        seTest.doSE(island, sm, null, new int[]{IpoptSeAlg.VARIABLE_VTHETA_PQ}, false, false);
        AVector result = seTest.getAlg().getFinalVTheta();
        for (MeasureInfo info : MeasureUtil.getTwoMeasureList(sm)[0])
            System.out.println(info.getValue() + "\t" + info.getValue_est() + "\t" + (info.getValue() - info.getValue_est()));
        for (MeasureInfo info : MeasureUtil.getTwoMeasureList(sm)[1])
            System.out.println(info.getValue() + "\t" + info.getValue_est() + "\t" + (info.getValue() - info.getValue_est()));
    }

    public void testSE_wls() {
        SeTest_IeeeCase seTest = new SeTest_IeeeCase("seTest");
        seTest.doSE(island, sm, null, new int[]{IpoptSeAlg.VARIABLE_VTHETA}, false, false);
        AVector result = seTest.getAlg().getFinalVTheta();
        checkNormalWls(result);

        WlsSeTest_IeeeCase wlsTest = new WlsSeTest_IeeeCase("seTest");
        wlsTest.doSE(island, sm, null);
        result = wlsTest.getAlg().getFinalVTheta();
        checkNormalWls(result);
    }

    private void checkNormalWls(AVector result) {
        assertTrue(Math.abs(result.getValue(0) - 1.1147) < tolerance);
        assertTrue(Math.abs(result.getValue(1) - 1.1034) < tolerance);
        assertTrue(Math.abs(result.getValue(2) - 1.1170) < tolerance);
        assertTrue(Math.abs(result.getValue(3) - 1.1232) < tolerance);
        assertTrue(Math.abs(result.getValue(4) - 0.0) < tolerance);
        assertTrue(Math.abs(result.getValue(5) - (-0.0069)) < tolerance);
        assertTrue(Math.abs(result.getValue(6) - 0.014) < tolerance);
        assertTrue(Math.abs(result.getValue(7) - 0.1379) < tolerance);
    }

    public void testHessian() {
        AVector state = new AVector(8);
        state.setValue(0, 1);
        state.setValue(1, 1);
        state.setValue(2, 1);
        state.setValue(3, 1);
        state.setValue(4, 0);
        state.setValue(5, 0);
        state.setValue(6, 0);
        state.setValue(7, 0);
        testHessian(state);
        state.setValue(0, 1 + Math.random());
        state.setValue(1, 1 + Math.random());
        state.setValue(2, 1 + Math.random());
        state.setValue(3, 1 + Math.random());
        state.setValue(4, 0 + Math.random());
        state.setValue(5, 0 + Math.random());
        state.setValue(6, 0 + Math.random());
        state.setValue(7, 0 + Math.random());
        testHessian(state);
    }

    public void testHessian(AVector state) {
        MeasVector meas = new MeasVector();
        meas.setMeasureOrder(new int[]{
                TYPE_BUS_VOLOTAGE,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_LINE_FROM_REACTIVE,
                TYPE_LINE_TO_REACTIVE,
                TYPE_BUS_ANGLE,
                TYPE_BUS_ACTIVE_POWER,
                TYPE_LINE_FROM_ACTIVE,
                TYPE_LINE_TO_ACTIVE,
        });
        meas.setBus_v_pos(new int[]{1, 3});
        meas.setBus_q_pos(new int[]{1, 3});
        meas.setLine_from_q_pos(new int[]{1, 2, 4});
        meas.setLine_to_q_pos(new int[]{2, 3});
        meas.setBus_a_pos(new int[0]);
        meas.setBus_p_pos(new int[]{1, 3});
        meas.setLine_from_p_pos(new int[]{1, 2, 4});
        meas.setLine_to_p_pos(new int[]{2, 3});
        meas.setZ(new AVector(new double[]{1.1215, 1.1099, 0.3405, -0.4193, 0.3796, -0.0224, -0.7108, -0.1074, 0.4261,
                0.1871, -0.4949, 0.4179, -0.191, -1.9322, 0.1755, 1.3257}));
        meas.setZ_estimate(new AVector(meas.getZ().getN()));

        ASparseMatrixLink[] admittance = y.getAdmittance();
        StateCalByPolar.getEstimatedZ(meas, y, state);
        SparseDoubleMatrix2D jac = JacobianMakerPC.getJacobianOfVTheta(meas, y, state);
        ASparseMatrixLink2D jacobian = new ASparseMatrixLink2D(jac.rows(), jac.columns());
        ColtMatrixUtil.toMyMatrix(jac, jacobian);
        ASparseMatrixLink2D[] result = new ASparseMatrixLink2D[jacobian.getM()];

        ASparseMatrixLink[] h = HessianMakerPCSlow.getHessian(meas, y, jacobian, state);

        for (int i = 0; i < 2; i++) {
            result[i] = new ASparseMatrixLink2D(state.getN(), state.getN());
        }
        result[2] = new ASparseMatrixLink2D(state.getN(), state.getN());
        result[2].setValue(0, 0, -2 * admittance[1].getValue(0, 0));
        result[2].setValue(0, 1, jacobian.getValue(2, 1) / state.getValue(0));
        result[2].setValue(0, 2, jacobian.getValue(2, 2) / state.getValue(0));
        result[2].setValue(1, 0, jacobian.getValue(2, 1) / state.getValue(0));
        result[2].setValue(2, 0, jacobian.getValue(2, 2) / state.getValue(0));

        result[2].setValue(4, 0, jacobian.getValue(2, 4) / state.getValue(0));
        result[2].setValue(4, 1, -jacobian.getValue(2, 5) / state.getValue(1));
        result[2].setValue(4, 2, -jacobian.getValue(2, 6) / state.getValue(2));
        result[2].setValue(5, 0, jacobian.getValue(2, 5) / state.getValue(0));
        result[2].setValue(6, 0, jacobian.getValue(2, 6) / state.getValue(0));
        result[2].setValue(5, 1, jacobian.getValue(2, 5) / state.getValue(1));
        result[2].setValue(6, 2, jacobian.getValue(2, 6) / state.getValue(2));
        result[2].setValue(4, 4, jacobian.getValue(9, 4));
        result[2].setValue(4, 5, jacobian.getValue(9, 5));
        result[2].setValue(4, 6, jacobian.getValue(9, 6));
        result[2].setValue(5, 4, jacobian.getValue(9, 5));
        result[2].setValue(6, 4, jacobian.getValue(9, 6));
        result[2].setValue(5, 5, -jacobian.getValue(9, 5));
        result[2].setValue(6, 6, -jacobian.getValue(9, 6));

        result[9] = new ASparseMatrixLink2D(state.getN(), state.getN());
        result[9].setValue(0, 0, 2 * admittance[0].getValue(0, 0));
        result[9].setValue(0, 1, jacobian.getValue(9, 1) / state.getValue(0));
        result[9].setValue(0, 2, jacobian.getValue(9, 2) / state.getValue(0));
        result[9].setValue(1, 0, jacobian.getValue(9, 1) / state.getValue(0));
        result[9].setValue(2, 0, jacobian.getValue(9, 2) / state.getValue(0));

        result[9].setValue(4, 0, jacobian.getValue(9, 4) / state.getValue(0));
        result[9].setValue(4, 1, -jacobian.getValue(9, 5) / state.getValue(1));
        result[9].setValue(4, 2, -jacobian.getValue(9, 6) / state.getValue(2));
        result[9].setValue(5, 0, jacobian.getValue(9, 5) / state.getValue(0));
        result[9].setValue(6, 0, jacobian.getValue(9, 6) / state.getValue(0));
        result[9].setValue(5, 1, jacobian.getValue(9, 5) / state.getValue(1));
        result[9].setValue(6, 2, jacobian.getValue(9, 6) / state.getValue(2));

        result[9].setValue(4, 4, -jacobian.getValue(2, 4));
        result[9].setValue(4, 5, -jacobian.getValue(2, 5));
        result[9].setValue(4, 6, -jacobian.getValue(2, 6));
        result[9].setValue(5, 4, -jacobian.getValue(2, 5));
        result[9].setValue(6, 4, -jacobian.getValue(2, 6));
        result[9].setValue(5, 5, jacobian.getValue(2, 5));
        result[9].setValue(6, 6, jacobian.getValue(2, 6));

        int[] toCompare = new int[]{2, 9};
        for (int i = 0; i < toCompare.length; i++) {
            ASparseMatrixLink m1 = h[toCompare[i]];
            ASparseMatrixLink m2 = result[toCompare[i]];
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    //System.out.println("m1(" + j + " + " + k + ") = " + m1.getValue(j, k));
                    //System.out.println("m2(" + j + " + " + k + ") = " + m2.getValue(j, k));
                    assertTrue(Math.abs(m1.getValue(j, k) - m2.getValue(j, k)) < 10e-8);
                }
            }
        }
    }
}
