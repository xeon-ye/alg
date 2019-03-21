package zju.matrix;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import zju.ieeeformat.DefaultIcfParser;
import zju.ieeeformat.IEEEDataIsland;
import zju.measure.*;
import zju.util.NumberOptHelper;
import zju.util.YMatrixGetter;

import java.io.InputStream;
import java.net.URL;

/**
 * Created by IntelliJ IDEA.
 * author: wangbin
 * date: 2010-7-14
 */
public class ASparseMatrixTest_db implements MeasTypeCons {

    //public static Logger log = LogManager.getLogger(ASparseMatrixTest_db.class);
    private static boolean bUsingColt = true;
    ASparseMatrixLink2D result;          // 109ms 172ms(t43)
    DoubleMatrix2D resultColt;           // 21 ms 47ms(t43)

    @Before
    public void setup() {
    }

    @Test
    public void testJacobinMatrixTime() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/20101016_0050_island.txt");
        InputStream measFile = this.getClass().getResourceAsStream("/measfiles/20101016_0050_meas.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "GBK");
        SystemMeasure sm = MeasureFileRw.parse(measFile);

        NumberOptHelper numOpt = new NumberOptHelper();
        numOpt.simple(island);
        numOpt.trans(island);
        MeasureUtil.trans(sm, numOpt.getOld2new());

        MeasVector meas = new MeasVectorCreator().getMeasureVector(sm);
        YMatrixGetter admittanceGetter = new YMatrixGetter(island);
        admittanceGetter.formYMatrix();
        long start = System.currentTimeMillis();
        //Y.getAdmittance()[0].formLinkStructure();
        //Y.getAdmittance()[1].formLinkStructure();
        System.out.println("Time used for forming link structure of admittance matrix: " + (System.currentTimeMillis() - start) + "ms");
        int n = island.getBuses().size();
        int m = meas.getZ().getN();
        result = new ASparseMatrixLink2D(m, 2 * n);
        resultColt = DoubleFactory2D.sparse.make(m, 2 * n);
        int index = 0;
        long s = System.currentTimeMillis();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        setValue(index, meas.getBus_a_pos()[i] + n - 1, 1);
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        setValue(index, meas.getBus_v_pos()[i] - 1, 1);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        ASparseMatrixLink[] admittance = admittanceGetter.getAdmittance();
                        int num = meas.getBus_p_pos()[i] - 1;//num starts from 0
                        int k = admittance[0].getIA()[num];
                        while (k != -1) {
                            int j = admittance[0].getJA().get(k);
                            if (num == j) {
                                setValue(index, num, 1);
                                setValue(index, num + n, 1);
                                k = admittance[0].getLINK().get(k);
                                continue;
                            }
                            setValue(index, j, 1);
                            setValue(index, j + n, 1);
                            k = admittance[0].getLINK().get(k);
                        }
                        setValue(index, num + n, 1, true);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        ASparseMatrixLink[] admittance = admittanceGetter.getAdmittance();
                        int num = meas.getBus_q_pos()[i] - 1;//num starts from 0
                        int k = admittance[0].getIA()[num];
                        while (k != -1) {
                            int j = admittance[0].getJA().get(k);
                            if (num == j) {
                                setValue(index, num, 1);
                                setValue(index, num + n, 1);
                                k = admittance[0].getLINK().get(k);
                                continue;
                            }
                            setValue(index, j, 1);
                            setValue(index, j + n, 1);
                            k = admittance[0].getLINK().get(k);
                        }
                        setValue(index, num + n, 1, true);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int[] ij = admittanceGetter.getFromTo(meas.getLine_from_p_pos()[k]);
                        int i = ij[0] - 1;
                        int j = ij[1] - 1;
                        setValue(index, i, 1, true);
                        setValue(index, j, 1, true);
                        setValue(index, i + n, 1, true);
                        setValue(index, j + n, 1, true);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int[] ij = admittanceGetter.getFromTo(meas.getLine_from_q_pos()[k]);
                        int i = ij[0] - 1;
                        int j = ij[1] - 1;
                        setValue(index, i, 1, true);
                        setValue(index, j, 1, true);
                        setValue(index, i + n, 1, true);
                        setValue(index, j + n, 1, true);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int[] ij = admittanceGetter.getFromTo(meas.getLine_to_p_pos()[k]);
                        int i = ij[0] - 1;
                        int j = ij[1] - 1;
                        setValue(index, i, 1, true);
                        setValue(index, j, 1, true);
                        setValue(index, i + n, 1, true);
                        setValue(index, j + n, 1, true);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int[] ij = admittanceGetter.getFromTo(meas.getLine_to_q_pos()[k]);
                        int i = ij[0] - 1;
                        int j = ij[1] - 1;
                        setValue(index, i, 1, true);
                        setValue(index, j, 1, true);
                        setValue(index, i + n, 1, true);
                        setValue(index, j + n, 1, true);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
        //result.formLinkStructure();
        System.out.println("Time used for form jac " + (System.currentTimeMillis() - s) + "ms");
    }

    private void setValue(int index, int p1, int v) {
        setValue(index, p1, v, false);
    }

    private void setValue(int index, int i, int v, boolean b) {
        if (!bUsingColt) result.setValue(index, i, v, b);
        else {
            if (!b) resultColt.setQuick(index, i, v);
            else {
                resultColt.setQuick(index, i, resultColt.getQuick(index, i) + v);
            }
        }
    }

    @Test
    public void getUrl() {
        URL url = ASparseMatrixTest_db.class.getResource("/db/ieee.txt");
        Assert.assertNotNull(url);
        System.out.println(url.getPath());
    }
}
