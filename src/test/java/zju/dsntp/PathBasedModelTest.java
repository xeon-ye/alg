package zju.dsntp;

import junit.framework.TestCase;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsModelCons;
import zju.dsmodel.IeeeDsInHand;

import java.io.*;

import static zju.dsmodel.IeeeDsInHand.createDs;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2016/9/12
 */
public class PathBasedModelTest extends TestCase implements DsModelCons {

    public PathBasedModelTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testDscase13() throws Exception {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER13);
        model.buildPathes(5000);
        assertEquals(12, model.getPathes().size());
    }

    public void testDscase34() throws Exception {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER34);
        model.buildPathes(5000);
        assertEquals(33, model.getPathes().size());
    }

    public void testDscase37() throws Exception {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER37);
        model.buildPathes(5000);
        assertEquals(36, model.getPathes().size());
    }

    public void testDscase4() throws Exception {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER4_DD_B);
        model.buildPathes(5000);
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_DD_UNB);
        model.buildPathes(5000);
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_DGrY_B);
        model.buildPathes(5000);
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_DGrY_UNB);
        model.buildPathes(5000);
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_GrYGrY_B);
        model.buildPathes(5000);
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_GrYGrY_UNB);
        model.buildPathes(5000);
        assertEquals(3, model.getPathes().size());
    }

    public void testDscase123() throws Exception {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER123);
        model.buildPathes(5000);
        assertEquals(193, model.getPathes().size());
    }

    public void testCase10() throws Exception {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase10/graphtest.txt");
        testsys = createDs(ieeeFile, "150", 100);
        supplyID = new String[]{"150", "451"};
        Double[] supplyBaseKv = new Double[]{100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        PathBasedModel model = new PathBasedModel(testsys);
        long start = System.currentTimeMillis();
        model.buildPathes(5000);
        System.out.println((System.currentTimeMillis() - start) + "ms");
    }

    public boolean convertFile(String inputFile, String outputFile) throws IOException {
        boolean isSucess = false;
        FileOutputStream out = null;
        OutputStreamWriter osw = null;
        BufferedWriter bw = null;

        InputStream is = new FileInputStream(inputFile);
        BufferedReader r = new BufferedReader(new InputStreamReader(is));
        String str;
        String[] customerList;
        try {
            out = new FileOutputStream(outputFile);
            osw = new OutputStreamWriter(out);
            bw = new BufferedWriter(osw);

            bw.write("Line" + "\t" + "Segment" + "\t" + "Data" + "\t" + "2526" + "\t" + "items" + "\t" + "feet");
            bw.newLine();

            r.readLine();
            r.readLine();
            while ((str = r.readLine()) != null) {
                customerList = str.split(",");
                bw.write(customerList[1] + "\t" + customerList[3] + "\t" + customerList[5] + "\t" +"4");
                bw.newLine();
            }
            r.close();
            isSucess = true;
        } catch (Exception e) {
            isSucess = false;
        } finally {
            if(bw != null) {
                try {
                    bw.close();
                    bw = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(osw != null) {
                try {
                    osw.close();
                    osw = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(out!=null) {
                try {
                    out.close();
                    out = null;
                } catch(IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return isSucess;
    }
}
