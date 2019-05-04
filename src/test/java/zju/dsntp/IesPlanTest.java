package zju.dsntp;

import junit.framework.TestCase;
import zju.dsmodel.DistriSys;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static zju.dsmodel.IeeeDsInHand.createDs;

public class IesPlanTest extends TestCase {

    public void testChangeFileFormat() throws IOException {
        String readPath = this.getClass().getResource("/iesfiles/originEdges.txt").getPath();
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(readPath)));
        String data;
        String writePath = "C:/Users/Administrator.2013-20160810IY/IdeaProjects/alg/src/test/resources/iesfiles/edges.txt";
        Writer writer = new OutputStreamWriter(new FileOutputStream(new File(writePath)));

        while((data = br.readLine()) != null) {
            String[] content = data.split("\t");
            writer.write(content[0].split("-")[0] + "\t" + content[0].split("-")[1] + "\t" + content[1] + "\t" + "2\n");
        }
        writer.close();
    }

    public void testCase1() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/iesfiles/case1/graph.txt");
        testsys = createDs(ieeeFile, "116", 10);
        supplyID = new String[]{"116", "124", "165", "177", "187", "192", "196", "205"};
        Double[] supplyBaseKv = new Double[]{10., 10., 10., 10., 10., 10., 10., 10.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        IesPlan model = new IesPlan(testsys);
        String loadsPath = this.getClass().getResource("/iesfiles/case1/loads.txt").getPath();
        Map<String, ArrayList<Double>> loads = new HashMap<String, ArrayList<Double>>();
        readLoads(loadsPath, loads);
        model.setLoads(loads);
        model.doOpt();
    }

    public void readLoads(String path, Map<String, ArrayList<Double>> loads) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        while ((data = br.readLine()) != null) {
            String[] newdata = data.split("\t", 4);
            String cnId = newdata[0];
            ArrayList<Double> cnLoads = new ArrayList<>(3);
            cnLoads.set(0, Double.parseDouble(newdata[1]));
            cnLoads.set(1, Double.parseDouble(newdata[2]));
            cnLoads.set(2, Double.parseDouble(newdata[3]));
            loads.put(cnId, cnLoads);
        }
    }
}
