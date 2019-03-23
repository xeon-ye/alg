package zju.bpamodel;

import junit.framework.TestCase;
import zju.bpamodel.swi.Exciter;
import zju.bpamodel.swi.Generator;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * BpaSwiModelWriter Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/17/2012</pre>
 */
public class BpaSwiModelRwTest extends TestCase {

    public List<String> toFound;

    public BpaSwiModelRwTest(String name) {
        super(name);
        toFound = new ArrayList<String>();
        toFound.add("后石");
        toFound.add("可门");
        toFound.add("江阴");
        toFound.add("华能");
        toFound.add("鸿山");
        toFound.add("嵩屿");
        toFound.add("湄电"); //湄洲湾
        toFound.add("坑口");
        toFound.add("漳平");
        toFound.add("前云");
        toFound.add("石圳");
        toFound.add("新店");
        toFound.add("水口");
        toFound.add("安砂");
        toFound.add("周宁");

        toFound.add("池潭");
        toFound.add("棉电");//棉滩
        toFound.add("沙电");//沙溪口
        toFound.add("南埔");
        toFound.add("宁核");//宁德核电
        toFound.add("西抽");//西苑抽蓄
        toFound.add("福核");//福清核电
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testReadAndWriter() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/fj/2012年稳定20111103.swi"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/fj/2012年稳定20111103.swi");
        //FileOutputStream out = new FileOutputStream("2012年稳定20111103-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);

        for (Exciter exciter : exciters) {
            if (exciter.getBusName().equals("闽江阴_1")) {
                exciter.setXc(-0.0396259400157145);
                System.out.println(exciter.toString());
            }
        }
    }

    public void testSwi003() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/003_bus/bpa/003bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/003_bus/bpa/003bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("003bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testSwi009() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/009_bus/bpa/009bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/009_bus/bpa/009bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("009bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testSwi039() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/039_bus/bpa/039bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/039_bus/bpa/039bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("039bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testSwi145() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/145_bus/bpa/145bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/145_bus/bpa/145bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("145bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testSwi162() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/162_bus/bpa/162bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/162_bus/bpa/162bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("162bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testXJ() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/示范区BPA运行方式/XIAOJIN.SWI"), "GBK");
        assertNotNull(model);
//        BpaSwiModel modifiedModel = new BpaSwiModel();
//        modifiedModel.setGenerators(new ArrayList<Generator>());
//        modifiedModel.setExciters(exciters);
//        InputStream in = this.getClass().getResourceAsStream("/bpafiles/示范区BPA运行方式/XIAOJIN.SWI");
//        //FileOutputStream out = new FileOutputStream("162bpaswi-opted.swi");
//        ByteArrayOutputStream out = new ByteArrayOutputStream();
//        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
//        assertTrue(r);
    }
}
