package zju.dsieeeformat;


import java.io.InputStream;

/**
 * @Description:
 * @Author: Fang Rui
 * @Date: 2018/1/6
 * @Time: 15:03
 */
public class Ds8500CaseWriterApplication {

    public static void main(String[] args) {

        DsIeee8500NodeWriter dsIeee8500NodeWriter = new DsIeee8500NodeWriter("src/main/resources/dsieee/8500node/", "case8500");
        InputStream linesResource = DsIeee8500NodeWriter.class.getResourceAsStream("/dsieee/8500node/Lines.csv");
        InputStream capacitorsResource = DsIeee8500NodeWriter.class.getResourceAsStream("/dsieee/8500node/Capacitors.csv");
        InputStream transformersResource = DsIeee8500NodeWriter.class.getResourceAsStream("/dsieee/8500node/Transformers.csv");
        InputStream loadXfmrsResource = DsIeee8500NodeWriter.class.getResourceAsStream("/dsieee/8500node/LoadXfmrs.csv");
        InputStream solutionBalancedResource = DsIeee8500NodeWriter.class.getResourceAsStream("/dsieee/8500node/Solutions/Solution-Balanced.xls");
        InputStream solutionUnbalancedResource = DsIeee8500NodeWriter.class.getResourceAsStream("/dsieee/8500node/Solutions/Solution-unBalanced.xls");

        dsIeee8500NodeWriter.readRawData(linesResource, capacitorsResource, transformersResource, loadXfmrsResource);
        dsIeee8500NodeWriter.readSolutionBalancedData(solutionBalancedResource);

        dsIeee8500NodeWriter.createFile();

        dsIeee8500NodeWriter.writeLineSegmentData();
        dsIeee8500NodeWriter.writeSpotLoadData();
        dsIeee8500NodeWriter.writeDistributedLoadData();
        dsIeee8500NodeWriter.writeCapacitorData();
        dsIeee8500NodeWriter.writeTransformerData();
        dsIeee8500NodeWriter.writeRegulatorData();

        DsIeee8500NodeConfigWriter dsIeee8500NodeConfigWriter = new DsIeee8500NodeConfigWriter("src/main/resources/dsieee/common/","feederconfig8500");
        InputStream configResource = DsIeee8500NodeConfigWriter.class.getResourceAsStream("/dsieee/8500node/LineCodes2.dat");
        dsIeee8500NodeConfigWriter.readRawData(configResource);

        dsIeee8500NodeConfigWriter.createFile();
        dsIeee8500NodeConfigWriter.writeAll();

    }
}
