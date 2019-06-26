package zju.bpamodel;

import java.io.File;
import java.util.HashMap;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;

/**
 * @author BinLuo@EIRI
 * @version v1.0
 * @time 2019-6-20 下午6:26:15
 */
public class XMLparse {

    public HashMap getDailyPlanHash(String xmlPath ){
		if(xmlPath == null || xmlPath.equals(""))
			return null;
		Document doc;
        try {
        	File file = new File(xmlPath);
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            doc = builder.parse(file);
            NodeList nlst = doc.getElementsByTagName("string");
    		int planNum = nlst.getLength();
    		HashMap<String, double[]> resHash = new HashMap<>();
    		for (int i = 0; i < planNum; i++) {
    			String planName  = doc.getElementsByTagName("string").item(i).getFirstChild().getNodeValue();
    			double[] plans = new double[96];
    			for (int j = 0; j < plans.length; j++) {
    				String value  = doc.getElementsByTagName("v" + j).item(i).getFirstChild().getNodeValue();
    				plans[j] = Double.parseDouble(value.trim());
    			}
    			resHash.put(planName, plans);
    		}
            return resHash;
        } catch (Exception e) {
            System.err.println("读取该NARI发电计划失败");
            e.printStackTrace();
            return null;
        }
	}
}
