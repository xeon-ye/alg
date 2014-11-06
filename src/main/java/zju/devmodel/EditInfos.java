package zju.devmodel;

import java.io.*;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2011-4-10
 */
public class EditInfos implements Serializable {
    Map<String, MOEditInfo> type2EditInfo;
    List<String> typeDisplayNames;
    List<String> types;

    public MOEditInfo getEditInfo(String type) {
        return type2EditInfo.get(type);
    }
    
    public void loadProperty(File file) {
        loadProperty(file.getPath());
    }

    public void loadProperty(String file) {
        try {
            loadProperty(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            setType2EditInfo(new HashMap<String, MOEditInfo>(0));
            setTypeDisplayNames(new ArrayList<String>(0));
            setTypes(new ArrayList<String>(0));
        }
    }

    public void loadProperty(InputStream stream) {
        Properties p = new Properties();
        try {
            p.load(stream);
            int num = Integer.parseInt(p.getProperty("totalTypeNum"));
            type2EditInfo = new HashMap<String, MOEditInfo>(num);
            typeDisplayNames = new ArrayList<String>(num);
            types = new ArrayList<String>(num);

            for (int i = 1; i <= num; i++) {
                String type = p.getProperty("type" + i);
                String displayName = p.getProperty("type" + i + ".displayName");
                typeDisplayNames.add(displayName);
                types.add(type);
                MOEditInfo info = new MOEditInfo(type);
                info.loadProperties(p);
                type2EditInfo.put(type, info);
            }
        } catch (IOException e) {
            e.printStackTrace();
            setType2EditInfo(new HashMap<String, MOEditInfo>(0));
            setTypeDisplayNames(new ArrayList<String>(0));
            setTypes(new ArrayList<String>(0));
        }
    }

    public Map<String, MOEditInfo> getType2EditInfo() {
        return type2EditInfo;
    }

    public void setType2EditInfo(Map<String, MOEditInfo> type2EditInfo) {
        this.type2EditInfo = type2EditInfo;
    }

    public List<String> getTypeDisplayNames() {
        return typeDisplayNames;
    }

    public void setTypeDisplayNames(List<String> typeDisplayNames) {
        this.typeDisplayNames = typeDisplayNames;
    }

    public List<String> getTypes() {
        return types;
    }

    public void setTypes(List<String> types) {
        this.types = types;
    }
}
