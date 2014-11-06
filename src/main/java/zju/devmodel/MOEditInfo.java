package zju.devmodel;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2011-4-2
 */
public class MOEditInfo implements Serializable {
    String type;
    List<String> editableKeys;
    List<String> displayNames;
    Map<String, String[]> choices;

    public void loadProperties(Properties p) {
        String property = p.getProperty(type + ".editableKeyNum");
        if(property == null){
            setEditableKeys(new ArrayList<String>(0));
            setDisplayNames(new ArrayList<String>(0));
            setChoices(new HashMap<String, String[]>(0));
            return;
        }
        int num = Integer.parseInt(property);
        int choiceNum = Integer.parseInt(p.getProperty(type + ".choiceNum"));
        List<String> keys = new ArrayList<String>(num);
        List<String> names = new ArrayList<String>(num);
        Map<String, String[]> choices = new HashMap<String, String[]>(choiceNum);

        for(int i = 1; i <= num; i++) {
            keys.add(p.getProperty(type + ".editableKey" + i));
            names.add(p.getProperty(type + ".displayName" + i));
        }
        for(int i = 1; i <= choiceNum; i++) {
            String key = p.getProperty(type + ".choiceKey" + +i);
            String v = p.getProperty(type + ".choiceContent" + i);
            choices.put(key, v.split(";"));
        }
        setEditableKeys(keys);
        setDisplayNames(names);
        setChoices(choices);
    }

    public void loadProperties(InputStream stream) {
        Properties p = new Properties();
        try {
            p.load(stream);
            loadProperties(p);
        } catch (IOException e) {
            e.printStackTrace();
            setEditableKeys(new ArrayList<String>(0));
            setDisplayNames(new ArrayList<String>(0));
            setChoices(new HashMap<String, String[]>(0));
        }
    }

    /**
     * @return key=index in editable keys, value=choices
     */
    public Map<Integer, String[]> getChoices2() {
        java.util.List<String> keys = getEditableKeys();
        Map<String, String[]> choices = getChoices();
        Map<Integer, String[]> choices2 = new HashMap<Integer, String[]>(choices.size());
        int col = 0;
        for (String key : keys) {
            if (choices.containsKey(key))
                choices2.put(col, choices.get(key));
            col++;
        }
        return choices2;
    }

    public MOEditInfo() {
    }

    public MOEditInfo(String type) {
        this.type = type;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public List<String> getEditableKeys() {
        return editableKeys;
    }

    public List<String> getDisplayNames() {
        return displayNames;
    }

    public Map<String, String[]> getChoices() {
        return choices;
    }

    public void setEditableKeys(List<String> editableKeys) {
        this.editableKeys = editableKeys;
    }

    public void setDisplayNames(List<String> displayNames) {
        this.displayNames = displayNames;
    }

    public void setChoices(Map<String, String[]> choices) {
        this.choices = choices;
    }
}
