package zju.util;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 11-8-17
 */
public class SerializableUtil {
    public static final String LIST_SEPARATOR = ";";
    public static final String ENTITY_SEPARATOR = "-@-";
    public static final String PROP_SEPARATOR = "-=-";

    public static final int TYPE_INTEGER = 1;
    public static final int TYPE_LONG = 2;
    public static final int TYPE_DOUBLE = 3;
    public static final int TYPE_STRING = 4;
    public static final int TYPE_STRING_LIST = 5;
    public static final int TYPE_INT_LIST = 6;
    public static final int TYPE_LONG_LIST = 7;
    public static final int TYPE_DOUBLE_LIST = 8;

    public static String formString(Map map) {
        StringBuilder buffer = new StringBuilder();
        for (Object key : map.keySet()) {
            Object value = map.get(key);
            if (value == null)
                continue;
            buffer.append(key).append(PROP_SEPARATOR);
            if(value instanceof Collection) {
                Collection l = (Collection) value;
                int i = 0;
                Iterator iter = l.iterator();
                while (iter.hasNext()) {
                    if (i == l.size() - 1)
                        buffer.append(iter.next());
                    else
                        buffer.append(iter.next()).append(LIST_SEPARATOR);
                    i++;
                }
                buffer.append(ENTITY_SEPARATOR);
            } else
                buffer.append(value).append(ENTITY_SEPARATOR);
        }
        return buffer.toString();
    }

    public static String formString(Collection<String> l) {
        StringBuilder buffer = new StringBuilder();
        int i = 0;
        Iterator<String> iter = l.iterator();
        while (iter.hasNext()) {
            if (i == l.size() - 1)
                buffer.append(iter.next());
            else
                buffer.append(iter.next()).append(LIST_SEPARATOR);
            i++;
        }
        return buffer.toString();
    }

    public static List<String> parseList(String content) {
        StringTokenizer tokenizer = new StringTokenizer(content, LIST_SEPARATOR);
        List<String> result = new ArrayList<String>(tokenizer.countTokens());
        while (tokenizer.hasMoreTokens()) {
            String s = tokenizer.nextToken();
            result.add(s);
        }
        return result;
    }

    public static List parseList(String content, int valueType) {
        StringTokenizer tokenizer = new StringTokenizer(content, LIST_SEPARATOR);
        List result = new ArrayList(tokenizer.countTokens());
        while (tokenizer.hasMoreTokens()) {
            String s = tokenizer.nextToken();
            switch (valueType) {
                case TYPE_INTEGER:
                    result.add(Integer.parseInt(s));
                    break;
                case TYPE_LONG:
                    result.add(Long.parseLong(s));
                    break;
                case TYPE_DOUBLE:
                    result.add(Double.parseDouble(s));
                    break;
                case TYPE_STRING:
                    result.add(s);
                    break;
                default:
                    break;
            }
        }
        return result;
    }

    /**
     * @param content
     * @return
     */
    public static Map createMap(String content, int keyType, int valueType) {
        Map result = new HashMap();
        String tem;
        int index = content.indexOf(ENTITY_SEPARATOR);
        do {
            if (index == -1) {
                tem = content;
            } else {
                tem = content.substring(0, index);
                content = content.substring(index + ENTITY_SEPARATOR.length());
                index = content.indexOf(ENTITY_SEPARATOR);
            }
            try {
                if (tem.equals(""))
                    continue;
                int index2 = tem.indexOf(PROP_SEPARATOR);
                String s = tem.substring(0, index2);
                String value = tem.substring(index2 + PROP_SEPARATOR.length());
                Object key = null;
                switch (keyType) {
                    case TYPE_STRING:
                        key = s;
                        break;
                    case TYPE_INTEGER:
                        key = Integer.parseInt(s);
                        break;
                    case TYPE_LONG:
                        key = Long.parseLong(s);
                        break;
                    default:
                        break;
                }
                switch (valueType) {
                    case TYPE_INTEGER:
                        result.put(key, Integer.parseInt(value));
                        break;
                    case TYPE_LONG:
                        result.put(key, Long.parseLong(value));
                        break;
                    case TYPE_DOUBLE:
                        result.put(key, Double.parseDouble(value));
                        break;
                    case TYPE_STRING:
                        result.put(key, value);
                        break;
                    case TYPE_STRING_LIST:
                        result.put(key, parseList(value));
                        break;
                    case TYPE_INT_LIST:
                        result.put(key, parseList(value, TYPE_INTEGER));
                        break;
                    case TYPE_LONG_LIST:
                        result.put(key, parseList(value, TYPE_LONG));
                        break;
                    case TYPE_DOUBLE_LIST:
                        result.put(key, parseList(value, TYPE_DOUBLE));
                        break;
                    default:
                        break;
                }
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        } while (index != -1);
        return result;
    }
}

