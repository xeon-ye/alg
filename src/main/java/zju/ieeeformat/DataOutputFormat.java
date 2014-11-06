package zju.ieeeformat;

import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;

/**
 * Created by IntelliJ IDEA.
 * User: roylee
 * Date: 2006-7-17
 */
public class DataOutputFormat {
    public static final DataOutputFormat format = new DataOutputFormat();

    //private static final String EOL = System.getProperty("line.separator");
    public char fillChar = ' ';
    private String charset = "GBK";
    private char alignment;
    private boolean separator;
    private char base;
    private boolean scientific;
    private boolean zeroFill;
    private int width;
    private int decimals;
    private String pattern;

    public String getCharset() {
        return charset;
    }

    public void setCharset(String charset) {
        this.charset = charset;
    }

    /**
     * **********************************************************
     * Formats the passed value using the passed format descriptor
     * and returns the result as a string.
     *
     * @param value the value to be formatted.
     * @param fd    the format descriptor.
     * @return the formatted value as a string.
     *         ************************************************************
     */
    public String format(byte value, String fd) {
        return formatInteger(value, fd, 8);
    }

    /**
     * **********************************************************
     * Formats the passed value using the passed format descriptor
     * and returns the result as a string.
     *
     * @param value the value to be formatted.
     * @param fd    the format descriptor.
     * @return the formatted value as a string.
     *         ************************************************************
     */
    public String format(char value, String fd) {
        return formatInteger(value, fd, 16);
    }

    /**
     * **********************************************************
     * Formats the passed value using the passed format descriptor
     * and returns the result as a string.
     *
     * @param value the value to be formatted.
     * @param fd    the format descriptor.
     * @return the formatted value as a string.
     *         ************************************************************
     */
    public String format(double value, String fd) {
        extractAttributes(fd);
        String s1;
        if (base == 'B') {
            s1 = Long.toBinaryString(Double.doubleToLongBits(value));
            s1 = repeat(64, '0') + s1;
            s1 = s1.substring(s1.length() - 64);
        } else if (base == 'X') {
            s1 = Long.toHexString(Double.doubleToLongBits(value));
            s1 = repeat(16, '0') + s1;
            s1 = s1.substring(s1.length() - 16);
        } else {
            pattern = decimals != -1 ? "." + repeat(decimals, '0') : ".#";
            if (scientific)
                pattern = "0" + pattern + "E0";
            else
                pattern = separator ? "#,##0" + pattern : pattern;
            s1 = (new DecimalFormat(pattern)).format(value);
        }
        return size(s1);
    }

    public String format(String str, String fd) {
        extractAttributes(fd);
        return size(str);
    }

    /**
     * **********************************************************
     * Formats the passed value using the passed format descriptor
     * and returns the result as a string.
     *
     * @param value the value to be formatted.
     * @param fd    the format descriptor.
     * @return the formatted value as a string.
     *         ************************************************************
     */
    public String format(float value, String fd) {
        extractAttributes(fd);
        String s1;
        if (base == 'B') {
            s1 = Integer.toBinaryString(Float.floatToIntBits(value));
            s1 = repeat(32, '0') + s1;
            s1 = s1.substring(s1.length() - 32);
        } else if (base == 'X') {
            s1 = Integer.toHexString(Float.floatToIntBits(value));
            s1 = repeat(8, '0') + s1;
            s1 = s1.substring(s1.length() - 8);
        } else {
            pattern = decimals != -1 ? "." + repeat(decimals, '0') : ".#";
            if (scientific)
                pattern = "0" + pattern + "E0";
            else
                pattern = separator ? "#,##0" + pattern : pattern;
            s1 = (new DecimalFormat(pattern)).format(value);
        }
        return size(s1);
    }

    /**
     * **********************************************************
     * Formats the passed value using the passed format descriptor
     * and returns the result as a string.
     *
     * @param value the value to be formatted.
     * @param fd    the format descriptor.
     * @return the formatted value as a string.
     *         ************************************************************
     */
    public String format(int value, String fd) {
        return formatInteger(value, fd, 32);
    }

    /**
     * **********************************************************
     * Formats the passed value using the passed format descriptor
     * and returns the result as a string.
     *
     * @param value the value to be formatted.
     * @param fd    the format descriptor.
     * @return the formatted value as a string.
     *         ************************************************************
     */
    public String format(long value, String fd) {
        return formatInteger(value, fd, 64);
    }

    /**
     * **********************************************************
     * Formats the passed value using the passed format descriptor
     * and returns the result as a string.
     *
     * @param value the value to be formatted.
     * @param fd    the format descriptor.
     * @return the formatted value as a string.
     *         ************************************************************
     */
    public String format(short value, String fd) {
        return formatInteger(value, fd, 16);
    }

    /* Formats the passed value using the passed format descriptor
    * and returns the result as a string.
    */
    private String formatInteger(long l, String s, int i) {
        extractAttributes(s);
        String s1;
        if (base == 'B') {
            s1 = Long.toBinaryString(l);
            s1 = repeat(64, '0') + s1;
            s1 = s1.substring(s1.length() - i);
        } else if (base == 'X') {
            s1 = Long.toHexString(l);
            //s1 = repeat(16, '0') + s1;
            //s1 = s1.substring(s1.length() - i / 4);
        } else if (separator) {
            s1 = (new DecimalFormat("#,###")).format(l);
        } else {
            s1 = String.valueOf(l);
            if (zeroFill)
                s1 = repeat(width - s1.length(), '0') + s1;
        }
        return size(s1);
    }

    // Gets information from the passed format descriptor.
    private void extractAttributes(String s) {
        s = s.toUpperCase();
        alignment = 'R';
        separator = false;
        base = 'D';
        scientific = false;
        zeroFill = false;
        width = -1;
        decimals = -1;
        int i = s.indexOf('L', 0);
        if (i > -1) {
            alignment = 'L';
            s = s.substring(0, i) + s.substring(i + 1);
        }
        i = s.indexOf('C', 0);
        if (i > -1) {
            alignment = 'C';
            s = s.substring(0, i) + s.substring(i + 1);
        }
        i = s.indexOf(',', 0);
        if (i > -1) {
            separator = true;
            pattern = pattern + ",###";
            s = s.substring(0, i) + s.substring(i + 1);
        }
        i = s.indexOf('X', 0);
        if (i > -1) {
            base = 'X';
            s = s.substring(0, i) + s.substring(i + 1);
        } else {
            i = s.indexOf('B', 0);
            if (i > -1) {
                base = 'B';
                s = s.substring(0, i) + s.substring(i + 1);
            }
        }
        i = s.indexOf('S', 0);
        if (i > -1) {
            scientific = true;
            s = s.substring(0, i) + s.substring(i + 1);
        }
        i = s.indexOf('Z', 0);
        if (i > -1) {
            zeroFill = true;
            s = s.substring(0, i) + s.substring(i + 1);
        }
        i = s.indexOf('.', 0);
        if (i > -1) {
            decimals = Integer.parseInt(s.substring(i + 1));
            s = s.substring(0, i);
        }
        if (s.length() > 0)
            width = Integer.parseInt(s);
    }

    /**
     * **********************************************************
     * Output the passed value to the standard output device using
     * the passed format descriptor. No trailing End-Of-Line
     * character is printed.
     *
     * @param value the value to be printed.
     * @param fd    the format descriptor.
     *              ************************************************************
     */
    public String getFormatStr(byte value, String fd) {
        return (format(value, fd));

    }

    /**
     * **********************************************************
     * Output the passed value to the standard output device using
     * the passed format descriptor. No trailing End-Of-Line
     * character is printed.
     *
     * @param value the value to be printed.
     * @param fd    the format descriptor.
     *              ************************************************************
     */
    public String getFormatStr(char value, String fd) {
        return (format(value, fd));

    }

    public String getFormatStr(String value, String fd) {
        return (format(value, fd));
    }

    /**
     * **********************************************************
     * Output the passed value to the standard output device using
     * the passed format descriptor. No trailing End-Of-Line
     * character is printed.
     *
     * @param value the value to be printed.
     * @param fd    the format descriptor.
     *              ************************************************************
     */
    public String getFormatStr(double value, String fd) {
        return (format(value, fd));

    }

    /**
     * **********************************************************
     * Output the passed value to the standard output device using
     * the passed format descriptor. No trailing End-Of-Line
     * character is printed.
     *
     * @param value the value to be printed.
     * @param fd    the format descriptor.
     *              ************************************************************
     */
    public String getFormatStr(float value, String fd) {
        return (format(value, fd));

    }

    /**
     * **********************************************************
     * Output the passed value to the standard output device using
     * the passed format descriptor. No trailing End-Of-Line
     * character is printed.
     *
     * @param value the value to be printed.
     * @param fd    the format descriptor.
     *              ************************************************************
     */
    public String getFormatStr(int value, String fd) {
        return (format(value, fd));

    }

    /**
     * **********************************************************
     * Output the passed value to the standard output device using
     * the passed format descriptor. No trailing End-Of-Line
     * character is printed.
     *
     * @param value the value to be printed.
     * @param fd    the format descriptor.
     *              ************************************************************
     */
    public String getFormatStr(long value, String fd) {
        return (format(value, fd));

    }

    /**
     * **********************************************************
     * Output the passed value to the standard output device using
     * the passed format descriptor. No trailing End-Of-Line
     * character is printed.
     *
     * @param value the value to be printed.
     * @param fd    the format descriptor.
     *              ************************************************************
     */
    public String getFormatStr(short value, String fd) {
        return (format(value, fd));

    }

    // Returns the string padding with 'fillChar'.
    private String size(String s) {
        char[] arrchar = s.toCharArray();
        if (arrchar.length > 2) {
            if (arrchar[0] == '0' && arrchar[1] == '.' && arrchar[2] != '0') {
                String tmp = s;
                s = tmp.substring(1);
            } else {
                if (arrchar.length == 3 && arrchar[0] == '0' && arrchar[1] == '.' && arrchar[2] == '0') {
                    String tmp = s;
                    s = tmp.substring(0, 2);
                }
            }
        }
        byte[] bytes = new byte[0];
        try {
            bytes = s.getBytes(charset);
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        byte[] toShow = new byte[width];
        int i = width - bytes.length;
        if (i < 0)
            System.arraycopy(bytes, 0, toShow, 0, width);
        else if (alignment == 'L') {
            return s + repeat(i, fillChar);
        } else if (alignment == 'R') {
            return repeat(i, fillChar) + s;
        } else
            return repeat(i / 2, fillChar) + s + repeat(i / 2 + i % 2, fillChar);
        return new String(toShow);
    }

    // Repeats the passed character 'times' times.
    public static String repeat(int times, char c) {
        String s = "";
        for (int i = 0; i < times; i++)
            s = s + c;

        return s;
    }

    public char getFillChar() {
        return fillChar;
    }

    public void setFillChar(char fillChar) {
        this.fillChar = fillChar;
    }

    public char getAlignment() {
        return alignment;
    }

    public void setAlignment(char alignment) {
        this.alignment = alignment;
    }

    public boolean isSeparator() {
        return separator;
    }

    public void setSeparator(boolean separator) {
        this.separator = separator;
    }

    public char getBase() {
        return base;
    }

    public void setBase(char base) {
        this.base = base;
    }

    public boolean isScientific() {
        return scientific;
    }

    public void setScientific(boolean scientific) {
        this.scientific = scientific;
    }

    public boolean isZeroFill() {
        return zeroFill;
    }

    public void setZeroFill(boolean zeroFill) {
        this.zeroFill = zeroFill;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getDecimals() {
        return decimals;
    }

    public void setDecimals(int decimals) {
        this.decimals = decimals;
    }

    public String getPattern() {
        return pattern;
    }

    public void setPattern(String pattern) {
        this.pattern = pattern;
    }
}
