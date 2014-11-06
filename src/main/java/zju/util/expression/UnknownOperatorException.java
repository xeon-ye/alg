package zju.util.expression;

/**
 * 本类用来表达未知的运算符号
 *
 * @author lyman
 * @version 0.9, 2003/4/19
 */
public class UnknownOperatorException extends Exception {
    public UnknownOperatorException(String s) {
        super(s);
    }
}