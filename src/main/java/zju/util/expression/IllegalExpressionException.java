package zju.util.expression;

/**
 * 本类用来表现非法表达式异常,例如无法解析
 *
 * @author lyman
 * @version 0.9, 2003/4/22
 */
public class IllegalExpressionException extends Exception {
    public IllegalExpressionException(String s) {
        super(s);
    }
}