package zju.util.expression;

/**
 * 本类用来表现表达式的不可计算性异常，例如表达式中含有变量，操作数缺少或多余
 *
 * @author lyman
 * @version 0.9, 2003/4/23
 */
public class IncalculableExpressionException extends Exception {
    public IncalculableExpressionException(String s) {
        super(s);
    }
}