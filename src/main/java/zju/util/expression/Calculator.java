package zju.util.expression;

import java.util.Map;
import java.util.Stack;

/**
 * 后缀表达式计算器。提供两个方法：校验和计算，输入参数均是后缀表达式。
 *
 * @author lyman
 * @version 0.97, 2004/8/24
 */
public class Calculator {
    private boolean isDebug = false;

    public Number eval(String infixExpress) throws IncalculableExpressionException, UnknownOperatorException, IllegalExpressionException {
        return eval(infixExpress, null);
    }

    public Number eval(String infixExpress, Map<String, Number> varValues) throws IncalculableExpressionException, UnknownOperatorException, IllegalExpressionException {
        DefaultInfixParser parser = new DefaultInfixParser();
        Object[] infix = parser.parse(infixExpress);
        if (isDebug) {
            System.out.print("The infix expression:");
            printExpressionOfElements(infix);
        }

        Converter converter = new Converter();
        Object[] postfix = converter.convert(infix);
        if (isDebug) {
            System.out.print("The postfix expression:");
            printExpressionOfElements(postfix);
        }
        return eval(postfix, varValues);
    }

    private void printExpressionOfElements(Object[] expressionOfElements) {
        for (int i = 0; i < expressionOfElements.length; i++) {
            Object element = expressionOfElements[i];
            System.out.print(element.toString() + " "); // 元素之间以逗号分隔
        }
        System.out.println();
        System.out.println();
    }

    public Number eval(Object[] postfixExpr) throws IncalculableExpressionException {
        return eval(postfixExpr, null);
    }

    /**
     * 计算后缀表达式的值(要求不含变量)，基于栈的经典算法。
     * 这个算法是：创建一个工作栈，
     * 从左到右读后缀表达式，读到数值元素就压入栈S中；
     * 读到运算符则从栈中依次弹出N个数，计算出结果，再压入栈S中，N是该运算符号的元数；
     * 如果后缀表达式未读完，就重复上面过程，最后输出栈顶的数值则为结束。
     *
     * @param postfixExpr 后缀表达式（要求其中的元素不是Operator就是Number）
     * @return expression result
     * @throws IllegalArgumentException
     * @throws IncalculableExpressionException
     *                                  如果表达式中含有除运算符和数值之外的元素，
     *                                  或者操作数与运算符的个数不匹配
     * @throws ArithmeticException
     */
    public Number eval(Object[] postfixExpr, Map<String, Number> varValues)
            throws IncalculableExpressionException {
        Stack<Number> stack = new Stack<Number>();
        int currPosition = 0;
        while (currPosition < postfixExpr.length) {
            Object element = postfixExpr[currPosition++];
            if (element instanceof Number) {
                stack.push((Number) element);
            } else if (element instanceof Variable) {
                if (varValues != null)
                    stack.push(varValues.get(((Variable) element).getName()));
                else
                    stack.push(((Variable) element).getValue());
            } else if (element instanceof DefaultOperator) {
                DefaultOperator op = (DefaultOperator) element;
                int dimensions = op.getDimension();
                if (dimensions < 1 || stack.size() < dimensions)
                    throw new IncalculableExpressionException("lack operand(s) for operator '" + op + "'");

                Number[] operands = new Number[dimensions];
                for (int j = dimensions - 1; j >= 0; j--) {
                    operands[j] = stack.pop();
                }
                stack.push(op.eval(operands));
            } else
                throw new IncalculableExpressionException("Unknown element: " + element);
        }
        if (stack.size() != 1)
            throw new IncalculableExpressionException("redundant operand(s)");

        return stack.pop();
    }

    /**
     * check postfix expression is valied
     *
     * @param postfixExpr postfix expression
     * @return 0-pass, 1-lack operand(s) 2-operand(s) redundant(or say lack operator(s))
     */
    public int verify(Object[] postfixExpr) {
        if (postfixExpr == null)
            throw new IllegalArgumentException("invalid argument");
        Stack<Object> stack = new Stack<Object>();
        int currPosition = 0;
        Object mockNumber = new Object();
        while (currPosition < postfixExpr.length) {
            Object element = postfixExpr[currPosition++];
            if (element instanceof DefaultOperator) {
                DefaultOperator op = (DefaultOperator) element;
                int dimensions = op.getDimension();
                if (dimensions < 1 || stack.size() < dimensions)
                    return 1;

                for (int j = dimensions - 1; j >= 0; j--) {
                    stack.pop();
                }
                stack.push(mockNumber);
            } else { // treated as a number
                stack.push(mockNumber);
            }
        }
        if (stack.size() != 1)
            return 2;

        return 0;
    }
}