package zju.util.expression;

import java.util.ArrayList;
import java.util.Stack;

/**
 * 中缀到后缀表达式的转换
 */
public class Converter {// what about Infix2Postfix ?

    private boolean isAddOrSub(Operator op) {
        int operator = ((DefaultOperator) op).getOperator();
        return operator == DefaultOperator.OPERATOR_ADD || operator == DefaultOperator.OPERATOR_SUB;
    }

    public void setPriorityInnerStatck(Operator op) {
        if (op instanceof Bracket) {
            if (((Bracket) op).isLeftBracket())
                op.setPriority(1);
            else
                op.setPriority(8);
            return;
        }
        DefaultOperator operator = (DefaultOperator) op;
        if (isAddOrSub(op)) {
            op.setPriority(3);
            return;
        }
        switch (operator.getOperator()) {
            case DefaultOperator.OPERATOR_POW:
            case DefaultOperator.OPERATOR_EXP:
            case DefaultOperator.OPERATOR_SQRT:
            case DefaultOperator.OPERATOR_SIN:
            case DefaultOperator.OPERATOR_COS:
            case DefaultOperator.OPERATOR_TAN:
                op.setPriority(7);
                break;
            default:
                op.setPriority(5);
                break;
        }
    }

    /**
     * 设置栈内优先级
     *
     * @param op 操作符
     * @throws UnknownOperatorException
     */
    public void setPriorityOuterStatck(Operator op) {
        if (op instanceof Bracket) {
            if (((Bracket) op).isLeftBracket())
                op.setPriority(8);
            else
                op.setPriority(1);
            return;
        }
        DefaultOperator operator = (DefaultOperator) op;
        if (isAddOrSub(op)) {
            op.setPriority(2);
            return;
        }
        switch (operator.getOperator()) {
            case DefaultOperator.OPERATOR_POW:
            case DefaultOperator.OPERATOR_EXP:
            case DefaultOperator.OPERATOR_SQRT:
            case DefaultOperator.OPERATOR_SIN:
            case DefaultOperator.OPERATOR_COS:
            case DefaultOperator.OPERATOR_TAN:
                op.setPriority(6);
                break;
            default:
                op.setPriority(4);
                break;
        }
    }

    public Object[] convert(Object[] infixExpr)
            throws IllegalExpressionException, UnknownOperatorException {
        return convert(infixExpr, 0, infixExpr.length);
    }

    /**
     * 将中缀表达式(infix-expr)转换为后缀表达式(postfix-expr)，基于栈的经典算法。
     * 这个算法是：创建一个工作栈，
     * 当读到数值或者变量时，直接送至输出队列；
     * 当读到运算符t时，
     * a.将栈中所有优先级高于t的运算符弹出，送到输出队列中；
     * b.t进栈
     * c.栈中所有优先级和t相同，则对消
     * 例如对中缀表达式：a + (b -c) * d，转换成后缀形式为：abc-d*+
     *
     * @param infixExpr 中缀表达式，其中的元素是可以是以下之一：Operator，Numeral，Bracket。
     *                  非以上三种之一的被认为是变量类型
     * @return 后缀表达式
     * @throws IllegalArgumentException
     * @throws IllegalExpressionException 由于非法表达式而导致转换不能成功（例如：左右括号不匹配）
     */
    public Object[] convert(Object[] infixExpr, int offset, int len)
            throws IllegalExpressionException, UnknownOperatorException {
        if (infixExpr == null || infixExpr.length - offset < len)
            throw new IllegalArgumentException("invalid argument");

        // 创建一个输出表达式，用来存放结果
        ArrayList<Object> output = new ArrayList<Object>();

        // 创建一个工作栈
        Stack<Operator> stack = new Stack<Operator>();

        int currInputPosition = offset;  // 当前位置（于输入队列）

        while (currInputPosition < offset + len) {
            Object currInputElement = infixExpr[currInputPosition++];
            if (currInputElement instanceof Number) { // 数值元素直接输出
                output.add(currInputElement);
                // System.out.println("Number:"+currInputElement);//TEMP!
            } else if (currInputElement instanceof Operator) {
                Operator currInputOperator = (Operator) currInputElement;
                setPriorityOuterStatck(currInputOperator);
                boolean isCanceled = false;
                while (!stack.empty()) {
                    Operator stackElement = stack.peek();
                    if (currInputOperator.getPriority() < stackElement.getPriority()) {
                        stack.pop();
                        output.add(stackElement);
                    } else if (currInputOperator.getPriority() == stackElement.getPriority()) {    // 优先级别低于当前的，没有可以弹出的了
                        stack.pop();
                        isCanceled = true;
                        break;
                    } else
                        break;
                }
                if (!isCanceled) {
                    stack.push(currInputOperator);
                    setPriorityInnerStatck(currInputOperator);
                }
            } else // 其它一律被认为变量，变量也直接输出
                output.add(currInputElement);

        }
        // 将栈中剩下的元素(运算符)弹出至输出队列
        while (!stack.empty()) {
            Object stackElement = stack.pop();
            output.add(stackElement);
        }
        return output.toArray();
    }
}