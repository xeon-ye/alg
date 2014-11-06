package zju.util.expression;

import java.util.ArrayList;
import java.util.List;

public class DefaultInfixParser {
    private static final int MAX_LEN_VAR_NAME = 200;

    public Object[] parseVariable(char[] expr, int off, List<Object> r) {
        if (expr == null || off >= expr.length)
            throw new IllegalArgumentException("invalid argument ");
        Object[] result = new Object[2];
        Variable var = null;
        int newOff = off;
        if (expr.length - off > 3 && expr[newOff] == 'v' && expr[newOff + 1] == 'a' && expr[newOff + 2] == 'r') {
            var = new Variable();
            newOff += 3;
            if (expr[newOff] != '(') {
                newOff = 0;
            } else {
                int start = newOff + 1;
                while (expr[newOff] != ')') {
                    newOff++;
                    if (newOff - start > MAX_LEN_VAR_NAME) {
                        newOff = 0;
                        break;
                    }
                }
                if (newOff != 0) {
                    String name = new String(expr, start, newOff - start);
                    var.setName(name);
                    newOff++;
                }
            }
        }
        // ignore blank char
        result[0] = newOff;
        result[1] = var;
        return result;
    }

    public Object[] parseOperator(char[] expr, int off, List<Object> r) {
        if (expr == null || off >= expr.length)
            throw new IllegalArgumentException("invalid argument ");

        int newOff = off;

        Object[] result = new Object[2];
        Operator value = null;

        if (expr[newOff] == '+') {
            value = new DefaultOperator("+", DefaultOperator.OPERATOR_ADD);
            newOff++;
        } else if (expr[newOff] == '-') {
            int i = r.size() - 1;
            if (i == -1 || ((r.get(i) instanceof Bracket) && ((Bracket) r.get(i)).isLeftBracket()))
                r.add(new Float(0.0));
            value = new DefaultOperator("-", DefaultOperator.OPERATOR_SUB);
            newOff++;
        } else if (expr[newOff] == '*') {
            value = new DefaultOperator("*", DefaultOperator.OPERATOR_MUL);
            newOff++;
        } else if (expr[newOff] == '/') {
            value = new DefaultOperator("/", DefaultOperator.OPERATOR_DIV);
            newOff++;
        } else if (expr[newOff] == '^') {
            value = new DefaultOperator("^", DefaultOperator.OPERATOR_POW);
            newOff++;
        } else if (expr[newOff] == '%') {
            value = new DefaultOperator("%", DefaultOperator.OPERATOR_MOD);
            newOff++;
        } else if (expr.length - off > 3 && expr[newOff] == 'e' && expr[newOff + 1] == 'x' && expr[newOff + 2] == 'p') {
            value = new DefaultOperator("exp", DefaultOperator.OPERATOR_EXP);
            newOff += 3;
        } else
        if (expr.length - off > 4 && expr[newOff] == 's' && expr[newOff + 1] == 'q' && expr[newOff + 2] == 'r' && expr[newOff + 3] == 't') {
            value = new DefaultOperator("sqrt", DefaultOperator.OPERATOR_SQRT);
            newOff += 4;
        } else if (expr.length - off > 3 && expr[newOff] == 's' && expr[newOff + 1] == 'i' && expr[newOff + 2] == 'n') {
            value = new DefaultOperator("sin", DefaultOperator.OPERATOR_SIN);
            newOff += 3;
        } else if (expr.length - off > 3 && expr[newOff] == 'c' && expr[newOff + 1] == 'o' && expr[newOff + 2] == 's') {
            value = new DefaultOperator("cos", DefaultOperator.OPERATOR_COS);
            newOff += 3;
        } else if (expr.length - off > 3 && expr[newOff] == 't' && expr[newOff + 1] == 'a' && expr[newOff + 2] == 'n') {
            value = new DefaultOperator("tan", DefaultOperator.OPERATOR_TAN);
            newOff += 3;
        } else {
            newOff = 0;
        }

        // ignore blank char
        result[0] = newOff;
        result[1] = value;
        return result;
    }

    public Object[] parseNumber(char[] expr, int off, List<Object> r) {
        if (expr == null || off >= expr.length)
            throw new IllegalArgumentException("Invalid argument");

        int newOff = off;

        if (expr[newOff] == '-') {
            if (expr.length <= newOff + 1 || !(expr[newOff + 1] >= '0' && expr[newOff + 1] <= '9'))
                return new Object[]{0};
            if (r.size() == 0)
                newOff++;
            else if ((r.get(r.size() - 1) instanceof Bracket) && ((Bracket) r.get(r.size() - 1)).isLeftBracket())
                newOff++;
        }
        int numberType = 1; // 1-Integer 2-Float
        boolean isSientificNotation = false;
        while (newOff < expr.length) {
            if (expr[newOff] >= '0' && expr[newOff] <= '9') {
                newOff++;
            } else if (expr[newOff] == '.') {
                newOff++;
                numberType = 2;
            } else
            if (newOff > off && expr[newOff - 1] >= '0' && expr[newOff - 1] <= '9' && expr[newOff] == 'e' || expr[newOff] == 'E') {
                newOff++;
                isSientificNotation = true;
            } else if (isSientificNotation && expr[newOff] == '-') {
                newOff++;
                numberType = 2;
            } else
                break;
        }

        // ignore blank char
        Object[] result = new Object[2];
        Number value = null;

        if (newOff == off) newOff = 0;
        else {
            String numStr = new String(expr, off, newOff - off);
            try {
                if (numberType == 1) // Integer
                    value = new Integer(numStr);
                else // numberType == 2 Float
                    value = new Float(numStr);
            } catch (Exception e) {
                System.out.println("Can not convert string to number:" + e.getMessage());//TEMP!
            }
        }

        result[0] = newOff;
        result[1] = value;

        return result;
    }

    public Object[] parseBracket(char[] expr, int off, List<Object> r) {
        if (expr == null || off >= expr.length)
            throw new IllegalArgumentException("invalid argument ");

        int newOff = off;

        Object[] result = new Object[2];
        Bracket value = null;

        if (expr[newOff] == '(') {
            value = Bracket.createLeftBracket();
            newOff++;
        } else if (expr[newOff] == ')') {
            value = Bracket.createRightBracket();
            newOff++;
        } else {
            newOff = 0;
        }

        // ignore blank char
        result[0] = newOff;
        result[1] = value;

        return result;
    }

    /**
     * 解析给定的中缀表达式字符串，得到中缀表达式元素序列。
     * 过程：分析表达式字符串，提取出一个个基本元素，组成元素序列。
     * 当遇到空白符时，跳过；否则
     * 若遇到的是左右括号时，直接得到对应的元素实例；否则
     * 首先调用变量解析器。若不成功，调用运算符解析器。若仍不成功，调用数值解析器。
     * 若依然不成功，抛出IllegalExpressionException异常。
     *
     * @param expr 中缀表达式字符串
     * @return 中缀表达式元素序列
     * @throws zju.util.expression.IllegalExpressionException
     *          不合法的表达式
     */
    public Object[] parse(String expr) throws IllegalExpressionException {
        char[] chars = new char[expr.length()];
        expr.getChars(0, expr.length(), chars, 0);

        int currPosition = 0;
        int totalLength = expr.length();
        ArrayList<Object> elementList = new ArrayList<Object>();

        while (true) {
            // 跳过空白符：空格，TAB，回车换行符
            while (currPosition < totalLength) {
                if (chars[currPosition] == '\t' || chars[currPosition] == ' '
                        || chars[currPosition] == '\r' || chars[currPosition] == '\n')
                    currPosition++;
                else
                    break;
            }
            if (currPosition >= totalLength)
                break;

            // 非空白符，将依次按照变量，运算符(包括自定义的)，数值，括号来尝试解析。
            try {
                Object[] result;
                do {
                    result = parseNumber(chars, currPosition, elementList);
                    if ((Integer) result[0] != 0)
                        break;
                    result = parseOperator(chars, currPosition, elementList);
                    if ((Integer) result[0] != 0)
                        break;
                    result = parseBracket(chars, currPosition, elementList);
                    if ((Integer) result[0] != 0)
                        break;
                    result = parseVariable(chars, currPosition, elementList);
                    if ((Integer) result[0] != 0)
                        break;
                } while (false);
                if ((Integer) result[0] == 0) {
                    int leftLength = chars.length - currPosition;
                    String unknownString = new String(chars, currPosition, leftLength < 32 ? leftLength : 32);
                    throw new IllegalExpressionException("can not parse symbol: " + unknownString);
                }
                for (int i = 1; i < result.length; i++) {
                    elementList.add(result[i]);
                }
                currPosition = (Integer) result[0];
            } catch (Exception e) {
                throw new IllegalExpressionException("Exception occurs on parsing: " + e.getMessage());
            }
        }
        return elementList.toArray();
    }
}