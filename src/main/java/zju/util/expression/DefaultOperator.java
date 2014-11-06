package zju.util.expression;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-6-12
 */
public class DefaultOperator extends Operator {
    public static final int OPERATOR_ADD = 1;
    public static final int OPERATOR_SUB = 2;
    public static final int OPERATOR_MUL = 3;
    public static final int OPERATOR_DIV = 4;
    public static final int OPERATOR_MOD = 6;
    public static final int OPERATOR_POW = 5;

    public static final int OPERATOR_EXP = 7;
    public static final int OPERATOR_SQRT = 8;

    public static final int OPERATOR_SIN = 9;
    public static final int OPERATOR_COS = 10;
    public static final int OPERATOR_TAN = 11;

    private static final int MODE_INT = 1;
    private static final int MODE_LONG = 2;
    private static final int MODE_FLOAT = 3;
    private static final int MODE_DOUBLE = 4;

    private int operator;
    private int mode = MODE_DOUBLE;

    public DefaultOperator(String name, int operator) {
        super(name);
        this.operator = operator;
    }

    /**
     * @return dimension of the current operator
     */
    public int getDimension() {
        switch (operator) {
            case OPERATOR_ADD:
            case OPERATOR_SUB:
            case OPERATOR_MUL:
            case OPERATOR_DIV:
            case OPERATOR_POW:
            case OPERATOR_MOD:
                return 2;
            default:
                return 1;
        }
    }

    /**
     * evaluate the result
     *
     * @param operands operands
     * @param offset  offset in operands
     * @return compuation result
     * @throws IllegalArgumentException when this is not enough operands in input the exception is throw
     */
    public Number eval(Number[] operands, int offset) {
        if ((operands.length - offset) < getDimension())
            throw new IllegalArgumentException("Lack operand(s), given: " + (operands.length - offset));
        switch (operator) {
            case OPERATOR_ADD:
                return add(operands, offset);
            case OPERATOR_SUB:
                return sub(operands, offset);
            case OPERATOR_MUL:
                return mul(operands, offset);
            case OPERATOR_DIV:
                return div(operands, offset);
            case OPERATOR_MOD:
                return mod(operands, offset);
            case OPERATOR_POW:
                return Math.pow(operands[offset].doubleValue(), operands[offset + 1].doubleValue());
            case OPERATOR_EXP:
                return Math.exp(operands[offset].doubleValue());
            case OPERATOR_SQRT:
                return Math.sqrt(operands[offset].doubleValue());
            case OPERATOR_SIN:
                return Math.sin(operands[offset].doubleValue());
            case OPERATOR_COS:
                return Math.cos(operands[offset].doubleValue());
            case OPERATOR_TAN:
                return Math.tan(operands[offset].doubleValue());
            default:
                return null;
        }
    }

    public Number eval(Number[] oprands) {
        return eval(oprands, 0);
    }

    private Number add(Number[] oprands, int offset) {
        Number oprand1 = oprands[offset];
        Number oprand2 = oprands[offset + 1];
        switch (mode) {
            case MODE_INT:
                return oprand1.intValue() + oprand2.intValue();
            case MODE_LONG:
                return oprand1.intValue() + oprand2.intValue();
            case MODE_FLOAT:
                return oprand1.floatValue() + oprand2.floatValue();
            case MODE_DOUBLE:
                return oprand1.doubleValue() + oprand2.doubleValue();
        }
        return null;
    }

    private Number sub(Number[] oprands, int offset) {
        Number oprand1 = oprands[offset];
        Number oprand2 = oprands[offset + 1];
        switch (mode) {
            case MODE_INT:
                return oprand1.intValue() - oprand2.intValue();
            case MODE_LONG:
                return oprand1.intValue() - oprand2.intValue();
            case MODE_FLOAT:
                return oprand1.floatValue() - oprand2.floatValue();
            case MODE_DOUBLE:
                return oprand1.doubleValue() - oprand2.doubleValue();
        }
        return null;
    }

    private Number mul(Number[] oprands, int offset) {
        Number oprand1 = oprands[offset];
        Number oprand2 = oprands[offset + 1];

        switch (mode) {
            case MODE_INT:
                return oprand1.intValue() * oprand2.intValue();
            case MODE_LONG:
                return oprand1.intValue() * oprand2.intValue();
            case MODE_FLOAT:
                return oprand1.floatValue() * oprand2.floatValue();
            case MODE_DOUBLE:
                return oprand1.doubleValue() * oprand2.doubleValue();
        }
        return null;
    }

    private Number div(Number[] oprands, int offset) {
        Number oprand1 = oprands[offset];
        Number oprand2 = oprands[offset + 1];
        switch (mode) {
            case MODE_INT:
                return oprand1.intValue() / oprand2.intValue();
            case MODE_LONG:
                return oprand1.intValue() / oprand2.intValue();
            case MODE_FLOAT:
                return oprand1.floatValue() / oprand2.floatValue();
            case MODE_DOUBLE:
                return oprand1.doubleValue() / oprand2.doubleValue();
        }
        return null;
    }

    private Number mod(Number[] oprands, int offset) {
        Number oprand1 = oprands[offset];
        Number oprand2 = oprands[offset + 1];
        switch (mode) {
            case MODE_INT:
                return oprand1.intValue() % oprand2.intValue();
            case MODE_LONG:
                return oprand1.intValue() % oprand2.intValue();
            case MODE_FLOAT:
                return oprand1.floatValue() % oprand2.floatValue();
            case MODE_DOUBLE:
                return oprand1.doubleValue() % oprand2.doubleValue();
        }
        return null;
    }

    public int getOperator() {
        return operator;
    }

    public int getMode() {
        return mode;
    }

    public void setMode(int mode) {
        this.mode = mode;
    }
}
