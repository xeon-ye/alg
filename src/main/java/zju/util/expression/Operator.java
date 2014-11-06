package zju.util.expression;

/**
 * 运算符类。这是一个抽象类。
 * 所有具体的运算符从Operator继承，实现getDimension()以及eval()方法。
 *
 * @author lyman
 * @version 1.01, 2004/08/24
 */
public abstract class Operator {
    private String name;
    private int priority;

    protected Operator(String name) {
        this.name = name;
    }

    public int getPriority() {
        return priority;
    }

    public void setPriority(int priority) {
        this.priority = priority;
    }

    public String toString() {
        return name;
    }

    public String getName() {
        return name;
    }
}