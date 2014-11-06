package zju.util.expression;

/**
 * 本类用来表达"括号"这样的表达式元素。只有两种：左括号和右括号。括号可用来改变运算符的优先顺序。
 * 本类被声明为final（或等价于final），不可继承。
 *
 * @author lyman
 * @version 0.9, 2003/4/19
 */
public class Bracket extends Operator {

    /**
     * 声明为私有函数，阻止从外部创建一个实例
     * @param bracketNotation 括号的记号
     */
    private Bracket(String bracketNotation) {
        super(bracketNotation);
    }

    public static Bracket createLeftBracket() {
        return new Bracket("(");
    }


    public static Bracket createRightBracket() {
        return new Bracket(")");
    }

    public boolean isLeftBracket() {
        return getName().equals("(");
    }

    public boolean isRightBracket() {
        return getName().equals(")");
    }
}