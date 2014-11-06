package zju.opf;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-12-1
 */
public class OpfModel {
    /**
     * OPF计算参数，参数中的母线号都是节点编号前的
     */
    protected OpfPara para;

    //下面三个参数均是节点编号后的母线号
    public int[] pControl;

    public int[] vControl;

    public int[] qControl;


}
