#define OPERATOR_PLUS 1.0
#define OPERATOR_MINUS 2.0
#define OPERATOR_MULTIPLY 3.0
#define OPERATOR_DIVIDE 4.0
#define OPERATOR_SIN 5.0
#define OPERATOR_COS 6.0

__global__ void graph_compute(float* node, int* parent, int start, int end) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = (end - start) / 2;
    if (tid < threadNum) {
        float left = node[start + tid * 2];
        float right = node[start + tid * 2 + 1];
        int parentId = parent[start / 2 + tid];
        float oper = node[parentId];
        float result;
        switch (oper) {
            case OPERATOR_PLUS:
                result = left + right;
                break;
            case OPERATOR_MINUS:
                result = left - right;
                break;
            case OPERATOR_MULTIPLY:
                result = left * right;
                break;
            case OPERATOR_DIVIDE:
                result = left / right;
                break;
            case OPERATOR_SIN:
                result = sinf(left);
                break;
            case OPERATOR_PLUS:
                result = cosf(left);
                break;
            default:
                result = 0;
        }
        node[parentId] = result;
    }
}