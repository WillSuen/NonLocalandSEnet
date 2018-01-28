import mxnet as mx


class EyeLikeOp(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        n = x.shape[0]
        assert x.ndim == 2, "Check for 2D"
        assert n ==  x.shape[1], "Check for square array"
        out = mx.nd.one_hot(mx.nd.arange(n), depth=n)
        self.assign(out_data[0], req[0], out)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        n = x.shape[0]
        grad = mx.nd.zeros((n, n))
        self.assign(in_grad[0], req[0], grad)
        
        
@mx.operator.register("eye_like")
class EyeLikeProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(EyeLikeProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return EyeLikeOp()