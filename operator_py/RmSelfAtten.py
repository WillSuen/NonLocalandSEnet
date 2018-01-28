import mxnet as mx


class RmSelfAttenOp(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        n = x.shape[0]
        m = x.shape[1]
        assert x.ndim == 3, "Check for 3D"
        assert m ==  x.shape[2], "Check for square array"
        eye = mx.nd.one_hot(mx.nd.arange(m), depth=m)
        batch_eye = mx.nd.broadcast_axis(mx.nd.reshape(eye, shape=(1, m, m)), axis=0, size=n)
        rmin = mx.nd.min(x, axis=2, keepdims=True)
        out = mx.nd.broadcast_axis(rmin, axis=2, size=m) * batch_eye + x * (1 - eye)
        self.assign(out_data[0], req[0], out)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = out_grad[0]
        n = x.shape[0]
        m = x.shape[1]
        out = out_grad[0]
        eye = mx.nd.one_hot(mx.nd.arange(m), depth=m)
        batch_eye = mx.nd.broadcast_axis(mx.nd.reshape(eye, shape=(1, m, m)), axis=0, size=n)
        grad = (1 - batch_eye) * out
        self.assign(in_grad[0], req[0], grad)
        
@mx.operator.register("RmSelfAtten")
class RmSelfAttenProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(RmSelfAttenProp, self).__init__(need_top_grad=False)

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
        return RmSelfAttenOp()
    
    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [out_grad[0]]