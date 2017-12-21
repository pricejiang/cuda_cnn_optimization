
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 24
#define X_tile_width 28
#define H_out 24
#define W_out 24


namespace mxnet
{
namespace op
{

__constant__ float W_constant[50][5][5];


__global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    // #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
    __shared__ float X_shared[28][28];

    int n;
    int tx,ty;
    int i;
    int p,q;

    n = blockIdx.x;

    tx = threadIdx.x;
    ty = threadIdx.y;

    int t = ty*32 + tx;

    if(t < 784){
        X_shared[t/28][t%28] = x4d(n,0,t/28,t%28);
    }

    __syncthreads();

    float acc0, acc1;

    for(i = 0;i < 18;i++){
        acc0 = 0.0;
        acc1 = 0.0;
        for(p = 0;p < 5;p++){
            for(q = 0;q < 5;q++){
                acc0 += W_constant[ty][p][q] * X_shared[(i*32+tx)/24 + p][(i*32+tx)%24 + q];
                acc1 += W_constant[ty+25][p][q] * X_shared[(i*32+tx)/24 + p][(i*32+tx)%24 + q];
            }
        }
        y4d(n,ty,(i*32+tx)/24,(i*32+tx)%24) = acc0;
        y4d(n,ty+25,(i*32+tx)/24,(i*32+tx)%24) = acc1;
    }
    

    #undef y4d
    #undef x4d
    // #undef k4d
}



/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    // int H_out = H - K + 1;
    // int W_out = W - K + 1;

    // int H_grid = (H_out - 1)/TILE_WIDTH + 1;
    // int W_grid = (W_out - 1)/TILE_WIDTH + 1;

    // int Z = H_grid * W_grid;

    // Set the kernel dimensions
    dim3 DimGrid(B, 1, 1);
    dim3 DimBlock(32, 25, 1);
    // size_t shmem_size = sizeof(float) * ((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1) + K*K);

    cudaMemcpyToSymbol(W_constant, w.dptr_, 50*5*5*sizeof(float));

    // Call the kernel
    forward_kernel<<<DimGrid, DimBlock, 0, s>>>(y.dptr_,x.dptr_,B,M,C,H,W);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif