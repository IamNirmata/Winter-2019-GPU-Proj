// c, c++
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <sstream>
#include <chrono>
#include <iostream>
#include <vector>
// cuda, cudnn
#include <cudnn.h>
#include <cublas_v2.h>

#define checkCUDNN(expression)                                         \
{                                                                      \
    cudnnStatus_t status = (expression);                               \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      std::cerr << "Error on line " << __LINE__ << ": "                \
      << cudnnGetErrorString(status) << std::endl;                     \
	  std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
}

#define checkCudaErrors(expression)                                    \
{                                                                      \
    uint32_t status = (expression);                                    \
    if (status != 0) {                                                 \
      std::cerr << "Error on line " << __LINE__ << ": "                \
      << "Cuda failure: " << status << std::endl;                      \
	  std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
}

#define bswap(x) __builtin_bswap32(x);

struct ImageMetaData
{
	uint32_t magicNum;
	uint32_t size;
	uint32_t height;
	uint32_t width;

	void swap()
	{
		magicNum = bswap(magicNum);
		size = bswap(size);
		height = bswap(height);
		width = bswap(width);
	}
};

struct LabelMetaData
{
	uint32_t magicNum;
	uint32_t size;

	void swap()
	{
		magicNum = bswap(magicNum);
		size = bswap(size);
	}
};

struct ConvolutionLayer
{
	int channelsIn, channelsOut, kernelSize;
	int widthIn, heightIn, widthOut, heightOut;

	std::vector<float> weight, bias;

	ConvolutionLayer(int channelsIn_, int channelsOut_, int kernelSize_, int width, int height) :
		             weight(channelsIn_ * kernelSize_ * kernelSize_ * channelsOut_),
		             bias(channelsOut_)
	{
		channelsIn = channelsIn_;
		channelsOut = channelsOut_;
		kernelSize = kernelSize_;
		widthIn = width;
		heightIn = height;
		widthOut = width - kernelSize_ + 1;
		heightOut = height - kernelSize_ + 1;
	}

	bool loadWeights(const char *filePrefix)
	{
		std::stringstream s1, s2;
		s1 << filePrefix << ".bin";
		s2 << filePrefix << ".bias.bin";

		FILE *fp = fopen(s1.str().c_str(), "rb");
		if (!fp) return false;
		fread(&weight[0], sizeof(float), channelsIn * channelsOut * kernelSize * kernelSize, fp);
		fclose(fp);

		fp = fopen(s2.str().c_str(), "rb");
		if (!fp) return false;
		fread(&bias[0], sizeof(float), channelsOut, fp);
		fclose(fp);

		return true;
	}
};

struct PoolingLayer
{
	int size, stride;
	PoolingLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

struct FullyConnectedLayer
{
	int inputs, outputs;
	std::vector<float> weight, bias;

	FullyConnectedLayer(int inputs_, int outputs_) :
		                inputs(inputs_), outputs(outputs_),
		                weight(inputs_ * outputs_), bias(outputs_) {}

	bool loadWeights(const char *filePrefix)
	{
		std::stringstream s1, s2;
		s1 << filePrefix << ".bin";
		s2 << filePrefix << ".bias.bin";

		FILE *fp = fopen(s1.str().c_str(), "rb");
		if (!fp) return false;
		fread(&weight[0], sizeof(float), inputs * outputs, fp);
		fclose(fp);

		fp = fopen(s2.str().c_str(), "rb");
		if (!fp) return false;
		fread(&bias[0], sizeof(float), outputs, fp);
		fclose(fp);

		return true;
	}
};

struct Lenet
{
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;

	cudnnTensorDescriptor_t inputTensor;

	cudnnTensorDescriptor_t conv1_OutputTensor, conv2_OutputTensor, conv1_BiasTensor, conv2_BiasTensor;
	cudnnFilterDescriptor_t conv1_FilterDesc, conv2_FilterDesc;
	cudnnConvolutionDescriptor_t conv1_Desc, conv2_Desc;
	cudnnActivationDescriptor_t conv1_ActDesc, conv2_ActDesc;

	cudnnTensorDescriptor_t pool1_OutputTensor, pool2_OutputTensor;
	cudnnPoolingDescriptor_t pool1_Desc, pool2_Desc;

	cudnnTensorDescriptor_t fc1_OutputTensor, fc2_OutputTensor;
	cudnnActivationDescriptor_t fc1_ActDesc;

	cudnnConvolutionFwdAlgo_t conv1_AlgoDesc, conv2_AlgoDesc;

	size_t m_workSpaceSize;

	ConvolutionLayer *m_conv1, *m_conv2;
	PoolingLayer *m_pool1, *m_pool2;
	FullyConnectedLayer *m_fc1, *m_fc2;

	float *conv1_data, *conv1_relu_data, *pool1_data,
		  *conv2_data, *conv2_relu_data, *pool2_data,
		  *fc1_data, *fc1relu_data, *fc2_data;

	float *conv1_weight, *conv1_bias, *conv2_weight, *conv2_bias;
	float *fc1_weight, *fc1_bias, *fc2_weight, *fc2_bias;

	float *vecter;
	void *workspace;

	Lenet(int channels, int width, int height)
	{
		m_conv1 = new ConvolutionLayer((int)channels, 6, 5, (int)width, (int)height);
		m_pool1 = new PoolingLayer(2, 2);
		m_conv2 = new ConvolutionLayer(m_conv1->channelsOut, 16, 5, m_conv1->widthOut / m_pool1->stride, m_conv1->heightOut / m_pool1->stride);
		m_pool2 = new PoolingLayer(2, 2);
		m_fc1 = new FullyConnectedLayer((m_conv2->channelsOut*m_conv2->widthOut*m_conv2->heightOut) / (m_pool2->stride * m_pool2->stride), 500);
		m_fc2 = new FullyConnectedLayer(m_fc1->outputs, 10);
		m_conv1->loadWeights("conv1");
		m_conv2->loadWeights("conv2");
		m_fc1->loadWeights("fc1");
		m_fc2->loadWeights("fc2");

		m_workSpaceSize = 0;
		Setup();
		MemoryLocate();
	}

	~Lenet()
	{
		checkCudaErrors(cublasDestroy(cublasHandle));
		checkCUDNN(cudnnDestroy(cudnnHandle));
		checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv1_OutputTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv2_OutputTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv1_BiasTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv2_BiasTensor));
		checkCUDNN(cudnnDestroyFilterDescriptor(conv1_FilterDesc));
		checkCUDNN(cudnnDestroyFilterDescriptor(conv2_FilterDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1_Desc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2_Desc));
		checkCUDNN(cudnnDestroyActivationDescriptor(conv1_ActDesc));
		checkCUDNN(cudnnDestroyActivationDescriptor(conv2_ActDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(pool1_OutputTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(pool2_OutputTensor));
		checkCUDNN(cudnnDestroyPoolingDescriptor(pool1_Desc));
		checkCUDNN(cudnnDestroyPoolingDescriptor(pool2_Desc));
		checkCUDNN(cudnnDestroyTensorDescriptor(fc1_OutputTensor));
		checkCUDNN(cudnnDestroyTensorDescriptor(fc2_OutputTensor));
		checkCUDNN(cudnnDestroyActivationDescriptor(fc1_ActDesc));

		checkCudaErrors(cudaFree(conv1_data));
		checkCudaErrors(cudaFree(conv1_relu_data));
		checkCudaErrors(cudaFree(pool1_data));
		checkCudaErrors(cudaFree(conv2_data));
		checkCudaErrors(cudaFree(conv2_relu_data));
		checkCudaErrors(cudaFree(pool2_data));
		checkCudaErrors(cudaFree(fc1_data));
		checkCudaErrors(cudaFree(fc1relu_data));
		checkCudaErrors(cudaFree(fc2_data));

		checkCudaErrors(cudaFree(conv1_weight));
		checkCudaErrors(cudaFree(conv1_bias));
		checkCudaErrors(cudaFree(conv2_weight));
		checkCudaErrors(cudaFree(conv2_bias));
		checkCudaErrors(cudaFree(fc1_weight));
		checkCudaErrors(cudaFree(fc1_bias));
		checkCudaErrors(cudaFree(fc2_weight));
		checkCudaErrors(cudaFree(fc2_bias));

		checkCudaErrors(cudaFree(vecter));
		checkCudaErrors(cudaFree(workspace));

		delete m_conv1, m_conv2, m_pool1, m_pool2, m_fc1, m_fc2;
	}

	void Setup()
	{
		int batchSize = 1;
		size_t sizeBytes = 0;

		checkCudaErrors(cublasCreate(&cublasHandle));
		checkCUDNN(cudnnCreate(&cudnnHandle));
                     
		// conv 1
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1_BiasTensor));
		checkCUDNN(cudnnCreateActivationDescriptor(&conv1_ActDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
		checkCUDNN(cudnnCreateFilterDescriptor(&conv1_FilterDesc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1_Desc));
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1_OutputTensor));

		int conv1_channelsIn = m_conv1->channelsIn;
		int conv1_channelsOut = m_conv1->channelsOut;
		int conv1_heightIn = m_conv1->heightIn;
		int conv1_widthIn = m_conv1->widthIn;
		int conv1_kernelSize = m_conv1->kernelSize;

		checkCUDNN(cudnnSetTensor4dDescriptor(conv1_BiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, conv1_channelsOut,
			                                  1, 1));

		checkCUDNN(cudnnSetActivationDescriptor(conv1_ActDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.01));

		checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, conv1_channelsIn,
			                                  conv1_heightIn, conv1_widthIn));

		checkCUDNN(cudnnSetFilter4dDescriptor(conv1_FilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			                                  conv1_channelsOut, conv1_channelsIn,
			                                  conv1_kernelSize, conv1_kernelSize));

		checkCUDNN(cudnnSetConvolution2dDescriptor(conv1_Desc, 0, 0, 1, 1, 1, 1,
			                                       CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

		checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv1_Desc, inputTensor, conv1_FilterDesc,
			                                             &batchSize, &conv1_channelsIn,
			                                             &conv1_heightIn, &conv1_widthIn));

		checkCUDNN(cudnnSetTensor4dDescriptor(conv1_OutputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, conv1_channelsIn,
			                                  conv1_heightIn, conv1_widthIn));

		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
			                                           inputTensor,
			                                           conv1_FilterDesc,
			                                           conv1_Desc,
			                                           conv1_OutputTensor,
			                                           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			                                           0,
			                                           &conv1_AlgoDesc));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
			                                               inputTensor,
			                                               conv1_FilterDesc,
		                                                   conv1_Desc,
			                                               conv1_OutputTensor,
			                                               conv1_AlgoDesc,
			                                               &sizeBytes));

		m_workSpaceSize = std::max(m_workSpaceSize, sizeBytes);

		// pool 1
		checkCUDNN(cudnnCreatePoolingDescriptor(&pool1_Desc));

		checkCUDNN(cudnnSetPooling2dDescriptor(pool1_Desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
			                                   m_pool1->size, m_pool1->size,
			                                   0, 0,
			                                   m_pool1->stride, m_pool1->stride));

		// conv 2
		checkCUDNN(cudnnCreateTensorDescriptor(&conv2_BiasTensor));
		checkCUDNN(cudnnCreateActivationDescriptor(&conv2_ActDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&pool1_OutputTensor));
		checkCUDNN(cudnnCreateFilterDescriptor(&conv2_FilterDesc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2_Desc));
		checkCUDNN(cudnnCreateTensorDescriptor(&conv2_OutputTensor));

		int conv2_channelsIn = m_conv2->channelsIn;
		int conv2_channelsOut = m_conv2->channelsOut;
		int conv2_heightIn = m_conv2->heightIn;
		int conv2_widthIn = m_conv2->widthIn;
		int conv2_kernelSize = m_conv2->kernelSize;

		checkCUDNN(cudnnSetTensor4dDescriptor(conv2_BiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, m_conv2->channelsOut,
			                                  1, 1));

		checkCUDNN(cudnnSetActivationDescriptor(conv2_ActDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.01));

		checkCUDNN(cudnnSetTensor4dDescriptor(pool1_OutputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, conv2_channelsIn,
			                                  conv2_heightIn, conv2_widthIn));

		checkCUDNN(cudnnSetFilter4dDescriptor(conv2_FilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			                                  conv2_channelsOut, conv2_channelsIn,
			                                  conv2_kernelSize, conv2_kernelSize));

		checkCUDNN(cudnnSetConvolution2dDescriptor(conv2_Desc, 0, 0, 1, 1, 1, 1,
			                                       CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

		checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv2_Desc, pool1_OutputTensor, conv2_FilterDesc,
			                                             &batchSize, &conv2_channelsIn,
			                                             &conv2_heightIn, &conv2_widthIn));

		checkCUDNN(cudnnSetTensor4dDescriptor(conv2_OutputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, conv2_channelsIn,
			                                  conv2_heightIn, conv2_widthIn));

		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
			                                           pool1_OutputTensor,
			                                           conv2_FilterDesc,
			                                           conv2_Desc,
			                                           conv2_OutputTensor,
			                                           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			                                           0,
			                                           &conv2_AlgoDesc));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
			                                               pool1_OutputTensor,
			                                               conv2_FilterDesc,
		                                                   conv2_Desc,
			                                               conv2_OutputTensor,
			                                               conv2_AlgoDesc,
			                                               &sizeBytes));

		m_workSpaceSize = std::max(m_workSpaceSize, sizeBytes);

		// pool 2
		checkCUDNN(cudnnCreateTensorDescriptor(&pool2_OutputTensor));
		checkCUDNN(cudnnCreatePoolingDescriptor(&pool2_Desc));

		checkCUDNN(cudnnSetTensor4dDescriptor(pool2_OutputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, m_conv2->channelsOut,
			                                  m_conv2->heightOut / m_pool2->stride,
			                                  m_conv2->widthOut / m_pool2->stride));

		checkCUDNN(cudnnSetPooling2dDescriptor(pool2_Desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
			                                   m_pool2->size, m_pool2->size,
			                                   0, 0,
			                                   m_pool2->stride, m_pool2->stride));
		// fc 1
		checkCUDNN(cudnnCreateTensorDescriptor(&fc1_OutputTensor));
		checkCUDNN(cudnnCreateActivationDescriptor(&fc1_ActDesc));


		checkCUDNN(cudnnSetTensor4dDescriptor(fc1_OutputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, m_fc1->outputs, 1, 1));

		checkCUDNN(cudnnSetActivationDescriptor(fc1_ActDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.01));

		// fc 2
		checkCUDNN(cudnnCreateTensorDescriptor(&fc2_OutputTensor));
		
		checkCUDNN(cudnnSetTensor4dDescriptor(fc2_OutputTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			                                  1, m_fc2->outputs, 1, 1));
	}

	void MemoryLocate()
	{
		checkCudaErrors(cudaMalloc(&conv1_data, sizeof(float)* m_conv1->channelsOut * m_conv1->heightOut                  * m_conv1->widthOut));
		checkCudaErrors(cudaMalloc(&conv1_relu_data, sizeof(float) * m_conv1->channelsOut * m_conv1->heightOut                  * m_conv1->widthOut));
		checkCudaErrors(cudaMalloc(&pool1_data, sizeof(float) * m_conv1->channelsOut * (m_conv1->heightOut / m_pool1->stride) * (m_conv1->widthOut / m_pool1->stride)));
		checkCudaErrors(cudaMalloc(&conv2_data, sizeof(float) * m_conv2->channelsOut * m_conv2->heightOut                  * m_conv2->widthOut));
		checkCudaErrors(cudaMalloc(&conv2_relu_data, sizeof(float) * m_conv2->channelsOut * m_conv2->heightOut                  * m_conv2->widthOut));
		checkCudaErrors(cudaMalloc(&pool2_data, sizeof(float) * m_conv2->channelsOut * (m_conv2->heightOut / m_pool2->stride) * (m_conv2->widthOut / m_pool2->stride)));
		checkCudaErrors(cudaMalloc(&fc1_data, sizeof(float) * m_fc1->outputs));
		checkCudaErrors(cudaMalloc(&fc1relu_data, sizeof(float) * m_fc1->outputs));
		checkCudaErrors(cudaMalloc(&fc2_data, sizeof(float) * m_fc2->outputs));

		checkCudaErrors(cudaMalloc(&conv1_weight, sizeof(float) * m_conv1->weight.size()));
		checkCudaErrors(cudaMalloc(&conv1_bias, sizeof(float) * m_conv1->bias.size()));
		checkCudaErrors(cudaMalloc(&conv2_weight, sizeof(float) * m_conv2->weight.size()));
		checkCudaErrors(cudaMalloc(&conv2_bias, sizeof(float) * m_conv2->bias.size()));
		checkCudaErrors(cudaMalloc(&fc1_weight, sizeof(float) * m_fc1->weight.size()));
		checkCudaErrors(cudaMalloc(&fc1_bias, sizeof(float) * m_fc1->bias.size()));
		checkCudaErrors(cudaMalloc(&fc2_weight, sizeof(float) * m_fc2->weight.size()));
		checkCudaErrors(cudaMalloc(&fc2_bias, sizeof(float) * m_fc2->bias.size()));

		checkCudaErrors(cudaMalloc(&vecter, sizeof(float)));
		checkCudaErrors(cudaMalloc(&workspace, m_workSpaceSize));

		checkCudaErrors(cudaMemcpyAsync(conv1_weight, &m_conv1->weight[0], sizeof(float) * m_conv1->weight.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(conv1_bias, &m_conv1->bias[0], sizeof(float) * m_conv1->bias.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(conv2_weight, &m_conv2->weight[0], sizeof(float) * m_conv2->weight.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(conv2_bias, &m_conv2->bias[0], sizeof(float) * m_conv2->bias.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(fc1_weight, &m_fc1->weight[0], sizeof(float) * m_fc1->weight.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(fc1_bias, &m_fc1->bias[0], sizeof(float) * m_fc1->bias.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(fc2_weight, &m_fc2->weight[0], sizeof(float) * m_fc2->weight.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(fc2_bias, &m_fc2->bias[0], sizeof(float) * m_fc2->bias.size(), cudaMemcpyHostToDevice));
	}

	void ForwardPass(float *input_data, float *softmax_data)
	{
		float alpha = 1.0f, beta = 0.0f;

		// conv 1
		checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, inputTensor, input_data, 
			                               conv1_FilterDesc, conv1_weight, conv1_Desc, conv1_AlgoDesc,
			                               workspace, m_workSpaceSize, &beta, conv1_OutputTensor, conv1_data));

		checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1_BiasTensor, conv1_bias, &alpha, conv1_OutputTensor, conv1_data));

		checkCUDNN(cudnnActivationForward(cudnnHandle, conv1_ActDesc, &alpha, 
			                              conv1_OutputTensor, conv1_data, &beta, conv1_OutputTensor, conv1_relu_data));

		// pool 1
		checkCUDNN(cudnnPoolingForward(cudnnHandle, pool1_Desc, &alpha, conv1_OutputTensor, 
			                           conv1_relu_data, &beta, pool1_OutputTensor, pool1_data));

		// conv 2
		checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1_OutputTensor, pool1_data,
			                               conv2_FilterDesc, conv2_weight, conv2_Desc, conv2_AlgoDesc,
			                               workspace, m_workSpaceSize, &beta, conv2_OutputTensor, conv2_data));

		checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2_BiasTensor, conv1_bias, &alpha, conv2_OutputTensor, conv2_data));

		checkCUDNN(cudnnActivationForward(cudnnHandle, conv2_ActDesc, &alpha,
			                              conv2_OutputTensor, conv2_data, &beta, conv2_OutputTensor, conv2_relu_data));

		// pool 2
		checkCUDNN(cudnnPoolingForward(cudnnHandle, pool2_Desc, &alpha, conv2_OutputTensor,
			                           conv2_relu_data, &beta, pool2_OutputTensor, pool2_data));

		// fc 1
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			                        m_fc1->outputs, 1, m_fc1->inputs,
			                        &alpha, fc1_weight, m_fc1->inputs,
			                        pool2_data, m_fc1->inputs, &beta,
			                        fc1_data, m_fc1->outputs));

		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			                        m_fc1->outputs, 1, 1,
			                        &alpha, fc1_bias, m_fc1->outputs,
			                        vecter, 1, &alpha,
			                        fc1_data, m_fc1->outputs));

		checkCUDNN(cudnnActivationForward(cudnnHandle, fc1_ActDesc, &alpha,
			                              fc1_OutputTensor, fc1_data, &beta, fc1_OutputTensor, fc1relu_data));

		// fc 2
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			                        m_fc2->outputs, 1, m_fc2->inputs,
			                        &alpha, fc2_weight, m_fc2->inputs,
			                        fc1relu_data, m_fc2->inputs, &beta,
			                        fc2_data, m_fc2->outputs));

		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			                        m_fc2->outputs, 1, 1,
			                        &alpha, fc2_bias, m_fc2->outputs,
			                        vecter, 1, &alpha,
			                        fc2_data, m_fc2->outputs));

		// softmax
		checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
			                           &alpha, fc2_OutputTensor, fc2_data, &beta, fc2_OutputTensor, softmax_data));
	}
};

int main(int argc, char **argv)
{
	checkCudaErrors(cudaSetDevice(0));

	printf("Loading data...\n");

	FILE *fp_image = fopen("t10k-images-idx3-ubyte", "rb");
	if (!fp_image) return -1;

	ImageMetaData imageMetaData;
	if (fread(&imageMetaData, sizeof(ImageMetaData), 1, fp_image) != 1)
	{
		fclose(fp_image);
		return -1;
	}
	imageMetaData.swap();

	FILE *fp_label = fopen("t10k-labels-idx1-ubyte", "rb");
	if (!fp_label) return -1;

	LabelMetaData labelMetaData;
	if (fread(&labelMetaData, sizeof(LabelMetaData), 1, fp_label) != 1)
	{
		fclose(fp_label);
		return -1;
	}
	labelMetaData.swap();

	size_t width = imageMetaData.width;
	size_t height = imageMetaData.height;
	size_t testSize = imageMetaData.size;
	size_t channels = 1;

	std::vector<uint8_t> test_images(testSize * width * height * channels);
	if (fread(&test_images[0], sizeof(uint8_t), testSize * width * height, fp_image) != testSize * width * height)
	{
		fclose(fp_image);
		return -1;
	}

	std::vector<uint8_t> test_labels(testSize);
	if (fread(&test_labels[0], sizeof(uint8_t), testSize, fp_label) != testSize)
	{
		fclose(fp_label);
		return -1;
	}

	fclose(fp_image);
	fclose(fp_label);
	printf("Testing dataset size: %d\n", (int)testSize);

	Lenet LENET(channels, width, height);

	float *inputData, *softMax;
	checkCudaErrors(cudaMalloc(&inputData, sizeof(float) * channels * height * width));
	checkCudaErrors(cudaMalloc(&softMax, sizeof(float) * LENET.m_fc2->outputs));

	int correctCount = 0;
	auto t_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < (int)testSize; ++i)
	{
		std::vector<float> image(width * height);

		for (int j = 0; j < width * height; ++j)
			image[j] = (float)test_images[i * width * height * channels + j] / 255.0f;

		checkCudaErrors(cudaMemcpyAsync(inputData, &image[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));

		LENET.ForwardPass(inputData, softMax);

		std::vector<float> outVector(10);

		checkCudaErrors(cudaMemcpy(&outVector[0], softMax, sizeof(float) * 10, cudaMemcpyDeviceToHost));

		int predict = 0;
		for (int label = 1; label < 10; ++label)
		{
			if (outVector[predict] < outVector[label]) predict = label;
		}

		if (predict == test_labels[i])
			++correctCount;
	}
	auto t_end = std::chrono::high_resolution_clock::now();

	printf("Accuracy: %.2f %% \n", (float)correctCount / (float)testSize * 100.0f);
	printf("Cost time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0f);

	checkCudaErrors(cudaFree(inputData));
	checkCudaErrors(cudaFree(softMax));

	return 0;
}