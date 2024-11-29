#include <stdio.h>
#include <cub/cub.cuh>
#include "cuda_helper.h"

extern "C" void DownsampDataArray(float *currentdsdata_gpu, float *currentdata_gpu, int worklen, int bs, int nsub);
extern "C" void float_dedisp_gpu(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets, cudaStream_t stream);
extern "C" void float_dedisp_waterfall_gpu(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets, cudaStream_t stream);
extern "C" void float_dedisp_waterfall_gpu2(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets, float *channel_means_gpu, float *channel_means_std_gpu, float *channel_means_mean_gpu, cudaStream_t stream);
extern "C" void dsamp_in_time_gpu(int batchsize, int numchan, int dsamp_len, long long start_index, int one_ms_len, float *inputs, float *outputs, cudaStream_t stream);
extern "C" void dsamp_in_time_gpu_bg(int batchsize, int numchan, int dsamp_len, long long start_index, int one_ms_len, float *inputs, float *outputs, int tmp, cudaStream_t stream);
extern "C" void boxcar_two_prefix_gpu(int batchsize, int numchan, float *matrix, float *bg, float *sums, cudaStream_t stream);
extern "C" void boxcar_two_suffix_gpu(int batchsize, int numchan, int BASSET_Ls_num, int *BASSET_Ls, float *sums, float *corr, cudaStream_t stream);
extern "C" void get_SNR_gpu(int batchsize, float *corr, float *corr_bg, int numchan, int *BASSET_Ls, int *widths_gpu, int BASSET_Ls_num, float *SNRs_gpu, cudaStream_t stream);
extern "C" void convolve_gpu(int batchsize, float *input, size_t input_size, float *output, size_t kernel_size, cudaStream_t stream);
extern "C" void autocorrelation_gpu(int batchsize, float *input, size_t input_size, float *output, cudaStream_t stream);

__global__ void DownsampDataArray_kernel(float *currentdsdata_gpu, float *currentdata_gpu, int nsub, int worklen, int bs)
{
    int ThrPerBlk = blockDim.x;
    int MYbid = blockIdx.x;
    int MYtid = threadIdx.x;
    int MYgtid = ThrPerBlk * MYbid + MYtid;
    if (MYgtid >= worklen * nsub)
        return;

    int j;
    float val = 0;
    int jj = (int)(MYgtid / worklen);
    int ii = MYgtid - jj * worklen;

    int index = jj + ii * bs * nsub;
    int dsindex = jj + ii * nsub;
    for (j = 0; j < bs; j++)
    {
        val += (currentdata_gpu[index]);
        index += nsub;
    }
    currentdsdata_gpu[dsindex] = val / bs;
}

__global__ void float_dedisp_gpu_kernel(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets)
{
    int i, j, b;
    int ThrPerBlk = blockDim.x;
    int MYbid = blockIdx.x;
    int MYtid = threadIdx.x;
    int MYgtid = ThrPerBlk * MYbid + MYtid;
    if (MYgtid >= worklen * numdms)
        return;

    float val = 0;
    int dmi = (int)(MYgtid / worklen); // dms
    int xi = MYgtid - dmi * worklen;   // worklen
    for (i = 0; i < nsub; i++)
    {
        int offset = offsets[dmi * nsub + i];
        int jj = i + (xi + offset) * nsub;
        if (xi < (worklen - offset))
        {
            val += lastdsdata_gpu[jj];
        }
        else
        {
            val += currentdsdata_gpu[i + (xi - (worklen - offset)) * nsub];
        }
    }
    outdata_gpu[MYgtid] = val;
}

__global__ void float_dedisp_waterfall_gpu_kernel(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets)
{
    int i, j, b;
    int ThrPerBlk = blockDim.x;
    int MYbid = blockIdx.x;
    int MYtid = threadIdx.x;
    int MYgtid = ThrPerBlk * MYbid + MYtid;
    if (MYgtid >= worklen * numdms)
        return;

    int dmi = (int)(MYgtid / worklen); // dms
    int xi = MYgtid - dmi * worklen;   // worklen
    for (i = 0; i < nsub; i++)
    {
        int offset = offsets[dmi * nsub + i];
        int jj = i + (xi + offset) * nsub;
        if (xi < (worklen - offset))
        {
            outdata_gpu[MYgtid * nsub + i] = lastdsdata_gpu[jj];
        }
        else
        {
            outdata_gpu[MYgtid * nsub + i] = currentdsdata_gpu[i + (xi - (worklen - offset)) * nsub];
        }
    }
}

void DownsampDataArray(float *currentdsdata_gpu, float *currentdata_gpu, int worklen, int bs, int nsub)
{
    int BlkPerRow = (worklen * nsub - 1 + 1024) / 1024;
    DownsampDataArray_kernel<<<BlkPerRow, 1024>>>(currentdsdata_gpu, currentdata_gpu, nsub, worklen, bs);
}

void float_dedisp_gpu(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets, cudaStream_t stream)
{
    int Blocksize = 1024;
    int BlkPerRow = (worklen * numdms - 1 + Blocksize) / Blocksize;
    float_dedisp_gpu_kernel<<<BlkPerRow, Blocksize, 0, stream>>>(currentdsdata_gpu, lastdsdata_gpu, outdata_gpu, worklen, nsub, numdms, offsets);
}

void float_dedisp_waterfall_gpu(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets, cudaStream_t stream)
{
    int Blocksize = 1024;
    int BlkPerRow = (worklen * numdms - 1 + Blocksize) / Blocksize;
    float_dedisp_waterfall_gpu_kernel<<<BlkPerRow, Blocksize, 0, stream>>>(currentdsdata_gpu, lastdsdata_gpu, outdata_gpu, worklen, nsub, numdms, offsets);
}

__global__ void waterfall_result_channel_kernel(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets, float *channel_means_gpu)
{
    int i, j, b;
    int ThrPerBlk = blockDim.x;
    int MYbid = blockIdx.x;
    int MYtid = threadIdx.x;
    int MYgtid = ThrPerBlk * MYbid + MYtid;
    if (MYgtid >= worklen * numdms)
        return;

    int dmi = (int)(MYgtid / worklen); // dms
    int xi = MYgtid - dmi * worklen;   // worklen
    for (i = 0; i < nsub; i++)
    {
        int offset = offsets[dmi * nsub + i];
        int jj = i + (xi + offset) * nsub;
        if (xi < (worklen - offset))
        {
            outdata_gpu[MYgtid * nsub + i] = lastdsdata_gpu[jj];
        }
        else
        {
            outdata_gpu[MYgtid * nsub + i] = currentdsdata_gpu[i + (xi - (worklen - offset)) * nsub];
        }
        float val = atomicAdd(&channel_means_gpu[dmi * nsub + i], outdata_gpu[MYgtid * nsub + i]);
    }
}

__global__ void waterfall_mean_std_kernel(float *data, int length, int worklen, float *means, float *stddevs3)
{
    int arrayIndex = blockIdx.x;
    int tid = threadIdx.x;
    float *array = &data[arrayIndex * length];

    extern __shared__ float sharedData[];
    float *sharedSums = &sharedData[0];
    int *sharedCounts = (int *)&sharedData[blockDim.x];

    float sum = 0;
    int count = 0;

    // 每个线程处理一部分数据，并计算非零元素的总和和计数
    for (int i = tid; i < length; i += blockDim.x)
    {
        if (array[i] != 0)
        {
            array[i] = array[i] / worklen;
            sum += array[i];
            count++;
        }
    }
    sharedSums[tid] = sum;
    sharedCounts[tid] = count;

    __syncthreads(); // 确保所有线程都写入shared memory

    // 使用线程0对block中的结果进行归约
    if (tid == 0)
    {
        float totalSum = 0;
        int totalCount = 0;
        for (int i = 0; i < blockDim.x; i++)
        {
            totalSum += sharedSums[i];
            totalCount += sharedCounts[i];
        }

        means[arrayIndex] = (totalCount > 0) ? (totalSum / totalCount) : 0; // 避免除以0

        float varianceSum = 0;
        for (int i = 0; i < length; i++)
        {
            if (array[i] != 0)
            {
                float diff = array[i] - means[arrayIndex];
                varianceSum += diff * diff;
            }
        }
        stddevs3[arrayIndex] = (totalCount > 1) ? 3 * sqrtf(varianceSum / (totalCount - 1)) : 0; // 样本标准差
    }
}

__global__ void waterfall_norm_result_kernel(float *data, float *channel_means, int nsub, int worklen, int numdms, float *means, float *stddevs3)
{
    int i;
    int ThrPerBlk = blockDim.x;
    int MYbid = blockIdx.x;
    int MYtid = threadIdx.x;
    int MYgtid = ThrPerBlk * MYbid + MYtid;
    if (MYgtid >= worklen * numdms)
        return;

    int dmi = (int)(MYgtid / worklen); // dms
    int xi = MYgtid - dmi * worklen;   // worklen
    float *result = &data[MYgtid * nsub];
    for (i = 0; i < nsub; i++)
    {

        float deviation = channel_means[dmi * nsub + i] - means[dmi];
        if (fabs(deviation) < stddevs3[dmi])
        {
            result[i] -= channel_means[i];
        }
        else
        {
            result[i] = 0;
        }
    }
}

void float_dedisp_waterfall_gpu2(float *currentdsdata_gpu, float *lastdsdata_gpu, float *outdata_gpu, int worklen, int nsub, int numdms, int *offsets, float *channel_means_gpu, float *channel_means_std_gpu, float *channel_means_mean_gpu, cudaStream_t stream)
{
    int Blocksize = 256;
    int BlkPerRow = (worklen * numdms - 1 + Blocksize) / Blocksize;
    waterfall_result_channel_kernel<<<BlkPerRow, Blocksize, 0, stream>>>(currentdsdata_gpu, lastdsdata_gpu, outdata_gpu, worklen, nsub, numdms, offsets, channel_means_gpu);

    waterfall_mean_std_kernel<<<numdms, Blocksize, 2 * Blocksize * sizeof(float), stream>>>(channel_means_gpu, nsub, worklen, channel_means_mean_gpu, channel_means_std_gpu);

    waterfall_norm_result_kernel<<<BlkPerRow, Blocksize, 0, stream>>>(outdata_gpu, channel_means_gpu, nsub, worklen, numdms, channel_means_mean_gpu, channel_means_std_gpu);
}

__global__ void dsamp_in_time_gpu_kernel(int batchsize, int numchan, int dsamp_len, long long start_index, int one_ms_len, float *inputs, float *outputs)
{
    // 计算全局线程索引
    int batch_index = blockIdx.x;                           // 直接映射到 batchsize
    int chan_index = blockIdx.y * blockDim.x + threadIdx.x; // 映射到处理的 numchan 部分

    // 确保索引在有效范围内
    if (batch_index >= batchsize || chan_index >= numchan)
    {
        return;
    }

    float val = 0;
    float *data = &inputs[numchan * (batch_index + start_index) * one_ms_len];
    // 遍历每个 dsamp_len 处理数据
    for (int i = 0; i < dsamp_len; i++)
    {
        val += data[chan_index + i * numchan];
    }
    outputs[batch_index * numchan + chan_index] = val;
}

void dsamp_in_time_gpu(int batchsize, int numchan, int dsamp_len, long long start_index, int one_ms_len, float *inputs, float *outputs, cudaStream_t stream)
{
    int Blocksize = 1024;
    dim3 blocks_dim(batchsize, (numchan + Blocksize - 1) / Blocksize, 1);
    dsamp_in_time_gpu_kernel<<<blocks_dim, Blocksize, 0, stream>>>(batchsize, numchan, dsamp_len, start_index, one_ms_len, inputs, outputs);
}

__global__ void dsamp_in_time_gpu_bg_kernel(int batchsize, int numchan, int dsamp_len, long long start_index, int one_ms_len, float *inputs, float *outputs, int tmp)
{
    // 计算全局线程索引
    int batch_index = blockIdx.x;                           // 直接映射到 batchsize
    int chan_index = blockIdx.y * blockDim.x + threadIdx.x; // 映射到处理的 numchan 部分

    // 确保索引在有效范围内
    if (batch_index >= batchsize || chan_index >= numchan)
    {
        return;
    }

    float val = 0;
    float *data = &inputs[numchan * (batch_index + start_index - 1 - tmp) * one_ms_len];
    // 遍历每个 dsamp_len 处理数据
    for (int i = 0; i < dsamp_len; i++)
    {
        val += data[chan_index + i * numchan];
    }
    outputs[batch_index * numchan + chan_index] = val;
}

void dsamp_in_time_gpu_bg(int batchsize, int numchan, int dsamp_len, long long start_index, int one_ms_len, float *inputs, float *outputs, int tmp, cudaStream_t stream)
{
    int Blocksize = 1024;
    dim3 blocks_dim(batchsize, (numchan + Blocksize - 1) / Blocksize, 1);
    dsamp_in_time_gpu_bg_kernel<<<blocks_dim, Blocksize, 0, stream>>>(batchsize, numchan, dsamp_len, start_index, one_ms_len, inputs, outputs, tmp);
}

__global__ void convolve_gpu_kernel(int batchsize, float *input, size_t input_size, float *output, size_t start_index, size_t kernel_size)
{
    // 计算全局线程索引
    int batch_index = blockIdx.x;                           // 直接映射到 batchsize
    int chan_index = blockIdx.y * blockDim.x + threadIdx.x; // 映射到处理的 numchan 部分

    // 确保索引在有效范围内
    if (batch_index >= batchsize || chan_index >= input_size)
    {
        return;
    }

    float val = 0;
    for (int j = 0; j < kernel_size; ++j)
    {
        int input_idx = chan_index - j + start_index; // Adjust index to simulate 'full' mode
        if (input_idx >= 0 && input_idx < input_size)
        {
            val += input[batch_index * input_size + input_idx];
        }
    }
    output[batch_index * input_size + chan_index] = val;
}

void convolve_gpu(int batchsize, float *input, size_t input_size, float *output, size_t kernel_size, cudaStream_t stream)
{
    int Blocksize = 1024;
    dim3 blocks_dim(batchsize, (input_size + Blocksize - 1) / Blocksize, 1);
    size_t start_index = (kernel_size - 1) / 2;
    convolve_gpu_kernel<<<blocks_dim, Blocksize, 0, stream>>>(batchsize, input, input_size, output, start_index, kernel_size);
}

__global__ void autocorrelation_gpu_kernel(int batchsize, float *input, float *output, int result_size, int gap, int input_size)
{
    // 计算全局线程索引
    int batch_index = blockIdx.x;                          // 直接映射到 batchsize
    int res_index = blockIdx.y * blockDim.x + threadIdx.x; // 映射到处理的 result_size 部分

    // 确保索引在有效范围内
    if (batch_index >= batchsize || res_index >= result_size)
    {
        return;
    }

    float val = 0;
    int prefix = batch_index * input_size;
    if (res_index < gap)
    {
        for (int j = 0; j < res_index + 1; ++j)
        {
            val += input[prefix + gap - j] * input[prefix + res_index - j];
        }
    }
    else
    {
        for (int j = 0; j < result_size - res_index; ++j)
        {
            val += input[prefix + j] * input[prefix + j + res_index - gap];
        }
    }
    output[batch_index * result_size + res_index] = val;
}

void autocorrelation_gpu(int batchsize, float *input, size_t input_size, float *output, cudaStream_t stream)
{
    int result_size = 2 * input_size - 1;
    int gap = input_size - 1;

    int Blocksize = 1024;
    dim3 blocks_dim(batchsize, (result_size + Blocksize - 1) / Blocksize, 1);
    autocorrelation_gpu_kernel<<<blocks_dim, Blocksize, 0, stream>>>(batchsize, input, output, result_size, gap, input_size);
}

__global__ void boxcar_two_prefix_gpu_kernel(int batchsize, int numchan, float *matrix, float *bg, float *sums)
{
    // 计算全局线程索引
    int array_index = blockIdx.x;
    int batch_index = blockIdx.y * blockDim.x + threadIdx.x;
    // 确保索引在有效范围内
    if (batch_index >= batchsize || array_index >= 2)
    {
        return;
    }
    float *arr;
    if (array_index == 0)
        arr = matrix;
    else
        arr = bg;

    int sums_prefix = batchsize * (numchan + 1) * array_index + batch_index * (numchan + 1);
    sums[sums_prefix] = 0.0;
    for (int ii = 0; ii < numchan; ++ii)
    {
        sums[sums_prefix + ii + 1] = sums[sums_prefix + ii] + arr[batch_index * numchan + ii];
    }
}

void boxcar_two_prefix_gpu(int batchsize, int numchan, float *matrix, float *bg, float *sums, cudaStream_t stream)
{
    int Blocksize = 256;
    dim3 blocks_dim(2, (batchsize + Blocksize - 1) / Blocksize, 1);
    boxcar_two_prefix_gpu_kernel<<<blocks_dim, Blocksize, 0, stream>>>(batchsize, numchan, matrix, bg, sums);
}

__global__ void boxcar_two_suffix_gpu_kernel(int batchsize, int numchan, int BASSET_Ls_num, int *BASSET_Ls, float *sums, float *corr)
{
    // 计算全局线程索引
    int array_index = blockIdx.x;
    int thread_index = blockIdx.y * blockDim.x + threadIdx.x;
    // 确保索引在有效范围内
    if (thread_index >= batchsize * BASSET_Ls_num || array_index >= 2)
    {
        return;
    }

    int batch_index = thread_index / BASSET_Ls_num;
    int widths_index = thread_index % BASSET_Ls_num;

    float *convolution_result = &corr[array_index * BASSET_Ls_num * numchan * batchsize + BASSET_Ls_num * numchan * batch_index];
    float *arr = &sums[batchsize * (numchan + 1) * array_index + batch_index * (numchan + 1)];

    int width = BASSET_Ls[widths_index];
    for (int ii = 0; ii < numchan - width + 1; ++ii)
    {
        convolution_result[widths_index * numchan + ii] = arr[ii + width] - arr[ii];
    }
}

void boxcar_two_suffix_gpu(int batchsize, int numchan, int BASSET_Ls_num, int *BASSET_Ls, float *sums, float *corr, cudaStream_t stream)
{
    int Blocksize = 256;
    dim3 blocks_dim(2, (batchsize * BASSET_Ls_num + Blocksize - 1) / Blocksize, 1);
    boxcar_two_suffix_gpu_kernel<<<blocks_dim, Blocksize, 0, stream>>>(batchsize, numchan, BASSET_Ls_num, BASSET_Ls, sums, corr);
}


__global__ void SNR_max_kernel(float *corr, int batchsize, int numchan, int *widths, int num_widths, float *SNRs_gpu)
{
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    // 确保索引在有效范围内
    if (thread_index >= batchsize * num_widths)
    {
        return;
    }

    int batch_index = thread_index / num_widths;
    int widths_index = thread_index % num_widths;

    int width = widths[widths_index];
    float *arr = &corr[batch_index * num_widths * numchan + widths_index * numchan];
    int size = numchan - width + 1;

    float max_value = arr[0];

    for (int i = 1; i < size; ++i)
    {
        if (arr[i] > max_value)
        {
            max_value = arr[i];
        }
    }
    SNRs_gpu[batch_index * num_widths + widths_index] = max_value;
}


const int nTPB = 512;
const int nIPT = 8;
__global__ void SNR_median_kernel(float *corr_bg, int batchsize, int size, int numchan, int num_widths, int widths_index, float *SNRs_gpu)
{
    int batch_index = blockIdx.x;
    float *data = &corr_bg[batchsize * numchan * num_widths + batch_index * num_widths * numchan + widths_index * numchan];

    // Specialize BlockRadixSort for a 1D block of 512 threads owning 8 integer items each
    using BlockRadixSort = cub::BlockRadixSort<float, nTPB, nIPT>;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Obtain a segment of consecutive items across threads
    float thread_keys[nIPT];
    for (int i = 0; i < nIPT; i++)
    {
        int index = i * nTPB + threadIdx.x;
        if (index < size)
        {
            thread_keys[i] = data[index];
        }
        else
        {
            thread_keys[i] = FLT_MAX; // 使用FLT_MAX填充超出实际长度部分
        }
    }
    // Collectively sort the keys
    BlockRadixSort(temp_storage).Sort(thread_keys);

    int half_size = size / 2;
    int median_threadIdx = (half_size / nIPT) % nTPB;
    int median_local_index = half_size % nIPT;
    if (size % 2 == 1)
    {
        if (threadIdx.x == median_threadIdx)
        {
            SNRs_gpu[batch_index * num_widths + widths_index] -= thread_keys[median_local_index];
        }
    }
    else if (median_local_index != 0)
    {
        if (threadIdx.x == median_threadIdx)
        {
            SNRs_gpu[batch_index * num_widths + widths_index] -= (thread_keys[median_local_index] + thread_keys[median_local_index - 1]) / 2.0;
        }
    }
    else
    {
        if (threadIdx.x == median_threadIdx)
        {
            atomicAdd(&SNRs_gpu[batch_index * num_widths + widths_index], -(thread_keys[0] / 2.0));
        }
        else if (threadIdx.x == median_threadIdx - 1)
        {
            atomicAdd(&SNRs_gpu[batch_index * num_widths + widths_index], -(thread_keys[0] / 2.0));
        }
    }
}

__global__ void SNR_standard_kernel(float *corr_bg, int batchsize, int size, int numchan, int num_widths, int widths_index, float *SNRs_gpu)
{
    int tid = threadIdx.x;
    int batch_index = blockIdx.x;
    float *data = &corr_bg[batchsize * numchan * num_widths + batch_index * num_widths * numchan + widths_index * numchan];

    extern __shared__ float sharedData[];
    float *sharedSums = &sharedData[0];

    float sum = 0;

    // 每个线程处理一部分数据，并计算非零元素的总和和计数
    for (int i = tid; i < size; i += blockDim.x)
    {
        sum += data[i];
    }
    sharedSums[tid] = sum;

    __syncthreads(); // 确保所有线程都写入shared memory

    // 使用线程0对block中的结果进行归约
    if (tid == 0)
    {
        float totalSum = 0;
        for (int i = 0; i < blockDim.x; i++)
        {
            totalSum += sharedSums[i];
        }

        float mean = totalSum / size;

        float varianceSum = 0;
        for (int i = 0; i < size; i++)
        {
            float diff = data[i] - mean;
            varianceSum += diff * diff;
        }
        SNRs_gpu[batch_index * num_widths + widths_index] = SNRs_gpu[batch_index * num_widths + widths_index] / (sqrtf(varianceSum / (size - 1))); // 样本标准差
    }
}

void get_SNR_gpu(int batchsize, float *corr, float *corr_bg, int numchan, int *BASSET_Ls, int *widths_gpu, int BASSET_Ls_num, float *SNRs_gpu, cudaStream_t stream)
{
    int Blocksize = 256;
    dim3 blocks_dim((batchsize * BASSET_Ls_num + Blocksize - 1) / Blocksize, 1, 1);
    SNR_max_kernel<<<blocks_dim, Blocksize, 0, stream>>>(corr, batchsize, numchan, widths_gpu, BASSET_Ls_num, SNRs_gpu);

    Blocksize = 512; // 最大线程数
    for (int jj = 0; jj < BASSET_Ls_num; ++jj)
    {
        int width = BASSET_Ls[jj];
        dim3 blocks_dim_median(batchsize, 1, 1);
        SNR_median_kernel<<<batchsize, Blocksize, 0, stream>>>(corr, batchsize, numchan - width + 1, numchan, BASSET_Ls_num, jj, SNRs_gpu);
        SNR_standard_kernel<<<batchsize, Blocksize, Blocksize * sizeof(float), stream>>>(corr, batchsize, numchan - width + 1, numchan, BASSET_Ls_num, jj, SNRs_gpu);
    }
}