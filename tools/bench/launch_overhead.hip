// launch_overhead.hip -- kernel launch / sync / event overhead microbenchmarks.
//
// (a) 1000 back-to-back empty kernels on one stream: reports both host enqueue time (wall time to
//     issue all 1000 launches) and total completion time (host wall time until the stream drains).
// (b) the same 1000 launches captured into a hipGraph and replayed 10 times: reports us/launch on
//     replay.
// (c) hipMemcpyAsync D2H of 4 bytes + hipStreamSynchronize round-trip latency.
// (d) hipEventRecord/hipEventQuery overhead.
//
// Usage: launch_overhead.exe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

#define HIP_CHECK(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
      fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(_e)); \
      exit(1); \
    } \
  } while (0)

__global__ void empty_kernel() {}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main() {
    const int N = 1000;
    const int TRIALS = 3;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ---- (a) 1000 back-to-back empty kernel launches ----
    std::vector<double> enqueue_us_trials, completion_us_trials;
    for (int t = 0; t < TRIALS; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipStreamSynchronize(stream));
        auto t2 = std::chrono::high_resolution_clock::now();
        double enqueue_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        double completion_us = std::chrono::duration<double, std::micro>(t2 - t0).count() / N;
        enqueue_us_trials.push_back(enqueue_us);
        completion_us_trials.push_back(completion_us);
    }
    printf("(a) empty_kernel_launch: enqueue_us_per_launch_median=%.3f completion_us_per_launch_median=%.3f\n",
           median(enqueue_us_trials), median(completion_us_trials));
    printf("    raw enqueue_us=[");
    for (double v : enqueue_us_trials) printf("%.3f ", v);
    printf("] raw completion_us=[");
    for (double v : completion_us_trials) printf("%.3f ", v);
    printf("]\n");
    fflush(stdout);

    // ---- (b) same 1000 launches captured into a hipGraph, replayed 10x ----
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int i = 0; i < N; ++i) {
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
    }
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // warmup replay
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> replay_us_trials;
    for (int t = 0; t < TRIALS; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < 10; ++r) {
            HIP_CHECK(hipGraphLaunch(graphExec, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double us_per_launch = std::chrono::duration<double, std::micro>(t1 - t0).count() / (10.0 * N);
        replay_us_trials.push_back(us_per_launch);
    }
    printf("(b) graph_replay(10x, 1000 launches/replay): us_per_launch_median=%.3f raw=[",
           median(replay_us_trials));
    for (double v : replay_us_trials) printf("%.3f ", v);
    printf("]\n");
    fflush(stdout);

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));

    // ---- (c) hipMemcpyAsync D2H of 4 bytes + hipStreamSynchronize round trip ----
    int* d_val;
    HIP_CHECK(hipMalloc(&d_val, sizeof(int)));
    HIP_CHECK(hipMemset(d_val, 0, sizeof(int)));
    int h_val = 0;
    const int M = 1000;
    std::vector<double> d2h_us_trials;
    for (int t = 0; t < TRIALS; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < M; ++i) {
            HIP_CHECK(hipMemcpyAsync(&h_val, d_val, sizeof(int), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / M;
        d2h_us_trials.push_back(us);
    }
    printf("(c) memcpyAsync_D2H_4B+sync roundtrip: us_median=%.3f raw=[", median(d2h_us_trials));
    for (double v : d2h_us_trials) printf("%.3f ", v);
    printf("]\n");
    fflush(stdout);
    HIP_CHECK(hipFree(d_val));

    // ---- (d) hipEventRecord / hipEventQuery overhead ----
    hipEvent_t ev;
    HIP_CHECK(hipEventCreate(&ev));
    const int E = 1000;
    std::vector<double> record_us_trials, query_us_trials;
    for (int t = 0; t < TRIALS; ++t) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < E; ++i) {
            HIP_CHECK(hipEventRecord(ev, stream));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipStreamSynchronize(stream));
        record_us_trials.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / E);

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < E; ++i) {
            (void)hipEventQuery(ev); // ignore result, just measure call overhead
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        query_us_trials.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count() / E);
    }
    printf("(d) hipEventRecord: us_per_call_median=%.3f raw=[", median(record_us_trials));
    for (double v : record_us_trials) printf("%.3f ", v);
    printf("]\n");
    printf("(d) hipEventQuery: us_per_call_median=%.3f raw=[", median(query_us_trials));
    for (double v : query_us_trials) printf("%.3f ", v);
    printf("]\n");
    fflush(stdout);

    HIP_CHECK(hipEventDestroy(ev));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
