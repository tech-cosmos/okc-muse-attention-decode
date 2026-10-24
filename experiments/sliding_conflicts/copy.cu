// Offline diagnostic only: copies bytes, does not implement attention.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
constexpr int BLOCKS=64, BYTES=4096;
cudaStream_t probe_stream;
template<int DEST> __device__ unsigned map(unsigned i) {
 if constexpr(DEST==1) return i ^ (((i>>7)&3)<<4); // production K SW64
 if constexpr(DEST==2) return (i/16)*128+i%16; // deliberate 8-way stress
 return i; // production Vraw
}
template<int SOURCE,int DEST> __global__ void copy_probe(const unsigned char* src,unsigned char* dst) {
 __shared__ __align__(512) unsigned char sm[32768];
 unsigned tid=threadIdx.x;
 #pragma unroll
 for(int j=0;j<2;j++) {
  unsigned i=tid+j*128,row=i/4,piece=(i%4)*16;
  unsigned source_row=SOURCE==2?((row*17+blockIdx.x*67)%8192):blockIdx.x*64+row;
  unsigned source_stride=SOURCE==0?64:128;
  unsigned address=(unsigned)__cvta_generic_to_shared(sm+map<DEST>(row*64+piece));
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(address),"l"(src+source_row*source_stride+piece):"memory");
 }
 asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");
 __syncthreads();
 for(unsigned i=tid;i<BYTES;i+=128) dst[blockIdx.x*BYTES+i]=sm[map<DEST>(i)];
}
void launch(int source,int dest,unsigned char* src,unsigned char* out) {
 #define CALL(S,D) copy_probe<S,D><<<BLOCKS,128,0,probe_stream>>>(src,out)
 #define ROW(S) if(dest==0){CALL(S,0);}else if(dest==1){CALL(S,1);}else{CALL(S,2);}
 if(source==0){ROW(0);}else if(source==1){ROW(1);}else{ROW(2);}
 CK(cudaGetLastError());
}
int main(int argc,char** argv) {
 int source=argc>1?atoi(argv[1]):0,dest=argc>2?atoi(argv[2]):0;
 if(source<0||source>2||dest<0||dest>2) return 2;
 int lock=open("/tmp/okc-gpu.lock",O_CREAT|O_WRONLY,0666);
 if(lock<0||flock(lock,LOCK_EX)) return 3;
 unsigned char *src,*out; CK(cudaMalloc(&src,8192*128)); CK(cudaMalloc(&out,BLOCKS*BYTES));
 CK(cudaStreamCreate(&probe_stream));
 std::vector<unsigned char> input(8192*128),output(BLOCKS*BYTES);
 cudaGraph_t graph; cudaGraphExec_t exec;
 CK(cudaStreamBeginCapture(probe_stream,cudaStreamCaptureModeGlobal)); launch(source,dest,src,out);
 CK(cudaStreamEndCapture(probe_stream,&graph)); CK(cudaGraphInstantiate(&exec,graph,0));
 for(int seed=0;seed<3;seed++) {
  for(unsigned i=0;i<input.size();i++)input[i]=(i*19+(i>>8)*37+seed*71)%256;
  CK(cudaMemcpy(src,input.data(),input.size(),cudaMemcpyHostToDevice));
  CK(cudaMemset(out,0xcc,output.size()));
  CK(cudaGraphLaunch(exec,probe_stream)); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(output.data(),out,output.size(),cudaMemcpyDeviceToHost));
  for(int b=0;b<BLOCKS;b++) for(int i=0;i<BYTES;i++) {
   int row=i/64,sr=source==2?((row*17+b*67)%8192):b*64+row;
   int address=sr*(source==0?64:128)+i%64;
   if(output[b*BYTES+i]!=input[address]){fprintf(stderr,"FAIL %d %d %d %d\n",source,dest,b,i);return 4;}
  }
 }
 // Profile one ordinary call after independently checked fresh graph replays.
 CK(cudaProfilerStart()); launch(source,dest,src,out); CK(cudaDeviceSynchronize()); CK(cudaProfilerStop());
 CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph)); CK(cudaFree(src)); CK(cudaFree(out));
 CK(cudaStreamDestroy(probe_stream));
 printf("PASS source=%d dest=%d fresh_graphs=3 bytes_per_graph=%d\n",source,dest,BLOCKS*BYTES);
 flock(lock,LOCK_UN);close(lock);
}
