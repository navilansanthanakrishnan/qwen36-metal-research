#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
static const char *SRC =
"#include <metal_stdlib>\n using namespace metal;\n"
"kernel void probe(device float* out [[buffer(0)]],\n"
"                  threadgroup float* s [[threadgroup(0)]],\n"
"                  uint tid [[thread_index_in_threadgroup]],\n"
"                  uint lane[[thread_index_in_simdgroup]]) {\n"
"  for (uint i = tid; i < 64; i += 32) { s[i] = (float) i; }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  simdgroup_float8x8 m;\n"
"  simdgroup_load(m, s, 8);\n"
"  auto e = m.thread_elements();\n"
"  uint n = sizeof(e)/sizeof(float);\n"
"  out[lane*8 + 0] = (float) n;\n"
"  for (uint k = 0; k < n && k < 7; ++k) { out[lane*8 + 1 + k] = e[k]; }\n"
"}\n";
int main(){@autoreleasepool{
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  NSError *e=nil;
  id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:[MTLCompileOptions new] error:&e];
  if(!lib){ printf("compile failed: %s\n", e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&e];
  id<MTLBuffer> out=[d newBufferWithLength:32*8*sizeof(float) options:MTLResourceStorageModeShared];
  memset(out.contents,0,32*8*sizeof(float));
  id<MTLCommandQueue> q=[d newCommandQueue];
  id<MTLCommandBuffer> cb=[q commandBuffer];
  id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
  [en setComputePipelineState:p]; [en setBuffer:out offset:0 atIndex:0];
  [en setThreadgroupMemoryLength:64*sizeof(float) atIndex:0];
  [en dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
  float *o=(float*)out.contents;
  printf("elements per lane = %d\n", (int)o[0]);
  printf("lane : elements (value == row*8+col of the 8x8)\n");
  for(int l=0;l<32;l++){
    printf("  %2d :", l);
    int n=(int)o[l*8];
    for(int k=0;k<n&&k<7;k++){ int v=(int)o[l*8+1+k]; printf(" %2d(r%d,c%d)", v, v/8, v%8); }
    printf("\n");
  }
}return 0;}
