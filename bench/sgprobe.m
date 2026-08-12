#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
static const char *variants[][2] = {
 {"thread-addrspace simdgroup_load",
  "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device float*o){ thread float a[64]; for(int i=0;i<64;i++)a[i]=i; simdgroup_float8x8 m; simdgroup_load(m,(thread float*)a,8); o[0]=1; }"},
 {"threadgroup simdgroup_load (baseline)",
  "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device float*o, threadgroup float* s [[threadgroup(0)]]){ simdgroup_float8x8 m; simdgroup_load(m,s,8); o[0]=1; }"},
 {"simdgroup_matrix element write via thread_elements()",
  "#include <metal_stdlib>\nusing namespace metal;\nkernel void k(device float*o){ simdgroup_float8x8 m=make_filled_simdgroup_matrix<float,8>(0.f); auto e=m.thread_elements(); e[0]=1.f; o[0]=e[0]; }"},
};
int main(){@autoreleasepool{
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();
 for(int i=0;i<3;i++){
   NSError*e=nil;
   [d newLibraryWithSource:[NSString stringWithUTF8String:variants[i][1]] options:[MTLCompileOptions new] error:&e];
   printf("%-46s : %s\n", variants[i][0], e? "REJECTED":"COMPILES");
   if(e){ NSString*m=e.localizedDescription; NSArray*L=[m componentsSeparatedByString:@"\n"];
          for(NSString*l in L){ if([l containsString:@"error:"]){ printf("      %s\n",[l UTF8String]); break; } } }
 }
}return 0;}
