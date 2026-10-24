// Match the table's FP32 angle multiplication, then reduce that angle in FP64.
// Hardware sin/cos receive only [-pi,pi]; no large-argument fast-intrinsic error.
__device__ __forceinline__ float2 rope_pair(int position,float frequency) {
 float angle=__fmul_rn((float)position,frequency);
 double turns=(double)__double2int_rn((double)angle*0.15915494309189533577);
 float reduced=__double2float_rn(__fma_rn(-turns,6.28318530717958647693,(double)angle));
 float2 result;
 asm("cos.approx.ftz.f32 %0,%1;":"=f"(result.x):"f"(reduced));
 asm("sin.approx.ftz.f32 %0,%1;":"=f"(result.y):"f"(reduced));
 return result;
}
