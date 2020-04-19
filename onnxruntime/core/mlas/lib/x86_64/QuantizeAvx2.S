/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QuantizeAvx2.S

Abstract:

    This module implements the kernels for the quantized linear add
    for element type int8_t and uint8_t.

    This implementation uses AVX2 instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

//
// Stack frame layout for the QLinearBinaryElementwise kernel.
//
        .equ    QLinearBinaryElementwiseFrame_ZeroPointA, 0
        .equ    QLinearBinaryElementwiseFrame_ZeroPointB, 8
        .equ    QLinearBinaryElementwiseFrame_ZeroPointC, 16
        .equ    QLinearBinaryElementwiseFrame_ReturnAddress, 24
        .equ    QLinearBinaryElementwiseFrame_N, 32

        .text

        .p2align 5
.LMM256PackBytesControlData:
        .byte 0,4,8,12, 255,255,255,255, 255,255,255,255, 255,255,255,255
        .byte 255,255,255,255, 0,4,8,12, 255,255,255,255, 255,255,255,255
.LMM256PackBytesControlDataPermPS:
        .int  0, 5, 2, 3, 4, 1, 6, 7
/*++
    Generate function for QLienarBinaryElementwiseOp()
        DataType : Uint8 or Int8
        OpName:  Add, Mul, etc
        OpInstruction: vaddps, vmulps, etc
*/
        .macro QLinearBinaryElementwiseOpAvx2 DataType, OpName, OpInstruction

/*++
Routine Description:

    This routine implements the kernels for the Quantize Linear \OpName\()
    for element type \DataType\(), vector on vector.

Arguments:

    InputA (rdi) - Supplies the address of InputA.

    ScaleA (xmm0) - 

    ZeroPointA (rsi) - 

    InputB (rdx) - Supplies the address of InputB.

    ScaleB (xmm1) -

    ZeroPointB (rcx) - 

    ScaleC (xmm2) - 

    ZeroPointC (r8) - 

    OutputC (r9) - Supplies the address of OutputC.

    N - Supplies the number of elements to calculate.

Return Value:

    None.
--*/
        .globl  C_UNDERSCORE(MlasQLinear\OpName\()Avx2\DataType\())
C_UNDERSCORE(MlasQLinear\OpName\()Avx2\DataType\()):
        sub     rsp,QLinearBinaryElementwiseFrame_ReturnAddress

        mov     QLinearBinaryElementwiseFrame_ZeroPointA[rsp],rsi
        mov     QLinearBinaryElementwiseFrame_ZeroPointB[rsp],rcx
        mov     QLinearBinaryElementwiseFrame_ZeroPointC[rsp],r8
        mov     rsi,rdx
        mov     rdx,r9
        mov     rcx,QLinearBinaryElementwiseFrame_N[rsp]

        vbroadcastss ymm0,xmm0                  # VectorScaleA
        vbroadcastss ymm1,xmm1                  # VectorScaleB
        vbroadcastss ymm2,xmm2                  # VectorScaleC
        
        vbroadcastss ymm3,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointA[rsp]
        vbroadcastss ymm4,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointB[rsp]
        vbroadcastss ymm5,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        vmovaps ymm10,.LMM256PackBytesControlData[rip]
        vmovaps ymm11,.LMM256PackBytesControlDataPermPS[rip]

.ifeqs "\DataType\()","Int8"
        mov     r8d,DWORD PTR 127
.else
.ifeqs "\DataType\()","UInt8"
        mov     r8d,DWORD PTR 255
.else
        Invalid Datatype specified, please choose from Int8 or UInt8
.endif
.endif
        sub     r8d,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm6,r8d
        vbroadcastss ymm6,xmm6

.ifeqs "\DataType\()","Int8"
        mov     r9d,DWORD PTR -128
.else
        mov     r9d,DWORD PTR 0
.endif
        sub     r9d,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm7,r9d
        vbroadcastss ymm7,xmm7

        sub     rcx,8
        jb      .LQLinear\OpName\()Avx2\DataType\()ProcessRemainingCount

.LQLinear\OpName\()Avx2\DataType\()Process8Loop:
.ifeqs "\DataType\()","Int8"
        vpmovsxbd ymm8,[rdi]
.else
        vpmovzxbd ymm8,[rdi]
.endif
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vmulps  ymm8,ymm8,ymm0

.ifeqs "\DataType\()","Int8"
        vpmovsxbd ymm9,[rsi]
.else
        vpmovzxbd ymm9,[rsi]
.endif
        vpsubd  ymm9,ymm9,ymm4
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm9,ymm9,ymm1

        \OpInstruction  ymm8,ymm8,ymm9

        vdivps  ymm8,ymm8,ymm2
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5

        # TODO: unpack's better way
        vpshufb ymm8,ymm8,ymm10
        vpermps ymm8,ymm11,ymm8
#        vperm2f128 ymm9,ymm8,ymm8,1
#        vorps   ymm8,ymm8,ymm9

#.ifeqs "\DataType\()","Int8"
#        vperm2f128 ymm9,ymm8,ymm8,1
#        vpackssdw ymm8,ymm8,ymm9
#        vpacksswb xmm8,xmm8,xmm8
#.else
#        vperm2f128 ymm9,ymm8,ymm8,1
#        vpackusdw ymm8,ymm8,ymm9
#        vpackuswb xmm8,xmm8,xmm8
#.endif
        movsd   QWORD PTR [rdx],xmm8

        add     rdi,8
        add     rsi,8
        add     rdx,8
        sub     rcx,8
        jae     .LQLinear\OpName\()Avx2\DataType\()Process8Loop

.LQLinear\OpName\()Avx2\DataType\()ProcessRemainingCount:
        add     rcx,8
        jz      .LQLinear\OpName\()Avx2\DataType\()Exit

.ifeqs "\DataType\()","Int8"
        vpmovsxbd ymm8,[rdi]
.else
        vpmovzxbd ymm8,[rdi]
.endif
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vmulps  ymm8,ymm8,ymm0

.ifeqs "\DataType\()","Int8"
        vpmovsxbd ymm9,[rsi]
.else
        vpmovzxbd ymm9,[rsi]
.endif
        vpsubd  ymm9,ymm9,ymm4
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm9,ymm9,ymm1

        \OpInstruction  ymm8,ymm8,ymm9

        vdivps  ymm8,ymm8,ymm2
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5

        vpshufb ymm8,ymm8,ymm10
        vpermps ymm8,ymm11,ymm8
#        vperm2f128 ymm9,ymm8,ymm8,1
#        vorps   ymm8,ymm8,ymm9

#.ifeqs "\DataType\()","Int8"
#        vperm2f128 ymm9,ymm8,ymm8,1
#        vpackssdw ymm8,ymm8,ymm9
#        vpacksswb xmm8,xmm8,xmm8
#.else
#        vperm2f128 ymm9,ymm8,ymm8,1
#        vpackusdw ymm8,ymm8,ymm9
#        vpackuswb xmm8,xmm8,xmm8
#.endif

        pextrq  rax,xmm8,0

.LQLinear\OpName\()Avx2\DataType\()StoreLoop:
        mov     BYTE PTR [rdx],al
        shr     rax,8
        inc     rdx
        dec     rcx
        jnz     .LQLinear\OpName\()Avx2\DataType\()StoreLoop

.LQLinear\OpName\()Avx2\DataType\()Exit:
        vzeroupper
        add     rsp,QLinearBinaryElementwiseFrame_ReturnAddress
        ret

        .endm


/*++
    Generate function for QLienarVectorOpScalar()
        DataType : Uint8 or Int8
        OpName:  Add, Mul, etc
        OpInstruction: vaddps, vmulps, etc
*/
        .macro QLienarVectorOpScalarAvx2 DataType, OpName, OpInstruction

/*++
Routine Description:

    This routine implements the kernels for the Quantize Linear \OpName\()
    for element type \DataType\(), Vector on Scalar.

Arguments:

    InputA (rdi) - Supplies the address of InputA.

    ScaleA (xmm0) - 

    ZeroPointA (rsi) - 

    InputB (rdx) - Supplies the address of InputB. Note currently InputB has only one element

    ScaleB (xmm1) -

    ZeroPointB (rcx) - 

    ScaleC (xmm2) - 

    ZeroPointC (r8) - 

    OutputC (r9) - Supplies the address of OutputC.

    N - Supplies the number of elements to calculate.

Return Value:

    None.
--*/
        .globl  C_UNDERSCORE(MlasQLinear\OpName\()ScalarAvx2\DataType\())
C_UNDERSCORE(MlasQLinear\OpName\()ScalarAvx2\DataType\()):
        sub     rsp,QLinearBinaryElementwiseFrame_ReturnAddress

        mov     QLinearBinaryElementwiseFrame_ZeroPointA[rsp],rsi
        mov     QLinearBinaryElementwiseFrame_ZeroPointB[rsp],rcx
        mov     QLinearBinaryElementwiseFrame_ZeroPointC[rsp],r8
        mov     rsi,rdx
        mov     rdx,r9
        mov     rcx,QLinearBinaryElementwiseFrame_N[rsp]

        vbroadcastss ymm0,xmm0                  # VectorScaleA
        vbroadcastss ymm1,xmm1                  # VectorScaleB
        vbroadcastss ymm2,xmm2                  # VectorScaleC
        
        vbroadcastss ymm3,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointA[rsp]
        vbroadcastss ymm4,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointB[rsp]
        vbroadcastss ymm5,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]

.ifeqs "\DataType\()","Int8"
        mov     r8d,DWORD PTR 127
.else
.ifeqs "\DataType\()","UInt8"
        mov     r8d,DWORD PTR 255
.else
        Invalid Datatype specified, please choose from Int8 or UInt8
.endif
.endif
        sub     r8d,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm6,r8d
        vbroadcastss ymm6,xmm6

.ifeqs "\DataType\()","Int8"
        mov     r9d,DWORD PTR -128
.else
        mov     r9d,DWORD PTR 0
.endif
        sub     r9d,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm7,r9d
        vbroadcastss ymm7,xmm7

.ifeqs "\DataType\()","Int8"
        movsx   eax,BYTE PTR[rsi]
.else
        movzx   eax,BYTE PTR[rsi]
.endif
        mov     DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointB[rsp],eax
        vbroadcastss ymm9,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointB[rsp]
        vpsubd  ymm4,ymm9,ymm4
        vcvtdq2ps ymm4,ymm4
        vmulps  ymm4,ymm4,ymm1   # Vector for InputB's value

        sub     rcx,8
        jb      .LQLinearVector\OpName\()ScalarAvx2\DataType\()ProcessRemainingCount

.LQLinearVector\OpName\()ScalarAvx2\DataType\()Process8Loop:
.ifeqs "\DataType\()","Int8"
        vpmovsxbd ymm8,[rdi]
.else
        vpmovzxbd ymm8,[rdi]
.endif
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vmulps  ymm8,ymm8,ymm0

        \OpInstruction  ymm8,ymm8,ymm4

        vdivps  ymm8,ymm8,ymm2
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5

        vperm2f128 ymm9,ymm8,ymm8,1
.ifeqs "\DataType\()","Int8"
        vpackssdw ymm8,ymm8,ymm9
        vpacksswb xmm8,xmm8,xmm8
.else
        vpackusdw ymm8,ymm8,ymm9
        vpackuswb xmm8,xmm8,xmm8
.endif
        movsd   QWORD PTR [rdx],xmm8

        add     rdi,8
        add     rdx,8
        sub     rcx,8
        jae     .LQLinearVector\OpName\()ScalarAvx2\DataType\()Process8Loop

.LQLinearVector\OpName\()ScalarAvx2\DataType\()ProcessRemainingCount:
        add     rcx,8
        jz      .LQLinearVector\OpName\()ScalarAvx2\DataType\()Exit

.ifeqs "\DataType\()","Int8"
        vpmovsxbd ymm8,[rdi]
.else
        vpmovzxbd ymm8,[rdi]
.endif
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vmulps  ymm8,ymm8,ymm0

        \OpInstruction  ymm8,ymm8,ymm4

        vdivps  ymm8,ymm8,ymm2
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5

        vperm2f128 ymm9,ymm8,ymm8,1
.ifeqs "\DataType\()","Int8"
        vpackssdw ymm8,ymm8,ymm9
        vpacksswb xmm8,xmm8,xmm8
.else
        vpackusdw ymm8,ymm8,ymm9
        vpackuswb xmm8,xmm8,xmm8
.endif
        pextrq  rax,xmm8,0

.LQLinearVector\OpName\()ScalarAvx2\DataType\()StoreLoop:
        mov     BYTE PTR [rdx],al
        shr     rax,8
        inc     rdx
        dec     rcx
        jnz     .LQLinearVector\OpName\()ScalarAvx2\DataType\()StoreLoop

.LQLinearVector\OpName\()ScalarAvx2\DataType\()Exit:
        vzeroupper
        add     rsp,QLinearBinaryElementwiseFrame_ReturnAddress
        ret

        .endm

//
// Generate the QLinearAdd Avx2 Int8 kernel.
//
QLinearBinaryElementwiseOpAvx2 Int8,Add,vaddps
QLienarVectorOpScalarAvx2 Int8,Add,vaddps

//
// Generate the QLinearAdd Avx2 UInt8 kernel.
//
QLinearBinaryElementwiseOpAvx2 UInt8,Add,vaddps
QLienarVectorOpScalarAvx2 UInt8,Add,vaddps

        .end
