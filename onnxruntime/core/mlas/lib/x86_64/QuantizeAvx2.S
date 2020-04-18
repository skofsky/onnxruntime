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

/*++

Routine Description:

    This routine implements the kernels for the quantized linear add
    for element type int8_t.

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

        .globl  C_UNDERSCORE(MlasQLinearAddAvx2Int8)
C_UNDERSCORE(MlasQLinearAddAvx2Int8):
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
        mov     eax,DWORD PTR 127
        sub     eax,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm6,eax
        vbroadcastss ymm6,xmm6
        mov     eax,DWORD PTR -128
        sub     eax,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm7,eax
        vbroadcastss ymm7,xmm7

        sub     rcx,8
        jb      .LQLinearAddAvx2Int8ProcessRemainingCount

.LQLinearAddAvx2Int8Process8Loop:
        vpmovsxbd ymm8,[rdi]
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vmulps  ymm8,ymm8,ymm0
        vpmovsxbd ymm9,[rsi]
        vpsubd  ymm9,ymm9,ymm4
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm9,ymm9,ymm1
        vaddps  ymm8,ymm8,ymm9

        vdivps  ymm8,ymm8,ymm2
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5
        vperm2f128 ymm9,ymm8,ymm8,1
        vpackssdw ymm8,ymm8,ymm9
        vpacksswb xmm8,xmm8,xmm8
        movsd   QWORD PTR [rdx],xmm8

        add     rdi,8
        add     rsi,8
        add     rdx,8
        sub     rcx,8
        jae     .LQLinearAddAvx2Int8Process8Loop

.LQLinearAddAvx2Int8ProcessRemainingCount:
        add     rcx,8
        jz      .LQLinearAddAvx2Int8Exit

        vpmovsxbd ymm8,[rdi]
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vmulps  ymm8,ymm8,ymm0
        vpmovsxbd ymm9,[rsi]
        vpsubd  ymm9,ymm9,ymm4
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm9,ymm9,ymm1
        vaddps  ymm8,ymm8,ymm9

        vdivps  ymm8,ymm8,ymm2
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5
        vperm2f128 ymm9,ymm8,ymm8,1
        vpackssdw ymm8,ymm8,ymm9
        vpacksswb xmm8,xmm8,xmm8
        pextrq  rax,xmm8,0

.LQLinearAddAvx2Int8StoreLoop:
        mov     BYTE PTR [rdx],al
        shr     rax,8
        inc     rdx
        dec     rcx
        jnz     .LQLinearAddAvx2Int8StoreLoop

.LQLinearAddAvx2Int8Exit:
        vzeroupper
        add     rsp,QLinearBinaryElementwiseFrame_ReturnAddress
        ret

/*++

Routine Description:

    This routine implements the kernels for the quantized linear add
    for element type int8_t.

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

        .globl  C_UNDERSCORE(MlasQLinearAddAvx2UInt8)
C_UNDERSCORE(MlasQLinearAddAvx2UInt8):
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
        mov     eax,DWORD PTR 255
        sub     eax,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm6,eax
        vbroadcastss ymm6,xmm6
        mov     eax,DWORD PTR 0
        sub     eax,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm7,eax
        vbroadcastss ymm7,xmm7

        sub     rcx,8
        jb      .LQLinearAddAvx2UInt8ProcessRemainingCount

.LQLinearAddAvx2UInt8Process8Loop:
        vpmovzxbd ymm8,[rdi]
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vmulps  ymm8,ymm8,ymm0
        vpmovzxbd ymm9,[rsi]
        vpsubd  ymm9,ymm9,ymm4
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm9,ymm9,ymm1
        vaddps  ymm8,ymm8,ymm9

        vdivps  ymm8,ymm8,ymm2
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5
        vperm2f128 ymm9,ymm8,ymm8,1
        vpackusdw ymm8,ymm8,ymm9
        vpackuswb xmm8,xmm8,xmm8
        movsd   QWORD PTR [rdx],xmm8

        add     rdi,8
        add     rsi,8
        add     rdx,8
        sub     rcx,8
        jae     .LQLinearAddAvx2UInt8Process8Loop

.LQLinearAddAvx2UInt8ProcessRemainingCount:
        add     rcx,8
        jz      .LQLinearAddAvx2UInt8Exit

        vpmovzxbd ymm8,[rdi]
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vmulps  ymm8,ymm8,ymm0
        vpmovzxbd ymm9,[rsi]
        vpsubd  ymm9,ymm9,ymm4
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm9,ymm9,ymm1
        vaddps  ymm8,ymm8,ymm9

        vdivps  ymm8,ymm8,ymm2
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5
        vperm2f128 ymm9,ymm8,ymm8,1
        vpackusdw ymm8,ymm8,ymm9
        vpackuswb xmm8,xmm8,xmm8
        pextrq    rax,xmm8,0

.LQLinearAddAvx2UInt8StoreLoop:
        mov     BYTE PTR [rdx],al
        shr     rax,8
        inc     rdx
        dec     rcx
        jnz     .LQLinearAddAvx2UInt8StoreLoop

.LQLinearAddAvx2UInt8Exit:
        vzeroupper
        add     rsp,QLinearBinaryElementwiseFrame_ReturnAddress
        ret

        .end
