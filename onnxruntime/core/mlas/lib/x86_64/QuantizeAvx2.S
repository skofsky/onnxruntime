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
        .byte 0,4,8,12,        255,255,255,255, 255,255,255,255, 255,255,255,255
        .byte 255,255,255,255, 0,4,8,12,        255,255,255,255, 255,255,255,255
.LMM256PackBytesControlDataPermPS:
        .int  0, 5, 2, 3, 4, 1, 6, 7

        .macro UnpackBytesDWords Target, Source, DataType
.ifeqs "\DataType\()","Int8"
        vpmovsxbd \Target\(),\Source\()
.else
        vpmovzxbd \Target\(),\Source\()
.endif
        .endm

        .macro SetMaxInt8ValueDW Target, DataType
.ifeqs "\DataType\()","Int8"
        mov \Target\(),DWORD PTR 127
.else
        mov \Target\(),DWORD PTR 255
.endif
        .endm

        .macro SetMinInt8ValueDW Target, DataType
.ifeqs "\DataType\()","Int8"
        mov \Target\(),DWORD PTR -128
.else
        mov \Target\(),DWORD PTR 0
.endif
        .endm

        .macro CheckInt8DataType DataType
.ifeqs "\DataType\()","Int8"
.else
.ifeqs "\DataType\()","UInt8"
.else
        Invalid Datatype specified, please choose from Int8 or UInt8
.endif
.endif
        .endm

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
        CheckInt8DataType \DataType\()

        sub     rsp,QLinearBinaryElementwiseFrame_ReturnAddress
        mov     QLinearBinaryElementwiseFrame_ZeroPointA[rsp],rsi
        mov     QLinearBinaryElementwiseFrame_ZeroPointB[rsp],rcx
        mov     QLinearBinaryElementwiseFrame_ZeroPointC[rsp],r8
        mov     rcx,QLinearBinaryElementwiseFrame_N[rsp]

        vbroadcastss ymm0,xmm0                  # VectorScaleA
        vbroadcastss ymm1,xmm1                  # VectorScaleB
        vbroadcastss ymm2,xmm2                  # VectorScaleC
        vbroadcastss ymm3,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointA[rsp]
        vbroadcastss ymm4,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointB[rsp]
        vbroadcastss ymm5,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]

        # vector for pack to int8
        vmovaps ymm10,.LMM256PackBytesControlData[rip]
        vmovaps ymm11,.LMM256PackBytesControlDataPermPS[rip]

        SetMaxInt8ValueDW r8d,\DataType\()
        sub     r8d,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm6,r8d
        vbroadcastss ymm6,xmm6

        SetMinInt8ValueDW r8d,\DataType\()
        sub     r8d,DWORD PTR QLinearBinaryElementwiseFrame_ZeroPointC[rsp]
        cvtsi2ss xmm7,r8d
        vbroadcastss ymm7,xmm7
        mov rax,QWORD PTR 8

.LQLinear\OpName\()Avx2\DataType\()Process8Loop:
        test    rcx,rcx
        jz      .LQLinear\OpName\()Avx2\DataType\()Exit

        UnpackBytesDWords ymm8,[rdi],\DataType\()
        UnpackBytesDWords ymm9,[rdx],\DataType\()
        vpsubd  ymm9,ymm9,ymm4
        vpsubd  ymm8,ymm8,ymm3
        vcvtdq2ps ymm8,ymm8
        vcvtdq2ps ymm9,ymm9
        vmulps  ymm8,ymm8,ymm0
        vmulps  ymm9,ymm9,ymm1

        \OpInstruction  ymm8,ymm8,ymm9

        # Quantize
        vdivps  ymm8,ymm8,ymm2
        add     rdi,rax
        add     rdx,rax
        vmaxps  ymm8,ymm8,ymm7
        vminps  ymm8,ymm8,ymm6
        vcvtps2dq ymm8,ymm8
        vpaddd  ymm8,ymm8,ymm5
        # pack 32bits integers into 8bit integers
        vpshufb ymm8,ymm8,ymm10
        vpermps ymm8,ymm11,ymm8

        sub     rcx,rax
        jb      .LQLinear\OpName\()Avx2\DataType\()StoreLessThan8

        movsd   QWORD PTR [r9],xmm8
        add     r9,rax
        jmp     .LQLinear\OpName\()Avx2\DataType\()Process8Loop

.LQLinear\OpName\()Avx2\DataType\()StoreLessThan8:
        add     rcx,rax
        pextrq  rax,xmm8,0

.LQLinear\OpName\()Avx2\DataType\()StoreLoop:
        mov     BYTE PTR [r9],al
        shr     rax,8
        inc     r9
        dec     rcx
        jnz     .LQLinear\OpName\()Avx2\DataType\()StoreLoop

.LQLinear\OpName\()Avx2\DataType\()Exit:
        vzeroupper
        add     rsp,QLinearBinaryElementwiseFrame_ReturnAddress
        ret

        .endm


//
// Generate the QLinearAdd Avx2 Int8 kernel.
//
QLinearBinaryElementwiseOpAvx2 Int8,Add,vaddps

//
// Generate the QLinearAdd Avx2 UInt8 kernel.
//
QLinearBinaryElementwiseOpAvx2 UInt8,Add,vaddps

        .end
